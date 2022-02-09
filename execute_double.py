# LIBRARIES USED
import json
import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch_pruning as tp
import torchvision.models as models
from torchsummary import summary
import time
#from caption.py
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
#from scipy.misc import imread, imresize ##########depracated
from imageio import imread
from skimage.transform import resize
from PIL import Image
import torch.quantization 
import ResnetQuant
import camera
from enum import Enum
 # MACROS
img = r"./img_test3.jpeg"
word_map = r"./WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"
model = r"./BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"
beam_size = 5
device = "cpu"
torch.backends.quantized.engine = "qnnpack"
# INFERENCE FUNCTIONS
def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = beam_size
    vocab_size = len(word_map)
    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
   # img = imresize(img, (256, 256))  #- comment out this line since it always generates the same caption
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    tic = time.time()
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    t_encode = time.time() - tic
    print('Elapsed time for encoding:', t_encode, ' seconds.\n')
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        ###############################
        prev_word_inds = prev_word_inds.long()
        next_word_inds = next_word_inds.long()
        ###############################
        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1
    t_decode = time.time() - t_encode - tic
    print('Elapsed time for decoding and generating sequences:', t_decode, ' seconds.\n')
    T = time.time()-tic
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]
    print('Elapsed time for inference:', T, '\n')
    return seq, alphas


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(int(np.ceil(len(words) / 5.)), 5, t + 1)    ### added int func

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show(block=False)


def inference(encoder, decoder, img, rev_word_map, word_map=word_map, beam_size=beam_size, visualize = False):
    seq, alphas = caption_image_beam_search(encoder, decoder, img, word_map, beam_size)
    alphas = torch.FloatTensor(alphas)
    words = [rev_word_map[ind] for ind in seq]
    sentence = ""
    for word in words:
        sentence = sentence + " " + word
    if visualize == True:
    	alphas = torch.FloatTensor(alphas)
	# Visualize caption and attention of best sequence
    	visualize_att(img, seq, alphas, rev_word_map, smooth=True)    
    return sentence
    
# ENCODER CLASS
class Encoder(torch.nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
        #resnet = resnext101_32x8d()
        resnet = models.resnet101()
        # resnet.load_state_dict(torch.load("resnet101-2.pth"))

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = torch.nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size))

        self.fine_tune()
        #self.qconfig = quantization.get_default_qconfig('fbgemm')
        # set the qengine to control weight packing
        #torch.backends.quantized.engine = 'fbgemm'

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
#         images = quantization.QuantStub(images)
        #out = quantization.QuantStub(images)
        # (batch_size, 2048, image_size/32, image_size/32)
        out = self.resnet(images)
        #out = quantization.DeQuantStub(out)
        # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = self.adaptive_pool(out)
        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        out = out.permute(0, 2, 3, 1)
#         out = quantization.DeQuantStub(out)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

# State enum class
class State(Enum):
    INIT = 1
    IMAGE = 2
    VIDEO = 3

# main function
def main():
    # MACROS
    #img = r"./img_test3.jpeg"
    word_map = r"./WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"
    model = r"./BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"
    beam_size = 5
    device = "cpu"
    # Loading the models
    myencoder = Encoder()
    myencoder = torch.load('EncoderPruned.pth')
    myencoder = myencoder.to(device)
    myencoder.eval()
    decoder = torch.load("DecoderQuantized.pth")
    CAM = camera.Camera(128,128)
    # Load word map (word2ix)
    with open(word_map, 'r') as j:
        word_map = json.load(j)
        rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
    entry_message = "*************************************\n*************************************\nGreetings and welcome!\n*************************************\n*************************************\n"
    print(entry_message)
    currstate = State.INIT
    while(True):
        try:
            if currstate == State.INIT:
                startcmd = input("\nFor image captioning press 'i'. \nFor quasi video captioning / multi image captioning press 'v'. :\n")
                if startcmd == 'i' or startcmd == 'I':
                    currstate = State.IMAGE
                elif startcmd == 'v' or startcmd == 'V':
                    currstate = State.VIDEO
                else:
                    print("\nWrong input!\n")
            elif currstate == State.IMAGE:
                cmd = input("\nPlease enter a command [ 'c' -> Captures and captions an image from camera. | 'x' -> Exit program. ]:\n")
                if cmd == 'c' or cmd == 'C':
                    CAPTURE = CAM.capture(True)
                    plt.clf()
                    sent = inference(myencoder, decoder, CAPTURE, rev_word_map, word_map, beam_size, visualize = True)
                    sent = sent[9:-6]
                    print(sent)
                elif cmd == 'x' or 'X':
                    currstate = State.INIT
                else:
                    print("\nWrong input!\n") 
            elif currstate == State.VIDEO:
                cmd = input("\nPlease enter a command [ 'l' -> Captions a video with l setences. | 'x' -> Exit program. ]:\n")
                if cmd.isdigit():
                    l = int(cmd)
                    print("\nPrinting {} sentence long paragraph for the video.\n".format(l))
                    paragraph = ''
                    for i in range(l):
                        CAPTURE = CAM.capture(True)
                        sent = inference(myencoder, decoder, CAPTURE, rev_word_map, word_map, beam_size, visualize = False)
                        sent = sent[9:-6]
                        paragraph = paragraph + sent + '. '
                    print(paragraph)
                elif cmd == 'x' or cmd == 'X':
                    currstate = State.INIT
                else:
                    print("\nWrong input!\n")
            else:
                currstate = State.INIT            
        except(KeyboardInterrupt):
            break  


if __name__ == '__main__':
    main()
