# LIBRARIES USED
import json
import torch
import torchvision.models as models
import caption

# MACROS
word_map = r"./WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"
model = r"./BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"
beam_size = 5
device = "cpu"

# INFERENCE FUNCTION


def inference(encoder, decoder, img, word_map=word_map, beam_size=beam_size):
    seq, alphas = caption.caption_image_beam_search(
        encoder, decoder, img, word_map, beam_size)
    alphas = torch.FloatTensor(alphas)
    words = [rev_word_map[ind] for ind in seq]
    sentence = ""
    for word in words:
        sentence = sentence + " " + word
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
        # resnet = resnext101_32x8d()
        resnet = models.resnet101()
        # resnet.load_state_dict(torch.load("resnet101-2.pth"))

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = torch.nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size))

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
#         images = quantization.QuantStub(images)
        # out = quantization.QuantStub(images)
        # (batch_size, 2048, image_size/32, image_size/32)
        out = self.resnet(images)
        # out = quantization.DeQuantStub(out)
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


def sparsity(encoder):
    total_encoder_sparsity = 0
    den = 0
    num = 0
    for name, module in encoder.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print("Sparsity in {} (shape = {}): {:.2f}%".format
                  (name, module.weight.shape,
                   100. * float(torch.sum(module.weight == 0)) /
                   float(module.weight.nelement())
                   ))
            num += float(torch.sum(module.weight == 0))
            den += float(module.weight.nelement())
        elif isinstance(module, torch.nn.Linear):
            print("Sparsity in {} (shape = {}): {:.2f}%".format
                  (name, module.weight.shape,
                   100. * float(torch.sum(module.weight == 0)) /
                   float(module.weight.nelement())
                   ))
            num += float(torch.sum(module.weight == 0))
            den += float(module.weight.nelement())
    total_encoder_sparsity = 100.*num/den
    print("Total sparsity in the pruned encoder: {:.2f}%".format(
        total_encoder_sparsity))


# Loading the models
myencoder = Encoder()
myencoder = myencoder.to(device)
myencoder.eval()
myencoder.load_state_dict(torch.load("pruned_model.pth"))
decoder = torch.load("decoderQuantized.pth")
# decoder.eval()

sparsity(myencoder)

# Load word map (word2ix)
with open(word_map, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

while(True):
    try:
        myimg = input("image path: ")
        print(inference(myencoder, decoder, myimg, word_map, beam_size))
    except(KeyboardInterrupt):
        break
