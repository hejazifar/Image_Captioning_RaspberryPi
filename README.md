# Image_Captioning_RaspberryPi
In this project we tried to compress a full model for image captioning and put it on Raspberry Pi. The original model is created for computer and we get it from 
this repository: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

To make the project work you need to download the models from the following link: https://drive.google.com/drive/folders/1U9LJyL4YBXLel-nP237_6U8ypZHd4oxH?usp=sharing

In order to run the model you need to choose among the offering models which one you want to use for inferencing an image. The options are quantized encoder, quantized decoder
pruned encoder, normal models.

In case you want to change the models you gave to change the path in execute files, and then you can run the execute files.

For running this project you need to have a camera, you can also use this project to caption an image from the internet but for that you need to do some modifications.


After adjusting the paths and choosing the models of choice the program to execute is execute_double.

**execute_double.py**

1) At the beginning you get a welcoming message.
2) Next you are prompted to choose between doing image captioning or video captioning.
  i) If you chose image captioning you can enter 'c' or 'C' to capture an image from the PiCamera and you should get the captionining sentence along with a describing picture.
  ii) If you chose video captioning you should enter a positive integer and later you should get a paragraph describing what was happening in front of the PiCamera with 
  a number of sentences equal to the input integer
3) You can enter 'x' or 'X' to escape from the current mini program and go to point 2).
4) Press ctrl + 'c' to exit the program definitely.
