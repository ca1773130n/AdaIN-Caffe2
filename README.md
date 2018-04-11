# AdaIN-Caffe2

A Caffe2 implementation of the paper "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization".

## Requirements

* Python 3
* Caffe2
* numpy
* skimage
* scipy

## Usage

    python adain_caffe2.py <arguments>

* --vgg_init (-vi) <filepath> : VGG-19 (normalized) Caffe2 init net PB file
* --vgg_predict (-vp) <filepath> : VGG-19 (normalized) Caffe2 predict net PB file
* --decoder_init (-di) <filepath> : Decoder Caffe2 init net PB file
* --decoder_predict (-dp) <filepath> : Decoder Caffe2 predict net PB file
* --content (-c) <filepath> : input content image (JPG/PNG)
* --content_size (-cs) <num_pixels> : resize resolution (short side) for the content image (default: 512)
* --style (-s) <filepath> : input style image (JPG/PNG)
* --style_size (-ss) <num_pixels> : resize resolution (short side) for the style image (default: 512)
* --gpu (-g) <gpuid> : GPU device ID for CUDNN (default: -1, CPU)

## Acknowledgement

Tested environment
 * Ubuntu 16.04
 * Anaconda Python 3.6
