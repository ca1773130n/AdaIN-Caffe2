import skimage.io
import skimage.transform

import caffe2
from caffe2.proto import caffe2_pb2
import argparse
import numpy as np
import os
import shutil
import skimage
import caffe2.python.predictor.predictor_exporter as pe
from caffe2.python import core, workspace
from scipy.misc import imread, imresize, imsave
import ntpath

from caffe2.python import (
    brew,
    core,
    model_helper,
    optimizer,
    workspace,
)

def load_image(img_path,scale,crop,mean=0.5):
    # load and transform image
    img = skimage.img_as_float(skimage.io.imread(img_path)).astype(np.float32)
    h, w, _ = img.shape

    if h < w:
        ratio = h / scale
        resizeh = scale
        resizew = round(w / ratio)
    else:
        ratio = w / scale
        resizeh = round(h / ratio)
        resizew = scale

    img = rescale(img, resizeh, resizew)

    if crop == True:
        if resizew > resizeh:
            crop_size = resizeh
        else:
            crop_size = resizew
        img = crop_center(img, crop_size, crop_size)

    print("After crop: " , img.shape)

    # switch to CHW
    img = img.swapaxes(1, 2).swapaxes(0, 1)

    # switch to BGR
    #img = img[(2, 1, 0), :, :]

    # add batch size
    img = img[np.newaxis, :, :, :].astype(np.float32)
    print("NCHW: ", img.shape)
    return img

def save_image(filename, image, data_format='channels_first'):
    #image = image[(2, 1, 0), :, :]
    if data_format == 'channels_first':
        image = np.transpose(image, [1, 2, 0]) # CHW --> HWC
    image *= 255
    image = np.clip(image, 0, 255)
    imsave(filename, image.astype(np.uint8))

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def rescale(img, input_height, input_width):
    print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
    print("Model's input shape is %dx%d" % (input_height, input_width))
    aspect = img.shape[1]/float(img.shape[0])
    print("Orginal aspect ratio: " + str(aspect))
    imgScaled = skimage.transform.resize(img, (input_height, input_width))
    print("New image shape:" + str(imgScaled.shape) + " in HWC")
    return imgScaled

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

ntpath.basename("a/b/c")
core.GlobalInit(['caffe2', '--caffe2_log_level=0'])

parser = argparse.ArgumentParser()
parser.add_argument('-vi', '--vgg_init', dest='vgg19_init_pb', type=str, default=None, help='VGG-19 init protobuf binary file (.pb)')
parser.add_argument('-vp', '--vgg_predict', dest='vgg19_predict_pb', type=str, default=None, help='VGG-19 predict protobuf binary file (.pb)')
parser.add_argument('-di', '--decoder_init', dest='decoder_init_pb', type=str, default=None, help='Decoder init protobuf binary file (.pb)')
parser.add_argument('-dp', '--decoder_predict', dest='decoder_predict_pb', type=str, default=None, help='Decoder predict protobuf binary file (.pb)')
parser.add_argument('-c', '--content', dest='content_image', type=str, default=None, help='Content image file')
parser.add_argument('-cs', '--content_size', dest='content_size', type=int, default=512, help='Resize scale (short side) for the content image')
parser.add_argument('-s', '--style', dest='style_image', type=str, default=None, help='Style image file')
parser.add_argument('-ss', '--style_size', dest='style_size', type=int, default=512, help='Resize scale (short side) for the style image')
parser.add_argument('-g', '--gpu', dest='gpu_id', type=int, default=-1, help='GPU device ID for CuDNN (default: -1, CPU)')
args = parser.parse_args()

device_opts = core.DeviceOption(caffe2_pb2.CPU if args.gpu_id == -1 else caffe2_pb2.CUDA)
with core.DeviceScope(device_opts):
    vgg_init_net = caffe2_pb2.NetDef()
    vgg_predict_net = caffe2_pb2.NetDef()
    decoder_init_net = caffe2_pb2.NetDef()
    decoder_predict_net = caffe2_pb2.NetDef()

    with open(args.vgg19_init_pb, mode='rb') as f:
        vgg_init_net.ParseFromString(f.read())
    with open(args.vgg19_predict_pb, mode='rb') as f:
        vgg_predict_net.ParseFromString(f.read())
    with open(args.decoder_init_pb, mode='rb') as f:
        decoder_init_net.ParseFromString(f.read())
    with open(args.decoder_predict_pb, mode='rb') as f:
        decoder_predict_net.ParseFromString(f.read())

    vgg_init_net.name = 'vgg_init_net'
    vgg_predict_net.name = 'vgg_predict_net'
    decoder_init_net.name = 'decoder_init_net'
    decoder_predict_net.name = 'decoder_predict_net'

    cs_op = core.CreateOperator(
        "ChannelStats",
        ["X"],
        ["sum", "sum_sq"]
    )
    
    in_op = core.CreateOperator(
        "InstanceNorm",
        ["X", "scale", "bias"],
        ["Y"],
        order = "NCHW",
        is_test = True,
        epsilon = 1e-5
    )
    
    content_img = load_image(args.content_image, args.content_size, False)
    style_img = load_image(args.style_image, args.style_size, False)
   
    vgg19_input_layer_name = '0'
    vgg19_conv41_layer_name = '51'

    workspace.CreateNet(vgg_init_net)
    workspace.RunNetOnce(vgg_init_net)
    workspace.CreateNet(vgg_predict_net)

    # VGG-19 feature extraction for both content and style
    workspace.FeedBlob(vgg19_input_layer_name, content_img, device_opts)
    workspace.RunNet(vgg_predict_net.name)
    content_encoded = workspace.FetchBlob(vgg19_conv41_layer_name)
    
    workspace.FeedBlob(vgg19_input_layer_name, style_img, device_opts)
    workspace.RunNet(vgg_predict_net.name)
    style_encoded = workspace.FetchBlob(vgg19_conv41_layer_name)
   
    # Get mean and variance of style feature maps
    workspace.FeedBlob("X", style_encoded, device_opts)
    workspace.RunOperatorOnce(cs_op)
    style_sum = workspace.FetchBlob("sum")
    style_sum_sq = workspace.FetchBlob("sum_sq")
    style_mean = style_sum / (style_encoded.shape[2] * style_encoded.shape[3])
    style_variance = np.sqrt(style_sum_sq / (style_encoded.shape[2] * style_encoded.shape[3]))
    print("Encoded content feature map's shape: ", content_encoded.shape)
    print("Encoded style feature map's shape: ", style_encoded.shape)
    
    # AdaIN
    workspace.FeedBlob("X", content_encoded, device_opts)
    workspace.FeedBlob("bias", style_mean, device_opts)
    workspace.FeedBlob("scale", style_variance, device_opts)
    workspace.RunOperatorOnce(in_op)
    instance_normalized = workspace.FetchBlob("Y")
    print("Instance normalized tensor's shape: ", instance_normalized.shape)
    
    # Decode and save to output image
    decoder_predictor = workspace.Predictor(decoder_init_net, decoder_predict_net)
    style_transferred_decoded = decoder_predictor.run({'0': instance_normalized})
    print("Style transferred decoded tensor's shape: ", style_transferred_decoded[0].shape)
   
    content_filename = path_leaf(args.content_image)
    style_filename = path_leaf(args.style_image)
    filename = '%s_stylized_by_%s.png' % (content_filename, style_filename)
    save_image(filename, style_transferred_decoded[0][0], data_format='channels_first')
    print('Output image saved at', filename)
