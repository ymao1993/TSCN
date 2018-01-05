#!/bin/python
# split train and development data
import numpy
import os
import caffe
import argparse
import cv2
import os, errno

model_def = 'models/Vgg16/VGG_ILSVRC_16_layers_deploy.prototxt'
model_weights = 'models/Vgg16/VGG_ILSVRC_16_layers.caffemodel'
model_mean_img = 'models/Vgg16/places365CNN_mean.binaryproto'
feature_layer_name = "fc7"


def mkdirs(newdir, mode=0777):
    try: os.makedirs(newdir, mode)
    except OSError, err:
        # Reraise the error unless it's about an already existing directory 
        if err.errno != errno.EEXIST or not os.path.isdir(newdir): 
            raise


def forward(net, imgs):
    count = len(imgs)
    net.blobs['data'].reshape(count,  # batch size
                              3,  # 3-channel (BGR) images
                              224, 224)  # image size is 224x224
    net.blobs['data'].data[...] = imgs
    net.forward(end=feature_layer_name)
    features = numpy.array(net.blobs[feature_layer_name].data)
    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_frames_folder')
    parser.add_argument('output_folder', help="output folder to dump pooled fc7 features for each video frame")
    parser.add_argument('batch_size', type=int, default=32,
                        help="output file to dump pooled fc7 features for each frame")
    parser.add_argument('gpu_idx', type=int, default=0,
                        help='index of the GPU')
    args = parser.parse_args()
    video_frames_folder_path = args.video_frames_folder
    output_folder = args.output_folder
    batch_size = args.batch_size
    gpu_idx = args.gpu_idx

    # set up gpu
    caffe.set_mode_gpu()
    caffe.set_device(gpu_idx)

    # initialize models
    print('loading model weights...')
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # load the mean image (as distributed with Caffe) for subtraction
    print('loading mean image...')
    blob = caffe.proto.caffe_pb2.BlobProto()
    mu = open(model_mean_img, 'rb').read()
    blob.ParseFromString(mu)
    mu = numpy.array(caffe.io.blobproto_to_array(blob))
    mu = mu.mean(0).mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))     # move image channels to outermost dimension
    transformer.set_mean('data', mu)                 # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)           # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    for root, _, files in os.walk(video_frames_folder_path):
        files = sorted(files)
        if len(files) == 0: continue
        output_file_path = root.replace(video_frames_folder_path, output_folder) + '.txt'
        output_dir = output_file_path[0:output_file_path.rfind('/')]
        features = []
        imgs = []
        count = 0
        for file in files:
            if not (file.endswith('.jpg') or file.endswith('.png')):
                continue
            file_path = os.path.join(root, file)
            print('extracting fc7 feature of %s to %s' % (file_path, output_file_path))
            img = caffe.io.load_image(file_path)
            img = cv2.resize(img, (224, 224))
            imgs.append(transformer.preprocess('data', img))
            count += 1
            if count % batch_size == 0:
                features.extend(forward(net, imgs))
                imgs = []
        if count % batch_size != 0:
            features.extend(forward(net, imgs))
        assert count == len(features), 'count != len(features) (%d != %d)' % (count, len(features))
        if count != 0:
            mkdirs(output_dir)
            fout = open(output_file_path, 'w')
            for i in xrange(0, len(features)):
                fout.write(','.join([str(x) for x in features[i]]) + '\n')
            fout.close()
