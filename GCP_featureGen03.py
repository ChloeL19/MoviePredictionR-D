#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 16:18:11 2018

@author: chloeloughridge

Goal is to create InceptionV3 feature data for AI Grant project.
"""

from keras.applications.resnet50 import ResNet50
import cv2
import numpy as np
import os

model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

# function for padding each input image
def pad(array, reference_shape, offsets):
    """
    array: Array to be padded
    reference_shape: tuple of size of ndarray to create
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    will throw a ValueError if offsets is too big and the reference_shape cannot handle the offsets
    """

    # Create an array of zeros with the reference shape
    result = np.zeros(reference_shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result


# extracting feature data for all of the movies
# iterate though each movie in the folder
num_movies = 14
max_width = 1280
max_height = 720

# set up master array
resnet50_feature_data = np.zeros([14, 5564, 30720])

for num in range(num_movies):
    count = 0
    mov = num 
    for file in os.listdir('./frames/{}/'.format(mov)):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            # extract the image data
            test_img = cv2.imread(os.path.join('./frames/{}/'.format(mov), filename))
            # pad the image to that it is of max width and max height
            test_img = pad(test_img,(max_width, max_height,3), (0,0,0)) # no offset
            data = (test_img/255)[np.newaxis, :, :, :]
            # send image data through the resnet model
            intermediate_output = model.predict(data)
            # reshape model output
            feature_data = intermediate_output.flatten()
            #print(feature_data.shape)
            # slot into the master away
            resnet50_feature_data[mov, count, :feature_data.shape[0]] = feature_data
            print(count)
            count = count + 1
            continue
        else:
            continue
    print(mov)