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


def extract_resnet():
    # bring in the resnet model!
    model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    
    # extracting feature data for all of the movies
    # iterate though each movie in the folder
    num_movies = 14
    max_timesteps = 5564
    data_dim = 30720 
    max_width = 1280
    max_height = 720
    
    # set up master array
    resnet50_feature_data = np.zeros([num_movies, max_timesteps, data_dim])
    
    mov = 0 
    # go through the movie files
    for file in os.listdir('movies/'): # for gcp
        count = 0 # keeping track of second number
    
        filename = os.fsdecode(file)
        
        vidcap = cv2.VideoCapture('movies/{}'.format(filename)) # for gcp
            
        framerate = vidcap.get(cv2.CAP_PROP_FPS)
            
        valid, frm = vidcap.read()
            
        # go through each frame in the movie
        while valid == True:
            
            if (vidcap.get(cv2.CAP_PROP_POS_FRAMES) % round(framerate) == 0): #going to miss frame from the last partial second
                # extract the image data
                test_img = frm
                print(test_img.shape)
                # pad the image to that it is of max width and max height
                test_img = pad(test_img,(max_height, max_width,3), (0,0,0)) # no offset
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
    
            valid, frm = vidcap.read()
    
        print(mov)
        mov = mov + 1
    return resnet50_feature_data