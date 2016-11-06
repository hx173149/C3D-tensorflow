# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""

import math
import re
import numpy as np

import tensorflow as tf

import random 

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 101 

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 112 
#IMAGE_SIZE = 28 
CHANNELS = 3
#CHANNELS = 1 
FRAMES = 16 
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNELS * FRAMES
#IMAGE_PIXELS = 28*28*1 


"-----------------------------------------------------------------------------------------------------------------------"


def conv3d(name, l_input, w, b):
  return tf.nn.bias_add(tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),b)
def conv3d_valid(name, l_input, w, b):
  return tf.nn.relu(tf.nn.bias_add(tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='VALID'),b), name=name)

def max_pool(name, l_input, k):
  return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

def inference_c3d(_X,_dropout,batch_size,_weights,_biases):
  # Convolution Layer
  conv1 = conv3d('conv1', _X, _weights['wc1'], _biases['bc1'])

  conv1 = tf.nn.relu(conv1,'relu1')
  pool1 = max_pool('pool1', conv1, k=1)

  # Convolution Layer
  conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
  conv2 = tf.nn.relu(conv2,'relu2')
  pool2 = max_pool('pool2', conv2, k=2)

  # Convolution Layer
  conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
  conv3 = tf.nn.relu(conv3,'relu3a')
  conv3 = conv3d('conv3b', conv3, _weights['wc3b'], _biases['bc3b'])
  conv3 = tf.nn.relu(conv3,'relu3b')
  pool3 = max_pool('pool3', conv3, k=2)

  # Convolution Layer
  conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
  conv4 = tf.nn.relu(conv4,'relu4a')
  conv4 = conv3d('conv4b', conv4, _weights['wc4b'], _biases['bc4b'])
  conv4 = tf.nn.relu(conv4,'relu4b')
  pool4 = max_pool('pool4', conv4, k=2)

  # Convolution Layer
  conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
  conv5 = tf.nn.relu(conv5,'relu5a')
  conv5 = conv3d('conv5b', conv5, _weights['wc5b'], _biases['bc5b'])
  conv5 = tf.nn.relu(conv5,'relu5b')

  pool5 = max_pool('pool5', conv5, k=2)
  
  # Fully connected layer
  pool5 = tf.transpose(pool5,perm=[0,1,4,2,3])
  dense1 = tf.reshape(pool5, [batch_size, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
  dense1 = tf.matmul(dense1, _weights['wd1']) + _biases['bd1'] 
  
  dense1 = tf.nn.relu(dense1, name='fc1') # Relu activation
  dense1 = tf.nn.dropout(dense1, _dropout)

  dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
  dense2 = tf.nn.dropout(dense2, _dropout)
  # Output, class prediction
  out = tf.matmul(dense2, _weights['out']) + _biases['out']
  return out
