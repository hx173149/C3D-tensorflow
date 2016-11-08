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

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile
import array

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import PIL.Image as Image
import random
import numpy as np

import cv
import cv2

from cStringIO import StringIO
import time

def getframesdata(filename,frames_num):
  minist_ret = []
  ret_arr = []
  s_index = 0
  for parent,dirnames,filenames in os.walk(filename):
    if(len(filenames)<frames_num):
      return [],s_index
    filenames = sorted(filenames)
    s_index = random.randint(0,len(filenames)-frames_num)
    for i in range(s_index,s_index+frames_num):
      image_name = str(filename) + '/' + str(filenames[i])
      img = Image.open(image_name)
      img_data = np.array(img)
      ret_arr.append(img_data)
  return ret_arr,s_index


def CropImageToSquare(src_img): # moses add image crop
    # crop
    if src_img.size[0] > src_img.size[1]:
      begin = int((src_img.size[0] - src_img.size[1]) / 2)
      end = int((src_img.size[0] + src_img.size[1]) / 2)
      src_img = src_img.crop((begin, 0, end, src_img.size[1]))
    elif src_img.size[0] < src_img.size[1]:
      begin = int((src_img.size[1] - src_img.size[0]) / 2)
      end = int((src_img.size[1] + src_img.size[0]) / 2)
      src_img = src_img.crop((0, begin, src_img.size[0], end))
    src_img = cv2.resize(np.array(src_img), (112, 112))
    return src_img


def ReadTestDataLabelFromFile(filename,batch_size,start_pos):
  lines = open(filename,'r')
  read_dirnames = []
  data = []
  label = []
  batch_index = 0
  lines = list(lines)
  np_mean = np.load('crop_mean.npy').reshape([16,112,112,3])
  start_indexs = []
  for index in range(start_pos,len(lines)):
    if(batch_index>=batch_size):
      next_batch_start = index 
      break
    line = lines[index].strip('\n').split()
    dirname = line[0]
    tmp_label = line[1]
    print (dirname)
    tmp_data,s_index = getframesdata(dirname,16)
    img_datas = [];
    if(len(tmp_data)!=0):
      start_indexs.append(s_index)
      for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if(img.width>img.height):
          scale = 112.0/float(img.height)
          img = np.array(cv2.resize(np.array(img),(int(img.width*scale+1),112))).astype(np.float32)
        else:
          scale = 112.0/float(img.width)
          img = np.array(cv2.resize(np.array(img),(112,int(img.height*scale+1)))).astype(np.float32)
        img = img[(img.shape[0]-112)/2:(img.shape[0]-112)/2+112,(img.shape[1]-112)/2:(img.shape[1]-112)/2+112,:]-np_mean[j]
        img_datas.append(img)
      data.append(img_datas)
      label.append(tmp_label)
      batch_index = batch_index + 1
      read_dirnames.append(dirname)
  np_arr_data = np.array(data)
  np_arr_label = np.array(label)
  return np_arr_data,np_arr_label,next_batch_start,read_dirnames 

def ReadDataLabelFromFile(filename,batch_size):
  lines = open(filename,'r')
  data = []
  label = []
  batch_index = 0
  lines = list(lines)
  random.seed(time.time())
  random.shuffle(lines)
  np_mean = np.load('crop_mean.npy').reshape([16,112,112,3])
  for line in lines:
    if(batch_index>=batch_size):
      break
    line = line.strip('\n').split()
    dirname = line[0]
    tmp_label = line[1]
    tmp_data,_ = getframesdata(dirname,16)
    img_datas = [];
    if(len(tmp_data)!=0):
      for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if(img.width>img.height):
          scale = 112.0/float(img.height)
          img = np.array(cv2.resize(np.array(img),(int(img.width*scale+1),112))).astype(np.float32)
        else:
          scale = 112.0/float(img.width)
          img = np.array(cv2.resize(np.array(img),(112,int(img.height*scale+1)))).astype(np.float32)
        img = img[(img.shape[0]-112)/2:(img.shape[0]-112)/2+112,(img.shape[1]-112)/2:(img.shape[1]-112)/2+112,:]-np_mean[j]
        img_datas.append(img)
      data.append(img_datas)
      label.append(int(tmp_label))
      batch_index = batch_index + 1
  np_arr_data = np.array(data)
  np_arr_data = np_arr_data.astype(numpy.float32)
  np_arr_label = np.array(label)
  np_arr_label = np_arr_label.astype(numpy.int64)
  return np_arr_data,np_arr_label
