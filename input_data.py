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

IMAGE_SIZE = 112
#import FileClient
from cStringIO import StringIO
import time


def CropImageToSquare(src_img): # moses add image crop
    if src_img.size[0] == IMAGE_SIZE and src_img.size[1] == IMAGE_SIZE:
        return src_img
    # crop
    if src_img.size[0] > src_img.size[1]:
      begin = int((src_img.size[0] - src_img.size[1]) / 2)
      end = int((src_img.size[0] + src_img.size[1]) / 2)
      src_img = src_img.crop((begin, 0, end, src_img.size[1]))
    elif src_img.size[0] < src_img.size[1]:
      begin = int((src_img.size[1] - src_img.size[0]) / 2)
      end = int((src_img.size[1] + src_img.size[0]) / 2)
      src_img = src_img.crop((0, begin, src_img.size[0], end))
    src_img = src_img.resize((IMAGE_SIZE, IMAGE_SIZE))
    return src_img


def getframesdata(filename,frames_num):
  #frames_num = 8 
  minist_ret = []
  ret_arr = []
  for parent,dirnames,filenames in os.walk(filename):
    if(len(filenames)<frames_num):
      return []
    filenames = sorted(filenames)
    s_index = random.randint(0,len(filenames)-frames_num)
    for i in range(s_index,s_index+frames_num):
      image_name = str(filename) + '/' + str(filenames[i])
      img = Image.open(image_name)
      img = CropImageToSquare(img)
      img_data = np.array(img)
      ret_arr.append(img_data)
  return ret_arr
  #return img_data 

def ReadDataLabelFromFile_16(filename,batch_size):
  lines = open(filename,'r')
  data = []
  label = []
  batch_index = 0
  lines = list(lines)
  random.seed(time.time())
  random.shuffle(lines)
  for line in lines:
    if(batch_index>=batch_size):
      break
    line = line.strip('\n').split()
    dirname = line[0]
    tmp_label = line[1]
    tmp_data = getframesdata(dirname,16)
    if(len(tmp_data)!=0):
      data.append(tmp_data)
      label.append(int(tmp_label))
      batch_index = batch_index + 1
  np_arr_data = np.array(data)
  np_arr_data = np_arr_data.astype(numpy.float32)
  np_arr_data = numpy.multiply(np_arr_data, 1.0 / 255.0)
  np_arr_label = np.array(label)
  np_arr_label = np_arr_label.astype(numpy.int64)
  return np_arr_data,np_arr_label


def ReadTestDataLabelFromFile(filename,batch_size,start_pos):
  lines = open(filename,'r')
  read_dirnames = []
  data = []
  label = []
  batch_index = 0
  lines = list(lines)
  next_batch_start = start_pos + batch_size
  for index in range(start_pos,len(lines)):
    if(batch_index>=batch_size):
      #next_batch_start = start_pos + index 
      break
    line = lines[index].strip('\n').split()
    dirname = line[0]
    tmp_label = line[1]
    tmp_data = getframesdata(dirname,16)
    #print (line[0]) + ': ' + str(len(tmp_data))
    if(len(tmp_data)!=0):
      data.append(tmp_data)
      label.append(int(tmp_label))
      batch_index = batch_index + 1
      read_dirnames.append(dirname)
  print (len(data))
  np_arr_data = np.array(data)
  np_arr_data = np_arr_data.astype(numpy.float32)
  np_arr_data = numpy.multiply(np_arr_data, 1.0 / 255.0)
  np_arr_label = np.array(label)
  return np_arr_data,np_arr_label,next_batch_start,read_dirnames
