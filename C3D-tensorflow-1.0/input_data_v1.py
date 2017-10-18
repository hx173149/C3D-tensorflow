# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
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
import os
from six.moves import xrange    # pylint: disable=redefined-builtin
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
mean_file='../crop_mean.npy'
np_mean = np.load(mean_file)

def get_frames_data(filename, num_frames_per_clip=16):
    ''' Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays '''
    ret_arr = []
    s_index = 0
    for parent, dirnames, filenames in os.walk(filename):
        if(len(filenames)<num_frames_per_clip):
            return [], s_index
        filenames = sorted(filenames)
        s_index = random.randint(0, len(filenames) - num_frames_per_clip)
        for i in range(s_index, s_index + num_frames_per_clip):
            image_name = str(filename) + '/' + str(filenames[i])
            img = Image.open(image_name)
            img_data = np.array(img)
            ret_arr.append(img_data)
    return ret_arr, s_index

def read_clip_and_label(filename, batch_size, start_pos=-1, num_frames_per_clip=16, height=228,width=171, shuffle=False):
    lines = open(filename,'r')
    read_dirnames = []
    data = []
    label = []
    batch_index = 0
    next_batch_start = -1
    lines = list(lines)
    # np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])  #(16, 112, 112, 3)
    np_mean = 0
    # Forcing shuffle, if start_pos is not specified
    if start_pos < 0:
        shuffle = True
    if shuffle:
        video_indices = range(len(lines))
        random.seed(time.time())
        random.shuffle(video_indices)
    else:
        # Process videos sequentially
        video_indices = range(start_pos, len(lines))
    for index in video_indices:
        if(batch_index>=batch_size):
            next_batch_start = index
            break
        line = lines[index].strip('\n').split()
        dirname = line[0]
        tmp_label = line[1]
        if not shuffle:
            print("Loading a video clip from {}...".format(dirname))
        tmp_data, _ = get_frames_data(dirname, num_frames_per_clip)   #one clip:(16, 112, 112, 3)
        img_datas = [];
        if(len(tmp_data)!=0):
            for j in range(len(tmp_data)):
                img = Image.fromarray(tmp_data[j].astype(np.uint8))
                if(img.width>img.height):
                    scale = float(height)/float(img.height)
                    img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), height))).astype(np.float32)
                else:
                    scale = float(width)/float(img.width)
                    img = np.array(cv2.resize(np.array(img),(width, int(img.height * scale + 1)))).astype(np.float32)
                # img = img[(img.shape[0] - crop_size)/2:(img.shape[0] - crop_size)/2 + crop_size, (img.shape[1] - crop_size)/2:(img.shape[1] - crop_size)/2 + crop_size,:] - np_mean[j]
                # img = img[(img.shape[0] - crop_size)/2:(img.shape[0] - crop_size)/2 + crop_size, (img.shape[1] - crop_size)/2:(img.shape[1] - crop_size)/2 + crop_size,:]
                # img = np.array(img).astype(np.float32)
                img = np.array(cv2.resize(np.array(img), (128,171))).astype(np.float32)
                img_datas.append(img)
            data.append(img_datas)
            label.append(int(tmp_label))
            batch_index = batch_index + 1
            read_dirnames.append(dirname)
        else:
            raise EOFError
    # pad (duplicate) data/label if less than batch_size
    valid_len = len(data)
    pad_len = batch_size - valid_len
    if pad_len:
        for i in range(pad_len):
            data.append(img_datas)
            label.append(int(tmp_label))

    np_arr_data = np.array(data).astype(np.float32)
    np_arr_label = np.array(label).astype(np.int64)

    return np_arr_data, np_arr_label, next_batch_start, read_dirnames, valid_len

def cropCenter(img, height, width):
    h,w,c = img.shape
    # print('h,w,c:', h,w,c)
    dx = (h-height)//2
    dy = (w-width )//2

    y1 = dy
    y2 = y1 + height
    x1 = dx
    x2 = x1 + width
    # img = img[x1:x2,y1:y2,:]
    img = img[y1:y2,x1:x2,:]
    # print('img.shape:',img.shape)
    return img
def RandomCrop(rand_seed,img, top,left,height=224, width=224,u=0.5,aug_factor=9/8):
    #first zoom in by a factor of aug_factor of input img,then random crop by(height,width)
    # if rand_seed < u:
    if 1:
        # h,w,c = img.shape
        # img = cv2.resize(img, (round(aug_factor*w), round(aug_factor*h)), interpolation=cv2.INTER_LINEAR)
        # h, w, c = img.shape

        new_h, new_w = height,width

        # top = np.random.randint(0, h - new_h)
        # left = np.random.randint(0, w - new_w)

        img = img[top: top + new_h,
              left: left + new_w]
    return img
def randomHorizontalFlip(rand_seed,img, u=0.5):
    if rand_seed < u:
        img = cv2.flip(img,1)  #np.fliplr(img)  #cv2.flip(img,1) ##left-right
    return img
def normalize(arr):
    arr=arr.astype('float32')
    if arr.max() > 1.0:
        arr/=255.0
    return arr
def sub_mean(batch_arr):
    # print('batch_arr.shape',batch_arr.shape)  #(32, 16, 112, 112, 3)
    for j in range(len(batch_arr)):
        batch_arr[j] -= np_mean
    return batch_arr
def train_aug(batch,is_train=True,Crop_heith=224,Crop_width=224,norm=True):
    new_batch=np.zeros((batch.shape[0],batch.shape[1],Crop_heith,Crop_width,3))
    # (16, 16, 112, 112, 3)
    rand_seed=random.random()
    random.seed(5)
    for i in range(batch.shape[0]):
        h, w, c = batch.shape[2:]
        new_h, new_w = Crop_heith, Crop_width
        dx = (h - Crop_heith) // 2
        dy = (w - Crop_width) // 2
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        for j in range(batch.shape[1]):
            if is_train:
                new_batch[i, j, :, :, :] = RandomCrop(rand_seed,batch[i, j, :, :, :],top,left,
                                                    height=Crop_heith, width=Crop_width)
                new_batch[i, j, :, :, :] = randomHorizontalFlip(rand_seed,new_batch[i, j, :, :, :])
            else:
                new_batch[i, j, :, :, :] = cv2.resize(batch[i, j, :, :, :],(Crop_width,Crop_heith))

    # return new_batch
    # return normalize(new_batch)
    if norm:
        return sub_mean(new_batch)
    else:
        return new_batch