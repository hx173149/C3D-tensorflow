#!/usr/bin/env python3
# encoding: utf-8
import cv2
import tensorflow as tf
import sys, os, h5py
import numpy as np
import tensorflow.contrib.layers as layers
import random
import  pandas as pd
from random import shuffle
from random import randint
from tqdm import  tqdm
import time
from input_data_v1 import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class C3dModel(object):
    def __init__(self,
            num_class = 101,
            keep_prob = 0.6,
            batch_size = 3,
            epoch=40,
            lr = 1e-4):
        self.IMG_WIDTH = 171
        self.IMG_HEIGHT = 128

        self.CROP_WIDTH = 112
        self.CROP_HEIGHT = 112
        self.graph = tf.Graph()
        self.num_class = num_class
        self.epoch = epoch
        self.CLIP_LENGTH = 16
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        decay_epoch=10   #每5个epoch改变一次学习率
        # train clip: 9537*5 CLIP=5
        # test  clip: 3783*5 CLIP=5
        # train clip: 9537*3 CLIP=3
        # test  clip: 3783*3 CLIP=3
        self.n_step_epoch=int( 9537/batch_size)
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32, [None, self.CLIP_LENGTH, self.CROP_HEIGHT, self.CROP_WIDTH, 3])
            self.labels = tf.placeholder(tf.int64, [batch_size,])

            self.initializer = layers.xavier_initializer()
            self.global_step = tf.Variable(0, trainable = False, name = "global_step")
            self.lr = tf.train.exponential_decay(lr, self.global_step, int(decay_epoch*self.n_step_epoch), 1e-1, True)
            tf.add_to_collection(tf.GraphKeys.GLOBAL_STEP, self.global_step)

    def conv3d(self, inputs, shape, name,w_name,b_name):
        with self.graph.as_default():
            with tf.variable_scope('var_name') as var_scope:
                W = tf.get_variable(name = w_name, shape = shape, initializer = self.initializer, dtype = tf.float32)
                b = tf.get_variable(name = b_name, shape = shape[-1], initializer = tf.zeros_initializer(), dtype = tf.float32)
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
                tf.add_to_collection(tf.GraphKeys.BIASES, b)
            return tf.nn.relu(tf.nn.bias_add(tf.nn.conv3d(inputs, W, strides = [1, 1, 1, 1, 1], padding = "SAME"), b))
            # filter:
            # [filter_depth, filter_height, filter_width, in_channels,out_channels]
    def fc(self, inputs, shape, name,w_name,b_name,activation = True):
        with self.graph.as_default():
            with tf.variable_scope('var_name') as var_scope:
                W = tf.get_variable(name = w_name, shape = shape, initializer = self.initializer, dtype = tf.float32)
                b = tf.get_variable(name = b_name, shape = shape[-1], initializer = tf.zeros_initializer(), dtype = tf.float32)
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
                tf.add_to_collection(tf.GraphKeys.BIASES, b)

            if activation:
                return tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, W), b))
            else:
                return tf.nn.bias_add(tf.matmul(inputs, W), b)

    # netstrucet is an orderdict with form {"conv": [shape, name]}
    def parseNet(self, net, netstruct, istraining = True):
        for key in netstruct:
            if key[0] == "conv":
                net = self.conv3d(net, key[2], key[1],key[3], key[4])
            elif key[0] == "fc":
                net = self.fc(net, key[2], key[1], key[3], key[4],activation = key[-1])
            elif key[0] == "maxpool":
                net = tf.nn.max_pool3d(net, ksize = key[2], strides = key[2], padding = "SAME", name = key[1])
            elif key[0] == "dropout" and istraining:
                net = tf.nn.dropout(net, key[2], name = key[1])
            elif key[0] == "reshape":
                net = tf.reshape(net, key[-1])
            elif key[0] == "softmax":
                net = tf.nn.softmax(net)
            elif key[0] == "transpose":
                net = tf.transpose(net, perm=key[-1])
        return net

    def test(self, test_list, modelpath):
        with self.graph.as_default():
            c3d_net = [
                ["conv", "conv1", [3, 3, 3, 3, 64], 'wc1', 'bc1'],
                ["maxpool", "pool1", [1, 1, 2, 2, 1]],
                ["conv", "conv2", [3, 3, 3, 64, 128], 'wc2', 'bc2'],
                ["maxpool", "pool2", [1, 2, 2, 2, 1]],
                ["conv", "conv3a", [3, 3, 3, 128, 256], 'wc3a', 'bc3a'],
                ["conv", "conv3b", [3, 3, 3, 256, 256], 'wc3b', 'bc3b'],
                ["maxpool", "pool3", [1, 2, 2, 2, 1]],
                ["conv", "conv4a", [3, 3, 3, 256, 512], 'wc4a', 'bc4a'],
                ["conv", "conv4b", [3, 3, 3, 512, 512], 'wc4b', 'bc4b'],
                ["maxpool", "pool4", [1, 2, 2, 2, 1]],
                ["conv", "conv5a", [3, 3, 3, 512, 512], 'wc5a', 'bc5a'],
                ["conv", "conv5b", [3, 3, 3, 512, 512], 'wc5b', 'bc5b'],
                ["maxpool", "pool5", [1, 2, 2, 2, 1]],
                ["transpose", [0, 1, 4, 2, 3]],  #only use it if you restore the sports1m_finetuning_ucf101.model, otherwise uncomment it,(e.g use conv3d_deepnetA_sport1m_iter_1900000_TF.model)
                ["reshape", [-1, 8192]],
                ["fc", "fc1", [8192, 4096], 'wd1', 'bd1', True],
                ["dropout", "dropout1", self.keep_prob],
                ["fc", "fc2", [4096, 4096],'wd2','bd2', True],
                ["dropout", "dropout2", self.keep_prob],
                ["fc", "fc3", [4096, self.num_class],'wout','bout',False],
            ]

            # print(tf.trainable_variables())
            # print(var_list)
            # print(tf.get_collection(tf.GraphKeys.WEIGHTS))

            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.9

            with tf.Session(config=config, graph=self.graph) as sess:
                logits = self.parseNet(self.inputs, c3d_net)
                softmax_logits = tf.nn.softmax(logits)
                # int_label = tf.one_hot(self.labels, self.num_class)
                int_label = self.labels  # [bs,101]-->[bs*4 or 8 or 16,101]
                # int_label=tf.concat(
                #     [int_label,int_label,int_label,int_label,],axis=0)

                # int_label=tf.cast(int_label,dtype=tf.int64)
                task_loss = tf.reduce_sum(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=int_label))
                # task_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = int_label))
                # task_loss = -tf.reduce_sum(int_label*tf.log(logits))
                acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(softmax_logits, axis=-1), int_label), tf.float32))
                right_count = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(softmax_logits, axis=1), int_label), tf.int32))
    
                reg_loss = layers.apply_regularization(layers.l2_regularizer(5e-4),
                                                       tf.get_collection(tf.GraphKeys.WEIGHTS))
                total_loss = task_loss + reg_loss
                # train_var_list = [v for v in tf.trainable_variables() if v.name.find("conv") == -1]
                train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(
                    total_loss, global_step=self.global_step)
                # train_op = tf.train.MomentumOptimizer(self.lr,0.9).minimize(
                #     total_loss, global_step = self.global_step,var_list=train_var_list)
    
    
                total_para = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
                print('total_para:', total_para)  # all CDC9 :28613120  #pool5 27655936

                # train clip:762960
                # test  clip:302640
                init = tf.global_variables_initializer()
                # var_list = [v for v in tf.trainable_variables() if v.name.find("conv") != -1]  # 初始化只加载卷积层参数
                # print(var_list)
                # saver = tf.train.Saver(tf.global_variables())
                sess.run(init)
                saver = tf.train.Saver(tf.trainable_variables())
                # saver.restore(sess, tf.train.latest_checkpoint(modelpath))
                saver.restore(sess, modelpath + "sports1m_finetuning_ucf101.model")
                print("Model Loading Done!")
                step = 0
                print_freq = 2
                next_start_pos = 0
                for one_epoch in range(1):
                    epostarttime = time.time()
                    starttime = time.time()
                    total_v = 0.0
                    test_correct_num = 0
                    for i in tqdm(range(int(3783 / self.batch_size))):
                        step += 1
                        total_v += self.batch_size
                        train_batch, label_batch, next_start_pos, _, _ = read_clip_and_label(
                            filename=test_list,
                            batch_size=self.batch_size,
                            num_frames_per_clip=self.CLIP_LENGTH,
                            height=self.IMG_HEIGHT,
                            width=self.IMG_WIDTH,
                            start_pos=next_start_pos,
                            shuffle=False
                        )
                        assert len(train_batch)==self.batch_size
                        train_batch = train_aug(train_batch, is_train=False, Crop_heith=self.CROP_HEIGHT,
                                                Crop_width=self.CROP_WIDTH,norm=True)
                        val_feed = {self.inputs: train_batch, self.labels: label_batch}
                        test_correct_num += sess.run(right_count, val_feed)
                        print('test acc:', test_correct_num / total_v, 'test_correct_num:', test_correct_num,
                              'total_v:', total_v)
            
if __name__ == "__main__":
    c3dnet = C3dModel()
    c3dnet.test(test_list="../list/test.list",
                 modelpath="../")