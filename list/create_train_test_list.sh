#!/usr/bin/env bash

DATA=train
cat /media/6TB/UCF-101/train_test_split/c3d_ucf101_${DATA}_split1.txt  | awk '{print $1 " " $3}' | sort | uniq > ${DATA}.list

DATA=test
cat /media/6TB/UCF-101/train_test_split/c3d_ucf101_${DATA}_split1.txt  | awk '{print $1 " " $3}' | sort | uniq > ${DATA}.list
