#!/bin/bash

# convert the images folder to the test.list and train.list file according to
#   the distribution, command will clear the train.list and test.list files first
#
#   Args:
#       path: the path to the video folder
#       factor: denominator that split the train and test data. if the number 
#               is 4, then 1/4 of the data will be written to test.list and the
#               rest of the data will be written to train.list
#   Usage:
#       ./convert_images_to_list.sh path/to/video 4
#   Example Usage:
#       ./convert_images_to_list.sh ~/document/videofile 4
#   Example Output(train.list and test.list):
#       /Volumes/passport/datasets/action_kth/origin_images/boxing/person01_boxing_d1_uncomp 0
#       /Volumes/passport/datasets/action_kth/origin_images/boxing/person01_boxing_d2_uncomp 0
#       ...
#       /Volumes/passport/datasets/action_kth/origin_images/handclapping/person01_handclapping_d1_uncomp 1
#       /Volumes/passport/datasets/action_kth/origin_images/handclapping/person01_handclapping_d2_uncomp 1
#       ...

> train.list
> test.list
COUNT=-1
for folder in $1/*
do
    COUNT=$[$COUNT + 1]
    for imagesFolder in "$folder"/*
    do
        if (( $(jot -r 1 1 $2)  > 1 )); then
            echo "$imagesFolder" $COUNT >> train.list
        else
            echo "$imagesFolder" $COUNT >> test.list
        fi        
    done
done