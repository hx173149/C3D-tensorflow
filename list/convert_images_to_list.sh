#!/bin/bash

# convert the images folder to the list file
#   Usage:
#       ./convert_images_to_list.sh path/to/video/ filename
#   Example Usage:
#       ./convert_images_to_list.sh ~/document/videofile kth_train
#   Example Output(kth_train.list):
#       /Volumes/passport/datasets/action_kth/origin_images/boxing/person01_boxing_d1_uncomp 0
#       /Volumes/passport/datasets/action_kth/origin_images/boxing/person01_boxing_d2_uncomp 0
#       ...
#       /Volumes/passport/datasets/action_kth/origin_images/handclapping/person01_handclapping_d1_uncomp 1
#       /Volumes/passport/datasets/action_kth/origin_images/handclapping/person01_handclapping_d2_uncomp 1
#       ...

COUNT=-1
for folder in $1/*
do
    COUNT=$[ $COUNT + 1 ]
    for imagesFolder in "$folder"/*
    do
        echo "$imagesFolder" $COUNT >> $2.list
    done
done