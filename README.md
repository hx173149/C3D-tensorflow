# C3D-tensorflow

This is a repository trying to implement [C3D-caffe][5] on tensorflow,useing models directly converted from original C3D-caffe.    
Be aware that there are about 5% video-level accuracy margin on UCF101 split1  between our implement in tensorflow and  the original C3D-caffe.  

## Requirements:

1. Have installed the tensorflow >= 1.2 version
2. You must have installed the following two python libs:
a) [tensorflow][1]
b) [Pillow][2]
3. You must have downloaded the [UCF101][3] (Action Recognition Data Set)
4. Each single avi file is decoded with 5FPS (it's depend your decision) in a single directory.
    - you can use the `./list/convert_video_to_images.sh` script to decode the ucf101 video files
    - run `./list/convert_video_to_images.sh .../UCF101 5`
5. Generate {train,test}.list files in `list` directory. Each line corresponds to "image directory" and a class (zero-based). For example:
    - you can use the `./list/convert_images_to_list.sh` script to generate the {train,test}.list for the dataset
    - run `./list/convert_images_to_list.sh .../dataset_images 4`, this will generate `test.list` and `train.list` files by a factor 4 inside the root folder

```
database/ucf101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01 0
database/ucf101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c02 0
database/ucf101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c03 0
database/ucf101/train/ApplyLipstick/v_ApplyLipstick_g01_c01 1
database/ucf101/train/ApplyLipstick/v_ApplyLipstick_g01_c02 1
database/ucf101/train/ApplyLipstick/v_ApplyLipstick_g01_c03 1
database/ucf101/train/Archery/v_Archery_g01_c01 2
database/ucf101/train/Archery/v_Archery_g01_c02 2
database/ucf101/train/Archery/v_Archery_g01_c03 2
database/ucf101/train/Archery/v_Archery_g01_c04 2
database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c01 3
database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c02 3
database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c03 3
database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c04 3
database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c01 4
database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c02 4
database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c03 4
database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c04 4
...
```

## Usage

1. `python train_c3d_ucf101.py` will train C3D model. The trained model will saved in `models` directory.
2. `python predict_c3d_ucf101.py` will test C3D model on a validation data set.
3.  `cd ./C3D-tensorflow-1.0 &&python Random_clip_valid.py` will get the random-clip accuracy on UCF101 test set with provided `sports1m_finetuning_ucf101.model`.
4. `C3D-tensorflow-1.0/Random_clip_valid.py` code is compatible with tensorflow 1.0+ , with a little bit different with the old repository
5. IMPORTANT NOTE: when you load the sports1m_finetuning_ucf101.model,you should use the tranpose operation like:` pool5 = tf.transpose(pool5, perm=[0,1,4,2,3])`,or in `Random_clip_valid.py` looks like:`["transpose", [0, 1, 4, 2, 3]]`, 
but if you load `conv3d_deepnetA_sport1m_iter_1900000_TF.model` or `c3d_ucf101_finetune_whole_iter_20000_TF.model`,you don't need tranpose operation,just comment that line code.  

##  Experiment result:
- Note:              
    1.All report results are done specific on UCF101 split1 (train videos:9537,test videos:3783).   
    2.ALL the results are video-level accuracy,unless stated otherwise.   
    3.We follow the same way to extract clips from video as the C3D paper saying:'To extract C3D feature, a video is split into 16 frame long clips with a 8-frame overlap between two consecutive clips.These clips are passed to the C3D network to extract fc6 activations. These clip fc6 activations are averaged to form a 4096-dim video descriptor which is then followed by an L2-normalization'   

- C3D as feature extractor:

|   platform  | feature extractor model | fc6+SVM |  fc6+SVM+L2 norm   | 
|:-----------:|:---------------:|:----------:|:----------------:|
|   caffe     | conv3d_deepnetA_sport1m_iter_1900000.caffemodel|   81.99%    |       83.39%      |
| tensorflow  | conv3d_deepnetA_sport1m_iter_1900000_TF.model  |   79.38%    |       81.44%      |
| tensorflow  | c3d_ucf101_finetune_whole_iter_20000_TF.model  |   79.67%    |       81.33%      |
| tensorflow  | sports1m_finetuning_ucf101.model  |    82.73%   |       85.35%      |

- finetune C3D network on UCF101 split1 with  pre-trained model:

|   platform  | pre-trained model |    train-strategy   |    video-accuracy    |     clip-accuracy     |    random-clip     | 
|:-----------:|:---------------:|:----------------:|:----------------:|:----------------:|:----------------:|
|   caffe     | c3d_ucf101_finetune_whole_iter_20000.caffemodel|    directly test   |    -   |       79.87%     |       -     |
| tensorflow  | c3d_ucf101_finetune_whole_iter_20000_TF.model  |    directly test  |    78.35%   |       72.77%    |       57.15%    |
| tensorflow-A  | conv3d_deepnetA_sport1m_iter_1900000_TF.caffemodel  |    whole finetuning   |    76.0%   |       71%    |       69.8%    |
| tensorflow-B  | sports1m_finetuning_ucf101.model  |    freeze conv,only finetune fc layers   |    79.93%  |       74.65%   |       76.6%     |


- Note:        
    1.the `tensorflow-A` model corresponding to the original C3D model pre-trained on UCF101 provided by @ [hx173149][7] .       
    2.the `tensorflow-B` model is just freeze the conv layers in `tensorflow-A` and finetuning  four more epochs on fc layers with `learning rate=1e-3`.   
    3.the `random-clip` column means random choose one clip from each video in UCF101 test split 1 ,so the result are not so robust.But according to the Law of Large Numbers,we may assume this items is positive correlated to your video-level accuracy.   
    4.it's obvious that it if you do more finetuning work based on `c3d_ucf101_finetune_whole_iter_20000_TF.model`,and you may achieve better performance,i didn't do it because of time limit.  
    5.with no doubt that you can get better result by appropriately finetuning the network   

## Trained models:
|   Model             |   Description     |   Clouds  |  Download   |
| ------------------- | ----------------- |  -------- | ------------|
| C3D sports1M  TF      |C3D sports1M converted from caffe C3D|  Dropbox  |[C3D sports1M ](https://www.dropbox.com/s/zvco2rfufryivqb/conv3d_deepnetA_sport1m_iter_1900000_TF.model?dl=0)       |
| C3D UCF101 TF  |C3D UCF101 trained model converted from caffe C3D|  Dropbox  |[C3D UCF101 ](https://www.dropbox.com/s/u5fxqzks2pkaolx/c3d_ucf101_finetune_whole_iter_20000_TF.model?dl=0 )       |
| C3D UCF101  TF train   |finetuning on UCF101 split1 use C3D sports1M model by  @ [hx173149][7]|  Dropbox  |[C3D UCF101 split1](https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0)       |
| split1 meanfile  TF   | UCF101 split1 meanfile converted from caffe C3D  |  Dropbox  |[UCF101 split1 meanfile](https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0)      |
| everything above    |  all four files above  |  baiduyun |[baiduyun](http://pan.baidu.com/s/1nuJe8vn)      |


## Other verions
- [C3D-estimator-sagemaker](https://github.com/frankgu/C3D-estimator-sagemaker): Use sagemaker to train C3D model.

## References:

- Thanks the author [Du tran][4]'s code: [C3D-caffe][5]
- [C3D: Generic Features for Video Analysis][6]


[1]: https://www.tensorflow.org/
[2]: http://pillow.readthedocs.io/en/3.1.x/reference/Image.html
[3]: http://crcv.ucf.edu/data/UCF101.php
[4]: https://github.com/dutran
[5]: https://github.com/facebook/C3D
[6]: http://vlg.cs.dartmouth.edu/c3d/
[7]:https://github.com/hx173149/C3D-tensorflow
