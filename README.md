# C3D-tensorflow

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

5. If you want to test my pre-trained model, you need to download my model from here: https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0

## Run command:

1. `python train_c3d_ucf101.py` will train C3D model. The trained model will saved in `models` directory.
2. `python predict_c3d_ucf101.py` will test C3D model on a validation data set.



Top-1 accuracy of 72.6% should be achieved for the validation dataset with this code and pre-trained from the sports1M model. You can download my pretrained UCF101 model and mean file from here:
https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0
https://www.dropbox.com/s/zvco2rfufryivqb/conv3d_deepnetA_sport1m_iter_1900000_TF.model?dl=0
baiduyun :http://pan.baidu.com/s/1nuJe8vn


##  Experiment result
From [the original paper](https://arxiv.org/pdf/1611.05431.pdf):

| cardinality | widen_factor | parameters |  Error cifar10   |   error cifar100  | default |
|:-----------:|:------------:|:----------:|:----------------:|:-----------------:|:-------:|
|      8      |       4      |    34.4M   |       3.65       |       17.77       |    x    |
|      16     |      64      |    68.1M   |       3.58       |       17.31       |         |

## Trained models
|   Model             |   Description     |  Download   |
| ------------------- | ----------------- | ------------|
| C3D sports1M        |  ~4.1  |  23.9        | ![C3D sports1M ](https://www.dropbox.com/s/zvco2rfufryivqb/conv3d_deepnetA_sport1m_iter_1900000_TF.model?dl=0)       |
| C3D UCF101 split1   |  ~4.1  |  22.2        | ![C3D UCF101 split1](https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0)       |
| split1 meanfile     |  ~7.8  |  22.0        | ![UCF101 split1 meanfile](https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0)      |
| everything above    |  ~7.8  |  21.2        | ![baiduyun](http://pan.baidu.com/s/1nuJe8vn)      |




## References:

- Thanks the author [Du tran][4]'s code: [C3D-caffe][5]
- [C3D: Generic Features for Video Analysis][6]


[1]: https://www.tensorflow.org/
[2]: http://pillow.readthedocs.io/en/3.1.x/reference/Image.html
[3]: http://crcv.ucf.edu/data/UCF101.php
[4]: https://github.com/dutran
[5]: https://github.com/facebook/C3D
[6]: http://vlg.cs.dartmouth.edu/c3d/
