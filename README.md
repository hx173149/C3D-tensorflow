# C3D-tensorflow

##requirements:

 1. you must have install the following two python libs:
 a) [tensorflow][1]
 b) [Pillow][2]
 2. you must have download the [UCF101][3] (Action Recognition Data Set)
 3. each single avi file is decoded with 5FPS(it's depend your decision) in a single directory
 like in the "ucf_101_list/train_ucf101.trainVideos":
 4. if you want to test my pretrained model, you need to download my model from here:
 https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0

> database/ucf101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01 0
> database/ucf101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c02 0
> database/ucf101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c03 0
> database/ucf101/train/ApplyLipstick/v_ApplyLipstick_g01_c01 1 
> database/ucf101/train/ApplyLipstick/v_ApplyLipstick_g01_c02 1 
> database/ucf101/train/ApplyLipstick/v_ApplyLipstick_g01_c03 1  
> database/ucf101/train/Archery/v_Archery_g01_c01 2  
> database/ucf101/train/Archery/v_Archery_g01_c02 2  
> database/ucf101/train/Archery/v_Archery_g01_c03 2  
> database/ucf101/train/Archery/v_Archery_g01_c04 2  
> database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c01 3 
> database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c02 3 
> database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c03 3
> database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c04 3 
> database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c01 4 
> database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c02 4 
> database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c03 4 
> database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c04 4

##run command:
    python train_c3d_ucf101.py  
the trained model will saved in models  

    python predict_c3d_ucf101.py  

##experiment result:
I can get a 72.6% accuracy in test dataset with this code and pretrained from the sports1M model, and you can download my pretrained UCF101 model and mean file from here:
https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0

##reference:

 - thanks the author [Du tran][4]'s code: [C3D-caffe][5]
 - [C3D: Generic Features for Video Analysis][6]


  [1]: https://www.tensorflow.org/
  [2]: http://pillow.readthedocs.io/en/3.1.x/reference/Image.html
  [3]: http://crcv.ucf.edu/data/UCF101.php
  [4]: https://github.com/dutran
  [5]: https://github.com/facebook/C3D
  [6]: http://vlg.cs.dartmouth.edu/c3d/
