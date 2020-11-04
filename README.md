# Space Analytics Engine #
Major project for Topics in Computer Science. Implementing machine learning for ship detection using Mask R-CNN framework.

## Mask R-CNN Implementation ##
Matterport has released a Mask R-CNN implemetation using TensorFlow and Keras in Python that will be used in this project. The original repository can be found at https://github.com/matterport/Mask_RCNN. The modified version for TensorFlow 2 that is used in this project can be found at https://github.com/akTwelve/Mask_RCNN.

## Project setup ##
To run this project in its entirety using an NVIDIA GPU, you will need:
* Airbus ship detection challenge data from https://www.kaggle.com/c/airbus-ship-detection/data
* Mask R-CNN weights trained on COCO dataset from https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
* Python 3.6
* CUDA and cuDNN
  * These must match your TensorFlow version and GPU driver version. Please check https://www.tensorflow.org/install/source#gpu and https://docs.nvidia.com/deploy/cuda-compatibility/index.html#cuda-application-compatibility to ensure that you get the correct CUDA and cuDNN version for your GPU driver version.
* TensorFlow
  * Once CUDA and cuDNN are installed ensure you install TensorFlow version >= 2.0.0. In this project I used version 2.2.0
* Keras 2.4.3
* Scikit image 0.16.2
* Scipy 1.4.1
* CV2
* ImgAug
* Matplotlib

## Training ##
Training on the Airbus ship detection challenge dataset can be done with the training.ipynb notebook. Please modify the paths near the beginning of the notebook to work with your directory structure. You are able to modify the training parameters as you need. Training on the Airbus Ship Detection Challenge Dataset is based on the code from https://github.com/abhinavsagar/kaggle-notebooks/blob/master/ship_segmentation.ipynb

## Testing ##
Testing of the trained model can be done with the testing.ipynb notebook. Please provide the location of the weights you would like to test. This notebook will also evaluate the model against the dataset to calculate the mAP and mAR.

## Pruning ##
Pruning of the trained model can be done with the prune.ipynb notebook. Please provide the location of the weights you would like to prune. Once the weights are pruned this notebook will re-train the model to regain accuracy. You are able to modify the training parameters as you need.
