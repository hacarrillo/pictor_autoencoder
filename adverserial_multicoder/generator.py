import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np # use numpy 1.11
import math
import os
import scipy
import matplotlib.pyplot as plt
import skimage.io as io
from time import sleep as sleep
from cnn_helper import *
import cv2

tfrecords_filename = "D://multicoder_dataset.tfrecords"

input_h = 128
input_w = 128
target_h = 256
target_w = 256
num_channels = 1
batch_size = 1
total_rounds = 95000


# this is for testing with one image
#data_path = "tensorflow_data//CRP//"
data_path = "tensorflow_data//CRP//"
pictor = "cr1_pictg.jpg"
cr1 = "cr1_cropg.jpg"

''' openCV to read in 32 bit float, green channel '''
# the input
_input_ = np.float32(cv2.imread(data_path + pictor,cv2.IMREAD_GRAYSCALE))
_input_ = cv2.normalize(_input_, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
_input_ = cv2.resize(input_, None,fx=0.08333333, fy= 0.11111111, interpolation=cv2.INTER_LANCZOS4)
_input_ = np.asarray(input_)
input_ = np.reshape(input_, (1,128,128,1))
# the target output
_target_ = np.float32(cv2.imread(data_path + cr1, cv2.IMREAD_GRAYSCALE))
_target_ = cv2.normalize(_target_, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
_target = cv2.resize(target_, None, fx=.0808080808, fy=0.0808080808, interpolation=cv2.INTER_CUBIC)
_target = np.asarray(target)
_target = np.reshape(target, (1,256,256,1))

