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


channels = 1
labels = 1
input_h = 128
input_w = 128
target_h = 256
target_w = 256
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
_input_ = cv2.resize(_input_, None,fx=0.08333333, fy= 0.11111111, interpolation=cv2.INTER_LANCZOS4)
_input_ = np.asarray(_input_)
_input_ = np.reshape(_input_, (1,128,128,1))
# the target output
_target_ = np.float32(cv2.imread(data_path + cr1, cv2.IMREAD_GRAYSCALE))
_target_ = cv2.normalize(_target_, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
_target_ = cv2.resize(_target_, None, fx=.0808080808, fy=0.0808080808, interpolation=cv2.INTER_CUBIC)
_target_ = np.asarray(_target_)
_target_ = np.reshape(_target_, (1,256,256,1))

in_image = tf.placeholder("float", [batch_size, input_h, input_w, channels])
target_image = tf.placeholder("float", [batch_size, target_h, target_w, channels])

# needs dropout layer
def discriminator(image, label):
	# first layer
	filters_1 = 128
	w_1 = weight_variable(shape = [3,3,1,filters_1])
	b_1 = bias_variable(shape = [1])
	out_1 = tf.nn.conv2d(image, w_1, strides = [1,2,2,1], padding = "SAME")
	out_1 = tf.nn.relu(out_1 + b_1)

	# second layer
	filters_2 = 64
	w_2 = weight_variable(shape = [3,3,filters_1, filters_2])
	b_2 = bias_variable(shape = [1])
	out_2 = tf.nn.conv2d(out_1, w_2, strides = [1,2,2,1], padding = "SAME")
	out_2 = tf.nn.relu(out_2 + b_2)
	out_2 = tf.nn.max_pool(out_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "VALID")

	# third layer
	filters_3 = 64
	w_3 = weight_variable(shape = [3,3,filters_2, filters_3])
	b_3 = bias_variable(shape = [1])
	out_3 = tf.nn.conv2d(out_2, w_3, strides = [1,2,2,1], padding = "SAME")
	out_3 = tf.nn.relu(out_3 + b_3)

	filters_4 = 32
	w_4 = weight_variable(shape = [3,3, filters_3, filters_4])
	b_4 = bias_variable(shape = [1])
	out_4 = tf.nn.conv2d(out_3, w_4, strides = [1,2,2,1], padding = "SAME")
	out_4 = tf.nn.relu(out_4 + b_4)
	
	filters_5 = 32
	w_5 = weight_variable(shape = [3,3, filters_4, filters_5])
	b_5 = bias_variable(shape = [1])
	out_5 = tf.nn.conv2d(out_4, w_5, strides = [1,2,2,1], padding = "SAME")
	out_5 = tf.nn.relu(out_5 + b_5)

	# fully connected layers
	# size at this point is [1,4,4,32], total 512 features
	out_4_flatten = tf.reshape(out_5, shape = [batch_size, -1])

	hidden_1 = 128
	wh_1 = weight_variable(shape = [512, hidden_1])
	bh_1 = bias_variable(shape = [1])
	outh_1 = tf.matmul(out_4_flatten, wh_1)
	outh_1 = tf.nn.relu(outh_1)

	hidden_2 = 16
	wh_2 = weight_variable(shape = [hidden_1, hidden_2])
	bh_2 = bias_variable(shape = [1])
	outh_2 = tf.matmul(outh_1, wh_2)
	outh_2 = tf.nn.relu(outh_2)

	hidden_3 = labels
	wh_3 = weight_variable(shape = [hidden_2, hidden_3])
	bh_3 = bias_variable(shape = [1])
	outh_3 = tf.matmul(outh_2, wh_3)
	outh_3 = tf.nn.sigmoid(outh_3)
	y = tf.squeeze(outh_3, [0] )

	return y

def generator(image):
	# input shape is 128 128
	# first layer
	filters_1 = 128
	w_1 = weight_variable(shape = [3,3,filters_1,channels])
	b_1 = bias_variable(shape = [1])
	out_1 = tf.nn.conv2d_transpose(image, w_1, output_shape = [batch_size, 128, 128, filters_1], strides = [1,1,1,1], padding = "SAME")
	out_1 = tf.nn.relu(out_1 + b_1)

	# second layer
	filters_2 = 64
	w_2 = weight_variable(shape = [3,3,filters_2, filters_1])
	b_2 = bias_variable(shape = [1])
	out_2 = tf.nn.conv2d_transpose(out_1, w_2, output_shape= [batch_size,128,128,filters_2], strides = [1,1,1,1], padding = "SAME")
	out_2 = tf.nn.relu(out_2 + b_2)

	# third layer
	filters_3 = 1
	w_3 = weight_variable(shape = [3,3,filters_3, filters_2])
	b_3 = bias_variable(shape = [1])
	out_3 = tf.nn.conv2d_transpose(out_2, w_3, output_shape = [batch_size, 256, 256, filters_3], strides = [1,2,2,1], padding = "SAME")
	y = tf.nn.sigmoid(out_3 + b_3)
	print("generated shape {0}".format(y.shape))

	return y

# 0 = fake , 1 = real
with tf.variable_scope('G'):
    G = generator(in_image)

with tf.variable_scope('D') as scope:
    d_real = discriminator(target_image, [1.0])
    scope.reuse_variables()
    d_fake = discriminator(G, [0.0])

# need to look into this 
# paper on adv nn : https://arxiv.org/pdf/1406.2661.pdf
# might be useful too : https://danieltakeshi.github.io/2017/03/05/understanding-generative-adversarial-networks/
# simple adv nnn tutorial : http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
loss_d = tf.reduce_mean(-tf.log(d_real) - tf.log(1 - d_fake))
loss_g = tf.reduce_mean(-tf.log(d_fake))

learning_rate = .01
train_discriminator = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_d)
train_generator = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_g)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	for i in range(total_rounds):
		# print("round: {0}".format(i))
		sess.run(train_discriminator, feed_dict={in_image: _input_, target_image: _target_})
		sess.run(train_generator, feed_dict={in_image: _input_, target_image: _target_})
		
		if i % 500 == 0:
			print("Descriminator loss: {0}".format(sess.run(loss_d, feed_dict={in_image: _input_, target_image: _target_})))
			print("Generator loss: {0}".format(sess.run(loss_g, feed_dict={in_image: _input_, target_image: _target_})))
			conv_img = sess.run(G, feed_dict={in_image: _input_, target_image: _target_})
			conv_arr = np.array(conv_img)
			conv_arr = conv_arr.reshape(256,256)
			conv_fig = plt.figure(6)
			conv_fig.suptitle('generator')
			plt.imshow(conv_arr, cmap='gray')
			conv_fig.show()
			old = conv_arr
			sleep(1)
			plt.close(conv_fig)
			real_fig = plt.figure(6)
			real_fig.suptitle("target")
			plt.imshow(np.reshape(_target_,(256, 256)), cmap='gray')
			real_fig.show()
			sleep(1)
			plt.close(real_fig)