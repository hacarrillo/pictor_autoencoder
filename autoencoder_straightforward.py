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
#_input_ = np.float32(cv2.imread(data_path + pictor, cv2.IMREAD_GRAYSCALE))
_input_ = np.float32(cv2.imread(data_path + pictor,cv2.IMREAD_GRAYSCALE))
iw, ih = _input_.shape
input_ = cv2.normalize(_input_, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
input_ = cv2.resize(input_, None,fx=0.08333333, fy= 0.11111111, interpolation=cv2.INTER_LANCZOS4)
input_ = np.asarray(input_)
input_ = np.reshape(input_, (1,128,128,1))
# the target output
_target_ = np.float32(cv2.imread(data_path + cr1, cv2.IMREAD_GRAYSCALE))
tw, th = _target_.shape
target_ = cv2.normalize(_target_, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
target = cv2.resize(target_, None, fx=.0808080808, fy=0.0808080808, interpolation=cv2.INTER_CUBIC)
target = np.asarray(target)
target = np.reshape(target, (1,256,256,1))

# pictor
x = tf.placeholder("float", [batch_size, input_h, input_w, num_channels])

# cannon
y = tf.placeholder("float", [batch_size, target_h, target_w, num_channels])

x_ = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]])

with tf.name_scope("pictor_encoder") as scope:
	# filters1 = 1
	# w1 = weight_variable([3,3,num_channels,filters1], name = "w1")
	# b1 = bias_variable([1], name = "b1")
	# out1 = tf.nn.conv2d(x_,w1,strides=[1,1,1,1], padding = "VALID")
	# out1 = tf.nn.relu(out1 + b1)

	filters1 = 32
	w1 = weight_variable([3,3,num_channels,filters1], name = "w1")
	b1 = bias_variable([1], name = "b1")
	# out1 = tf.nn.conv2d(x,w1,strides=[1,2,2,1], padding = "SAME")
	out1 = tf.nn.conv2d(x,w1,strides=[1,1,1,1], padding = "VALID")
	# out1 = tf.nn.tanh(out1 + b1)
	out1 = out1 + b1

	filters2 = 16
	w2 = weight_variable([3,3,filters1,filters2], name = "w2")
	b2 = bias_variable([1], name = "b2")
	# out2 = tf.nn.conv2d(out1,w2,strides=[1,2,2,1], padding = "SAME")
	out2 = tf.nn.conv2d(out1,w2,strides=[1,1,1,1], padding = "VALID")
	# out2 = tf.nn.tanh(out2 + b2)
	out2 = out2 + b2

	filters3 = 16
	w3 = weight_variable([3,3,filters2,filters3], name = "w3")
	b3 = bias_variable([1], name = "b3")
	# out3 = tf.nn.conv2d(out2,w3,strides=[1,2,2,1], padding = "SAME")
	out3 = tf.nn.conv2d(out2,w3,strides=[1,1,1,1], padding = "VALID")
	# out3 = tf.nn.tanh(out3 + b3)
	out3 = out3 + b3

	filters4 = 8
	w4 = weight_variable([3,3,filters3,filters4], name = "w4")
	b4 = bias_variable([1], name = "b4")
	# out4 = tf.nn.conv2d(out3,w4,strides=[1,2,2,1], padding = "SAME")
	out4 = tf.nn.conv2d(out3,w4,strides=[1,1,1,1], padding = "VALID")
	features = tf.nn.sigmoid(out4 + b4)

		# fully connected
		# hidden = 512
		# out4 = tf.reshape(out4, shape=[1,-1])
		# _, elements = out3.get_shape()
		# w5 = weight_variable(name='w5', shape=[elements, nhidden])
		# features = tf.matmul(out4, w5)

		# features = tf.reshape(out4, shape=[1,-1])

with tf.name_scope("pictor_decode") as scope:
	# filters1_ = 1
	# w1_ = weight_variable([3,3,filters1_,filters1],name = "w1_", )
	# b1_ = bias_variable([1], name = "b1_", )
	# out1_ = tf.nn.conv2d_transpose(out1,w1_, output_shape=[1,128,128,filters1_], strides=[1,1,1,1], padding = "SAME")
	# reconstruction = out1_ + b1_




error = tf.multiply(.5,tf.reduce_mean(tf.square(tf.subtract(reconstruction,x))))


learning_rate = .1
pictor_error = error
train_pictor = tf.train.GradientDescentOptimizer(learning_rate).minimize(pictor_error)
pictor_reconstruction = reconstruction

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=200)

# Even when reading in multiple threads, share the filename
# queue.
old = np.zeros((128,128))
_input_, _target_ = read_and_decode(filename_queue)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	for i in range(total_rounds):
		# print("round: {0}".format(i))
		pictor, cr1 = sess.run([_input_, _target_])
		sess.run(train_pictor, feed_dict={x: input_})
		
		if i % 1000 == 0:
			print("MSE: {0}".format(sess.run(pictor_error, feed_dict={x:input_})))
			conv_img = sess.run([pictor_reconstruction], feed_dict={x: input_})
			conv_arr = np.array(conv_img)
			conv_arr = conv_arr.reshape(128,128)
			conv_fig = plt.figure(6)
			conv_fig.suptitle('pictor to pictor')
			plt.imshow(conv_arr, cmap='gray')
			conv_fig.show()
			print("mean difference: {0}".format(np.mean(old-conv_arr)))
			# print("features: {0}".format(feature))
			old = conv_arr
			sleep(1)
			plt.close(conv_fig)
			real_fig = plt.figure(6)
			real_fig.suptitle("real pictor")
			plt.imshow(np.reshape(input_,(128,128)), cmap='gray')
			real_fig.show()
			sleep(1)
			plt.close(real_fig)

	coord.request_stop()
	coord.join(threads)