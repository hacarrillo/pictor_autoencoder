import tensorflow as tf


num_channels = 1

def weight_variable(shape, name):
    initial = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(initial, name = name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = name)


def read_and_decode(filename_queue):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features = {
        'input_height' : tf.FixedLenFeature([], tf.int64),
        'input_width': tf.FixedLenFeature([], tf.int64),
        'target_height': tf.FixedLenFeature([], tf.int64),
        'target_width': tf.FixedLenFeature([], tf.int64),
        'input_raw': tf.FixedLenFeature([], tf.string),
        'target_raw': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    input_ = tf.decode_raw(features['input_raw'], tf.float32)
    target = tf.decode_raw(features['target_raw'], tf.float32)
    
    input_height = tf.cast(features['input_height'], tf.int32)
    input_width = tf.cast(features['input_width'], tf.int32)

    target_height = tf.cast(features['target_height'], tf.int32)
    target_width = tf.cast(features['target_width'], tf.int32)
    
    # input_shape = tf.pack([height, width])
    # target_shape = tf.pack([height, width])

    # input_shape = tf.pack([height, width, 1])
    # target_shape = tf.pack([height, width, 1])
    
    # input_ = tf.reshape(input_, input_shape)
    # target = tf.reshape(target, target_shape)
    
    inh, inw,th,tw = 128,128,256,256

    # in_shape = tf.stack([input_height, input_width,num_channels])
    # target_shape = tf.stack([target_height, target_width,num_channels])

    in_shape = tf.stack([128,128,1])
    target_shape = tf.stack([256,256, 1])

    input_ = tf.reshape(input_, shape = in_shape)
    target = tf.reshape(target, shape = target_shape)

    print("{0} {1}".format(in_shape, target_shape))
    # input_.set_shape(in_shape)
    # target.set_shape(target_shape)


    images, annotations = tf.train.shuffle_batch([input_, target],
                                                 batch_size=1,
                                                 capacity=50,
                                                 num_threads=2,
                                                 min_after_dequeue=10)



    return images, annotations
