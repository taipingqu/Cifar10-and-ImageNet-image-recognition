#coding: utf-8
import tensorflow as tf
import cifar10_input
import os
import scipy.misc


def inputs_origin(data_dir):
    # only training set
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    # Wrap the list of file names into the form of the queue in TensorFlow
    filename_queue = tf.train.string_input_producer(filenames)
    # The result of the returned read_input attribute uint8image is the image of the Tensor
    read_input = cifar10_input.read_cifar10(filename_queue)
    # Convert image to real form
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    return reshaped_image


if __name__ == '__main__':
    with tf.Session() as sess:
        reshaped_image = inputs_origin('cifar10_data/cifar-10-batches-bin')
        # This queue must be started by start_queue_runners
        threads = tf.train.start_queue_runners(sess=sess)
        sess.run(tf.global_variables_initializer())
        if not os.path.exists('cifar10_data/raw/'):
            os.makedirs('cifar10_data/raw/')
        # save 30 images
        for i in range(30):
            image_array = sess.run(reshaped_image)
            scipy.misc.toimage(image_array).save('cifar10_data/raw/%d.jpg' % i)

