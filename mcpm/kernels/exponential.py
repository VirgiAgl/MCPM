import numpy as np
import tensorflow as tf

from mcpm import util
import kernel

class Exponential(kernel.Kernel):
    MAX_DIST = 1e8

    def __init__(self, input_dim, lengthscale=1.0, std_dev=1.0, white=0.01, input_scaling=False):
        if input_scaling:
            self.lengthscale = tf.Variable(lengthscale * tf.ones([input_dim]), name = 'lenghtscale')
        else:
            self.lengthscale = tf.Variable([lengthscale], dtype=tf.float32, name = 'lenghtscale')

        self.std_dev = tf.Variable([std_dev], dtype=tf.float32, name = 'std_dev')
        self.white = white

    def kernel(self, points1, points2=None): 
        if points2 is None:
            points2 = points1
            white_noise = self.white * util.eye(tf.shape(points1)[0]) 
        else: 
            white_noise = 0.0       

        # Compute the euclidian distances 
        magnitude_square1 = tf.expand_dims(tf.reduce_sum(points1 ** 2, 1), 1)
        magnitude_square2 = tf.expand_dims(tf.reduce_sum(points2 ** 2, 1), 1)
        distances_root = tf.sqrt((magnitude_square1 - 2.0 * tf.matmul(points1, tf.transpose(points2)) +
                     tf.transpose(magnitude_square2)))
        distances_root = tf.clip_by_value(distances_root, 0.0, self.MAX_DIST)
        
        kern = ((self.std_dev ** 2) * tf.exp(-distances_root/self.lengthscale))

        return kern + white_noise


    def diag_kernel(self, points):
        return ((self.std_dev ** 2) + self.white) * tf.ones([tf.shape(points)[0]])

    def get_params(self):
        return [self.lengthscale, self.std_dev]

