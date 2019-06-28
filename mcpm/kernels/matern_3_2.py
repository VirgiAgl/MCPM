import numpy as np
import tensorflow as tf

from mcpm import util
from . import kernel


# This function computes the Matern 3/2 kernel 

class Matern_3_2(kernel.Kernel):
    MAX_DIST = 1e8

    def __init__(self, input_dim, lengthscale=0.1, std_dev=1.0, white=0.1, input_scaling=False):
        if input_scaling:
            if lengthscale.size > 1:
                self.lengthscale = tf.Variable(lengthscale)
            else: 
                self.lengthscale = tf.Variable(lengthscale * tf.ones([input_dim]))
        else:
            self.lengthscale = tf.Variable([lengthscale], dtype=tf.float32)

        self.std_dev = tf.Variable([std_dev], dtype=tf.float32)
        self.white = white


    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = self.white * util.eye(tf.shape(points1)[0])
        else:
            white_noise = 0.0
        

        X = points1 / self.lengthscale
        Xs = tf.reduce_sum(tf.square(X), axis=1)


        X2 = points2 / self.lengthscale
        X2s = tf.reduce_sum(tf.square(X2), axis=1)
        r2 = -2.0 * tf.matmul(X, X2, transpose_b=True)
        r2 += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
        r2 = tf.clip_by_value(r2, 0.0, self.MAX_DIST)
        r = tf.sqrt(r2 + 1e-12)

        kernel_matrix = (self.std_dev ** 2) * (1. + np.sqrt(3.) * r) * tf.exp(-np.sqrt(3.) * r)


        return kernel_matrix + white_noise
    
    def kernel_split(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = self.white * util.eye(tf.shape(points1)[0])
        else:
            white_noise = 0.0
        


        X = points1 / self.lengthscale
        Xs = tf.reduce_sum(tf.square(X), axis=1)


        X2 = points2 / self.lengthscale
        X2s = tf.reduce_sum(tf.square(X2), axis=1)
        r2 = -2.0 * tf.matmul(X, X2, transpose_b=True)
        r2 += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
        r2 = tf.clip_by_value(r2, 0.0, self.MAX_DIST)
        r = tf.sqrt(r2 + 1e-12)

        kernel_matrix = (self.std_dev ** 2) * (1. + np.sqrt(3.) * r) * tf.exp(-np.sqrt(3.) * r)


        return (self.std_dev ** 2), (1. + np.sqrt(3.) * r),  r


    def diag_kernel(self, points):
        return ((self.std_dev ** 2) + self.white) * tf.ones([tf.shape(points)[0]])

    def get_params(self):
        return [self.lengthscale, self.std_dev]

