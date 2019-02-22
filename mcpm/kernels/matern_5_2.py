import numpy as np
import tensorflow as tf
import math

from mcpm import util
import kernel


class Matern_5_2(kernel.Kernel):
    MAX_DIST = 1e8

    def __init__(self, input_dim, lengthscale=1.0, std_dev=1.0, white=0.1, input_scaling=False):
        if input_scaling:
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

        # # Compute the euclidian distances 
        # magnitude_square1 = tf.expand_dims(tf.reduce_sum(points1 ** 2, 1), 1)
        # magnitude_square2 = tf.expand_dims(tf.reduce_sum(points2 ** 2, 1), 1)

        # distances_root = tf.sqrt((magnitude_square1 - 2.0 * tf.matmul(points1, tf.transpose(points2)) +
        #              tf.transpose(magnitude_square2))) #Numerical stability problem!!
        # distances_root = tf.clip_by_value(distances_root, 0.0, self.MAX_DIST);

        # # Compute constant for 3/2
        # constant = np.sqrt(5.0)/self.lengthscale

        # # Compute the two terms goving the kernel
        # first_term=(1.0 + constant*distances_root + (5.0/3.0)*distances_root**2)
        # second_term = tf.exp(-constant*distances_root)

        # kernel_matrix = tf.multiply(first_term,second_term) * (self.std_dev ** 2)
        
        X = points1 / self.lengthscale
        Xs = tf.reduce_sum(tf.square(X), axis=1)


        X2 = points2 / self.lengthscale
        X2s = tf.reduce_sum(tf.square(X2), axis=1)
        r2 = -2.0 * tf.matmul(X, X2, transpose_b=True)
        r2 += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
        r = tf.sqrt(r2 + 1e-12)

        kernel_matrix = (self.std_dev ** 2) * (1. + np.sqrt(5.) * r + 5./3. * r2) * tf.exp(-np.sqrt(5.) * r)

        return kernel_matrix + white_noise

    def diag_kernel(self, points):
        return ((self.std_dev ** 2) + self.white) * tf.ones([tf.shape(points)[0]])

    def get_params(self):
        return [self.lengthscale, self.std_dev]

