import numpy as np
import tensorflow as tf

from mcpm import util
from . import kernel

# This function computes the linear kernel 

class Linear(kernel.Kernel):
    MAX_DIST = 1e8
    
    def __init__(self, input_dim, variance, n_partitions = 1, white=0.01):
        self.input_dim = input_dim
        self.n_partitions = n_partitions
        self.variance = tf.Variable(1, dtype=tf.float32, name = 'variance', trainable = True)
        self.white = white

    def kernel(self, points1, points2=None): 
        if points2 is None:
            points2 = points1
            white_noise = self.white * util.eye(tf.shape(points1)[0]) 
        else: 
            white_noise = 0.0     
        kern = tf.matmul(points1 * self.variance, tf.transpose(points2))
        return kern + white_noise

    def diag_kernel(self, points):
        diag_kern = tf.reduce_sum(tf.square(points) * self.variance, 1)
        return diag_kern


    def get_params(self):
        return [self.variance]

