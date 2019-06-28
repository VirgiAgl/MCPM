import numpy as np
import tensorflow as tf

from mcpm import util
from . import kernel

class Periodic(kernel.Kernel):
    """
    The periodic kernel. Defined in  Equation (47) of
    D.J.C.MacKay. Introduction to Gaussian processes. In C.M.Bishop, editor,
    Neural Networks and Machine Learning, pages 133--165. Springer, 1998.
    Derived using an RBF kernel once mapped the original inputs through
    the mapping u=(cos(x), sin(x)).
    The resulting kernel can be expressed as:
    k_per(x, x') = variance * exp( -0.5 Sum_i sin^2((x_i-x'_i) * pi /period)/ell^2)
    (note that usually we have a factor of 4 instead of 0.5 in front but this is absorbed into ell
    hyperparameter).
    # I have changed 0.5 to 2
    """

    def __init__(self, period=1.0, variance=1.0, lengthscale=1.0, white =0.01):
        # No ARD support for lengthscale or period yet
        # Need to set it to positive. See what transform.positive does in gpflow
        self.variance = tf.Variable([variance], dtype=tf.float32, name = 'variance')

        self.lengthscale = tf.Variable([lengthscale], dtype=tf.float32, name = 'lenghtscale')

        self.period = tf.Variable([period], dtype=tf.float32, name = 'period')
        self.white = white

    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = self.white * util.eye(tf.shape(points1)[0]) 
        else: 
            white_noise = 0.0 
        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(points1, 1)  # now N x 1 x D
        f2 = tf.expand_dims(points2, 0)  # now 1 x M x D

        r = np.float32(np.pi) * (f - f2) / self.period
        r = tf.reduce_sum(tf.square(tf.sin(r) / self.lengthscale), 2)

        return self.variance * tf.exp(-2 * r) + white_noise

    def diag_kernel(self, points):
        return tf.fill(tf.stack([tf.shape(points)[0]]), tf.squeeze(self.variance))

    def get_params(self):
        return [self.lengthscale, self.variance, self.period]
