import numpy as np
import tensorflow as tf
from . import prior_w
from mcpm.util.util import *
from mcpm.util.normals import *




# Class for the Normal prior on the weights

class Normal(prior_w.Prior_w):
    MAX_DIST = 1e8
    
    def __init__(self, num_tasks, num_latent, prior_means = 0, prior_vars = 1):
        self.num_tasks = num_tasks
        self.num_latent = num_latent
        self.prior_means = prior_means
        self.prior_vars = prior_vars


    def build_samples(self, num_samples, means, covar):
        # This function creates the samples from the weights that are used in the computation of the ell term
        var = tf.transpose(tf.matrix_diag_part(covar))
        means = tf.transpose(means)
        return (means + tf.sqrt(var) * tf.random_normal([num_samples, self.num_tasks, self.num_latent], seed=5))
    
    def entropy(self, means, covar):
        sum_val = 0.0
        # This function is building the entropy for the weights - eq.
        for i in range(self.num_latent):
            chol = tf.cholesky(covar[i])
            trace = tf.reduce_sum(tf.matrix_diag_part(tf.cholesky_solve(chol,covar[i])))
            sum_val -= (CholNormal(means[i, :], chol).log_prob(means[i, :]) -
                        0.5 * trace)
        entropy = sum_val
        return entropy 
 
    def cross_entropy(self, means, covar, chol_var_weights, kernel_chol_weights):
        sum_val = 0.0
        for i in range(self.num_latent):
            trace = tf.reduce_sum(tf.matrix_diag_part(tf.cholesky_solve(chol_var_weights[i],covar[i])))
            sum_val += (CholNormal(means[i, :], chol_var_weights[i]).log_prob(0.0) -
                        0.5 * trace)
        cross_ent_weights = sum_val

        return cross_ent_weights

    def get_prior_params(self):
        return [self.prior_means, self.prior_vars]

