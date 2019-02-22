import numpy as np
import tensorflow as tf

from mcpm import util
import prior_w

# Class for the GP prior on the weights

class GP(prior_w.Prior_w):
    MAX_DIST = 1e8
    def __init__(self, num_tasks, num_latent, prior_means = 0, prior_vars = 1):
        self.num_tasks = num_tasks
        self.num_latent = num_latent
        self.prior_means = prior_means
        self.prior_vars = prior_vars
    
    def build_samples(self, num_samples, means, covar):
        # Starting from the the cholesky matrices for the weights' covariances matrices
        # We recostruct the complete matrix and get the diagonal for which we need to simulate.

        full_covar = util.init_list(0.0, [self.num_latent])

        for q in xrange(self.num_latent):
            covar_input = covar[q, :, :]
            full_covar[q] = tf.matmul(covar_input, tf.transpose(covar_input))
        full_covar = tf.stack(full_covar)

        means = tf.transpose(means)
        var = tf.transpose(tf.matrix_diag_part(full_covar))

        return (means + tf.sqrt(var) * tf.random_normal([num_samples, self.num_tasks, self.num_latent], seed=5))
    

    def entropy(self, means, covar):
        sum_val = 0.0
        for i in xrange(self.num_latent):
            # Recostruct the full covars_weights S_wp starting from its cholesky
            # check if correct
            full_covar = tf.matmul(covar[i, :, :], tf.transpose(covar[i, :, :]))
            trace = tf.reduce_sum(tf.matrix_diag_part(tf.cholesky_solve(covar[i, :, :],full_covar)))
            sum_val -= (util.CholNormal(means[i, :], covar[i, :, :]).log_prob(means[i, :]) -
                    0.5 * trace)
            entropy = sum_val
        return entropy 

    def cross_entropy(self, means, covar, chol_var_weights, kernel_chol_weights):
        sum_val = 0.0
        for i in xrange(self.num_latent):
            full_covar = tf.matmul(covar[i, :, :], tf.transpose(covar[i, :, :]))

            trace = tf.reduce_sum(tf.matrix_diag_part(tf.cholesky_solve(kernel_chol_weights[i, :, :],full_covar)))

            sum_val += (util.CholNormal(means[i, :], kernel_chol_weights[i, :, :]).log_prob(0.0) -
                    0.5 * trace)
        cross_ent_weights = sum_val

        return cross_ent_weights

    def get_prior_params(self):
        return [self.prior_means, self.prior_vars]


