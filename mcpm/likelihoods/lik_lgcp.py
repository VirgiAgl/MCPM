import numpy as np
import tensorflow as tf
import scipy

from . import likelihood
from mcpm.util.util import *


# Implementation of LGCP 

# p(y|f, w) = (lambda)^y exp(-lambda) / y!
# y is the number of events (points) within an area
# lambda = exp(f + offset)

# The offsets are task specific thus of dimension (Px1)
# The fuctions matrix is of dimension (NxP) (we have a GP for each task)

class Lik_LGCP(likelihood.Likelihood):
    def __init__(self, ytrain_non_missing_index, num_missing_data, offset_type = 'task', offsets=0.0, num_tasks = 1, 
        point_estimate = 'mean', trainable_offset = True): 
        self.trainable_offset = trainable_offset
        self.offsets = tf.Variable(offsets, name = 'offsets', trainable = self.trainable_offset) 
        self.offset_type = offset_type
        self.num_tasks = num_tasks
        self.num_missing_data = num_missing_data
        self.ytrain_non_missing_index = ytrain_non_missing_index
        self.point_estimate = point_estimate
   
        
    def log_cond_prob(self, num_samples, outputs, latent_samples, sample_means, sample_vars, weights_samples, means_w, covars_weights, *args):
                                
        var_w = tf.transpose(tf.expand_dims(tf.matrix_diag_part(covars_weights), axis = 2), [1,0,2])
        means_w = tf.transpose(tf.expand_dims(means_w, axis = 2), [1,0,2])
        outputs_no_missing = tf.boolean_mask(outputs, self.ytrain_non_missing_index)

        f_mu_noMissing = tf.boolean_mask(sample_means, self.ytrain_non_missing_index)
        f_var_noMissing = tf.boolean_mask(sample_vars, self.ytrain_non_missing_index)

        log_lik = (outputs_no_missing * f_mu_noMissing) - tf.exp(f_mu_noMissing+f_var_noMissing/2.) - tf.lgamma(outputs_no_missing + 1.0)
        ell = tf.reduce_sum(log_lik)

        return (ell, sample_means, sample_vars, means_w, var_w, self.offsets)

    def get_params(self):
        return [self.offsets]
    

    def predict(self, latent_means, latent_vars, *args):

        pred_vars = (tf.exp(latent_vars) - 1.) * tf.exp(2.0*latent_means + latent_vars) * tf.exp(2.0 * self.offsets)

        if self.point_estimate == 'mean':
            prediction = tf.exp(latent_means + 0.5 * latent_vars) * tf.exp(self.offsets)
        if self.point_estimate == 'median':
            prediction = tf.exp(latent_means) * tf.exp(self.offsets)
        if self.point_estimate == 'mode':
            prediction = tf.exp(latent_means - latent_vars) * tf.exp(self.offsets)
  
        return prediction, pred_vars, latent_means, latent_vars, self.offsets



