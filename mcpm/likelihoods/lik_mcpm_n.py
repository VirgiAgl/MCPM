import numpy as np
import tensorflow as tf

import likelihood

from mcpm.util.util import init_list

# Implementation of a Log Gaussian Cox process network where we assign each weight a normal prior distribution 
# We have an independent normal distribution on each weight

# p(y|f, w) = (lambda)^y exp(-lambda) / y!
# y is the number of events (points) within an area
# lambda = exp(w*f + offset)

# The offsets are task specific thus of dimension (Px1)
# The weights matrix is of dimension (PxQ)
# The fuctions matrix is of dimension (QxN)

class Lik_MCPM_N(likelihood.Likelihood):
    def __init__(self, ytrain_non_missing_index, num_missing_data, offset_type = 'task', offsets=1.0, num_tasks = 1, 
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
        # CHANGE TO THIS WHEN CONSTRAINING THE W MATRIX
        #var_w = tf.transpose(tf.expand_dims(tf.transpose(tf.matrix_band_part(tf.transpose(tf.matrix_diag_part(covars_weights)), -1, 0)), axis = 2), [1,0,2])
        
        means_w = tf.transpose(tf.expand_dims(means_w, axis = 2), [1,0,2])
        
        prediction = init_list(0.0, [self.num_tasks])

        outputs_no_missing = tf.boolean_mask(outputs, self.ytrain_non_missing_index)

        product_exp =tf.transpose(tf.matmul(means_w[:,:,0], tf.transpose(sample_means)) + self.offsets)

        sample_means = tf.transpose(tf.tile(tf.expand_dims(sample_means, axis = 0), [self.num_tasks,1,1]),  perm=[0, 2, 1])
        sample_vars = tf.transpose(tf.tile(tf.expand_dims(sample_vars, axis = 0), [self.num_tasks,1,1]),  perm=[0, 2, 1])

        first_term = tf.reduce_prod(tf.exp((tf.multiply(means_w, sample_means) +(tf.multiply(means_w**2,sample_vars) + tf.multiply(var_w,sample_means**2))/2)/((1.0 - tf.multiply(var_w,sample_vars)))), axis = 1)
        second_term = tf.reduce_prod(1.0/tf.sqrt(1.0 - tf.multiply(var_w,sample_vars)), axis = 1)

        prediction = tf.transpose(first_term * second_term * tf.exp(self.offsets))

        prediction_no_missing = tf.boolean_mask(prediction, self.ytrain_non_missing_index)
        product_exp_no_missing = tf.boolean_mask(product_exp, self.ytrain_non_missing_index)

        log_lik = product_exp_no_missing*outputs_no_missing - prediction_no_missing - tf.lgamma(outputs_no_missing + 1.0) 
        ell = tf.reduce_sum(log_lik)

        return (ell, sample_means, sample_vars, means_w, var_w, self.offsets)

    def get_params(self):
        return [self.offsets]
    

    def predict(self, latent_means, latent_vars, weights_means, weights_vars, *args):

        # Create two lists to store the values for the posterior means and the posterior square means of the intensities
        prediction = init_list(0.0, [self.num_tasks])
        pred_means_square = init_list(0.0, [self.num_tasks])

        weights_means = tf.transpose(weights_means) 
        # Weight means is P*Q and weight vars is Q*P*P

        weights_vars = tf.transpose(tf.matrix_diag_part(weights_vars))

        for p in xrange(self.num_tasks):

            # Get the values of the weights means and vars for process p (dim 1*Q)
            weights_mean_subset = weights_means[p,:] 
            weights_var_subset = weights_vars[p,:]

            # Compute the two terms giving the exact expressions for the posterior mean and var of the intesities  
            first_term = tf.exp((-1.0/(2.0*weights_var_subset))*((tf.square(weights_var_subset*latent_means + weights_mean_subset)/(weights_var_subset*latent_vars - 1.0)) + tf.square(weights_mean_subset)))
            first_term_square = tf.reduce_prod(tf.exp((-1.0/(2.0*weights_var_subset))*((tf.square(2.0*weights_var_subset*latent_means + weights_mean_subset)/(4.0*weights_var_subset*latent_vars - 1.0)) + tf.square(weights_mean_subset))), axis = 1)

            second_term = 1.0/tf.sqrt(1.0 - weights_var_subset*latent_vars)
            second_term_square = tf.reduce_prod(1.0/tf.sqrt(1.0 - 4.0*weights_var_subset*latent_vars), axis = 1)


            # WITH NO TRUNCATION
            first_term = tf.reduce_prod(first_term, axis = 1)
            second_term = tf.reduce_prod(second_term, axis = 1)

            if self.offset_type == 'task':
                # Assignin the found values to the lists
                prediction[p] = first_term * second_term * tf.exp(self.offsets[p])
                pred_means_square[p] = first_term_square * second_term_square * tf.exp(2.0*self.offsets[p])
            else:
                prediction[p] = first_term * second_term * tf.exp(self.offsets)
                pred_means_square[p] = first_term_square * second_term_square * tf.exp(2.0*self.offsets)

        # Stack the posterior means and vars for each task
        prediction = tf.transpose(tf.stack(prediction))
        pred_means_square = tf.transpose(tf.stack(pred_means_square))

        pred_vars = pred_means_square - tf.square(prediction)

  
        return prediction, pred_vars, latent_means, latent_vars, self.offsets

