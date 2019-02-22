import numpy as np
import tensorflow as tf

import likelihood

from mcpm.util.util import init_list


# Implementation of a Log Gaussian Cox process network where we a GP prior to the vector w_q of dimension P. 
# We have correlated weights for each Q across tasks

# p(y|f, w) = (lambda)^y exp(-lambda) / y!
# y is the number of events (points) within an area
# lambda = exp(w*f + offset)

# The offsets are task specific thus of dimension (Px1)
# The weights matrix is of dimension (PxQ)
# The fuctions matrix is of dimension (QxN)

class Lik_MCPM_GP(likelihood.Likelihood):
    def __init__(self, ytrain_non_missing_index, num_missing_data, offset_type = 'task', offsets=1.0, num_tasks = 1, num_latent =1, trainable_offset = True):
        self.trainable_offset = trainable_offset
        self.offsets = tf.Variable(offsets, name = 'offsets', trainable = self.trainable_offset)

        self.offset_type = offset_type
        self.num_tasks = num_tasks
        self.num_latent = num_latent
        self.num_missing_data = num_missing_data
        self.ytrain_non_missing_index = ytrain_non_missing_index

    def log_cond_prob(self, num_samples, outputs, latent_samples, sample_means, sample_vars, weights_samples, means_w, covars_weights, *args): 
        
        full_covar = init_list(0.0, [self.num_latent])

        for q in xrange(self.num_latent):
            covar_input = covars_weights[q, :, :]
            full_covar[q] = tf.matmul(covar_input, tf.transpose(covar_input))
        full_covar = tf.stack(full_covar)


        ### NOTE
        ### USE WITH TRUNCATION
        # Improve identifiability
        # full_covar = tf.matrix_band_part(full_covar, -1, 0)
        # means_w = tf.matrix_band_part(means_w, -1, 0)
        
        #cov_diagonal = tf.matrix_band_part(tf.transpose(tf.matrix_diag_part(full_covar)), -1, 0)
        #var_w = cov_diagonal

        means_w = tf.transpose(means_w)
        cov_diagonal = tf.transpose(tf.matrix_diag_part(full_covar))
        var_w = cov_diagonal

        prediction = init_list(0.0, [self.num_tasks])

        outputs_no_missing = tf.boolean_mask(outputs, self.ytrain_non_missing_index)
        product_exp =tf.transpose(tf.matmul(means_w, tf.transpose(sample_means)) + self.offsets)


        means_w = tf.expand_dims(means_w, axis = 2),
        var_w = tf.expand_dims(var_w, axis = 2)
        sample_means = tf.transpose(tf.tile(tf.expand_dims(sample_means, axis = 0), [self.num_tasks,1,1]),  perm=[0, 2, 1])
        sample_vars = tf.transpose(tf.tile(tf.expand_dims(sample_vars, axis = 0), [self.num_tasks,1,1]),  perm=[0, 2, 1])

        first_term = tf.reduce_prod(tf.exp((tf.multiply(means_w, sample_means)[0] +(tf.multiply(tf.square(means_w),sample_vars)[0] + tf.multiply(var_w,tf.square(sample_means)))/2)/((1.0 - tf.multiply(var_w,sample_vars)))), axis = 1)
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
        
        # For each latent function, recostruct the full covariance matrix of the weights starting from the lower triangular part (QxPxP)
        full_covar = init_list(0.0, [self.num_latent])
        for i in xrange(self.num_latent):
            covar_input = weights_vars[i, :, :]
            full_covar[i] = tf.matmul(covar_input, tf.transpose(covar_input))
        full_covar = tf.stack(full_covar)

        # Extract the diagonal terms (variances) for each weight (QxP)
        weights_variances = init_list(0.0, [self.num_latent, self.num_tasks])
        for q in xrange(self.num_latent):
            for p in xrange(self.num_tasks):
                weights_variances[q][p] = full_covar[q,p,p]
        weights_variances = tf.stack(weights_variances)

        # Transpose to have a dimension PxQ
        weights_means = tf.transpose(weights_means)
        weights_variances = tf.transpose(weights_variances)



        # Create two lists to store the values for the posterior means and the posterior square means of the intensities
        pred_means = init_list(0.0, [self.num_tasks])
        pred_means_square = init_list(0.0, [self.num_tasks])

        for p in xrange(self.num_tasks):

            # Get the values of the weights means and vars for process p (dim 1*Q)
            weights_mean_subset = weights_means[p,:] 
            weights_var_subset = weights_variances[p,:]


            # Compute the two terms giving the exact expressions for the posterior mean and var of the intesities                                                  
            first_term = tf.exp((-1/(2*weights_var_subset))*((tf.square(weights_var_subset*latent_means + weights_mean_subset)/(weights_var_subset*latent_vars - 1)) + tf.square(weights_mean_subset)))
            #first_term = tf.reduce_prod(tf.exp((weights_mean_subset*latent_means +(weights_mean_subset**2*latent_vars + latent_means**2*weights_var_subset)/2)/(tf.sqrt(1.0 - weights_var_subset*latent_vars))), axis = 1)

            first_term_square = tf.reduce_prod(tf.exp((-1/(2*weights_var_subset))*((tf.square(2*weights_var_subset*latent_means + weights_mean_subset)/(4*weights_var_subset*latent_vars - 1)) + tf.square(weights_mean_subset))), axis = 1)

            second_term = 1/tf.sqrt(1 - weights_var_subset*latent_vars)
            second_term_square = tf.reduce_prod(1/tf.sqrt(1 - 4*weights_var_subset*latent_vars), axis = 1)

            ### NOTE
            # depending on the truncation that we have, we need to change which Q are considered in the predictions
            # Truncation of QxP
            # first_term = tf.reduce_prod(first_term[:,p:], axis = 1)
            # second_term = tf.reduce_prod(second_term[:,p:], axis = 1)
            # Truncation of PxQ, even for trucation of non square matrix
            #first_term = tf.reduce_prod(first_term[:,:(p+1)], axis = 1)
            #second_term = tf.reduce_prod(second_term[:,:(p+1)], axis = 1)
            #print('first_term', first_term)
            #print('second_term', second_term)


            first_term = tf.reduce_prod(first_term, axis = 1)
            second_term = tf.reduce_prod(second_term, axis = 1)

            if self.offset_type == 'task':
                # Assignin the found values to the lists
                pred_means[p] = first_term * second_term * tf.exp(self.offsets[p])
                pred_means_square[p] = first_term_square * second_term_square * tf.exp(2*self.offsets[p])
            else:
                pred_means[p] = first_term * second_term * tf.exp(self.offsets)
                pred_means_square[p] = first_term_square * second_term_square * tf.exp(2*self.offsets)

        # Stack the posterior means and vars for each task
        pred_means = tf.transpose(tf.stack(pred_means))
        pred_means_square = tf.transpose(tf.stack(pred_means_square))

        pred_vars = pred_means_square - tf.square(pred_means)

        return pred_means, pred_vars, latent_means, latent_vars, self.offsets

