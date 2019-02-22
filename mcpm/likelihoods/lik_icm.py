import numpy as np
import tensorflow as tf

import likelihood
import scipy

from mcpm.util.utilities import *

from mcpm.util.util import init_list

# Implementation of ICM likelihood 

# p(y|f, w) = (lambda)^y exp(-lambda) / y!
# y is the number of events (points) within an area
# lambda = exp(f + offset)

# The offsets are task specific thus of dimension (Px1)
# The fuctions matrix is of dimension (NxP) (we have a GP for each task)


class Lik_ICM(likelihood.Likelihood):
    def __init__(self, ytrain_non_missing_index, num_missing_data, offset_type = 'task', offsets=0.0, num_tasks = 1, 
        point_estimate = 'mean', trainable_offset = False): 
        self.trainable_offset = trainable_offset
        self.offsets = tf.Variable(offsets, name = 'offsets', trainable = self.trainable_offset) 
        self.ytrain_non_missing_index = ytrain_non_missing_index
        self.point_estimate = point_estimate
   
        
    def log_cond_prob(self, num_samples, outputs, latent_samples, sample_means, sample_vars, weights_samples, means_w, covars_weights, weights_optim,
                        means, kernel_chol, inducing_inputs, kernel_params, sample_means_location, covars):
        
        ##### Discrete model
        if sample_means.shape[1] == 1: 
            outputs_no_missing = tf.boolean_mask(outputs, self.ytrain_non_missing_index)
            mean = tf.transpose(weights_optim*tf.transpose(sample_means) + self.offsets)
            var = tf.transpose(tf.square(weights_optim) * tf.transpose(sample_vars))
        else:
            outputs_no_missing = tf.boolean_mask(outputs, self.ytrain_non_missing_index)
            mean = tf.transpose(tf.matmul(tf.transpose(weights_optim), tf.transpose(sample_means)) + self.offsets)
            var = tf.transpose(tf.matmul(tf.square(tf.transpose(weights_optim)), tf.transpose(sample_vars)))

        mean_noMissing = tf.boolean_mask(mean, self.ytrain_non_missing_index)
        var_noMissing = tf.boolean_mask(var, self.ytrain_non_missing_index)

        log_lik = (outputs_no_missing * mean_noMissing) - tf.exp(mean_noMissing + 0.5*var_noMissing) - tf.lgamma(outputs_no_missing + 1.0)
        ell = tf.reduce_sum(log_lik)


        ##### Continuous model
        ##### Value of the integral with Erfi/Erf function
        
        # This is for 1 latent function
        full_covar = tf.matmul(covars[0, :, :], tf.transpose(covars[0, :, :])) # This is S
        full_kernel = tf.matmul(kernel_chol[0], tf.transpose(kernel_chol[0])) # This is K_zz

        # Get extremes of integration for integral - x_max and x_min
        maximum = tf.reduce_max(inducing_inputs[0])
        minimum = tf.reduce_min(inducing_inputs[0])


        # this gives k_zz^-1 S k_zz^-1
        inverse = tf.matrix_inverse(full_kernel)

        # These three ways of computing k_zz^-1 S k_zz^-1 are equivalent
        matrix_mult = tf.matmul(tf.cholesky_solve(kernel_chol[0], full_covar),inverse)

        # coefficient for x
        a = weights_optim[0,0]*kernel_params*tf.matmul(tf.transpose(tf.cholesky_solve(kernel_chol[0, :, :], inducing_inputs[0])), tf.expand_dims(means[0, :], 1))

        # coefficient for x^2
        second_b = tf.matmul(tf.matmul(tf.transpose(inducing_inputs[0]),tf.matrix_inverse(full_kernel)), inducing_inputs[0])
        third_b = tf.matmul(tf.matmul(tf.transpose(inducing_inputs[0]),matrix_mult), inducing_inputs[0])
        b = .5*tf.square(weights_optim[0,0])*(kernel_params - tf.square(kernel_params)*second_b + tf.square(kernel_params)*third_b)
        b = b[0,0]
        a = a[0,0]


        c = tf.cond(b < 0., lambda: -b, lambda: b)

        arg_max_erfi = (2.*c*maximum + a)/(2.* tf.sqrt(c))
        arg_min_erfi = (2.*c*minimum + a)/(2.* tf.sqrt(c))

        arg_max_erf = (2.*c*maximum - a)/(2.* tf.sqrt(c))
        arg_min_erf = (2.*c*minimum - a)/(2.* tf.sqrt(c))


        # arg_max_erfi = tf.clip_by_value(arg_max_erfi, -6., 6.)
        # arg_min_erfi = tf.clip_by_value(arg_min_erfi, -6., 6.)
        


        integral = tf.cond(b > 0, lambda: tf.sqrt(np.pi)/(2.*tf.sqrt(c)) * tf.exp(tf.clip_by_value(- tf.square(a)/(4.*c), -100.,88.))*(myerfi(arg_max_erfi) - myerfi(arg_min_erfi)), 
                             lambda: tf.minimum(tf.sqrt(np.pi)/(2.*tf.sqrt(c)) * tf.exp(tf.clip_by_value(tf.square(a)/(4.*c), -100.,88.)), 1e+38)*(tf.erf(arg_max_erf) - tf.erf(arg_min_erf)))

        # Dispnay gradients of the integral
        # grad_mean = tf.gradients(tf.sqrt(np.pi)/(2.*tf.sqrt(c)) * tf.exp(tf.clip_by_value(- tf.square(a)/(4.*c), -100.,88.))*(myerfi(arg_max_erfi) - myerfi(arg_min_erfi)), means)
        # integral = tf.Print(integral, [grad_mean], 'grad_mean: ')
        # integral = tf.cond(tf.abs(b) < 0.000005, lambda: (tf.exp(a*maximum)-tf.exp(a*minimum))/a, lambda: integral)

        mean_sum = weights_optim[0,0]*tf.reduce_sum(sample_means_location)

        ell =  - integral*tf.exp(self.offsets)[0] + mean_sum + (self.offsets*tf.cast(tf.shape(sample_means_location)[0],dtype = tf.float32))[0]


        ##### Upper bound
        # e = self.offsets*(maximum-minimum)

        # diff_1 = maximum**2 - minimum**2
        # diff_2 = maximum**3 - minimum**3

        # integral = - tf.exp(0.5*a*diff_1 + 1./3. * b *diff_2)

        # ell = - tf.exp(0.5*a*diff_1 + 1./3. * b *diff_2) + mean_sum


        return (ell, sample_means, sample_vars, weights_optim, weights_optim, self.offsets)


    def get_params(self):
        return [self.offsets]
    

    def predict(self, latent_means, latent_vars, means_w, covars_weights, weights_optim):
        
        #means_w and covars_weights are not used in this case

        mean = tf.transpose(weights_optim*tf.transpose(latent_means) + self.offsets)
        var = tf.transpose(tf.square(weights_optim) * tf.transpose(latent_vars))

        pred_vars = (tf.exp(var) - 1.) * tf.exp(2.0*mean + var)

        if self.point_estimate == 'mean':
            prediction = tf.exp(mean + 0.5 * var)
        if self.point_estimate == 'median':
            prediction = tf.exp(mean) 
        if self.point_estimate == 'mode':
            prediction = tf.exp(mean - var) 
  
        return prediction, pred_vars, latent_means, latent_vars, self.offsets



