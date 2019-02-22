import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
import sklearn.metrics.pairwise as sk
import time
import scipy 
from scipy.stats import poisson
import random
import math
import csv
from multiprocessing import Pool
from initialization_inducing import initialize_inducing_points
from itertools import product
import pyproj 


import mcpm
import methods
from methods import *
from mcpm.util.util import *
from mcpm.util.utilities import *
from mcpm.util.generate_data import *
from mcpm.util.process_results import *



np.random.seed(1500)

# This code does the following:
# import the count data for the crimes on a resolution 32x32
# generate 256 missing obs (for each task) in the outputs and train the model.

# N_all = total number of observations.
# n_missing_values = number of missing obs for each task. 
# n_tasks = number of tasks
# num_latent = number of latent functions used
# sparsity = sparsity in the inputs considering M training points
# inducing_on_inputs = inducing inputs must concide with some training points or not
# num_samples_ell = num of samples to evaluate the ell term. 
# num_components = number of component for the MOG variational distritbution on the latent functions
# diag_post = MOG has full covariance or not ? 
# var_steps = variational steps
# epochs= total number of epochs to be optimized for. Epochs are complete passes over the data.
# n_cores = number of cores to use in multiprocessing 

##### Import the data
crime_counts = np.genfromtxt('../Data/crime_experiments/crime_counts_final')

# Import index for cell inside NY (1) and cell outside NY (0)
cell_inside_ny = np.genfromtxt('../Data/crime_experiments/cell_inside_ny')

# Settings
N_all = crime_counts.shape[0]
N = int(crime_counts.shape[0]**.5)

n_tasks = 7
num_latent = 4
sparsity = False
sparsity_level = 1.0
inducing_on_inputs = True
optim_ind = False
num_samples_ell = 1
epochs = 1
var_steps = 1
display_step_nelbo = 1
n_sample_prediction = 100
n_bins = 100
inputs_dimension = 2
n_folds = 1
n_cores = 1
missing_exp = True
offset_type = 'task'
NAD83 = True
# Select Adam or RMSP
optimizer = "Adam"
intra_op_parallelism_threads = 0
inter_op_parallelism_threads = 0

# Interpolation or transfer experiment
if missing_exp == True:
	n_missing_values = N_all/4
else: 
	n_missing_values = 0

# Specify the quantity to use for predictions. Options are mean, median or mode. 
point_estimate = 'mean'
	
# Specify the type of prior to use. Can be "Normal" or "GP"
# Set to Normal when running LGCP
prior_mixing_weights = "Normal"

# Specify the type of method to use. Can be "MCPM", "LGCP"
method = 'LGCP'

if method == 'MCPM':
	fixed_weigths = False
	trainable_offset = True
else:
	fixed_weigths = True
	trainable_offset = False

# This should be RadialBasis, Matern_3_2, Matern_5_2 or Exponential
kernel_type = "Matern_3_2"

# Only available for RBF and Matern Kernels
input_scaling = True
if input_scaling == True:
	if method == 'MCPM':
		num_kernel_hyperpar = num_latent +  (num_latent*inputs_dimension)
		#num_kernel_hyperpar = 1 + inputs_dimension
	else:
		num_kernel_hyperpar = 1 + inputs_dimension
else:
	if method == 'MCPM':
		num_kernel_hyperpar = 2*(num_latent)
	else:
		num_kernel_hyperpar = 2


######### INITIALISATION
# Initialize the lik and kernel hyperparameters
if kernel_type == 'Matern_3_2' and input_scaling == True:
	# The distances are not different on the 2 dimensions so I am initialising both to one 
	lengthscale_initial = np.float32((1.0, 1.0))
	sigma_initial = np.float32(1.0)
else:
	#lengthscale is set = to the distance between two points
	lengthscale_initial = np.float32(1.0)
	sigma_initial = np.float32(1.0)

# Define the tasks specific offsets.
if offset_type == 'task':
	if method == 'LGCP':
		offset_initial = np.float32(np.repeat(0.0, n_tasks))[:,np.newaxis]
	else:
		offset_initial = np.float32(np.repeat(2.0, n_tasks))[:,np.newaxis]
else:
	offset_initial = np.float32(2.0)[:,np.newaxis]

# Set the white noise needed for the inversion of the kernel
white_noise = 0.01

# Initialize the kernel hyperparameters for the weight processes
lengthscale_initial_weights = np.float32(1.0)
sigma_initial_weights = np.float32(1.0)

# Save original inputs
original_inputs = crime_counts[:,1:3]

# Getting inputs (by converting lat and long in NAD83 coordinates) and outputs for the crime experiment
inputs = LatLong_NAD83(crime_counts[:,1:3])
outputs = crime_counts[:,3:(3 + n_tasks)]

# When using a GP prior we need to define some task_features as inputs from GPs on the weights
task_features = get_features(outputs)



np.savetxt('../Data/crime_experiments/original_inputs_crime', crime_counts[:,1:3])
np.savetxt('../Data/crime_experiments/inputs_crime', inputs)
np.savetxt('../Data/crime_experiments/outputs_crime', outputs)
np.savetxt('../Data/crime_experiments/task_features_crime', task_features)


# Define the inputs for training and testing
xtrain = inputs
xtest = inputs

# Determine the number of testing points and training points. 
# In the synthetic experiment they are both equal to N_all. 
num_train = xtrain.shape[0]
num_test = xtest.shape[0]

# Define the folds for 4-k fold cross validation
list_indeces_product = define_crime_folds(missing_exp, N)


total_list = np.concatenate((xtrain[:,0][:,np.newaxis], xtrain[:,1][:,np.newaxis], outputs), axis = 1)
total_outputs = np.zeros((n_tasks, N, N))
for i in xrange(n_tasks):
	total_outputs[i] = total_list[:,i + 2].reshape(N, N)

print('Ready to train')

######### TRAINING
if method == "MCPM":
	def Full_MCPM_learning(fold):
		# for fold in xrange(n_folds):
		# Generate missing values in the outputs, different locations for each fold
		outputs_na = generate_missing_data_crime(list_indeces_product, missing_exp, total_list, total_outputs, fold)
		outputs_inputs_na = np.concatenate((original_inputs, outputs_na), axis = 1)
		# Define the ytrain as the outputs with NA obs generated in this fold
		ytrain = outputs_na
		ytrain_non_missing_index = ~np.isnan(ytrain)

		np.save('../Data/crime_experiments/outputs_inputs_na_crime' + str(fold), outputs_inputs_na)
		return MCPM_learning(xtrain, xtest, ytrain, task_features, kernel_type, prior_mixing_weights, point_estimate, ytrain_non_missing_index, 
				n_missing_values, sparsity, sparsity_level, inducing_on_inputs, optim_ind, offset_type, offset_initial, n_tasks, num_latent, trainable_offset, lengthscale_initial, 
				sigma_initial, white_noise, input_scaling, lengthscale_initial_weights, sigma_initial_weights, prior_mixing_weights, num_samples_ell,
				epochs, var_steps, display_step_nelbo, intra_op_parallelism_threads, inter_op_parallelism_threads, fold)


	fold_list = list(range(0,n_folds,1))
	pool = Pool(processes = n_cores)
	results_single_fold_loop = pool.map(Full_MCPM_learning, fold_list)	
	
	## Process results
	# Extract results from the multiprocessing output
	# This function create tensors where to store the values for each task when using MCPM 
	# It extracts results from the multiprocessing output assigning them to the corresponding tensors
	(pred_mean, pred_var, latent_means, latent_vars, means_w, covars_weights, offsets, 
	time_iterations, nelbo_values) = post_process_results_MCPM(results_single_fold_loop, N_all, n_tasks, num_latent, num_train, num_test, epochs, 
																		  display_step_nelbo, num_kernel_hyperpar, n_missing_values, sparsity_level, n_folds, inputs_dimension, 
																		  method, prior_mixing_weights)

# Code for single task learning of the 4 tasks - use multiprocessing 
if method == "LGCP":
	def Full_tasks_LGCP_learning(fold, task):
		# Generate missing values in the outputs, different locations for each fold
		outputs_na = generate_missing_data_crime(list_indeces_product, missing_exp, total_list, total_outputs, fold)
		outputs_inputs_na = np.concatenate((original_inputs, outputs_na), axis = 1)
		# Define the ytrain as the outputs with NA obs generated in this fold
		ytrain = outputs_na
		ytrain_non_missing_index = ~np.isnan(ytrain)


		return LGCP_learning(xtrain, xtest, ytrain, task_features, kernel_type, point_estimate, ytrain_non_missing_index, sparsity, sparsity_level, inducing_on_inputs, optim_ind,
				offset_type, trainable_offset, lengthscale_initial, sigma_initial, white_noise, 
				input_scaling, lengthscale_initial_weights, sigma_initial_weights, prior_mixing_weights, num_samples_ell,
				epochs, var_steps, display_step_nelbo, intra_op_parallelism_threads, inter_op_parallelism_threads, task, fold)

	def Full_LGCP_learning(f_t):
		"""Convert `f([1,2])` to `f(1,2)` call."""
		return Full_tasks_LGCP_learning(*f_t)

	task_fold_list = list(product(range(0,n_folds,1), range(0,n_tasks,1)))
	pool = Pool(processes = n_cores)
	results_single_task_fold_loop = pool.map(Full_LGCP_learning, task_fold_list)	

	## Process results
	# This function create tensors where to store the values for each task when using ST
	# It extracts results from the multiprocessing output assigning them to the corresponding tensors
	(pred_mean, pred_var, latent_means, latent_vars, means_w, covars_weights, offsets, 
	time_iterations, nelbo_values) = post_process_results_LGCP(results_single_task_fold_loop, N_all, n_tasks, num_latent, num_train, num_test, epochs, 
																		 display_step_nelbo, num_kernel_hyperpar, n_missing_values, sparsity_level, 
																		 n_folds, inputs_dimension, 
																		 method, prior_mixing_weights)



######## SAVING RESULTS
folder = '../Data/crime_experiments/'
suffix = prior_mixing_weights + "_" + method + "_" + str(missing_exp)
suffix2 = prior_mixing_weights + "_" + method + "_" + str(missing_exp) + str(num_samples_ell)

# Create a dataset with data and predictions and save it 
final_dataset = np.zeros((n_folds, N_all, (n_tasks*3 + inputs_dimension)))
for i in xrange(n_folds):
		final_dataset[i] = np.concatenate((inputs, outputs, pred_mean[i], pred_var[i]), axis = 1)
np.save(folder + 'final_dataset_' + suffix, final_dataset)



# Save kernel info
# np.save(folder + 'kernel_params_final_' + prior_mixing_weights + "_" + method + "_" + str(missing_exp), kernel_params_final)
# np.save(folder + 'kernel_params_initial_' + prior_mixing_weights + "_" + method + "_" + str(missing_exp), kernel_params_initial)


# Save nelbo values, time iterations and variables' values over epochs
np.save(folder + 'nelbo_values_' + suffix2, nelbo_values)
np.save(folder + 'time_iterations_' + suffix2, time_iterations)
# np.save(folder + 'f_mu_' + suffix2, f_mu_tensor)
# np.save(folder + 'f_var_' + suffix2, f_var_tensor)
# np.save(folder + 'w_mean_' + suffix2, w_mean_tensor)
# np.save(folder + 'w_var_' + suffix2, w_var_tensor)
# np.save(folder + 'off_' + suffix2, off_tensor)


# Save latent functions and weights info
np.save(folder + 'latent_means_' + suffix, latent_means)
np.save(folder + 'latent_variances_' + suffix, latent_vars)
np.save(folder + 'means_weights_' + suffix, means_w)
np.save(folder + 'covars_weights_' + suffix, covars_weights)
np.save(folder + 'offsets_' + suffix, offsets)


if method == "MCPM":
	if prior_mixing_weights == "Normal":
		print('Results for MCPM with Normal prior')
	else:
		print('Results for MCPM with GP prior')

if method == "LGCP":
	print('LGCP')
if method == "Pooling":
	print('Pooling learning')



