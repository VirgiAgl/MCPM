import mcpm
import numpy as np
import tensorflow as tf
import time
import math
import csv
from multiprocessing import Pool
from initialization_inducing import initialize_inducing_points


def LGCP_learning(xtrain, xtest, ytrain, task_features, kernel_type, point_estimate, ytrain_non_missing_index, sparsity, sparsity_level, inducing_on_inputs, optim_ind,
				offset_type, trainable_offset, lengthscale_initial, sigma_initial, white_noise, 
				input_scaling, lengthscale_initial_weights, sigma_initial_weights, prior_mixing_weights, num_samples_ell,
				epochs, var_steps, display_step_nelbo, intra_op_parallelism_threads, inter_op_parallelism_threads, task, fold = 0, num_features = 2, events_location = None):
	t = task

	# Get the dimension of the inputs
	dim_inputs = xtrain.shape[1]

	num_latent = 1
	
	N_all = xtrain.shape[0]

	# Select the offset relative to one task
	offset_data_single_task = 0.0
	offset_initial_single_task = np.float32(offset_data_single_task)[np.newaxis]

	# Select only one output and the corresponding indeces for na/non-na
	ytrain_single_task = ytrain[:,t][:,np.newaxis]
	ytrain_non_missing_index_single_task = np.asarray(ytrain_non_missing_index)[:,t][:,np.newaxis]

	# The task feature will change because only one task is available
	task_features_single_task = task_features[t,:].reshape(1,num_features)

	# Train the model only with the non NA obs - Divide in trainind/test obs
	ytrain_single_task = ytrain_single_task[ytrain_non_missing_index[:,t]]
	ytrain_non_missing_index_single_task = ytrain_non_missing_index_single_task[ytrain_non_missing_index[:,t]]

	if dim_inputs == 1:
		xtrain = xtrain[ytrain_non_missing_index[:,t]]
	if dim_inputs == 2:
		xtrain = np.concatenate((xtrain[:,0][:,np.newaxis][ytrain_non_missing_index[:,t]], xtrain[:,1][:,np.newaxis][ytrain_non_missing_index[:,t]]), axis = 1)
	if dim_inputs > 2:
		result_array = np.ones((N_all - (N_all/16), 1))
		for i in xrange(dim_inputs):
			result = xtrain[:,i][:,np.newaxis][ytrain_non_missing_index[:,t]]
			result_array = np.append(result_array, result, axis=1)
		xtrain = result_array[:,1:]

	# Having erased the NA values and we are left with 0 missing values
	n_missing_values = 0
	num_train = xtrain.shape[0]


	# Define the dataset object. 
	# NB. We cannot shuffle the data at every training step. Passing an index variable of
	# non available/available data forces us to keep them fixed 
	data = mcpm.datasets.DataSet(xtrain, ytrain_single_task)


	# Initialize the likelihood function.
	likelihood = mcpm.likelihoods.Lik_LGCP(ytrain_non_missing_index = ytrain_non_missing_index_single_task,
										num_missing_data = n_missing_values, offset_type = offset_type, offsets = offset_data_single_task, 
										num_tasks = 1,  trainable_offset = False)

	weights = mcpm.Prior_w.Constant()




	num_train = xtrain.shape[0]
	num_test = xtest.shape[0]

	# Define the kernel object as a Radial basis kernel with the initial pars defined
	if kernel_type == "RadialBasis":
		kernel = [mcpm.kernels.RadialBasis(dim_inputs, lengthscale = lengthscale_initial, std_dev = sigma_initial, white = white_noise, input_scaling = input_scaling) for i in xrange(num_latent)] 
	if kernel_type == "Matern_5_2":
		kernel = [mcpm.kernels.Matern_5_2(dim_inputs, lengthscale = lengthscale_initial, std_dev = sigma_initial, white = white_noise, input_scaling = input_scaling) for i in xrange(num_latent)] 
	if kernel_type == "Matern_3_2":
		kernel = [mcpm.kernels.Matern_3_2(dim_inputs, lengthscale = lengthscale_initial, std_dev = sigma_initial, white = white_noise, input_scaling = input_scaling) for i in xrange(num_latent)] 
	if kernel_type == "Exponential":
		kernel = [mcpm.kernels.Exponential(dim_inputs, lengthscale = lengthscale_initial, std_dev = sigma_initial, white = white_noise, input_scaling = input_scaling) for i in xrange(num_latent)] 
	if kernel_type == 'Linear':
		kernel = [mcpm.kernels.Linear(dim_inputs, variance = sigma_initial) for i in xrange(num_latent)] 
	if kernel_type == 'Periodic':
		kernel = [mcpm.kernels.Periodic(period = 1.0, variance = sigma_initial, lengthscale = lengthscale_initial, white = white_noise) for i in xrange(num_latent)] 


	# Initialise the prior distribution on the weights. The prior mean is set to zero. 
	# The prior for the weights need to be given in a format PQX1. For the moment we give a var of 1 to all the weights
	prior_var_w_vector = np.ones(1*len(kernel), dtype=np.float32)

	kernel_weights = [mcpm.kernels.RadialBasis(num_features, lengthscale = lengthscale_initial_weights, std_dev = sigma_initial_weights, white = white_noise) for i in xrange(num_latent)] 


	# Define the levels of sparsity we want to train the model 
	# and initialise the inducing inputs
	if sparsity == False:
		sparsity_vector = np.array([1.0])
		inducing_inputs = xtrain 
	else:
		sparsity_vector = np.array([sparsity_level])
		inducing_number = int(sparsity_level*N_all)
		inducing_inputs, _ = initialize_inducing_points(xtrain,ytrain_single_task,inducing_on_inputs,num_latent,inducing_number,N_all-(N_all/4),dim_inputs) 
		# Optimisation of the inducing inputs is only introduced when using sparsity.


	# Define the model
	model = mcpm.mcpm(likelihood, kernel, weights, inducing_inputs, missing_data = n_missing_values, num_training_obs = num_train, num_testing_obs = num_test,
					ytrain_non_missing_index = ytrain_non_missing_index_single_task, events_location = events_location, offset_type = offset_type,
					prior_var_w_vector = prior_var_w_vector, kernel_funcs_weights = kernel_weights, task_features = task_features_single_task, 
					num_tasks = 1, optimization_inducing = optim_ind, 
					num_samples_ell = num_samples_ell,
					intra_op_parallelism_threads = intra_op_parallelism_threads, inter_op_parallelism_threads = inter_op_parallelism_threads)

	# Define the optimizer
	optimizer = tf.train.RMSPropOptimizer(0.005)


	# Start the training of the model
	start = time.time()

	(nelbo_values, time_iterations) = model.fit(data, optimizer, var_steps=var_steps, epochs=epochs, display_step=1, display_step_nelbo = display_step_nelbo)


	end = time.time()
	time_elapsed = end-start

	print("Total training finished in seconds", time_elapsed)



	# Predict outputs for the test inputs.
	# the command predict gives back the posterior means and vars for the latent functions, 
	# the posterior means and vars for the weights and 
	# the predicted means and vars for the intensities corresponding to the test set
	pred_mean, pred_var, latent_means, latent_vars, means_w, covars_weights, offsets = model.predict(xtest)

	return (task, fold, pred_mean, pred_var, latent_means, latent_vars, means_w, covars_weights, offsets, nelbo_values, time_iterations)




