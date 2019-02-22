import mcpm
import numpy as np
import tensorflow as tf
import time
import math
import csv
from multiprocessing import Pool



def Pooling_learning(xtrain, xtest, ytrain, stask_features, kernel_type, point_estimate, ytrain_non_missing_index, sparsity, sparsity_level, inducing_on_inputs, optim_ind,
				n_missing_values, offset_type, n_tasks, trainable_offset, lengthscale_initial, sigma_initial, white_noise, 
				input_scaling, lengthscale_initial_weights, sigma_initial_weights, prior_mixing_weights, num_samples_ell,
				epochs, var_steps, display_step_nelbo, intra_op_parallelism_threads, inter_op_parallelism_threads, fold = 0, num_features = 2):

	num_latent = 1

	N_all = xtrain.shape[0]

	xtrain_total = np.concatenate((xtrain, xtrain, xtrain, xtrain), axis = 0)
	ytrain_total = np.concatenate((ytrain[:,0], ytrain[:,1], ytrain[:,2], ytrain[:,3]), axis = 0)[:,np.newaxis]

	num_test = xtrain_total.shape[0]
	num_train = xtrain_total.shape[0]

	# Select the test set. In this case, because of the different subsets of available/non available points for each task
	# we consider all the points at the same time. 
	xtest = xtrain_total

	n_missing_values_pooling = n_missing_values*n_tasks

	offset_data_single_task = 0.0
	offset_initial_single_task = np.float32(offset_data_single_task)[np.newaxis]


	ytrain_non_missing_index = ~np.isnan(ytrain_total)


	ytrain_total = ytrain_total[ytrain_non_missing_index][:,np.newaxis]
	xtrain_total = xtrain_total[ytrain_non_missing_index][:,np.newaxis]
	ytrain_non_missing_index = ytrain_non_missing_index[ytrain_non_missing_index][:,np.newaxis]
	n_missing_values = 0
	num_train = int(xtrain_total.shape[0])

	output_toconsider = ytrain_total
	maximum = max(output_toconsider[~np.isnan(output_toconsider)])
	minimum = min(output_toconsider[~np.isnan(output_toconsider)])
	task_features = np.array([maximum, minimum]).reshape(1,num_features)


	# Define the dataset object. 
	# NB. We cannot shuffle the data at every training step. Passing an index variable of
	# non available/available data forces us to keep them fixed 
	data = mcpm.datasets.DataSet(xtrain_total, ytrain_total)



	# Initialize the likelihood function.
	likelihood = mcpm.likelihoods.Lik_LGCP(ytrain_non_missing_index = ytrain_non_missing_index,
									num_missing_data = n_missing_values, offset_type = offset_type, offsets = offset_data_single_task, 
									num_tasks = 1,  trainable_offset = False)

	weights = mcpm.Prior_w.Constant()
	# Get the dimension of the inputs
	dim_inputs = xtrain_total.shape[1]


	# Define the kernel object as a Radial basis kernel with the initial pars defined
	if kernel_type == "RadialBasis":
		kernel = [mcpm.kernels.RadialBasis(dim_inputs, lengthscale = lengthscale_initial, std_dev = sigma_initial, white = white_noise, input_scaling = input_scaling) for i in xrange(num_latent)] 
	if kernel_type == "Matern_5_2":
		kernel = [mcpm.kernels.Matern_5_2(dim_inputs, lengthscale = lengthscale_initial, std_dev = sigma_initial, white = white_noise, input_scaling = False) for i in xrange(num_latent)] 
	if kernel_type == "Matern_3_2":
		kernel = [mcpm.kernels.Matern_3_2(dim_inputs, lengthscale = lengthscale_initial, std_dev = sigma_initial, white = white_noise, input_scaling = False) for i in xrange(num_latent)] 
	if kernel_type == "Exponential":
		kernel = [mcpm.kernels.Exponential(dim_inputs, lengthscale = lengthscale_initial, std_dev = sigma_initial, white = white_noise, input_scaling = False) for i in xrange(num_latent)] 
		

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
		inducing_inputs, _ = initialize_inducing_points(xtrain,ytrain_single_task,inducing_on_inputs,num_latent,inducing_number,N_all-(N_all/4),inputs_dimension) 
		# Optimisation of the inducing inputs is only introduced when using sparsity.



	# Define the model
	model = mcpm.mcpm(likelihood, kernel, weights, inducing_inputs, missing_data = n_missing_values_pooling, num_training_obs = num_train, 
							num_testing_obs = num_test, ytrain_non_missing_index = ytrain_non_missing_index, offset_type = offset_type, 
							prior_var_w_vector = prior_var_w_vector, kernel_funcs_weights = kernel_weights, task_features = task_features, 
							num_tasks = 1, optimization_inducing = optim_ind, num_samples_ell = num_samples_ell,
							intra_op_parallelism_threads = intra_op_parallelism_threads, inter_op_parallelism_threads = inter_op_parallelism_threads)


	# Define the optimizer
	optimizer = tf.train.RMSPropOptimizer(0.005)


	# Start the training of the model
	start = time.time()

	# Train
	(nelbo_values, time_iterations) = model.fit(data, optimizer, var_steps=var_steps, epochs=epochs, display_step=1, display_step_nelbo = display_step_nelbo)

	end = time.time()
	time_elapsed = end-start

	time_iterations = np.asarray(time_iterations)[:,np.newaxis]
	print("Total training finished in seconds", time_elapsed)


	# Predictions
	pred_mean, pred_var, latent_means, latent_vars, means_w, covars_weights, offsets = model.predict(xtest)

	pred_mean = np.concatenate((pred_mean[0:(1*N_all),:], 
						pred_mean[(1*N_all):(2*N_all),:], 
						pred_mean[(2*N_all):(3*N_all),:], 
						pred_mean[(3*N_all):(4*N_all),:]), axis = 1)

	pred_var = np.concatenate((pred_var[0:(1*N_all),:], 
							pred_var[(1*N_all):(2*N_all),:], 
							pred_var[(2*N_all):(3*N_all),:], 
							pred_var[(3*N_all):(4*N_all),:]), axis = 1)

	return (fold, pred_mean, pred_var, latent_means, latent_vars, means_w, covars_weights, offsets, nelbo_values, time_iterations)


