import numpy as np
import gpflow
import time
from multiprocessing import Pool
import tensorflow as tf
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
np.random.seed(0)

data = 'synthetic'
use_inducing = False
n_cores = 1
n_folds = 1
covariates = False

maxiter = 1000



if data == 'synthetic':
	# sample from the model
	Q = 2
	D = 4

	X = np.loadtxt('../Data/synthetic_experiments/X_vector')
	Y = np.loadtxt('../Data/synthetic_experiments/Y_vector')
	Y = Y[:,np.newaxis]


	# build a sparse GP model
	M = 50
	k = gpflow.kernels.Matern32(1, active_dims=[0]) * gpflow.kernels.Coregion(input_dim=1, output_dim=D, rank=Q, active_dims=[1])
	k.coregion.W = np.random.randn(D, Q)  # random initialization for the weights


	k.coregion.W.read_value()

	lik = gpflow.likelihoods.Poisson()

	m = gpflow.models.SVGP(X=X, Y=Y.astype(np.float64), kern=k, likelihood=lik, Z=X)

	m.feature.trainable = False

	start = time.time()
	opt = gpflow.train.RMSPropOptimizer(0.005)
	opt.minimize(m)


	end = time.time()
	time_elapsed = end-start
	print('time_elapsed',time_elapsed)
	
	# # make some predictions and plot
	Xtest = np.loadtxt('../Data/synthetic_experiments/inputs')
	Xtest = Xtest[:,np.newaxis]

	for i in range(D):
		Xtest_i = np.hstack([Xtest, np.ones((200, 1)) * i])
		mu, var = m.predict_y(Xtest_i)
		
		np.savetxt('../Data/synthetic_experiments/mu_'+str(i), mu)
		np.savetxt('../Data/synthetic_experiments/var_'+str(i), var)


if data == 'crime':
	# sample from the model
	Q = 4
	D = 7
	M = 10

	X = np.loadtxt('../Data/crime_experiments/X_vector')
	Y = np.loadtxt('../Data/crime_experiments/Y_vector')
	Xtest = np.loadtxt('../Data/crime_experiments/inputs_crime')

	Y = Y.reshape(5376, 4)
	X = X.reshape(4, 5376,3)


	posterior_mean = np.zeros((n_folds, 1024, 7))
	posterior_var = np.zeros((n_folds, 1024, 7))
	posterior_mean_task = np.zeros((1024, 7))
	posterior_var_task = np.zeros((1024, 7))
	nelbo_values = np.zeros((n_folds, maxiter))
	time_tensor = np.zeros((n_folds, maxiter))


	# build a sparse GP model
	def training(fold):
		Y_single = Y[:, fold][:,np.newaxis]
		X_single = X[fold, :]


		k = gpflow.kernels.Matern32(2) * gpflow.kernels.Coregion(input_dim=1, output_dim=D, rank=Q, active_dims=[2])
		k.coregion.W = np.random.randn(D, Q)  # random initialization for the weights

		lik = gpflow.likelihoods.Poisson()

		if use_inducing == False:
			m = gpflow.models.SVGP(X=X_single, Y=Y_single.astype(np.float64), kern=k, likelihood=lik, Z=X_single)
		else:
			m = gpflow.models.SVGP(X=X_single, Y=Y_single.astype(np.float64), kern=k, likelihood=lik, Z=inducing_single)			

		m.feature.trainable = False

		start = time.time()
		opt = gpflow.train.RMSPropOptimizer(0.005)
		opt.minimize(m, maxiter=maxiter)

		end = time.time()
		time_elapsed = end-start
		print('time_elapsed',time_elapsed)



		for i in range(D):
			Xtest_i = np.hstack([Xtest, np.ones((1024, 1)) * i])
			mu, var = m.predict_y(Xtest_i)
			mu = np.sum(mu, axis = 1)
			var = np.sum(var, axis = 1)
			
			posterior_mean_task[:,i] = mu
			posterior_var_task[:,i] = var

		return (fold, posterior_mean_task, posterior_var_task, nelbo_values, time_tensor)


	# Construct a list of task id
	fold_list = range(0,n_folds)

	start_looping_cv = time.time()

	# Map the function for different tasks to different core
	pool = Pool(processes = n_cores)
	results = pool.map(training, fold_list)	

	end_looping_cv = time.time()

	time_elapsed_cv = end_looping_cv-start_looping_cv

	print("Execution finished in seconds", time_elapsed_cv)

	# Extract results from the multiprocessing output
	for i in range(len(results)):
		single_result = results[i]
		
		fold = single_result[0]
		posterior_mean[fold] = single_result[1]
		posterior_var[fold] = single_result[2]
		nelbo_values[fold] = single_result[3]
		time_tensor[fold] = single_result[4]


	np.save('../Data/crime_experiments/posterior_mean', posterior_mean)
	np.save('../Data/crime_experiments/posterior_var', posterior_var)
	np.save('../Data/crime_experiments/nelbo_values', nelbo_values)
	np.save('../Data/crime_experiments/time_tensor', time_tensor)


if data == 'btb':
	# sample from the model
	Q = 4
	D = 4
	M = 1152
	n_training = 15360
	

	if covariates == True:
		inputs_dimension = 6
		X = np.loadtxt('Data/btb_experiments/X_vector_cov')
		Y = np.loadtxt('Data/btb_experiments/Y_vector_cov')
		inducing = np.loadtxt('Data/btb_experiments/inducing_matrix_cov')
	else:
		inputs_dimension = 2
		X = np.loadtxt('Data/btb_experiments/X_vector')
		Y = np.loadtxt('Data/btb_experiments/Y_vector')
		inducing = np.loadtxt('Data/btb_experiments/inducing_matrix')
		
	Xtest = np.loadtxt('Data/btb_experiments/inputs_BTB')
	
	Y = Y.reshape(n_training, 16)
	X = X.reshape(16, n_training,inputs_dimension+1)
	inducing = inducing.reshape(16, M*D, inputs_dimension+1)


	posterior_mean = np.zeros((n_folds, 4096, D))
	posterior_var = np.zeros((n_folds, 4096, D))
	posterior_mean_task = np.zeros((4096, D))
	posterior_var_task = np.zeros((4096, D))
	nelbo_values = np.zeros((n_folds, maxiter))
	time_tensor = np.zeros((n_folds, maxiter))


	# build a sparse GP model
	def training(fold):
		Y_single = Y[:, fold][:,np.newaxis]
		X_single = X[fold, :]
		inducing_single = inducing[fold, :]
		print('Y', Y.shape)

		k = gpflow.kernels.Matern32(inputs_dimension) * gpflow.kernels.Coregion(input_dim=1, output_dim=D, rank=Q, active_dims=[inputs_dimension])
		k.coregion.W = np.random.randn(D, Q)  # random initialization for the weights

		lik = gpflow.likelihoods.Poisson()

		if use_inducing == False:
			m = gpflow.models.SVGP(X=X_single, Y=Y_single.astype(np.float64), kern=k, likelihood=lik, Z=X_single)
		else:
			m = gpflow.models.SVGP(X=X_single, Y=Y_single.astype(np.float64), kern=k, likelihood=lik, Z=inducing_single)			

		m.feature.trainable = False

		opt = gpflow.train.RMSPropOptimizer(0.005)
		nelbo_values, time_tensor = opt.minimize(m, maxiter=maxiter)

		for i in range(D):
			Xtest_i = np.hstack([Xtest, np.ones((4096, 1)) * i])
			mu, var = m.predict_y(Xtest_i)
			mu = np.sum(mu, axis = 1)
			var = np.sum(var, axis = 1)
			
			posterior_mean_task[:,i] = mu
			posterior_var_task[:,i] = var


		return (fold, posterior_mean_task, posterior_var_task, nelbo_values, time_tensor)


	# Construct a list of task id
	fold_list = range(0,n_folds)

	start_looping_cv = time.time()

	# Map the function for different tasks to different core
	pool = Pool(processes = n_cores)
	results = pool.map(training, fold_list)	

	end_looping_cv = time.time()

	time_elapsed_cv = end_looping_cv-start_looping_cv

	print("Execution finished in seconds", time_elapsed_cv)

	# Extract results from the multiprocessing output
	for i in range(len(results)):
		single_result = results[i]
		
		fold = single_result[0]
		posterior_mean[fold] = single_result[1]
		posterior_var[fold] = single_result[2]
		nelbo_values[fold] = single_result[3]
		time_tensor[fold] = single_result[4]


	np.save('../Data/btb_experiments/posterior_mean', posterior_mean)
	np.save('../Data/btb_experiments/posterior_var', posterior_var)
	np.save('../Data/btb_experiments/nelbo_values', nelbo_values)
	np.save('../Data/btb_experiments/time_tensor', time_tensor)

