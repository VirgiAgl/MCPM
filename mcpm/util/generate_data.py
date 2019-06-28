import numpy as np
from mcpm.util.util import init_list
import sklearn
import sklearn.metrics.pairwise as sk
from mcpm.util.utilities import *

#import matplotlib.pyplot as plt

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


def generate_synthetic_data(N_all, n_tasks, num_latent, num_features = 2):

	# Initialise required variables
	weights_data_task = init_list(0.0, [n_tasks])
	process_values = init_list(0.0, [num_latent])
	sample_intensity = init_list(0.0, [n_tasks])
	outputs = np.ones((N_all,n_tasks)) 
	task_features = np.zeros((n_tasks,num_features))


	# Define the tasks specific offsets. 
	offset_data = np.float32(np.array([2.0, 2.0, 2.6, 2.6]))

	# Select some weights to generate the data
	weights_data_task[0] = np.float32(np.array([+0.1,-0.12]))
	weights_data_task[1] = np.float32(np.array([-0.1,-0.1])) 
	weights_data_task[2] = np.float32(np.array([-0.1,+0.1]))
	weights_data_task[3] = np.float32(np.array([-0.2,+0.1]))

	# Random noise added to the true parameters in order to initialise the algorithm pars
	random_noise = np.random.normal(loc=0.0, scale=1.0, size=1)

	# Initiliaze the inputs and stardardize them 
	inputs = 5*np.linspace(0.0, 1.0, num=N_all)[:, np.newaxis]
	np.save('../Data/synthetic_experiments/original_inputs_synthetic', inputs)
	inputs_mean = np.transpose(np.mean(inputs, axis = 0)[:,np.newaxis])
	inputs_std = np.transpose(np.std(inputs, axis = 0)[:,np.newaxis])
	standard_inputs = (inputs - inputs_mean)/inputs_std
	inputs = standard_inputs

	# Define the kernel matrices for the two latent functions that we use to generate the data
	sigma1 = 50 * sk.rbf_kernel(inputs, inputs, gamma=25)
	sigma2 = 60 * sk.rbf_kernel(inputs, inputs, gamma=20)

	# Sample the true underlying GPs. 
	for i in range(num_latent):
		if i == 0:
			np.random.seed(10)
			process_values[i] = np.random.multivariate_normal(mean=np.repeat(0,N_all), cov=sigma1)
			process_values[i] = np.reshape(process_values[i], (N_all,1))
		if i == 1:
			process_values[i] = np.random.multivariate_normal(mean=np.repeat(0,N_all), cov=sigma2) 
			process_values[i] = np.reshape(process_values[i], (N_all,1))

	# Generate the intensities as a linear combination of the latent functions with the weights specific to the task + offset specific to the task
	for i in range(n_tasks):
		weighted_sum = 0.0 
		for j in range(num_latent):
			process_values_single = np.array(process_values[j])
			weights_data_task_single = np.array(weights_data_task[i])[:,np.newaxis]
			weighted_sum += weights_data_task_single[j,:]*process_values_single
		sample_intensity[i] = np.exp(weighted_sum + offset_data[i])


	# Generate the outputs by sampling from a Poisson with the constructed intensities
	for j in range(n_tasks):
		for i in range(N_all): 
			sample_intensity_single = sample_intensity[j]
			outputs[i,j] = np.random.poisson(lam = sample_intensity_single[i,0]) 



	# Define some task_features used when placing a GP prior on the mixing weights
	for i in range(n_tasks):
		output_toconsider = outputs[:,i]
		maximum = max(output_toconsider)
		minimum = min(output_toconsider)
		task_features[i,:] = np.array([maximum, minimum])


	return (inputs, outputs, sample_intensity, task_features, offset_data, random_noise, process_values, weights_data_task)

def generate_synthetic_data_noisy(N_all, n_tasks, num_latent, num_features = 2):

	# Initialise required variables
	weights_data_task = init_list(0.0, [n_tasks])
	process_values = init_list(0.0, [num_latent])
	sample_intensity = init_list(0.0, [n_tasks])
	final_process_value = init_list(0.0, [n_tasks])
	outputs = np.ones((N_all,n_tasks)) 
	random_noise_vector = np.ones((N_all,n_tasks)) 
	task_features = np.zeros((n_tasks,num_features))


	# Define the tasks specific offsets. 
	offset_data = np.float32(np.array([2.0, 2.0, 1.0, -0.5]))
	#offset_data = np.float32(np.array([1.2, 1.0, 1.0, -0.5]))
	np.save('../Data/synthetic_experiments/offset_data_noisy', offset_data)
	# Select some weights to generate the data
	weights_data_task[0] = np.float32(np.array([+0.1,-0.12]))
	weights_data_task[1] = np.float32(np.array([-0.1,-0.1])) 
	weights_data_task[2] = np.float32(np.array([-0.1,+0.1]))
	weights_data_task[3] = np.float32(np.array([-0.2,-0.15]))

	np.save('../Data/synthetic_experiments/weights_data_task_noisy', weights_data_task)
	# Random noise added to the true parameters in order to initialise the algorithm pars
	random_noise = np.random.normal(loc=0.0, scale=1.0, size=1)

	# Initiliaze the inputs and stardardize them 
	inputs = 5*np.linspace(0.0, 1.0, num=N_all)[:, np.newaxis]
	inputs_mean = np.transpose(np.mean(inputs, axis = 0)[:,np.newaxis])
	inputs_std = np.transpose(np.std(inputs, axis = 0)[:,np.newaxis])
	standard_inputs = (inputs - inputs_mean)/inputs_std
	inputs = standard_inputs

	# Define the kernel matrices for the two latent functions that we use to generate the data
	sigma1 = 50 * sk.rbf_kernel(inputs, inputs, gamma=25)
	sigma2 = 60 * sk.rbf_kernel(inputs, inputs, gamma=20)

	# Sample the true underlying GPs. 
	for i in range(num_latent):
		if i == 0:
			np.random.seed(10)
			process_values[i] = np.random.multivariate_normal(mean=np.repeat(0,N_all), cov=sigma1)
			process_values[i] = np.reshape(process_values[i], (N_all,1))
		if i == 1:
			process_values[i] = np.random.multivariate_normal(mean=np.repeat(0,N_all), cov=sigma2) 
			process_values[i] = np.reshape(process_values[i], (N_all,1))

	# Generate the intensities as a linear combination of the latent functions with the weights specific to the task + offset specific to the task
	for i in range(n_tasks):
		weighted_sum = 0.0 
		for j in range(num_latent):
			process_values_single = np.array(process_values[j])
			weights_data_task_single = np.array(weights_data_task[i])[:,np.newaxis]
			weighted_sum += weights_data_task_single[j,:]*process_values_single
			random_noise_vector[:,j] = np.random.normal(loc=0.0, scale=1.0, size=N_all)
		sample_intensity[i] = np.exp(weighted_sum + offset_data[i]) 
		final_process_value[i] = weighted_sum + offset_data[i] 
	np.save('../Data/synthetic_experiments/process_values_noisy', final_process_value)


	# Generate the outputs by sampling from a Poisson with the constructed intensities
	for j in range(n_tasks):
		for i in range(N_all): 
			sample_intensity_single = sample_intensity[j]
			random_noise_vector[i,j] = np.random.normal(loc=0.0, scale=2.0, size=1)
			outputs[i,j] = np.around(np.random.poisson(lam = sample_intensity_single[i,0]))
			#outputs[i,j] = np.random.poisson(lam = sample_intensity_single[i,0]) 
			#outputs[i,j] = 1.8*outputs[i,j]
			outputs[i,j] = np.random.poisson(lam = sample_intensity_single[i,0]) + np.random.normal(loc=0.0, scale=4.0, size=1) + 15




	# Define some task_features used when placing a GP prior on the mixing weights
	for i in range(n_tasks):
		output_toconsider = outputs[:,i]
		maximum = max(output_toconsider)
		minimum = min(output_toconsider)
		task_features[i,:] = np.array([maximum, minimum])


	return (inputs, outputs, sample_intensity, task_features, offset_data, random_noise, process_values, weights_data_task, random_noise_vector)


def generate_events_location(inputs, process_values, offset_data, sample_intensity, n_tasks, weights_data_task):

	subset_events_task = [None]*n_tasks

	sample_intensity = sample_intensity[:,:,0]
	print('sample_intensity',sample_intensity.shape)
	print('inputs shape',inputs.shape)

	# plt.plot(process_values)
	# plt.show()

	# plt.plot(sample_intensity)
	# plt.show()


	for t in range(n_tasks):
		## Here we do thinning
		range_inputs = np.max(inputs) - np.min(inputs)
		#max_intensity = np.max(intensity)
		max_intensity = np.max(sample_intensity[:,t])*1.0
		volume = max_intensity*range_inputs
		number_events = np.random.poisson(volume)
		uniform_events = np.sort(np.random.uniform(np.min(inputs), np.max(inputs), number_events)[:,np.newaxis], axis =0)

		sigma1 = 50 * sk.rbf_kernel(inputs, inputs, gamma=25)
		sigma2 = 60 * sk.rbf_kernel(inputs, inputs, gamma=20)

		sigma = weights_data_task[t,0]*sigma1 + weights_data_task[t,1]*sigma2

		# At this point we need to interpolate the intensity in this point to then compare the value of the intensity in the point and the maximum.
		K_xx = sk.linear_kernel(uniform_events, uniform_events) + np.eye(uniform_events.shape[0])* 0.001
		K_xz = sk.linear_kernel(uniform_events, inputs_partition) 
		K_zx = sk.linear_kernel(inputs, inputs) 
		K_zz_inverse = np.linalg.inv(sigma)
		K_zz_chol = np.linalg.cholesky(sigma)
		lt_chol = K_zz_chol[np.tril_indices(K_zz_chol.shape[0])]
	

		y_mean = process_values[:,t] - np.repeat(offset_data[t],N_all/n_partitions)[:,np.newaxis]
		mean_xx = np.repeat(offset_data[t],uniform_events.shape[0])[:,np.newaxis] + np.matmul(np.matmul(K_xz, np.linalg.inv(sigma)),y_mean)


		K = K_xx - np.matmul(np.matmul(K_xz,K_zz_inverse),np.transpose(K_xz))

		lt_chol = K[np.tril_indices(K.shape[0])]


		process_events = np.random.multivariate_normal(mean=mean_xx[:,0], cov=K) 
		# np.save('../Data/synthetic_experiments/process_events', process_events)
		intensity_events = np.exp(process_events)[:,np.newaxis]

		# plt.plot(process_events)
		# plt.show()

		# plt.plot(intensity_events)
		# plt.show()

	
		rejection_rule = intensity_events/max_intensity
		# print('rejection_rule.shape', rejection_rule.shape)
		# print('rejection_rule', rejection_rule)
		uniform = np.random.uniform(0.,1.,uniform_events.shape[0])

		subset_events = uniform_events[uniform[:,np.newaxis] < rejection_rule]

		subset_events_task[t] = subset_events

	return subset_events_task






def generate_from_piecewise_linear(N_all, n_tasks, num_latent, n_partitions, num_features = 2):

	# Initialise required variables

	np.random.seed(1500)
	# Define the tasks specific offsets. 
	#offset_data = np.float32(np.array([1.0]))
	offset_data = np.float32(np.array([1.0]))
	#offset_data = np.float32(np.array([20.0]))


	# Initiliaze the inputs and stardardize them 
	inputs = np.linspace(1.0, 4.0, num=N_all)[:, np.newaxis]
	inputs_mean = np.transpose(np.mean(inputs, axis = 0)[:,np.newaxis])
	inputs_std = np.transpose(np.std(inputs, axis = 0)[:,np.newaxis])
	standard_inputs = (inputs - inputs_mean)/inputs_std
	inputs = standard_inputs

	num_inputs = int(inputs.shape[0])/n_partitions
	print('num_inputs', num_inputs)

	inputs_list = [None]*n_partitions
	subset_events_list = [None]*n_partitions
	outputs_list = [None]*n_partitions
	task_features_list = [None]*n_partitions
	intensity_list = [None]*n_partitions
	ytrain_non_missing_index_list = [None]*n_partitions

	for i in range(n_partitions):
		if i == 1:
			np.random.seed(20)

		inputs_partition = inputs[(i*num_inputs):((i+1)*num_inputs)]
		print('inputs_partition', inputs_partition.shape)
		inputs_mean = np.transpose(np.mean(inputs_partition, axis = 0)[:,np.newaxis])
		inputs_std = np.transpose(np.std(inputs_partition, axis = 0)[:,np.newaxis])
		standard_inputs = (inputs_partition - inputs_mean)/inputs_std
		inputs_partition = standard_inputs

		inputs_list[i] = inputs_partition

		# Define the kernel matrices for the two latent functions that we use to generate the data
		sigma = sk.linear_kernel(inputs_partition, inputs_partition) + np.eye(inputs_partition.shape[0])* 0.001

		input_space_dim = inputs_partition.shape[0]

		# Sample the true underlying GPs. 
		mean_f = 2.
		mean_f_vector = np.float32(np.transpose(np.repeat(mean_f,input_space_dim)[:,np.newaxis]))
		print('mean_f', mean_f_vector.shape)
		process_values = np.random.multivariate_normal(mean=np.repeat(mean_f,input_space_dim), cov=sk.linear_kernel(inputs_partition, inputs_partition))
		process_values = np.reshape(process_values, (input_space_dim,1))

		# Generate the intensities as exp of the latent functions 
		intensity = np.exp(process_values)

		intensity_list[i]= intensity

		# plt.plot(process_values)
		# plt.show()

		# plt.plot(intensity)
		# plt.show()

		## Here we do thinning
		range_inputs = np.max(inputs_partition) - np.min(inputs_partition)
		#max_intensity = np.max(intensity)
		max_intensity = np.max(intensity)*1.0
		volume = max_intensity*range_inputs
		number_events = np.random.poisson(volume)
		uniform_events = np.sort(np.random.uniform(np.min(inputs_partition), np.max(inputs_partition), number_events)[:,np.newaxis], axis =0)



		# At this point we need to interpolate the intensity in this point to then compare the value of the intensity in the point and the maximum.
		K_xx = sk.linear_kernel(uniform_events, uniform_events) + np.eye(uniform_events.shape[0])* 0.001
		K_xz = sk.linear_kernel(uniform_events, inputs_partition) 
		K_zx = sk.linear_kernel(inputs_partition, uniform_events) 
		K_zz_inverse = np.linalg.inv(sigma)
		K_zz_chol = np.linalg.cholesky(sigma)


		y_mean = process_values - np.repeat(mean_f,N_all/n_partitions)[:,np.newaxis]
		mean_xx = np.repeat(mean_f,uniform_events.shape[0])[:,np.newaxis] + np.matmul(np.matmul(K_xz, np.linalg.inv(sigma)),y_mean)

		K = K_xx - np.matmul(np.matmul(K_xz,K_zz_inverse),np.transpose(K_xz))

		mean_f_vector = mean_xx


		process_events = np.random.multivariate_normal(mean=mean_xx[:,0], cov=K) 
		# np.save('../Data/synthetic_experiments/process_events', process_events)
		intensity_events = np.exp(process_events)[:,np.newaxis]

		# plt.plot(process_events)
		# plt.show()

		# plt.plot(intensity_events)
		# plt.show()


	
		rejection_rule = intensity_events/max_intensity

		uniform = np.random.uniform(0.,1.,uniform_events.shape[0])

		subset_events = uniform_events[uniform[:,np.newaxis] < rejection_rule]


		# Generate the outputs by sampling from a Poisson with the constructed intensities
		outputs = np.zeros((intensity.shape[0],1))
		for j in range(intensity.shape[0]):
			outputs[j] = np.random.poisson(lam = intensity[j]) 

		ytrain_non_missing_index_list[i] = ~np.isnan(outputs)

		# Define some task_features used when placing a GP prior on the mixing weights
		output_toconsider = outputs
		maximum = max(output_toconsider)
		minimum = min(output_toconsider)
		task_features = np.array([maximum, minimum])

		subset_events_list[i] = subset_events
		outputs_list[i] = outputs
		task_features_list[i] = task_features


	return (inputs_list, subset_events_list, intensity_list, offset_data, outputs_list, task_features_list, ytrain_non_missing_index_list)

def generate_locations_1d(inputs, outputs):


	events_location = []
	diff = (inputs[1] - inputs[0])

	for i in range(200):

		lower_extreme = inputs[i] - diff/2
		upper_extreme = inputs[i] + diff/2

		# print('lower_extreme', lower_extreme)
		# print('upper_extreme', upper_extreme)
		# print('lower_extreme-upper_exte', lower_extreme-upper_extreme)


		count = outputs[i]
		if count != 0:
			loc = np.random.uniform(low=lower_extreme, high=upper_extreme, size=int(count))

			events_location.append(loc)

	events_location = np.float32(np.hstack(events_location)[:,np.newaxis])
	return events_location



def generate_missing_data_synthetic(outputs, missing_experiment):

	index1 = range(10,60)
	index1_non_missing1 = range(0,10)
	index1_non_missing2 = range(60,200)

	index2 = range(30,80)
	index2_non_missing1 = range(0,30)
	index2_non_missing2 = range(80,200)

	index3 = range(140,190)
	index3_non_missing1 = range(0,140)
	index3_non_missing2 = range(190,200)

	index4 = range(50,100)
	index4_non_missing1 = range(0,50)
	index4_non_missing2 = range(100,200)

	if missing_experiment == True:
		outputs[index1,0] = np.nan
		outputs[index2,1] = np.nan
		outputs[index3,2] = np.nan
		outputs[index4,3] = np.nan

	# Define the indeces for non missing obs data
	ytrain_non_missing_index = ~np.isnan(outputs)

	return (outputs, ytrain_non_missing_index)

def generate_missing_data_crime_1D(list_indeces_product, missing_exp, total_list, total_outputs, fold):
	n_tasks = total_list.shape[1] - 1
	N_all = total_list.shape[0]
	num_missing_values = int(0.8*(N_all))

	outputs_na = np.zeros((N_all, n_tasks))
	if missing_exp == True:
		for i in range(n_tasks):
			total_outputs[i][(N_all-num_missing_values):] = np.nan
			outputs_na[:,i] = total_outputs[i].reshape(N_all,)
	else:
		for i in range(n_tasks):
			outputs_na[:,i] = total_list[:,i + 1]
	return outputs_na

def generate_missing_data_crime(list_indeces_product, missing_exp, total_list, total_outputs, fold):
	n_tasks = total_list.shape[1] - 2
	N_all = total_list.shape[0]

	outputs_na = np.zeros((N_all, n_tasks))
	if missing_exp == True:
		for i in range(n_tasks):
			if i == 0 or i == 1 or i == 2 or i == 3:
				total_outputs[i][list_indeces_product[fold + i]] = np.nan
			if i == 4 or i == 5 or i == 6:
				total_outputs[i][list_indeces_product[fold + i - 4]] = np.nan
			outputs_na[:,i] = total_outputs[i].reshape(N_all,)
	else:
		for i in range(n_tasks):
			outputs_na[:,i] = total_list[:,i + 2]
	return outputs_na

def generate_missing_data_btb(list_indeces_product, missing_exp, total_list, total_outputs, fold):
	n_tasks = total_list.shape[1] - 2
	N_all = total_list.shape[0]
	N = int(N_all**.5)


	total1 = total_list[:,2].reshape(N, N)
	total2 = total_list[:,3].reshape(N, N)
	total3 = total_list[:,4].reshape(N, N)
	total4 = total_list[:,5].reshape(N, N)

	# Generate missing values in the outputs, different location for each fold
	if missing_exp == True:
		total1[list_indeces_product[fold + 3]] = np.nan
		total2[list_indeces_product[fold + 13]] = np.nan
		total3[list_indeces_product[fold + 10]] = np.nan
		total4[list_indeces_product[fold + 12]] = np.nan

	outputs_na = np.concatenate((total1.reshape(N_all,1), 
									 total2.reshape(N_all,1), 
									 total3.reshape(N_all,1), 
									 total4.reshape(N_all,1)), axis = 1)
	return outputs_na





