from scipy.spatial import distance 
import tensorflow as tf 
import numpy as np
import pyproj 
import scipy 
from scipy.integrate import quad
from mcpm.util.tensor_initialisation import initialise_tensors


def post_process_results_LGCP(multi_processing_results, N_all, n_tasks, num_latent, num_train, num_test, epochs, display_step_nelbo, num_kernel_hyperpar, n_missing_values, 
                            sparsity_level, n_folds, inputs_dimension, method, prior_mixing_weights):
               
    
    # Initialising tensor to store results 
    (pred_mean, pred_var, latent_means, 
    latent_vars, means_w, covars_weights, 
    offsets, kernel_params_initial, kernel_params_final, 
    kernel_mat_initial, kernel_mat_final, time_iterations, 
    nelbo_values, f_mu_tensor, f_var_tensor, w_mean_tensor,
    w_var_tensor, off_tensor, 
    ell_tensor, integral_tensor, mean_sum_tensor, 
    crossent_tensor, ent_tensor, a_tensor, b_tensor, trace_cross_tensor, kernel_mat_tensor,
    full_covars_tensor, raw_means_tensor, sample_means_location_tensor)  = initialise_tensors(N_all, n_tasks, num_latent, num_train, num_test, epochs, 
                                                                 display_step_nelbo, num_kernel_hyperpar, n_missing_values, sparsity_level,
                                                                 n_folds, inputs_dimension, method, prior_mixing_weights)




    print('pred_mean', pred_mean.shape)
    print('latent_means', latent_means.shape)

    for i in range(len(multi_processing_results)):
        single_result = multi_processing_results[i]

        t = single_result[0]
        f = single_result[1]
        print('single_result[2]', single_result[2].shape)
        print('single_result[4]', single_result[4].shape)
        pred_mean[f,:,t] = np.sum(single_result[2],axis=1)
        pred_var[f,:,t] = np.sum(single_result[3],axis=1)

        latent_means[f,:,t] = np.sum(single_result[4],axis=1)
        latent_vars[f,:,t] = np.sum(single_result[5],axis=1)
        
        means_w[f,t] = single_result[6][0]
        covars_weights[f,t] = single_result[7][0]
        offsets[f,t] = single_result[8][0]

        nelbo_values[f,:,t] = np.sum(single_result[9],axis=1)
        time_iterations[f,:,t] = single_result[10]
        

    return (pred_mean, pred_var, latent_means, latent_vars, means_w, covars_weights, offsets, time_iterations, nelbo_values)


def post_process_results_MCPM(multi_processing_results, N_all, n_tasks, num_latent, num_train, num_test, epochs, display_step_nelbo, num_kernel_hyperpar, n_missing_values, sparsity_level, n_folds, inputs_dimension, method, prior_mixing_weights):
       
    # Initialising tensor to store results 
         
    (pred_mean, pred_var, latent_means, 
    latent_vars, means_w, covars_weights, 
    offsets, kernel_params_initial, kernel_params_final, 
    kernel_mat_initial, kernel_mat_final, time_iterations, 
    nelbo_values, f_mu_tensor, f_var_tensor, w_mean_tensor,
    w_var_tensor, off_tensor,
    ell_tensor, integral_tensor, mean_sum_tensor, 
    crossent_tensor, ent_tensor, a_tensor, b_tensor, trace_cross_tensor, kernel_mat_tensor,
    full_covars_tensor, raw_means_tensor, sample_means_location_tensor)  = initialise_tensors(N_all, n_tasks, num_latent, num_train, num_test, epochs, 
                                                                 display_step_nelbo, num_kernel_hyperpar, n_missing_values, sparsity_level,
                                                                 n_folds, inputs_dimension, method, prior_mixing_weights)

    for i in range(len(multi_processing_results)):
        single_result = multi_processing_results[i]

        f = single_result[0]

        pred_mean[f] = single_result[1]
        pred_var[f] = single_result[2]

        latent_means[f] = single_result[3]
        latent_vars[f] = single_result[4]
        

        means_w[f] = np.concatenate(single_result[5], axis=0)
        covars_weights[f] = np.concatenate(single_result[6], axis=0)
        offsets[f] = np.concatenate(single_result[7], axis=0)[:,0]

        nelbo_values[f] = single_result[8]
        time_iterations[f] = single_result[9]
        
        # kernel_params_final[f] = single_result[10]
        # kernel_params_initial[f] = single_result[11]

        # f_mu_tensor[f] = np.transpose(single_result[12][:,0,:,:], (0,2,1))
        # f_var_tensor[f] = np.transpose(single_result[13][:,0,:,:], (0,2,1))
        # w_mean_tensor[f] = single_result[14][:,:,:,0]
        # w_var_tensor[f] = single_result[15][:,:,:,0]
        # off_tensor[f] = single_result[16][:,:,0]

    return (pred_mean, pred_var, latent_means, latent_vars, means_w, covars_weights, offsets, time_iterations, nelbo_values)
  

def post_process_results_ICM(multi_processing_results, N_all, n_tasks, num_latent, num_train, num_test, epochs, display_step_nelbo, 
                                num_kernel_hyperpar, n_missing_values, sparsity_level, n_partition, inputs_dimension, method, prior_mixing_weights):
    # Initialising tensor to store results 
    (pred_mean, pred_var, latent_means, 
    latent_vars, means_w, covars_weights, 
    offsets, kernel_params_initial, kernel_params_final, 
    kernel_mat_initial, kernel_mat_final, time_iterations, 
    nelbo_values, f_mu_tensor, f_var_tensor, w_mean_tensor,
    w_var_tensor, off_tensor, ell_tensor, integral_tensor, mean_sum_tensor, 
    crossent_tensor, ent_tensor, a_tensor, b_tensor, trace_cross_tensor, kernel_mat_tensor,
    full_covars_tensor, raw_means_tensor, sample_means_location_tensor)  = initialise_tensors(N_all, n_tasks, 1, num_train, num_test, epochs, 
                                                                 display_step_nelbo, num_kernel_hyperpar, n_missing_values, sparsity_level,
                                                                 n_partition, inputs_dimension, method, prior_mixing_weights)


    for i in range(len(multi_processing_results)):
        single_result = multi_processing_results[i]

        partition = single_result[0]

        pred_mean[partition] = single_result[1]
        pred_var[partition] = single_result[2]

        latent_means[partition] = single_result[3]
        latent_vars[partition] = single_result[4]
        
        offsets[partition] = single_result[7][0]

        #nelbo_values[partition] = np.sum(np.sum(single_result[8],axis=1),axis=1)
        nelbo_values[partition] = np.sum(single_result[8],axis=1)
        time_iterations[partition] = single_result[9]
        

    return (pred_mean, pred_var, latent_means, latent_vars, means_w, covars_weights, offsets, nelbo_values, time_iterations)
