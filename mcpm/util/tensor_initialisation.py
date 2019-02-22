
from scipy.spatial import distance 
import tensorflow as tf 
import numpy as np
import pyproj 
import scipy 
from scipy.integrate import quad


def initialise_tensors(N_all, n_tasks, num_latent, num_train, num_test, epochs, display_step_nelbo, num_kernel_hyperpar, n_missing_values, sparsity_level, n_folds, inputs_dimension, method, prior_mixing_weights):
    

    data_final_with_na = np.zeros((n_folds, N_all, n_tasks + inputs_dimension))


    pred_mean = np.zeros((n_folds, num_test, n_tasks))
    pred_var = np.zeros((n_folds, num_test, n_tasks))
    
    latent_means = np.zeros((n_folds, num_test,num_latent))
    latent_vars = np.zeros((n_folds, num_test,num_latent))
    
    means_w = np.zeros((n_folds, num_latent, n_tasks))
    covars_weights = np.zeros((n_folds, num_latent, n_tasks, n_tasks))
 
    offsets = np.zeros((n_folds, n_tasks))
    
    kernel_params_final= np.zeros((n_folds, num_kernel_hyperpar))
    kernel_mat_final = np.zeros((n_folds, n_tasks, int(num_train*sparsity_level), int(num_train*sparsity_level)))

    kernel_params_initial= np.zeros((n_folds, num_kernel_hyperpar))
    kernel_mat_initial = np.zeros((n_folds, n_tasks, int(num_train*sparsity_level), int(num_train*sparsity_level)))


    f_mu_tensor = np.zeros((n_folds, epochs/display_step_nelbo, num_train, num_latent)) 
    f_var_tensor = np.zeros((n_folds, epochs/display_step_nelbo, num_train, num_latent))
    w_mean_tensor = np.zeros((n_folds, epochs/display_step_nelbo, n_tasks, num_latent)) 
    w_var_tensor = np.zeros((n_folds, epochs/display_step_nelbo, n_tasks, num_latent)) 
    off_tensor = np.zeros((n_folds, epochs/display_step_nelbo, n_tasks))

    # hyperparam_tensor= np.zeros((n_tasks, epochs/display_step_nelbo, num_kernel_hyperpar))
    # kernel_mat_tensor = np.zeros((n_tasks, epochs/display_step_nelbo, N_all, N_all))

    time_iterations = np.zeros((n_folds, epochs/display_step_nelbo))
    nelbo_values = np.zeros((n_folds, epochs/display_step_nelbo))
    ell_values = np.zeros((n_folds, epochs/display_step_nelbo))

    integral_values = np.zeros((n_folds, epochs/display_step_nelbo))
    mean_sum_values = np.zeros((n_folds, epochs/display_step_nelbo))

    crossent_tensor = np.zeros((n_folds, epochs/display_step_nelbo))
    ent_tensor = np.zeros((n_folds, epochs/display_step_nelbo))

    a_tensor = np.zeros((n_folds, epochs/display_step_nelbo)) 
    b_tensor = np.zeros((n_folds, epochs/display_step_nelbo))

    trace_cross_tensor = np.zeros((n_folds, epochs/display_step_nelbo))

    kernel_mat_tensor = np.zeros((n_folds, epochs/display_step_nelbo, num_train, num_train))

    full_covars_tensor = np.zeros((n_folds, epochs/display_step_nelbo, num_train, num_train))

    raw_means_tensor = np.zeros((n_folds, epochs/display_step_nelbo, num_train, 1))

    sample_means_location_tensor = np.zeros((n_folds, epochs/display_step_nelbo, num_train, 1))

    if method == 'LGCP':

        latent_means = np.zeros((n_folds, num_train,n_tasks))
        latent_vars = np.zeros((n_folds, num_train,n_tasks))

        means_w = np.zeros((n_folds, n_tasks))
        covars_weights = np.zeros((n_folds, n_tasks))

        kernel_params_final= np.zeros((n_folds, n_tasks, num_kernel_hyperpar))
        kernel_mat_final = np.zeros((n_folds, n_tasks, int(N_all*sparsity_level - n_missing_values), int(N_all*sparsity_level - n_missing_values)))

        kernel_params_initial= np.zeros((n_folds, n_tasks, num_kernel_hyperpar))
        kernel_mat_initial = np.zeros((n_folds, n_tasks, int(N_all*sparsity_level - n_missing_values), int(N_all*sparsity_level - n_missing_values)))

        time_iterations = np.zeros((n_folds, epochs/display_step_nelbo, n_tasks))
        nelbo_values = np.zeros((n_folds, epochs/display_step_nelbo, n_tasks))

        f_mu_tensor = np.zeros((n_folds, epochs/display_step_nelbo, N_all, n_tasks)) 
        f_var_tensor = np.zeros((n_folds, epochs/display_step_nelbo, N_all, n_tasks))
        w_mean_tensor = np.zeros((n_folds, epochs/display_step_nelbo, n_tasks)) 
        w_var_tensor = np.zeros((n_folds, epochs/display_step_nelbo, n_tasks)) 
        off_tensor = np.zeros((n_folds, epochs/display_step_nelbo, n_tasks))
        # hyperparam_tensor= np.zeros((n_tasks, epochs/display_step_nelbo, num_kernel_hyperpar))
        # kernel_mat_tensor = np.zeros((n_tasks, epochs/display_step_nelbo, N_all-n_missing_values, N_all-n_missing_values))

    return (pred_mean, pred_var, latent_means, latent_vars, means_w, covars_weights, offsets, kernel_params_initial, kernel_params_final, kernel_mat_initial, kernel_mat_final,
            time_iterations, nelbo_values, f_mu_tensor, f_var_tensor, w_mean_tensor, w_var_tensor, off_tensor, ell_values, integral_values, mean_sum_values, 
            crossent_tensor, ent_tensor, a_tensor, b_tensor, trace_cross_tensor, kernel_mat_tensor, full_covars_tensor, raw_means_tensor, sample_means_location_tensor)
