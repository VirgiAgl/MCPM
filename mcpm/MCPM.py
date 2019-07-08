from __future__ import print_function

import time

import tensorflow_probability as tfp

from mcpm.util import *
from . import Prior_w


class mcpm(object):
    """
    The class representing the MCPM model with Normal prior on the mixing weights.

    Parameters
    ----------
    likelihood_func : subclass of likelihoods.Likelihood
        An object representing the likelihood function p(y|f, w).
    kernel_funcs : list of subclasses of kernels.Kernel
        A list of one kernel per latent function.
    weights : subclass of prior_w.Prior_w
        An object representing the prior for the mixing weights
    inducing_inputs : ndarray   
        An array of initial inducing input locations. Dimensions: num_inducing * input_dim.
    prior_var_w_vector:
        Vector storing the prior vars for thw weights
    missing_data:
        The number of missing values in the training data. 
    num_testing_obs:
        Number of testing obs
    ytrain_non_missing_index:
        Indeces for the non missing observation in ytrain
    prior_mixing_weights: 
        Specify which prior we assign to the mixing weights. Can be Normal or GP. Default is Normal.
    convergence_check:
        Percentage difference required between to consecutive nelbo values in order for the alg to be considered converged.
    num_tasks:
        Number of tasks
    optimization_inducing:
        Bool. If False the location of the inducing inputs is not optimized. 
    num_samples_ell : int
        The number of samples to approximate the expected log likelihood of the posterior.
    intra_op_parallelism_threads, inter_op_parallelism_threads: 
        Number of threads to use in tensorflow computations. 0 leaves tensorflow free to optimise
        the number of threads to use. 
    """

    def __init__(self,
                 likelihood_func,
                 kernel_funcs,
                 weights,
                 inducing_inputs,
                 missing_data,
                 num_training_obs,
                 num_testing_obs, 
                 ytrain_non_missing_index,
                 events_location = None,  
                 offset_type = 'task',
                 prior_var_w_vector = None,
                 kernel_funcs_weights = None, 
                 task_features = None,               
                 convergence_check = 0.001,
                 num_tasks = 1,
                 optimization_inducing = False,
                 num_samples_ell = 100,
                 intra_op_parallelism_threads = 0, 
                 inter_op_parallelism_threads = 0):

        # Get the likelihood function 
        self.likelihood = likelihood_func

        # Get the list of kernel functions for each q
        self.kernels = kernel_funcs 

        # Get the prior for the mixing weights 
        self.weights = weights
        
        # Set a bound for the percentage change of the nelbo over the epochs
        #self.convergence_check = convergence_check
        #self.not_update = 0

        # Define the number of MonteCarlo samples used for the computation of the ell term
        self.num_samples_ell = num_samples_ell

        # Whether we want to optimize the inducing inputs or not. Optimise only when using a 
        # level of sparsity (num inducing inputs/number of training observations) smaller than 1.
        self.optimization_inducing = optimization_inducing

        # Repeat the inducing inputs for all latent processes if we haven't been given individually (dimension QxMxD)
        # specified inputs per process.
        if inducing_inputs.ndim == 2:
            inducing_inputs = np.tile(inducing_inputs[np.newaxis, :, :], [len(self.kernels), 1, 1])

        # Initialize all model dimension constants.
        self.num_tasks = num_tasks # P
        self.num_training_obs = num_training_obs
        self.num_latent = len(self.kernels) # Q
        self.num_inducing = inducing_inputs.shape[1] # M
        self.input_dim = inducing_inputs.shape[2] # D
        self.missing_data = missing_data # Number of NA in the training set 
        self.num_testing_obs = num_testing_obs
        self.ytrain_non_missing_index = ytrain_non_missing_index # Indeces for NA obs


        # Define all parameters that get optimized directly in raw form. Some parameters get
        # transformed internally to maintain certain pre-conditions (eg variance needs to be positive).

        #### Variational parameters for the latent functions
        # Means m_j for the latent functions, dimension is Q*M
        self.raw_means = tf.Variable(tf.ones([self.num_latent, self.num_inducing]))

        # Covariances S_j for the latent functions, dimension Q*M*M. We initialise the lower triangular matrix of the Cholesky decomposition. 
        init_vec = np.zeros([self.num_latent] +  util.tri_vec_shape(self.num_inducing), dtype=np.float32) 
        self.raw_covars = tf.Variable(init_vec)

        # Inducing inputs. They can be optimized or not depending on optimization_inducing var. 
        self.raw_inducing_inputs = tf.Variable(inducing_inputs, trainable = optimization_inducing, dtype=tf.float32)  

        # Get the likelihood par from the chosen likelihood function and the the kernel pars from the chosen cov matrices
        self.raw_likelihood_params = self.likelihood.get_params()
        self.raw_kernel_params = sum([k.get_params() for k in self.kernels], [])
        

        ### Normal prior
        # Prior parameters
        # In this case we are giving a Normal prior. When placing a Normal prior on the weights, we need to define the prior variances.
        # The prior variances for the weights need to be given in a format PQX1 where first we have the values for the task 1 
        # (task1,q=1....task1,q=Q,task2,q=1,...,task2,q=2,....). Rearranging by col we will get a matrix Q*P which represents the prior weights' vars.
        # The prior means for the weights are set to zero. 
        self.prior_var_w_vector = prior_var_w_vector
        prior_var_w_reshaped = tf.reshape(self.prior_var_w_vector, [self.num_latent, self.num_tasks])
        self.prior_var_w = tf.Variable(prior_var_w_reshaped, dtype=tf.float32)
        
        # Variational parameters
        # Variational parameters to be optimized for the mixing weights of the latent processes (dimension Q*P).
        # Means and variances are currenlty initialised to zero.         
        self.raw_means_w = tf.Variable(tf.zeros([self.num_latent, self.num_tasks]),dtype=np.float32)
        self.raw_var_w = tf.Variable(tf.zeros([self.num_latent, self.num_tasks]),dtype=np.float32)


        ### GP prior
        # When placing a GP prior on the weights, we have Q kernel functions on the weights whose inputs are the tasks features. 
        # The prior means for the weights are still initialized to zero.
        # Prior parameters
        self.kernels_weights = kernel_funcs_weights 
        if task_features.ndim == 2:
            task_features = np.tile(task_features[np.newaxis, :, :], [len(self.kernels_weights), 1, 1])

        # Variational parameters
        # The means are the same of the normal prior (raw_means_w).
        # The cov S_w for the weights processes. Dimension Q*P*P.
        # raw_covars_weights is the lower triangular matrix of the CHOLESKY decomposition of the variational covariance of the weights processes
        init_vec_weights = np.zeros([self.num_latent] + util.tri_vec_shape(self.num_tasks), dtype=np.float32) 
        self.raw_covars_weights = tf.Variable(init_vec_weights)


        # Inputs for the kernels on the weights. The dimension is P*D'. They are not optimized.
        self.raw_task_features = tf.Variable(task_features, trainable = False, dtype=tf.float32)  
        # Hyper parameters of the kernel matrices on the weights.
        self.raw_kernel_params_weights = sum([k.get_params() for k in self.kernels_weights], [])


        ## Define weighths to be optimized 
        self.weights_optim = tf.Variable(tf.ones([self.num_latent, self.num_tasks]),dtype=np.float32, trainable= False)


        # Define placeholder variables for training and predicting. 
        self.num_train = tf.placeholder(tf.float32, shape=[], name="num_train")
        self.train_inputs = tf.placeholder(tf.float32, shape=[self.num_training_obs, self.input_dim], name="train_inputs")
        self.train_outputs = tf.placeholder(tf.float32, shape=[self.num_training_obs, self.num_tasks], name="train_outputs")
        self.test_inputs = tf.placeholder(tf.float32, shape=[self.num_testing_obs, self.input_dim], name="test_inputs")

        if type(events_location) == type(None):
            self.events_location = np.float32(np.zeros((1,self.input_dim)))
        else:
            self.events_location = events_location

        # Build our computational graph. Notice that depending on the prior specification on the mixing weights, we will have different 
        # arguments for the prior and variational parameters on the weights. (raw_means_w, raw_var_w, prior_var_w for the normal prior
        # raw_task_features, raw_means_w and raw_covars_weights for the GP prior (with the raw kernel hyperparameters)). 
        # Thus all of these are needed to construct the computational graph
        (self.nelbo, self.entropy, self.entropy_weights, self.cross_ent, self.cross_ent_weights, 
        self.ell, self.gp_mean, self.gp_var, self.weights_mean, self.weights_var, 
        self.off_values, self.predictions, self.kernel_mat, self.covars_weights) = self._build_graph(self.raw_means,self.raw_covars,
                                                                self.raw_inducing_inputs,
                                                                self.train_inputs,
                                                                self.train_outputs,
                                                                self.num_train,
                                                                self.test_inputs,
                                                                raw_means_w = self.raw_means_w, 
                                                                raw_var_w = self.raw_var_w, 
                                                                prior_var_w = self.prior_var_w,
                                                                raw_task_features = self.raw_task_features,
                                                                raw_covars_weights = self.raw_covars_weights)



        # Do all the tensorflow bookkeeping. intra_op_parallelism_threads gives the number of cores to be used for one single operation (tf parallelises 
        # single steps within an op). inter_op_parallelism_threads gives the number of cores to be used across different operations within a single session. 
        # This is different from the multicore parallelisation. Multicore executes the code #n times in parallel on #n core. On each #n_i core tf split #m operations
        # on #m cores according to the inter_* parameter. Within each of the #m operations, tf splits the steps on #q cores. 
        # Need to pay attention to this when running on servers. Especially when doing CV in parallel, extreme parallelisation might slow down the algorithm because of cores trying to 
        # do different things at the same time. 
        session_conf = tf.ConfigProto(intra_op_parallelism_threads = intra_op_parallelism_threads, 
                                      inter_op_parallelism_threads = inter_op_parallelism_threads)

        self.session = tf.Session(config=session_conf)
        
        # Just initiliazing optimizer and train step
        self.optimizer = None
        self.train_step = None


    def fit(self, data, optimizer, var_steps=10, epochs=200, batch_size=None, display_step=1, display_step_nelbo = 100):
        """
        Fit the MCPMmodel to the given data.
        This function is returning the nelbo values over iterations and the itaration for which convergence is achieved.

        Parameters
        ----------
        data : subclass of datasets.DataSet
            The train inputs and outputs.
        optimizer : TensorFlow optimizer
            The optimizer to use in the fitting process.
        var_steps : int
            Number of steps to update variational parameters using variational objective (elbo).
            Set this to 1 when doing batch (all data used in optimisation once) optmisation. 
        epochs : int
            The number of epochs to optimize the model for. These give the number of complete pass through the data.
        batch_size : int
            The number of datapoints to use per mini-batch when training. If batch_size is None,
            then we perform batch gradient descent. Use all data together. 
        display_step : int
            The frequency at which the current iteration number is printed out.
        display_step_nelbo:
            The frequency at which current values are printed out and stored.
        

        Returns
        ----------
        nelbo_vector: np.array
            Values of the objective function over epochs
        kernel_params_initial, kernel_params_final: np.array
            Initial and final kernels' hyperparameters
        kernel_mat_initial, kernel_mat_final: np.array
            Initial and final kernels' matrices
        f_mu_tensor, f_var_tensor: np.array
            Approximate posterior mean and posterior vars for the GPs at every display_step_nelbo iter.
        w_mean_tensor, w_var_tensor: np.array
            Approximate posterior mean and posterior vars for the weights at every display_step_nelbo iter.
        off_tensor: np.array
            Optimized values of offsets at every display_step_nelbo iter.
        time_tensor: np.array
            Time to complete every display_step_nelbo iter.
        """
 
        num_train = data.num_examples
        if batch_size is None: 
            batch_size = num_train

        if self.optimizer != optimizer:
            self.optimizer = optimizer
            # This is defining the training step that tf should execute and INITIALISE (assign values once the graph is created) all the variables.
            self.train_step = optimizer.minimize(self.nelbo) 
            self.session.run(tf.global_variables_initializer())
    
        # Depending on the dataset object, the data are shuffled everytime we are taking a new batch. 
        # When we are not using batches, we are still shuffling the order of the training observations in doing optimisation.
        # Since we are using the first configuration of missing/non missing points we don't shuffle the data.
        # If we still want to shuffle tra training set for each epoch we need to determine in the code (maybe here)
        # the indeces of non missing training set.         ### NEED TO CHANGE

        # Initialise counters of epochs
        old_epoch = 0
        initial_epoch = 1

        # Define tensor to store the values of different objects over epochs
        nelbo_vector = []
        # Objective function values
        crossent_vector = []
        crossentweights_vector = []
        ent_vector = []
        entweight_vector = []
        ell_vector = []

        time_tensor = []

        # Evaluate initial values of objects
        # if old_epoch == 0:
        #     (kernel_mat_initial, kernel_params_initial) = self.session.run([self.kernel_mat,self.raw_kernel_params])
        #     print("kernel_mat_initial" + ' ' + repr(kernel_mat_initial), end=" ")
        #     print(' ')
        #     print("kernel_params_initial" + ' ' + repr(kernel_params_initial), end=" ")
        #     print(' ')
        
        # Time training step
        start = time.time()
        # Start training phase over epochs
        while data.epochs_completed < epochs:
            # Notice that wen var steps > 1 this step is repeated for epochs + var steps. 
            # This means that for each epoch we pass throught the data var_steps times shuffling the training mini-batch. 
            num_epochs = data.epochs_completed + var_steps
            while data.epochs_completed < num_epochs:
                # Shuffling the training data and get a new batch. Next_batch is giving a list with the x of the batch as first element (need to be a tensor)
                # and with the y of the batch as second element  (a matrix)
                batch = data.next_batch(batch_size)



                # Execution of training step which is defined above as minimisation of negative elbo 
                self.session.run(self.train_step, feed_dict={self.train_inputs: batch[0], 
                                                             self.train_outputs: batch[1], 
                                                             self.num_train: num_train})
                # Time training step
                #end = time.time()
                #time_elapsed = end - start
                #print("Execution time per training epoch", time_elapsed)


                # # Printing the kernel hyper parameters at each iteration
                if data.epochs_completed % display_step == 0 and data.epochs_completed != old_epoch:
                    print(' ' + 'i=' + repr(data.epochs_completed) + ' ')
                    old_epoch = data.epochs_completed
    

                if data.epochs_completed % display_step_nelbo == 0 and data.epochs_completed != 0:
                    # Every display_step_nelbo times (excluding the initial epoch) this is evaluating and storing values of objects
                    # Evaluating intermediate values of objective function, f mean and var, w mean and var, kernel info

                    (nelbo_value, crossent, crossentweights, ent, entweight, ell) = self._print_current_state(data, num_train)

                    # Append values to save them
                    nelbo_vector.append(nelbo_value)
                    crossent_vector.append(crossent)
                    crossentweights_vector.append(crossentweights)
                    ent_vector.append(ent)
                    entweight_vector.append(entweight)
                    ell_vector.append(ell)

                    


                ### Uncomment this part to ass a convergence criteria check
                # if data.epochs_completed > 1:
                #     abs_change = np.abs(nelbo_vector[data.epochs_completed - 2]-nelbo_vector[data.epochs_completed - 1])
                #     abs_change_perc = np.abs((nelbo_vector[data.epochs_completed - 2]-nelbo_vector[data.epochs_completed - 1])/nelbo_vector[data.epochs_completed - 2])
                #     print('abs_change', abs_change)
                #     print('abs_change_perc', abs_change_perc)
                    
                    # Convergence criteria 
                    # if abs_change_perc < 0.000000000025:
                    #     print('Convergence reached at: ', data.epochs_completed)
                    #     data.epochs_completed = epochs

                        # Time training step
        end = time.time()
        time_elapsed = end - start
        time_tensor.append(time_elapsed)

        # Once the optimisation is finished, convert objects to be saved in arrays
        nelbo_vector = np.asarray(nelbo_vector)
        time_tensor = np.asarray(time_tensor)
        crossent_vector = np.asarray(crossent_vector)
        crossentweights_vector = np.asarray(crossentweights_vector)
        ent_vector = np.asarray(ent_vector)
        entweight_vector = np.asarray(entweight_vector)
        ell_vector = np.asarray(ell_vector)
        

        return (nelbo_vector, time_tensor)



    def predict(self, test_inputs, batch_size=None):
        """
        After training, predict outputs given testing inputs.

        Parameters
        ----------
        test_inputs : ndarray
            Points on which we wish to make predictions. Dimensions: num_test * D.
        batch_size : int
            The size of the batches we make predictions on. If batch_size is None, predict on the entire test set at once.

        Returns
        -------
        pred_means: np.array
            Predicted intensity mean of the test inputs. Dimensions: num_test * P.
        pred_vars: ndarray
            Predicted intensity variance of the test inputs. Dimensions: num_test * P.
        latent_means: ndarray
            Approximate posterior means of the GPs. Dimensions: num_test * Q.
        latent_vars: ndarray
            Approximate posterior vars of the GPs. Dimensions: num_test * Q.
        means_w: ndarray
            Approximate posterior means of the weight. Dimensions: P * Q.
        covars_weights: ndarray
            Approximate posterior vars of the weights. Dimensions: P * Q.
        offsets: ndarray
            Task specific offsets. Dimensions: P.
        """

        # If batch size is None, we are prediction on the all test inputs at once. 
        if batch_size is None:
            num_batches = 1
        else:
            # If instedad we consider batches, the number of batches is equal to the number of testing points 
            # divided by the number of points we want to consider for each batch.
            num_batches = util.ceil_divide(test_inputs.shape[0], batch_size)

        # Define lists for the objects we want to evaluate. We split each list depending on the number of batches. 
        test_inputs = np.array_split(test_inputs, num_batches)
        #Predicted mean and var of the intensity for each task at each test point
        pred_means = util.init_list(0.0, [num_batches])  
        pred_vars = util.init_list(0.0, [num_batches])  

        # Posterior mean and var for the GPs and the mixing weights. These are combined internally to compute the predicted intensity.
        # However, they are needed separately in order to evaluate alternative performance metrics (MC NLPL RMSE). 
        latent_means = util.init_list(0.0, [num_batches])  
        latent_vars = util.init_list(0.0, [num_batches])
        means_w = util.init_list(0.0, [num_batches]) 
        covars_weights = util.init_list(0.0, [num_batches])

        # Optimised offsets 
        offsets = util.init_list(0.0, [num_batches])

        for i in range(num_batches):
            (pred_means[i], pred_vars[i], latent_means[i], latent_vars[i], means_w[i], 
                covars_weights[i], offsets[i]) = self.session.run(self.predictions, feed_dict={self.test_inputs: test_inputs[i]})

        return (np.concatenate(pred_means, axis=0), np.concatenate(pred_vars, axis=0), 
                np.concatenate(latent_means, axis=0), np.concatenate(latent_vars, axis=0), 
                means_w, covars_weights, offsets)
        # need to change previous row
 

    def _print_current_state(self, data, num_train):
        # This is evaluating the value of the elbo with the overall dataset. 
        # It is different from the one computed during optimisation only when using mini-batches.
        # All the variables that we want to evaluate at each epoch need to be added here
        (nelbo_value, crossent, crossentweights, ent, entweight, ell, 
            weights_means, weights_var, kernel_pars, cov_w) = self.session.run([self.nelbo, self.cross_ent, self.cross_ent_weights, self.entropy, 
                                                                          self.entropy_weights, self.ell, self.weights_mean, self.weights_var, 
                                                                          self.raw_kernel_params, self.covars_weights],
                                                                          feed_dict={self.train_inputs: data.X,
                                                                          self.train_outputs: data.Y,
                                                                          self.num_train: num_train})

        print('nelbo_value: ' + str(nelbo_value))
        print('cross_entropy: '+ str(crossent) + ' ' + 'cross_entropy_weights: ' + str(crossentweights))
        print('entropy:' + ' ' + str(ent) + ' ' + 'entropy_weights: ' + str(entweight))    
        print('ell: ' + str(ell))

        #print('weights_means:' + ' ' + str(weights_means)) 
        #print('weights_var:' + ' ' + str(weights_var)) 
        #print('kernel_pars:' + ' ' + str(kernel_pars)) 
        #print('cov_w:' + ' ' + str(cov_w)) 

        return (nelbo_value,  crossent, crossentweights, ent, entweight, ell)


    def _build_graph(self, raw_means, raw_covars, raw_inducing_inputs,
                     train_inputs, train_outputs, num_train, test_inputs, 
                     raw_means_w, raw_var_w, prior_var_w, raw_task_features, raw_covars_weights):

        # This function is building the computational graph that will be then evaluated and optimized.
        # The computational graph depends on the prior specification for the mixing weights. 

        # First transform all raw variables into their internal form. The optimisation is realized on the unconstrained variables. 
        # Variables are then brought back to the acceptable regions. (eg positive values for variances)

        ### Variables for the latent GPs
        # The cholesky has positive DIAGONAL entries thus we substitute the diagonal element of the chelesky 
        # with their exponential in order to garantee the positive definitness.
        # We use vec_to_tri(raw_covars) to go from one vector to a lower triangular matrix. 
        # We only optimize over the lower triangular portion of the Cholesky.
        # NB. We note that we will always operate over the cholesky space internally!!!

        # Variational covariances
        mat = util.forward_tensor(raw_covars, self.num_inducing)

        diag_mat = tf.matrix_diag(tf.matrix_diag_part(mat))
        exp_diag_mat = tf.matrix_diag(tf.exp(tf.matrix_diag_part(mat)))
        covars = mat - diag_mat + exp_diag_mat 
        
        # These can varies freely
        means = raw_means
        inducing_inputs = raw_inducing_inputs
 
        kernel_mat = [self.kernels[i].kernel(inducing_inputs[i, :, :]) for i in range(self.num_latent)]  
        kernel_chol = tf.stack([tf.cholesky(k) for k in kernel_mat], 0)


        #### Variables for the mixing weights
        # The variance terms for the weights need to be positive so we transform them exponentiating them.
        # Depending on the prior specification we initialise the covars_weights.
        
        # These are the variational parameters
        means_w = raw_means_w
        

        if type(self.weights) == Prior_w.GP:
            mat_weights = util.forward_tensor(raw_covars_weights, self.num_tasks)
            diag_mat_weights = tf.matrix_diag(tf.matrix_diag_part(mat_weights))
            exp_diag_mat_weights = tf.matrix_diag(tf.exp(tf.matrix_diag_part(mat_weights))) * 0.3
            covars_weights = mat_weights - diag_mat_weights + exp_diag_mat_weights
        else:
            # In the normal case we have a diagonal structure in the P * Q * Q tensor.
            var_w = 0.3 * tf.exp(raw_var_w)
            covars_weights = tf.matrix_diag(var_w)
        
        

        # These are the prior parameters
        # When using a Normal prior we give a vector of PRIOR variances that needs to be positive. We write it in tensor form and get the cholesky decomposition
        # which is used in the cross entropy evaluation 
        chol_var_weights = tf.cholesky(tf.matrix_diag(tf.exp(prior_var_w)))
        
        # When using a GP prior we also have inputs for the GP covariances
        task_features = raw_task_features

        # In the GP case this is defining the PRIOR covariance matrices for the GP on the weights
        kernel_mat_weights = [self.kernels_weights[i].kernel(task_features[i, :, :]) for i in range(self.num_latent)]  
        kernel_chol_weights = tf.stack([tf.cholesky(k) for k in kernel_mat_weights], 0)


        # Computation of the terms composing the nelbo
        entropy = self._build_entropy(means, covars)

        entropy_weights = self._build_entropy_weights(means_w = means_w, covars_weights  = covars_weights)
 
        cross_ent = self._build_cross_ent(means, covars, kernel_chol)
        
        cross_ent_weights = self._build_cross_ent_weights(means_w = means_w, covars_weights = covars_weights, 
                                                    chol_var_weights = chol_var_weights, kernel_chol_weights = kernel_chol_weights) 

         # Build the ell term.
        (ell, gp_mean, gp_var, weights_mean, weights_var, off_values) = self._build_ell(means, covars, inducing_inputs,
                                                                                              kernel_chol, train_inputs, train_outputs,
                                                                                              means_w = means_w, covars_weights = covars_weights)

        # Get the number of training observations
        batch_size = tf.to_float(tf.shape(train_inputs)[0]) 


        nelbo = -((batch_size / num_train) * (entropy + entropy_weights + cross_ent + cross_ent_weights) + ell)


        # Finally, build the prediction function.
        predictions = self._build_predict(means, covars, inducing_inputs,
                                            kernel_chol, test_inputs,
                                            means_w = means_w, covars_weights = covars_weights)

        return (nelbo, entropy, entropy_weights, cross_ent, cross_ent_weights, ell, 
                gp_mean, gp_var, weights_mean, weights_var, off_values, predictions, kernel_mat, covars_weights)


    def _build_predict(self, means, covars, inducing_inputs,
                       kernel_chol, test_inputs,
                       means_w, covars_weights):


        # This is giving the prior parameters for the inducing processes giving the optimised hypermerameters 
        kern_prods, kern_sums, kern_prods_location = self._build_interim_vals(kernel_chol, inducing_inputs, test_inputs)
        # Given kern_prods and kern_sums this is reconstructing the parameters of the variational distribution on F 
        # to use to construct the predictions
        sample_means, sample_vars, sample_means_location = self._build_sample_GP_info(kern_prods, kern_sums, means, covars, kern_prods_location)

        pred_means, pred_vars, latent_means, latent_vars, offsets = self.likelihood.predict(sample_means, sample_vars, means_w, covars_weights, self.weights_optim) 

        return (pred_means, pred_vars, sample_means, sample_vars, means_w , covars_weights, offsets)


    def _build_entropy(self, means, covars): 
        # This function is building the entropy for the latent functions Eq()[logq()]
        sum_val = 0.0
        for i in range(self.num_latent):
            # Recostruct the full covars S starting from its cholesky
            full_covar = tf.matmul(covars[i, :, :], tf.transpose(covars[i, :, :]))
            trace = tf.reduce_sum(tf.matrix_diag_part(tf.cholesky_solve(covars[i, :, :],full_covar)))
            # trace = tf.reduce_sum(tf.eye(200))

            sum_val -= (CholNormal(means[i,:], covars[i, :, :]).log_prob(means[i,:]) - 0.5 * trace)
        return sum_val
        

    def _build_entropy_weights(self, means_w, covars_weights):
        # This function is building the entropy for the weights
        return self.weights.entropy(means_w, covars_weights) 


    def _build_cross_ent(self, means, covars, kernel_chol):
        # This function is building the cross entropy for the latent functions 
        sum_val = 0.0
        for i in range(self.num_latent):
            full_covar = tf.matmul(covars[i, :, :], tf.transpose(covars[i, :, :]))

            trace = tf.reduce_sum(tf.matrix_diag_part(tf.cholesky_solve(kernel_chol[i, :, :],full_covar)))
            sum_val += (CholNormal(means[i, :], kernel_chol[i, :, :]).log_prob(0.0) - 0.5 * trace)
        return sum_val



    def _build_cross_ent_weights(self, means_w, covars_weights, chol_var_weights, kernel_chol_weights):
        # This function is building the cross entropy for the weights 
        return self.weights.cross_entropy(means_w, covars_weights, chol_var_weights, kernel_chol_weights)


    def _build_ell(self, means, covars, inducing_inputs,
                   kernel_chol, train_inputs, train_outputs, 
                   means_w = None, covars_weights = None):

        # Construct values for the GPs
        kern_prods, kern_sums, kern_prods_location = self._build_interim_vals(kernel_chol, inducing_inputs, train_inputs)

        sample_means, sample_vars, sample_means_location = self._build_sample_GP_info(kern_prods, kern_sums, means, covars, kern_prods_location)
        # Construct samples for the GPs
        latent_samples = self._build_samples_GP(kern_prods, kern_sums, means, covars, kern_prods_location)
        # Construct samples for the weights
        weights_samples = self.weights.build_samples(self.num_samples_ell, means_w, covars_weights)

        (ell, f_mu, f_var, w_mean, w_var, off) = self.likelihood.log_cond_prob(self.num_samples_ell, train_outputs, latent_samples, sample_means, 
                                                                                    sample_vars, weights_samples, means_w, covars_weights, self.weights_optim,
                                                                                    means, kernel_chol, inducing_inputs, self.raw_kernel_params, 
                                                                                    sample_means_location, covars)

        return (ell, f_mu, f_var, w_mean, w_var, off)


    def _build_interim_vals(self, kernel_chol, inducing_inputs, inputs):
        # Starting from the values of means and vars for the inducing process we compute intermediate values given by 
        # product of kernels that are then needed to get the parameter values for q(F).
        # Create list to save the intermediate values
        kern_prods = util.init_list(0.0, [self.num_latent])
        kern_sums = util.init_list(0.0, [self.num_latent])
        kern_prods_location = util.init_list(0.0, [self.num_latent])

        for i in range(self.num_latent):
            # Compute the term kzx and Kzx_location
            ind_train_kern = self.kernels[i].kernel(inducing_inputs[i, :, :], inputs)
            location_inducing_kern = self.kernels[i].kernel(inducing_inputs[i, :, :], self.events_location)

            # Compute A = Kxz.Kzz^(-1) = (Kzz^(-1).Kzx)^T. for x and x_location
            # Note that kernel_chol is the cholesky of kzz. full_covar = K_zz
            kern_prods[i] = tf.transpose(tf.cholesky_solve(kernel_chol[i, :, :], ind_train_kern))
            kern_prods_location[i] = tf.transpose(tf.cholesky_solve(kernel_chol[i, :, :], location_inducing_kern))

            # Diagonal components of kxx - AKzx 
            kern_sums[i] = (self.kernels[i].diag_kernel(inputs) - util.diag_mul(kern_prods[i], ind_train_kern))
            
        # For each latent function q, this gives the Aq (NxN)
        kern_prods = tf.stack(kern_prods, 0) 
        kern_prods_location = tf.stack(kern_prods_location, 0)
        # For each latent function q, this gives the diagonal elements of k^q_xx - AqK^qzx (Nx1)
        kern_sums = tf.stack(kern_sums, 0)


        return kern_prods, kern_sums, kern_prods_location


    def _build_samples_GP(self, kern_prods, kern_sums, means, covars, kern_prods_location):
        # This function creates the samples from the latent functions that are used in the computation of the ell term
        sample_means, sample_vars, sample_means_location = self._build_sample_GP_info(kern_prods, kern_sums, means, covars, kern_prods_location)
        batch_size = tf.shape(sample_means)[0]
        return (sample_means + tf.sqrt(sample_vars) * tf.random_normal([self.num_samples_ell, batch_size, self.num_latent], seed=1))


    def _build_sample_GP_info(self, kern_prods, kern_sums, means, covars, kern_prods_location): 
        # This function is used to get the means and the cov matrices for the latent functions
        # starting from the values of the means and the cov matrices of the inducing processes.
        # This are then used in building the samples for the latent functions and thus in evaluating ell
        # sample means and sample vars are the parameters of q(F)
        sample_means = util.init_list(0.0, [self.num_latent])
        sample_means_location = util.init_list(0.0, [self.num_latent])
        sample_vars = util.init_list(0.0, [self.num_latent])

        for i in range(self.num_latent):
            # From the cholesky, we get back the full covariance for the inducing processes. This gives S.
            full_covar = tf.matmul(covars[i, :, :], tf.transpose(covars[i, :, :]))
            # quad form is giving the terms in (23), second formula, second term
            quad_form = util.diag_mul(tf.matmul(kern_prods[i, :, :], full_covar), tf.transpose(kern_prods[i, :, :]))
            quad_form_location = util.diag_mul(tf.matmul(kern_prods_location[i, :, :], full_covar), tf.transpose(kern_prods_location[i, :, :]))

            # (23), first formula
            sample_means[i] = tf.matmul(kern_prods[i, :, :], tf.expand_dims(means[i, :], 1))
            sample_means_location[i] = tf.matmul(kern_prods_location[i, :, :], tf.expand_dims(means[i, :], 1)) 
            
            # (23), second formula
            sample_vars[i] = tf.expand_dims(kern_sums[i, :] + quad_form, 1)

        # The means for each process and the diagonal terms of the covariance matrices for each process
        sample_means = tf.concat(sample_means, 1)
        sample_means_location = tf.concat(sample_means_location, 1) 
        sample_vars = tf.concat(sample_vars, 1)

        return sample_means, sample_vars, sample_means_location
