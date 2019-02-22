    
import numpy as np
import sklearn.cluster
import warnings

np.random.seed(1500)

def initialize_inducing_points(train_inputs, train_outputs, inducing_on_inputs, num_latent, num_inducing, num_data_points, input_dim):
    """
    Initialize the position of inducing points and the initial posterior distribution means.
    Parameters
    ----------
    train_inputs : ndarray
        Input data. Dimensions: num_train * input_dim.
    train_outputs : ndarray
        Output data. Dimensions: num_train * output_dim.
    inducing_on_inputs : bool
        If True, initializes the inducing points on the input data otherwise, inducing points
        are initialized using clustering.
    num_latent : num 
        num of latent_functions
    num_inducing : num
        num of required inducing points
    num_data_points : num
        Number of training points
    input_dim : num
        Dimension of the inputs D

    Returns
    -------
    inducing_locations : ndarray
        An array of inducing point locations. Dimensions: num_latent * num_inducing * input_dim.
    initial_mean : ndarray
        Initial value for the mean of the posterior distribution.
        Dimensions: num_inducing * num_latent.
    """
    # Notice that we are generating inducing points which are specific for each latent process
    inducing_locations = np.zeros([num_latent, num_inducing, input_dim], dtype=np.float32)
    initial_mean = np.empty([num_latent, num_inducing], dtype=np.float32)

    if inducing_on_inputs or num_inducing == num_data_points:
        # Initialize inducing points on training data.
        for i in xrange(num_latent):
            inducing_index = np.random.permutation(num_data_points)[:num_inducing]
            inducing_locations[i] = train_inputs[inducing_index]
        for i in xrange(num_inducing):
            initial_mean[:, i] = np.mean(train_outputs[inducing_index[i]], axis = 0)
    else:
        # Initialize inducing points using clustering.
        mini_batch = sklearn.cluster.MiniBatchKMeans(num_inducing)
        with warnings.catch_warnings():
            # Squash deprecation warning in some older versions of scikit.
            warnings.simplefilter("ignore")
            cluster_indices = mini_batch.fit_predict(train_inputs)

        for i in xrange(num_latent):
            inducing_locations[i] = mini_batch.cluster_centers_
        # for i in xrange(num_inducing):
        #     data_indices, = np.where(cluster_indices == i)
        #     if data_indices.shape[0] == 0:
        #         # No points in this cluster so set the mean across all data points.
        #         initial_mean[:, i] = np.mean(train_outputs, axis = 0)
        #     else:
        #         initial_mean[:, i] = np.mean(train_outputs, axis = 0)
        #         #initial_mean[:, i] = np.mean(train_outputs[data_indices], axis = 0)

    return inducing_locations, initial_mean


