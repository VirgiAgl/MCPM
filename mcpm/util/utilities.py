from scipy.spatial import distance 
import tensorflow as tf 
import numpy as np
import pyproj 
import scipy 
from scipy.integrate import quad
import scipy.special
from tensorflow.python.framework import ops

from mcpm.util.util import init_list

def euclidean(vector1, vector2):
    '''calculate the euclidean distance, no numpy
    input: numpy.arrays or lists
    return: euclidean distance
    '''
    dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
    dist = math.sqrt(sum(dist))
    return dist

def euclidean2(vector1, vector2):
    ''' use scipy to calculate the euclidean distance. '''
    dist = distance.euclidean(vector1, vector2)
    return dist



def empirical_median(N_all, n_tasks, num_latent, GP_mean, GP_var, means_w, var_w, offsets, n_bins, n_sample_prediction):
                
    GP_samples = np.random.normal(GP_mean, np.sqrt(GP_var), size=(n_sample_prediction, N_all, num_latent))
    weights_samples = np.random.normal(means_w, np.sqrt(var_w), size=(n_sample_prediction, num_latent, n_tasks))
    offsets = np.sum(offsets, axis = 1)
    # Get samples from lambda 
    lambda_sample = np.transpose(np.exp(np.matmul(np.transpose(weights_samples, (0,2,1)),np.transpose(GP_samples, (0,2,1))) + offsets[:,np.newaxis]), (0,2,1)) 
    
    median_predictions = np.zeros((N_all, n_tasks))
    for t in xrange(n_tasks):
        for i in xrange(N_all):
            range_bins = (np.min(lambda_sample[:,i,t]),np.max(lambda_sample[:,i,t]))
            values = lambda_sample[:,i,t]
            freq, bins = np.histogram(values, bins = int(n_bins), range=range_bins)
            median_predictions[i,t] = (bins[np.where(freq == freq[~(np.cumsum(freq) < n_sample_prediction/2)][0])[0][0]] + bins[np.where(freq == freq[~(np.cumsum(freq) < n_sample_prediction/2)][0])[0][0] + 1])/2
    return median_predictions


def empirical_mode(N_all, n_tasks, num_latent, GP_mean, GP_var, means_w, var_w, offsets, n_bins):
    GP_samples = np.random.normal(GP_mean, np.sqrt(GP_var), size=(n_sample_prediction, N_all, num_latent))
    weights_samples = np.random.normal(means_w, np.sqrt(var_w), size=(n_sample_prediction, num_latent, n_tasks))
    offsets = np.sum(offsets, axis = 1)
    # Get samples from lambda 
    lambda_sample = np.transpose(np.exp(np.matmul(np.transpose(weights_samples, (0,2,1)),np.transpose(GP_samples, (0,2,1))) + offsets[:,np.newaxis]), (0,2,1)) 
    
    mode_predictions = np.zeros((N_all, n_tasks))
    for t in xrange(n_tasks):
        for i in xrange(N_all):
            range_bins = (np.min(lambda_sample[:,i,t]),np.max(lambda_sample[:,i,t]))
            values = lambda_sample[:,i,t]
            freq, bins = np.histogram(values, bins = int(n_bins), range=range_bins)
            mode_predictions[i,t] = (bins[np.where(freq == np.max(freq))[0][0] + 1] + bins[np.where(freq == np.max(freq))[0][0]])/2
    return mode_predictions


# Converting lat and long in NAD83 coordinates
def LatLong_NAD83(inputs):
    inProj = pyproj.Proj(init='epsg:4326') # this is lat long
    outProj = pyproj.Proj(init='epsg:26918') # nad 83 zone 18N
    lat = inputs[:,1]
    lon = inputs[:,0]
    new_lat, new_log = pyproj.transform(inProj,outProj, lat, lon)
    
    new_inputs = np.zeros((inputs.shape[0], inputs.shape[1]))
    new_inputs[:,0] = new_log
    new_inputs[:,1] = new_lat
    inputs = new_inputs
        
    inputs_mean = np.transpose(np.mean(inputs, axis = 0)[:,np.newaxis])
    inputs_std = np.transpose(np.std(inputs, axis = 0)[:,np.newaxis])

    standard_inputs = (inputs - inputs_mean)/inputs_std
        
    inputs = standard_inputs

    return inputs

def get_features(outputs, num_features = 2):
    n_tasks = outputs.shape[1]
    print("n_tasks",n_tasks)
    task_features = np.zeros((n_tasks,num_features))
    for i in xrange(n_tasks):
        output_toconsider = outputs[:,i]
        maximum = max(output_toconsider)
        minimum = min(output_toconsider)
        task_features[i,:] = np.array([maximum, minimum])
    return task_features


def define_crime_folds(missing_exp, N):
    if missing_exp == True:
        slice_x1 = slice(0,int(N/2.))
        slice_x2 = slice(int(N/2.),N)

        slice_y1 = slice(0,int(N/2.))
        slice_y2 = slice(int(N/2.),N)
    else:
        slice_x1 = slice(0,0)
        slice_x2 = slice(0,0)

        slice_y1 = slice(0,0)
        slice_y2 = slice(0,0)
    list_indeces_product = list([(slice_x1, slice_y1), (slice_x1, slice_y2),
                            (slice_x2, slice_y2), (slice_x2, slice_y1), 
                            (slice_x1, slice_y1), (slice_x1, slice_y2), 
                            (slice_x2, slice_y2), (slice_x2, slice_y1)])
    return list_indeces_product

def define_crime_folds_1D(missing_exp, N_all):
    if missing_exp == True:
        slice_x1 = slice(0,int(N_all/2.))
        slice_x2 = slice(int(N_all/2.),N_all)
    else:
        slice_x1 = slice(0)
        slice_x2 = slice(0)
    list_indeces_product = list([(slice_x1), (slice_x2)])
    return list_indeces_product


def define_btb_folds(missing_exp, N):
    if missing_exp == True:
        slice_x1 = slice(0,16)
        slice_x2 = slice(16,32)
        slice_x3 = slice(32,48)
        slice_x4 = slice(48,64)

        slice_y1 = slice(0,16)
        slice_y2 = slice(16,32)
        slice_y3 = slice(32,48)
        slice_y4 = slice(48,64)
    else:
        slice_x1 = slice(0,0)
        slice_x2 = slice(0,0)
        slice_x3 = slice(0,0)
        slice_x4 = slice(0,0)

        slice_y1 = slice(0,0)
        slice_y2 = slice(0,0)
        slice_y3 = slice(0,0)
        slice_y4 = slice(0,0)

    list_indeces_product = list([(slice_x1, slice_y1), (slice_x1, slice_y2), (slice_x1, slice_y3), (slice_x1, slice_y4) ,
                                (slice_x2, slice_y1), (slice_x2, slice_y2),(slice_x2, slice_y3), (slice_x2, slice_y4), 
                                (slice_x3, slice_y1), (slice_x3, slice_y2), (slice_x3, slice_y3), (slice_x3, slice_y4),
                                (slice_x4, slice_y1), (slice_x4, slice_y2), (slice_x4, slice_y3), (slice_x4, slice_y4), 
                                (slice_x1, slice_y1), (slice_x1, slice_y2), (slice_x1, slice_y3), (slice_x1, slice_y4) ,
                                (slice_x2, slice_y1), (slice_x2, slice_y2),(slice_x2, slice_y3), (slice_x2, slice_y4), 
                                (slice_x3, slice_y1), (slice_x3, slice_y2), (slice_x3, slice_y3), (slice_x3, slice_y4),
                                (slice_x4, slice_y1), (slice_x4, slice_y2), (slice_x4, slice_y3), (slice_x4, slice_y4)])

    return list_indeces_product


# def integrand_function(x, a, b):
#     return np.exp(a*x + b*x**2)

# def integrand_function_normal(x, c, lenghtscale, z):
#     second = np.exp(- (x-z)**2/(2.*lenghtscale))
#     value = np.matmul(np.transpose(c),second)
#     return value

# def tf_integration(a, b, z):
#     # This function gives the value of the erfi function corresponding to a specific x
#     x_max = np.max(z)
#     x_min = np.min(z)

#     value = np.float32(np.array((quad(integrand_function, x_min, x_max, args=(a,b))[0])))
#     return value

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates in the grads names:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    # print('rnd', rnd_name)

    # Get the gradient grad and give it the name rnd_name
    # gradient_override_map tells tf which gradient to consider for rnd_name
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def myDawErfi(x):
    # Numerically more stable implementation of the erfi function

    res = np.float32((2./np.sqrt(np.pi))*np.exp(x**2)*scipy.special.dawsn(x))
    # res = np.float32(np.minimum(res,1.0e+30))
    # res = np.float32(np.maximum(res,-1.0e+30))

    # res = np.float32(np.clip(res,-1.0e+30, 1.0e+30))

    return res

def myerfi(x, name=None):
    with ops.name_scope(name, "MyErfi", [x]) as name:
        # erfi_x = py_func(scipy.special.erfi,
        #                 [x],
        #                 [tf.float32],
        #                 name=name,
        #                 grad=_MyErfiGrad)  # <-- here's the call to the gradient
        erfi_x = py_func(myDawErfi,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=_MyErfiGrad)  # <-- here's the call to the gradient
        return erfi_x[0]


def _MyErfiGrad(op, grad):
    """Returns grad * 2/sqrt(pi) * exp(x**2)."""
    x = op.inputs[0]
    two_over_root_pi = 2. / np.sqrt(np.pi)
    x = tf.conj(x)
    return grad * two_over_root_pi * tf.exp(tf.square(x))


def construct_lag_counts(data):
    features = np.zeros((data.shape[0],data.shape[1]*2))

    print('features shape',features.shape)

    for t in xrange(data.shape[1]):
        # lag 1
        for i in xrange(data.shape[0]):
            if i == 0:
                features[i,t] = 0
            else:
                # event counts in the interval [i-1,i]
                features[i,t] = data[i-1,t] 

        #lag5
        for i in xrange(data.shape[0]):
            if i <= 5:
                if i == 0:
                    features[i,t+data.shape[1]] = 0
                else:
                    features[i,t+data.shape[1]] = np.sum(data[:(i),t])
                
            else:
                # event counts in the interval [i-5,i]
                features[i,t+data.shape[1]] = np.sum(data[(i-5):(i),t]) 
    return features

def construct_features(current_index, test_inputs, pred_means, train_outputs, num_tasks, num_training_obs):
    #Augment the train output with the current mean estimate
    train_outputs = tf.concat((train_outputs,pred_means), axis = 0)
    print('train_outputs in util', train_outputs)

    lag1 = init_list(0.0, [num_tasks])
    lag5 = init_list(0.0, [num_tasks])

    for i in xrange(num_tasks):
        lag1[i] = train_outputs[num_training_obs+current_index,i]
        lag5[i] = tf.reduce_sum(train_outputs[(num_training_obs+current_index-5):,i])

    lag1 = tf.stack(lag1)
    lag5 = tf.stack(lag5)
    lags = tf.concat((lag1,lag5), axis = 0)
    one_step_ahead_input = test_inputs[current_index+1]
    one_step_ahead_input = tf.transpose(tf.expand_dims(tf.concat((one_step_ahead_input,lags), axis = 0), axis = 1))

    return one_step_ahead_input, train_outputs

