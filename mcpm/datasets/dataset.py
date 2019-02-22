import numpy as np

# This is creating the object for the dataset

class DataSet():

    def __init__(self, X, Y, shuffle=False):
        
        # Get the number of training obs
        self._num_examples = X.shape[0]

        # If shuffle = True, we shuffle the training data. The current implementation does 
        # not allow for shuffling of the data case we have missing values and indeces for missing values that are fixed. 
        perm = np.arange(self._num_examples)
        # if (shuffle):
        #     np.random.shuffle(perm)
        self._X = X[perm,:]
        self._Y = Y[perm,:]

        # Set the initial value of iterations to zero
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
        # Gives the dimension for each obs
        self._Din = X.shape[1]
        # Gives the number of tasks
        self._Dout = Y.shape[1]

    def next_batch(self, batch_size):

        # When training, this functions pass the batch of data to use
        # If only one batch is used that it returns the overall dataset
        # It keeps track of the epochs completed with the variable _epochs_completed 
        # which is incremented by 1 everytime we have a complete pass over the data
        
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        
        # This is the case of batch opt in which we consider the overall dataset
        if (self._index_in_epoch > self._num_examples) and (start != self._num_examples):
            self._index_in_epoch = self._num_examples
        
        if self._index_in_epoch > self._num_examples:   # Finished epoch
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            #np.random.shuffle(perm)                  # Shuffle the data
            self._X = self._X[perm,:]
            self._Y = self._Y[perm,:]
            start = 0                               # Start next epoch
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._X[start:end,:], self._Y[start:end,:]

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def Din(self):
        return self._Din

    @property
    def Dout(self):
        return self._Dout

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y


