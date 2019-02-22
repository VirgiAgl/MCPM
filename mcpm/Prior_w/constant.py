import prior_w
import tensorflow as tf

# Class for the Normal prior on the weights

class Constant(prior_w.Prior_w):
    MAX_DIST = 1e8
    
    def __init__(self, num_tasks = 1, num_latent = 1):
        self.num_tasks = num_tasks
        self.num_latent = num_latent

    def build_samples(self, num_samples, *args):
        return (0. + 1. * tf.random_normal([num_samples, self.num_tasks, self.num_latent], seed=5))
    
    def entropy(self, *args):
        return tf.zeros(1)
 
    def cross_entropy(self, *args):
        return tf.zeros(1)

