import abc


class Likelihood:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def log_cond_prob(self, num_samples, outputs, latent, weights):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def get_params(self):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def predict(self, latent_means, latent_vars, weights_means, weights_vars):
        raise NotImplementedError("Subclass should implement this.")

