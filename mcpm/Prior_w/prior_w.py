import abc


class Prior_w:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def entropy(self, var_w):
        pass

    @abc.abstractmethod
    def cross_entropy(self, var):
        pass
