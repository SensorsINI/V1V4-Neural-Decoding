import numpy as np


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        data, label = sample
        data = (data - self.mean) / self.std
        return (np.float32(data), label)
