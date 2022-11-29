import numpy as np
import torch
from torchvision import transforms


class RDataset():
    """
    Dataset & Loader with augmentation options
    """

    def __init__(self, configuration, X, Y, norm_transform=None):
        self.configuration = configuration
        self.Labels = Y  # stores additional trial/label information
        self.Y = Y[:, 0]  # stores just labels
        self.X = X

        self.num_channels = X.shape[1]

        self.num_classes = len(np.unique(self.Y))
        self.size = self.Y.shape[0]

        transformations = []

        # Normalize
        if norm_transform:
            transformations.append(norm_transform)

        # Jittering
        jittering = configuration['jitter']['active']
        if jittering:
            jitter_factors = configuration['jitter']['sigmas']
            jitter_prob = configuration['jitter']['prob']
            transformations.append(RangeJittering(jitter_factors, jitter_prob))

        # Subsampling
        subsampling = configuration['subsample']['active']
        if subsampling:
            subsampling_rates = configuration['subsample']['rates']
            subsampling_prob = configuration['subsample']['prob']
            transformations.append(RandomRangeNeighbourSubsampling(subsampling_rates, subsampling_prob))

        self.transform = transforms.Compose(transformations)

    def __getitem__(self, index):
        series = self.X[index, :, :]
        label = self.Y[index]

        sample = (series, label)

        if self.transform:
            return self.transform(sample)

        return sample

    def __len__(self):
        return self.size

    def get_num_channels(self):
        return self.num_channels

    def get_num_classes(self):
        return self.num_classes

    def pre_epoch_callback(self, epoch):
        """Callback to be called before every epoch.
        """

    def post_epoch_callback(self, epoch):
        """Callback to be called after every epoch.
        """
