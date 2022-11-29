import numpy as np
import torch
from data.const import SESSION_1_TRIALS, SESSION_2_TRIALS, SESSION_3_TRIALS
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from transformers.normalize import Normalize

from dataset.dataset import RDataset
from dataset.pre_process import calculate_mean_std
from dataset.stratified_batch_sampler import create_stratified_batch_sampler


def create_datasets(X, Y, configuration):
    """
    Performs splitting of data into train/val/test datasets.
    Also calculates normalization parameters for training dataset.
    """
    test_split = configuration['train_test_split']
    val_split = configuration['train_val_split']
    seed = configuration['seed']

    # Split the dataset into training and test sets
    _x_train, x_test, _y_train, y_test = train_test_split(X,
                                                          Y,
                                                          test_size=test_split,
                                                          random_state=seed,
                                                          stratify=Y[:, 0])
    x_train, x_val, y_train, y_val = train_test_split(_x_train,
                                                      _y_train,
                                                      test_size=val_split,
                                                      random_state=seed,
                                                      stratify=_y_train[:, 0])

    # Normalization
    norm_transform = None

    if configuration['normalize'] == 'regular':
        means, stds = calculate_mean_std(x_train)
        norm_transform = Normalize(means, stds)
    else:
        print("No normalization...")

    print(y_train.shape, y_val.shape, y_train.shape)

    configuration['model_params']['weights'] = torch.ones(len(np.unique(Y[:, 0]))).tolist()

    train_data_loader = Loader(configuration['train_dataset_params'], x_train, y_train, norm_transform)
    train_dataset = train_data_loader.load_data()

    val_data_loader = Loader(configuration['val_dataset_params'], x_val, y_val, norm_transform)
    val_dataset = val_data_loader.load_data()

    test_data_loader = Loader(configuration['test_dataset_params'], x_test, y_test, norm_transform)
    test_dataset = test_data_loader.load_data()

    return train_dataset, val_dataset, test_dataset


def create_datasets_by_recording_day(X, Y, configuration, test_session=2):
    """
    Splits the training data and validation/test data by recording day.
    Also calculates normalization parameters for training dataset.
    """
    seed = configuration['seed']

    x_test = X[SESSION_2_TRIALS, :, :]
    y_test = Y[SESSION_2_TRIALS, :]

    remainig_trial_data = X[SESSION_1_TRIALS, :, :]
    remaining_trial_labels = Y[SESSION_1_TRIALS, :]

    x_train, x_val, y_train, y_val = train_test_split(remainig_trial_data,
                                                      remaining_trial_labels,
                                                      test_size=0.2,
                                                      random_state=seed,
                                                      stratify=remaining_trial_labels[:, 0])
    if test_session == 1:
        x_test = X[SESSION_1_TRIALS, :, :]
        y_test = Y[SESSION_1_TRIALS, :]

        remainig_trial_data = X[SESSION_2_TRIALS, :, :]
        remaining_trial_labels = Y[SESSION_2_TRIALS, :]

        x_train, x_val, y_train, y_val = train_test_split(remainig_trial_data,
                                                          remaining_trial_labels,
                                                          test_size=0.2,
                                                          random_state=seed,
                                                          stratify=remaining_trial_labels[:, 0])

    # Normalization
    norm_transform = None

    if configuration['normalize'] == 'regular':
        means, stds = calculate_mean_std(x_train)
        norm_transform = Normalize(means, stds)
    else:
        print("No normalization...")

    print(y_train.shape, y_val.shape, y_train.shape)

    configuration['model_params']['weights'] = torch.ones(len(np.unique(Y[:, 0]))).tolist()

    train_data_loader = Loader(configuration['train_dataset_params'], x_train, y_train, norm_transform)
    train_dataset = train_data_loader.load_data()

    val_data_loader = Loader(configuration['val_dataset_params'], x_val, y_val, norm_transform)
    val_dataset = val_data_loader.load_data()

    test_data_loader = Loader(configuration['test_dataset_params'], x_test, y_test, norm_transform)
    test_dataset = test_data_loader.load_data()

    return train_dataset, val_dataset, test_dataset


def create_datasets_splits(X, Y, configuration):
    seed = configuration['seed']
    kf = StratifiedKFold(n_splits=10)

    splits = list()

    for train_index, test_index in kf.split(X=X, y=Y[:, 0]):
        x_test = X[test_index, :, :]
        y_test = Y[test_index, :]

        x_train, x_val, y_train, y_val = train_test_split(X[train_index, :, :],
                                                          Y[train_index, :],
                                                          test_size=0.2,
                                                          random_state=seed,
                                                          stratify=Y[train_index, 0])
        # Normalization
        norm_transform = None

        if configuration['normalize'] == 'regular':
            means, stds = calculate_mean_std(x_train)
            norm_transform = Normalize(means, stds)
        else:
            print("No normalization...")

        train_data_loader = Loader(configuration['train_dataset_params'], x_train, y_train, norm_transform)
        train_dataset = train_data_loader.load_data()

        val_data_loader = Loader(configuration['val_dataset_params'], x_val, y_val, norm_transform)
        val_dataset = val_data_loader.load_data()

        test_data_loader = Loader(configuration['test_dataset_params'], x_test, y_test, norm_transform)
        test_dataset = test_data_loader.load_data()

        splits.append((train_dataset, val_dataset, test_dataset))
    return splits


class Loader():
    """Wrapper class of Dataset class that performs multi-threaded data loading
        according to the configuration.
    """

    def __init__(self, configuration, X, Y, norm_transform, batch_sampler=None):
        self.configuration = configuration
        self.dataset = RDataset(configuration, X, Y, norm_transform)

        if batch_sampler is None:
            self.dataloader = DataLoader(self.dataset, **configuration['loader_params'])
        else:
            self.dataloader = DataLoader(self.dataset, batch_sampler=batch_sampler)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data.
        """
        for batch in self.dataloader:
            yield batch
