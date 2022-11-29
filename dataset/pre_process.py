import numpy as np
import scipy
import scipy.ndimage.filters
from data.const import (SESSION_1_TRIALS, SESSION_2_TRIALS, SESSION_3_TRIALS, V1_CHANNEL_INDICES, V4_CHANNEL_INDECES)


def preprocess(configuration):
    X = np.load(f"{configuration['dataset_dir_path']}{configuration['X_file_name']}")
    Y = np.load(f"{configuration['dataset_dir_path']}{configuration['Y_file_name']}")

    print(f"Dataset of length: {len(X)}")

    # Start Indeces at 0
    Y = Y.astype(int)
    Y[:, 0] = Y[:, 0] - 1

    # NOTE: TEMPORARY REMOVING TRIALS
    trial_indeces = trials_for_sessions(configuration['recording_sessions'])
    X = X[trial_indeces, :, :]
    Y = Y[trial_indeces]

    # Drop false trials
    drop_false = configuration['drop_false_trials']
    monkey_performance_file = f"{configuration['dataset_dir_path']}{configuration['Performance_file_name']}"
    if drop_false:
        X, Y = drop_false_trials(X, Y, monkey_performance_file)

    # Drop unfamiliar stimuli
    drop_unfamiliar = configuration['familiar_stimuli_only']
    if drop_unfamiliar:
        X, Y = drop_unfamiliar_classes(X, Y)

    # Clip time series signal
    signal_start = configuration['signal_start_ms']
    signal_end = configuration['signal_end_ms']
    X = clip_signal(X, signal_start, signal_end)

    # V1 / V4 / All / Custom Split
    channels = configuration['channels']
    if isinstance(channels, str):
        X = drop_channels(X, channels)
    else:
        X = X[:, channels, :]
        print(f"Using custom channels: {X.shape}")

    # TEMPORARY SMOOTHING
    if configuration['smooth_signal'] != 0:
        X = scipy.ndimage.filters.gaussian_filter1d(X, configuration['smooth_signal'], axis=2)
        print("Smoothed Signal...")

    # Shuffle X & Y
    idx = np.random.permutation(len(X))
    X = X[idx]
    Y = Y[idx]

    return X, Y


def trials_for_sessions(sessions):
    """Based on recording session will return indeces of corresponding trials
    """
    trial_indeces = []
    if 1 in sessions:
        trial_indeces.extend(SESSION_1_TRIALS)
    if 2 in sessions:
        trial_indeces.extend(SESSION_2_TRIALS)
    if 3 in sessions:
        trial_indeces.extend(SESSION_3_TRIALS)
    return trial_indeces


def drop_channels(X, channels):
    if channels == 'v1':
        X = X[:, V1_CHANNEL_INDICES, :]
        print(f'Only using v1 channels: {X.shape}')
    elif channels == 'v4':
        X = X[:, V4_CHANNEL_INDECES, :]
        print(f'Only using v4 channels: {X.shape}')
    else:
        X = X
        print(f'Using all channels: {X.shape}')
    return X


def drop_false_trials(X, Y, monkey_performance_file):
    print("ENSURE THIS WORKS WITH THE NEW WAY!!!!!")
    monkey_performance = np.load(monkey_performance_file)
    correct_trial_indeces = monkey_performance[:, 3] == 1
    num_samples = len(X)
    X = X[correct_trial_indeces, :, :]
    Y = Y[correct_trial_indeces]
    num_correct_trials = num_samples - len(X)
    print(f"Dropping false trials ({num_correct_trials}): {num_samples} -> {len(X)}")

    return X, Y


def clip_signal(X, start, end):
    print(f"Clipping time series signal ({X.shape[2]}ms) =>  start: {start}ms, end: {end}ms")
    return X[:, :, start:end]


def drop_unfamiliar_classes(X, Y, max_class=8):
    familiar_indeces = np.where(Y[:, 0] < max_class)[0]
    numb_samples = len(X)
    num_familiar_samples = len(X) - len(familiar_indeces)
    X = X[familiar_indeces, :, :]
    Y = Y[familiar_indeces]
    print(f"Dropping unfamiliar stimuli ({num_familiar_samples}): {numb_samples} -> {len(X)}")

    return X, Y


def calculate_mean_std(data):
    """
    Calculate the mean of the data
    """
    return np.mean(data[:, :, :], axis=0), np.std(data[:, :, :], axis=0)


def _calculate_peak_per_channel(data):
    return np.amax(data, axis=1)


def _calculate_baseline_per_channel(data):
    return np.mean(data, axis=1)

