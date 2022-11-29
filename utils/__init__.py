import math

import torch
from data.const import (DOWN_SAMPLE_FREQUENCY, POST_STIMULUS_DURATION, PRE_STIMULUS_DURATION, SAMPLE_FREQUENCY,
                        STIMULUS_DURATION)
from torch.optim import lr_scheduler


def calc_accuracy(output, label):
    """
    Calculate accuracy of the model
    """
    predictions = torch.argmax(output, dim=1)

    correct = (predictions == label).sum()
    accuracy = correct.item() / label.size(0)

    correct_items = correct.item()
    total = label.size(0)

    return accuracy, correct_items, total


def intervals(trial_start_time,
              down_sample_frequency=DOWN_SAMPLE_FREQUENCY,
              sample_frequency=SAMPLE_FREQUENCY,
              pre_stimulus_duration=PRE_STIMULUS_DURATION,
              post_stimulus_duration=POST_STIMULUS_DURATION,
              stimulus_duration=STIMULUS_DURATION):
    """
        Given a starting time and a duration, return the start and end time of the interval
        """
    trial_start_time = math.floor(trial_start_time / down_sample_frequency)

    start_time = int(trial_start_time - (sample_frequency / down_sample_frequency * pre_stimulus_duration / 1000))

    end_time = int(trial_start_time + (sample_frequency / down_sample_frequency * stimulus_duration / 1000) +
                   (sample_frequency / down_sample_frequency * post_stimulus_duration / 1000))
    return start_time, end_time


def intervals_raw(trial_start_time,
                  pre_stimulus_duration=PRE_STIMULUS_DURATION,
                  post_stimulus_duration=POST_STIMULUS_DURATION,
                  stimulus_duration=STIMULUS_DURATION):
    """
        Given a starting time and a duration, return the start and end time of the interval
        """
    start_time = int(trial_start_time - (SAMPLE_FREQUENCY * pre_stimulus_duration / 1000))

    end_time = int(trial_start_time + (SAMPLE_FREQUENCY * stimulus_duration / 1000) +
                   (SAMPLE_FREQUENCY * post_stimulus_duration / 1000))
    return start_time, end_time


def get_scheduler(optimizer, configuration, last_epoch=-1):
    """Return a learning rate scheduler.
    """
    if configuration['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=configuration['lr_decay_iters'],
                                        gamma=0.1,
                                        last_epoch=last_epoch)
    elif configuration['lr_policy'] == 'lr_on_plateau':
        lr_patience = configuration['lr_patience']
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=lr_patience,
                                                   verbose=True)  # factor=0.5, min_lr=0.0001)
    elif configuration['lr_policy'] == 'none':
        scheduler = None
    else:
        return NotImplementedError(f'learning rate policy [{0}] is not implemented'.format(configuration['lr_policy']))
    return scheduler


def transfer_to_device(tensor, device):
    """Transfers pytorch tensors or lists of tensors to GPU. This
        function is recursive to be able to deal with lists of lists.
    """
    if isinstance(tensor, list):
        for i in range(len(tensor)):
            tensor[i] = transfer_to_device(tensor[i], device)
    else:
        tensor = tensor.to(device)
    return tensor
