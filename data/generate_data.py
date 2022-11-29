import argparse
import os
import sys

import numpy as np
from data.const import (CHECKERBOARD_DURATION, NUMBER_OF_AMPLIFIERS, NUMBER_OF_CHANNELS, NUMBER_OF_TIME_BINS,
                        NUMBER_OF_TRIALS, SESSION_2_START_TRIAL, SESSION_2_STARTS, SESSION_3_START_TRIAL,
                        SESSION_3_STARTS)
from scipy import io
from tqdm.auto import trange
from utils import intervals


def generate_dataset(data_dir, output_dir):
    """
    Consumes raw MUA data files and generates two numpy arrays:
    - data: a numpy array of shape (NUMBER_OF_TRIALS, NUMBER_OF_AMPLIFIERS * NUMBER_OF_CHANNELS,
         NUMBER_OF_TIME_BINS)
    - labels: a numpy array of shape (NUMBER_OF_TRIALS, 3)
    """
    neural_data = np.zeros((NUMBER_OF_TRIALS, NUMBER_OF_AMPLIFIERS * NUMBER_OF_CHANNELS, NUMBER_OF_TIME_BINS))
    labels = np.zeros((NUMBER_OF_TRIALS, 3))

    # Same label information is stored on each file
    label_file = f"{data_dir}/MUA_instance{1}_ch{1}_downsample.mat"
    label_data = io.loadmat(label_file)
    for trial in range(NUMBER_OF_TRIALS):
        trial_label = label_data['goodTrialCondsMatch'][trial]
        labels[trial, :] = trial_label

    for amplifier in trange(NUMBER_OF_AMPLIFIERS, desc='1st Loop: Amplifiers'):
        for channel in trange(NUMBER_OF_CHANNELS, desc='2nd Loop: Channels'):
            file_name = f"{data_dir}/MUA_instance{amplifier+1}_ch{channel+1}_downsample.mat"
            data = io.loadmat(file_name)
            channel_data = data['channelDataMUA'][0]
            trial_starting_times = data['timeStimOnsMatch'][0, :]

            for trial in range(0, NUMBER_OF_TRIALS):
                start_time, end_time = intervals(trial_starting_times[trial])

                #### ADAPT FOR SESSIONS 2 & 3 ####
                if trial >= SESSION_2_START_TRIAL:
                    start_time += SESSION_2_STARTS[amplifier]
                    end_time += SESSION_2_STARTS[amplifier]
                    if trial >= SESSION_3_START_TRIAL:
                        start_time += SESSION_3_STARTS[amplifier]
                        end_time += SESSION_3_STARTS[amplifier]
                #### ADAPT FOR SESSIONS 2 & 3 ####

                trial_channel_data = channel_data[start_time:end_time]
                neural_data[trial, amplifier * NUMBER_OF_CHANNELS + channel,] = trial_channel_data

    # Save neural data for all instances
    np.save(f'{output_dir}/rec_neural_data', neural_data)
    np.save(f'{output_dir}/rec_labels', labels)
    return neural_data, labels


def generate_checkerboard_response(data_dir, output_dir):
    """Generates response range of each channel for checkerboard stimuli
    """
    checkerboard_data = np.ones((NUMBER_OF_AMPLIFIERS * NUMBER_OF_CHANNELS, CHECKERBOARD_DURATION))

    for amplifier in range(NUMBER_OF_AMPLIFIERS):
        checkerbord_file = f'{data_dir}/mean_MUA_instance{amplifier+1}_new.mat'
        checker_data = io.loadmat(checkerbord_file)
        channel_data = checker_data['meanChannelMUA']

        start = amplifier * NUMBER_OF_CHANNELS
        end = amplifier * NUMBER_OF_CHANNELS + NUMBER_OF_CHANNELS
        checkerboard_data[start:end,] = channel_data

    np.save(f'{output_dir}/checkerboard_data', checkerboard_data)
    return checkerboard_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a dataset for recurrent models')
    parser.add_argument('--data_dir',
                        type=str,
                        default='./datasets/raw_recordings/',
                        help='Directory containing the data')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./datasets/recordings_processed/',
                        help='Directory to save the data')

    args = parser.parse_args()

    # Validate Config
    if not os.path.isdir(args.data_dir):
        print(f"{args.data_dir} does not exist")
        sys.exit()

    if not os.path.isdir(args.output_dir):
        print(f"{args.output_dir} does not exist")
        sys.exit()

    # Generate Data
    gen_data, gen_labels = generate_dataset(args.data_dir, args.output_dir)
    print("Data Generated:", gen_data.shape)
    print("Labels Generated:", gen_labels.shape)

    # Generate Checkerboard Data
    gen_checkerboard_data = generate_checkerboard_response(args.data_dir, args.output_dir)
    print("Checkerboard Data Generated:", gen_checkerboard_data.shape)
