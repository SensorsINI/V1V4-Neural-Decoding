"""This file should ensure that a proper data path is set and files required during runtime can be found.
From this file the data_dir_path should always be imported and used.
"""
import argparse
import os

from data.const import NUMBER_OF_AMPLIFIERS, NUMBER_OF_CHANNELS

RAW_DATA_DIR = None #'<ADD PATH HERE>'
DATA_DIR = None #'<ADD PATH HERE>'


def check_required_files(data_dir=DATA_DIR):
    """Ensure all required processed files are present.
    """
    neural_rec_data = f"{data_dir}/rec_neural_data.npy"
    if not os.path.isfile(neural_rec_data):
        raise Exception(f"File '{neural_rec_data}' does not exist. Run 'data/generate_data.py' first")

    labels = f"{data_dir}/rec_labels.npy"
    if not os.path.isfile(labels):
        raise Exception(f"File '{labels}' does not exist. Run 'data/generate_data.py' first")

    checker_board_range = f"{data_dir}/checkerboard_data.npy"
    if not os.path.isfile(checker_board_range):
        raise Exception(f"File '{checker_board_range}' does not exist. Run 'data/generate_data.py' first")

    print("All required processed/generated data files are present...")


def check_required_raw_files(raw_dir=RAW_DATA_DIR):
    """Ensure all raw data files are present storing the MUA data.
    """
    # MUA DATA
    for amplifier in range(1, NUMBER_OF_AMPLIFIERS + 1):
        for channel in range(1, NUMBER_OF_CHANNELS + 1):
            path = f"{raw_dir}/MUA_instance{amplifier}_ch{channel}_downsample.mat"
            if not os.path.isfile(path):
                raise Exception(f"File '{path}' does not exist")

    # MEAN DATA
    for amplifier in range(1, NUMBER_OF_AMPLIFIERS + 1):
        path = f"{raw_dir}/visual_response_instance{amplifier}.mat"
        if not os.path.isfile(path):
            raise Exception(f"File '{path}' does not exist")

    # CHECKERBOARD DATA
    for amplifier in range(1, NUMBER_OF_AMPLIFIERS + 1):
        path = f'{raw_dir}/mean_MUA_instance{amplifier}_new.mat'
        if not os.path.isfile(path):
            raise Exception(f"File '{path}' does not exist")

    print("All required raw data files are present...")


if __name__ == "__main__":
    check_required_raw_files(RAW_DATA_DIR)
    check_required_files(DATA_DIR)
