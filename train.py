import argparse
import os
import pickle
import random
import time


import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from data.const import NUMBER_OF_FAMILIAR_LABELS, NUMBER_OF_LABELS
from dataset import (Loader, create_datasets, create_datasets_by_recording_day, create_datasets_splits)
from dataset.pre_process import calculate_mean_std, preprocess
from models.model import Model
from se_util import save_importances, save_labels, save_predictions
from transformers.normalize import Normalize
from utils import calc_accuracy
from utils.configuration import parse_configuration


def run_cross_validation(X, Y, config, seed):
    sss = StratifiedKFold(n_splits=10)
    num_splits = sss.get_n_splits()
    print(f"Number of splits: {num_splits}")

    indeces = sss.split(X, Y[:, 0])

    i = 0
    accuracies = []

    run_name = config['run_name']

    for train_idx, test_idx in indeces:
        x_test = X[test_idx, :, :]
        y_test = Y[test_idx, :]

        x_train, x_val, y_train, y_val = train_test_split(X[train_idx, :, :],
                                                          Y[train_idx, :],
                                                          test_size=0.2,
                                                          random_state=seed,
                                                          stratify=Y[train_idx, 0])
        print(f"Test size {len(x_test)}")
        print(f"Train size {len(x_train)}")
        print(f"Validation size {len(x_val)}")

        config['model_params']['weights'] = torch.ones(len(np.unique(Y[:, 0]))).tolist()

        # Normalization
        norm_transform = None

        if config['normalize'] == 'regular':
            means, stds = calculate_mean_std(x_train)
            norm_transform = Normalize(means, stds)
        else:
            print("No normalization...")

        train_data_loader = Loader(config['train_dataset_params'], x_train, y_train, norm_transform)
        train_dataset = train_data_loader.load_data()
        val_data_loader = Loader(config['val_dataset_params'], x_val, y_val, norm_transform)
        val_dataset = val_data_loader.load_data()
        test_data_loader = Loader(config['test_dataset_params'], x_test, y_test, norm_transform)
        test_dataset = test_data_loader.load_data()

        config['run_name'] = f"{run_name}_split_{i}"
        acc, pred, labels, _ = train(config, train_dataset, val_dataset, test_dataset)
        np.save(f"{config['drop']}_predictions_{i}.npy", pred)
        np.save(f"{config['drop']}_labels_{i}.npy", labels)
        print(f"Accuracy for split {i}: {acc}")
        accuracies.append(acc)
        i += 1
    accuracy = np.sum(accuracies) / len(accuracies)
    print(f"Accuracy: {accuracy}")
    return accuracy


def train(configuration, train_loader, val_loader, test_loader):

    train_dataset_size = len(train_loader)
    val_dataset_size = len(val_loader)
    test_dataset_size = len(test_loader)

    training_data = train_loader
    validation_data = val_loader
    testing_data = test_loader

    print("train_dataset_size", train_dataset_size)
    print("val_dataset_size", val_dataset_size)
    print("test_dataset_size", test_dataset_size)

    num_stimuli = train_loader.dataset.get_num_classes()
    print(f'Number of stimuli: {num_stimuli}')

    num_channels = train_loader.dataset.get_num_channels()
    print(f'Number of channels: {num_channels}')

    print('Initializing model...')
    model_name = configuration['model_params']['model_name']
    model = Model(configuration['model_params'], num_stimuli, num_channels, model_name)
    model.setup()

    num_epochs = configuration['model_params']['max_epochs']
    patience = configuration['model_params']['patience']

    best_val_accuracy = 0
    best_val_loss = float('inf')
    best_acc_epoch = -1
    best_loss_epoch = -1
    patience_counter = 0  # counts how many epochs without improvement

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        total = 0
        correct = 0
        counter = 0
        losses = 0

        importance_weights = []
        importance_labels = []
        importance_predictions = []

        for _, data in enumerate(training_data):
            model.train()

            # unpack data from dataset and apply preprocessing
            model.set_input(data)
            model.forward()
            output, label, loss = model.backward()

            # calculate loss functions, get gradients, update network weights
            model.optimize_parameters()

            _, corr, tot = calc_accuracy(output, label)
            correct += corr
            total += tot
            counter += 1
            losses += loss
            if model_name == "CNNSE":
                importance = extract_importance(model)
                importance_weights.append(importance.detach().cpu().numpy())
                importance_labels.append(label.detach().cpu().numpy())
                importance_predictions.append(output.detach().cpu().numpy())

        for _, data in enumerate(validation_data):
            model.eval()
            model.set_input(data)
            model.test()

        # Validation Metrics get calculate here
        cur_val_accuracy, cur_val_loss, _ = model.post_epoch_callback(epoch)

        print("Validation Accuracy:", cur_val_accuracy)
        print("Valdiation Loss:", cur_val_loss)

        print(f'End of epoch {epoch} / {num_epochs} \t Time Taken: {time.time() - epoch_start_time} sec')

        # check for lr update every epoch
        lr = model.update_learning_rate(cur_val_accuracy)

        epsilon = 1e-3
        save_epoch = False

        # Keep track of patience
        if cur_val_accuracy > (best_val_accuracy + epsilon):
            best_acc_epoch = epoch
            best_val_accuracy = cur_val_accuracy
            patience_counter = 0
            save_epoch = True
            print("New best val acc")
            if model_name == "CNNSE":
                save_importances(importance_weights)
                save_labels(importance_labels)
                save_predictions(importance_predictions)
        elif cur_val_accuracy < (best_val_accuracy + epsilon):
            patience_counter += 1
            print(f"Patience left: {patience - patience_counter}")
            if patience_counter > patience:
                print(f'Early stopping at epoch {epoch}')
                break

        # Keep track of best val loss
        if (cur_val_loss + epsilon) < best_val_loss:
            best_loss_epoch = epoch
            best_val_loss = cur_val_loss

        if save_epoch:
            print(f'Saving model at the end of epoch {epoch}')
            model.save_network(epoch)
            model.save_optimizer(epoch)

    return inference(epoch=epoch,
                     best_epoch=best_acc_epoch,
                     testing_data=testing_data,
                     labels=test_loader.dataset.Y,
                     configuration=configuration,
                     NUM_CHANNELS=num_channels,
                     NUM_CLASSES=num_stimuli)


def inference(epoch, best_epoch, testing_data, labels, configuration, NUM_CHANNELS, NUM_CLASSES):
    print('Reinitializing model...')
    model_name = configuration['model_params']['model_name']
    model = Model(configuration['model_params'], NUM_CLASSES, NUM_CHANNELS, model_name)
    model.setup()
    model.load_network(best_epoch)
    model.load_optimizer(best_epoch)
    model.eval()
    for _, data in enumerate(testing_data):
        model.set_input(data)
        model.test()

    test_accuracy, test_loss, predictions = model.post_epoch_callback(epoch)

    print("Test Loss", test_loss)
    print("Test Accuracy", test_accuracy)

    # Save Predictions and Solution
    print('Saving predictions and solution...')
    results_dir = configuration['model_params']['save_dir']
    np.save(os.path.join(results_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(results_dir, 'labels.npy'), labels)

    ############
    predictions = np.load(os.path.join(results_dir, 'predictions.npy'))
    labels = np.load(os.path.join(results_dir, 'labels.npy'))

    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)

    predictions = torch.argmax(predictions, dim=1)

    ### ATTENTION WEIGHTS
    weights = None

    if model.name == "CNNSE":
        se = model.net.se
        weights = se.y[:, :]
        weights = weights.detach().cpu()

    return test_accuracy, predictions, labels, weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', help='path to the configfile')
    parser.add_argument('--run', help='description of the run', default=None, required=True)
    parser.add_argument('--drop', help='where to drop files or file prefix', default=None, required=True)

    args = parser.parse_args()

    # Ensure Config File Path exists
    if not os.path.exists(args.configfile):
        raise Exception('Config File Path does not exist')

    print('Reading config file...')
    config_params = parse_configuration(args.configfile)
    config_params['run_name'] = args.run
    config_params['drop'] = args.drop

    SEED = config_params['seed']
    if SEED is None:
        raise Exception('No seed determined!!')

    print(f"Using seed: {SEED}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # For Cuda Usage
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    X, Y = preprocess(configuration=config_params)

    # Run Cross Validation
    if config_params['cross_validation']:
        print('Running cross validation...')
        run_cross_validation(config=config_params, X=X, Y=Y, seed=SEED)
    else:
        print("Running single run")
        raise NotImplementedError("Implement single run")
