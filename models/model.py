import os

import numpy as np
import torch
from utils import calc_accuracy, get_scheduler, transfer_to_device

from models.cnn import CNN
from models.cnn_se import CNNSE
from models.gru import GRU


class Model():

    def __init__(
        self,
        configuration,
        number_of_stimuli,
        number_of_channels,
        network_name,
    ):
        self.configuration = configuration
        self.is_train = True
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0') if self.use_cuda else torch.device('cpu')
        print("DEVICE:", self.device)
        torch.backends.cudnn.benchmark = True
        self.save_dir = configuration['checkpoint_path']

        ####
        self.number_of_channels = number_of_channels
        self.number_of_stimuli = number_of_stimuli
        self.num_channels = number_of_channels
        self.name = network_name

        if self.name == "CNN":
            self.net = CNN(num_classes=self.number_of_stimuli, input_channels=self.number_of_channels)
        elif self.name == "CNNSE":
            self.net = CNNSE(num_classes=self.number_of_stimuli, input_channels=self.number_of_channels)
        elif self.name == "GRU":
            number_hidden = configuration['hidden_size']
            number_levels = configuration['num_layers']
            dropout = configuration['dropout']
            bidirectional = configuration['bidirectional']
            self.net = GRU(number_levels=number_levels,
                           number_of_stimuli=self.number_of_stimuli,
                           number_hidden=number_hidden,
                           NUMBER_OF_CHANNELS=number_of_channels,
                           dropout=dropout,
                           bidirectional=bidirectional)

        self.net = self.net.to(self.device)

        self.class_weights = torch.tensor(configuration['weights']).float().cuda()
        self.loss = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        self.model_loss = float('inf')
        self.input = None
        self.label = None
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=configuration['lr'],
                                          weight_decay=configuration['weight_decay'])

        self.val_predictions = []
        self.val_labels = []

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        self.input = transfer_to_device(input[0].float(), self.device)
        self.label = transfer_to_device(input[1], self.device)

    def forward(self):
        self.output = self.net(self.input)

    def backward(self):
        self.model_loss = self.loss(self.output, self.label)
        return self.output, self.label, self.model_loss.item()  # TODO: remove this return?

    def optimize_parameters(self):
        # Reset gradient
        self.optimizer.zero_grad()

        # Backward-pass
        self.model_loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()

    def setup(self):
        """Load and print networks; create schedulers.
        """
        if self.configuration['load_checkpoint'] >= 0:
            last_checkpoint = self.configuration['load_checkpoint']
        else:
            last_checkpoint = -1

        if last_checkpoint >= 0:
            # enable restarting training
            self.load_network(last_checkpoint)
            if self.is_train:
                self.load_optimizer(last_checkpoint)
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0][
                    'initial_lr']  # reset learning rate

        self.scheduler = get_scheduler(self.optimizer, self.configuration)

        if last_checkpoint > 0:
            if self.scheduler:
                for _ in range(last_checkpoint):
                    self.scheduler.step()
        self.print_network()

    def train(self):
        """Make model train mode during test time."""
        self.net.train()

    def eval(self):
        """Make models eval mode during test time."""
        self.net.eval()

    def test(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        """
        with torch.no_grad():
            self.forward()

        # self.val_data.append(self.input)
        self.val_predictions.append(self.output)
        self.val_labels.append(self.label)

    def update_learning_rate(self, metrics):
        """Update learning rates the network; called at the end of every epoch"""

        if self.scheduler:
            self.scheduler.step(metrics)

        learning_rate = self.optimizer.param_groups[0]['lr']
        return learning_rate

    def save_network(self, epoch, file_name=None):
        """Save the network to the disk.
        """
        # save_filename =
        if file_name == None:
            file_name = f'{epoch}_net_{self.name}.pth'
        save_path = os.path.join(self.save_dir, file_name)

        if self.use_cuda:
            torch.save(self.net.cpu().state_dict(), save_path)
            self.net.to(self.device)
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def load_network(self, epoch, load_filename=None):
        """Load all the networks from the disk.
        """
        if load_filename == None:
            load_filename = f'{epoch}_net_{self.name}.pth'
        load_path = os.path.join(self.save_dir, load_filename)

        if isinstance(self.net, torch.nn.DataParallel):
            self.net = self.net.module

        print(f'loading the model from {load_path}')
        state_dict = torch.load(load_path, map_location=self.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        self.net.load_state_dict(state_dict)
        return state_dict

    def save_optimizer(self, epoch, save_filename=None):
        """Save the optimizer to the disk for restarting training.
        """
        if save_filename == None:
            save_filename = f'{epoch}_optimizer_{self.name}.pth'
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(self.optimizer.state_dict(), save_path)

    def load_optimizer(self, epoch, load_filename=None):
        """Load all the optimizers from the disk.
        """
        if load_filename == None:
            load_filename = f'{epoch}_optimizer_{self.name}.pth'
        load_path = os.path.join(self.save_dir, load_filename)
        print(f'loading the optimizer from {load_path}')
        state_dict = torch.load(load_path)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        self.optimizer.load_state_dict(state_dict)

    def print_network(self):
        """Print the total number of parameters in the network and network architecture.
        """
        print('Network initialized')

        num_params = self.number_params()
        print(self.net)
        print('[Network {0}] Total number of parameters : {1:.3f} M'.format(self.name, num_params / 1e6))

    def number_params(self):
        """Count the number of parameters in the network.
        """
        num_params = 0
        for param in self.net.parameters():
            num_params += param.numel()
        return num_params

    def set_requires_grad(self, requires_grad=False):
        """Set requies_grad for all the networks to avoid unnecessary computations.
        """
        for param in self.net.parameters():
            param.requires_grad = requires_grad

    def get_current_loss(self):
        """Returns the current model loss.
        """
        return self.model_loss

    def pre_epoch_callback(self, epoch):
        pass

    def post_epoch_callback(self, epoch):
        total = 0
        tot_corr = 0
        losses = 0
        for _, (prediction, label) in enumerate(zip(self.val_predictions, self.val_labels)):
            _, correct, tot = calc_accuracy(prediction, label)
            tot_corr += correct
            total += tot
            losses += self.loss(prediction, label)

        losses /= len(self.val_predictions)
        accuracy = tot_corr / total
        val_preds = [val.cpu() for val in self.val_predictions]
        predictions = np.vstack(val_preds)

        self.val_predictions = []
        self.val_labels = []

        return float(accuracy), float(losses), predictions
