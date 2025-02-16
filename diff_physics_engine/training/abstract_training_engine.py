from typing import List, Dict, Tuple

import torch
import tqdm
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from diff_physics_engine.simulators.abstract_simulator import AbstractSimulator


class AbstractTrainingEngine(torch.nn.Module):

    def __init__(self,
                 trainable_params: Dict,
                 other_params: Dict,
                 simulator: AbstractSimulator,
                 train_data_loader: DataLoader,
                 val_data_loader: DataLoader,
                 optim_params: Dict,
                 criterion: _Loss):
        """

        :param trainable_params: Dict of torch.nn.Parameter(s) that can be updated by backprop
        :param other_params: Dict of torch.Tensors, other params needed for simulator that are not trainable
        :param simulator: function with step() to take in current state and external forces/pts to produce
                          next state
        :param train_data_loader: torch DataLoader object that can be iterated through for training
        :param val_data_loader: torch DataLoader object that can be iterated through for validation
        :param optim_params: Optimizer parameters for Adam
        :param criterion: Loss function
        """
        super().__init__()

        self.trainable_params = torch.nn.ParameterDict(trainable_params)
        self.other_params = other_params
        self.params = {**trainable_params, **other_params}

        self.simulator = self.init_simulator(simulator)
        self.optimizer = torch.optim.Adam(self.parameters(), **optim_params)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.994)
        self.criterion = criterion

        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

        self.device = "cpu"  # default

    def move_tensors(self, device):
        self.device = device
        self.to(device)
        self.other_params = {k: v.to(device) for k, v in self.other_params.items()}
        self.simulator.move_tensors(device)

    def init_simulator(self, simulator) -> AbstractSimulator:
        """
        Method to initiate simulator
        :param simulator: AbstractSimulator class or object
        :return: AbstractSimulator() object
        """
        return simulator

    def preprocess_data(self, batch_x):
        """
        Method to preprocess data from dataloader before feeding to forward_batch function
        :param batch_x: batch data of any type
        :return:
        """
        return batch_x

    def forward_batch(self, batch_x) -> List[torch.Tensor]:
        """
        Method to run forward dynamics over batch of states. Will run iteratively through time based on len of batch_x
        (not the datasize of batch)

        :param batch_x: List of tensors, Batch of input values. len(batch_x) is number of timesteps,
                        batch_x[0].shape is (batchsize, ...)
        :return: List of tensors, len(return) is number of timesteps
        """
        batch_state_preds = []

        state = batch_x[0]['curr_state']  # get first state
        for data_dict in batch_x:
            # Copy data dict and set last state as curr_state value
            data_dict_copy = dict(data_dict)
            data_dict_copy['curr_state'] = state

            for k, v in data_dict_copy.items():
                data_dict_copy[k] = v.to(self.device)

            # TODO remove
            data_dict_copy['external_forces'] = None
            data_dict_copy['external_pts'] = None

            # Run batch step forward with simulator
            state = self.simulator.step(**data_dict_copy)

            # Add predicted state to predications
            batch_state_preds.append(state)

        return batch_state_preds

    def post_process_preds(self, batch_preds: List[torch.Tensor]) -> torch.Tensor:
        """
        Method to post process state predictions from forward_batch() before using to compute loss
        :param batch_preds: List of tensors. Batch predictions
        :return:
        """
        return torch.stack(batch_preds, dim=-1)

    def compute_loss(self, pred_y: torch.Tensor, gt_y: torch.Tensor) -> torch.Tensor:
        """
        Method to do custom loss computation if needed

        :param pred_y: Predicted y-value
        :param gt_y: Ground truth y-value
        :return: Loss tensor
        """
        loss = self.criterion(pred_y, gt_y)
        return loss

    def forward(self, batch_data) -> torch.Tensor:
        """
        Forward method in torch.nn.Module

        :param batch_data: input data for batch processing
        :return:
        """
        loss = self.process_batch_data(batch_data)
        return loss

    def backward(self, loss: torch.Tensor) -> None:
        """
        Run back propagation with loss tensor

        :param loss: torch.Tensor
        """
        if loss.grad_fn is not None:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def process_batch_data(self, batch_data):
        """
        Method to process batch of input data and compute loss

        :param batch_data: tuple of tensors (input batch of x, batch of ground truth yy)
        :return: loss tensor
        """
        # Split batch data
        batch_x, batch_gt_y, batch_t = batch_data

        # Preprocess batch x to make sure data is ready to input into forward_batch()
        batch_x = self.preprocess_data(batch_x)

        # Run forward over batch to get predicted states
        batch_pred_states = self.forward_batch(batch_x)

        # Post process state data to match ground truth y
        batch_pred_y = self.post_process_forward(batch_pred_states)

        # Finally, compute loss
        batch_gt_y = batch_gt_y.to(self.device)
        loss = self.compute_loss(batch_pred_y, batch_gt_y)

        return loss

    def run_one_epoch(self, data_loader: DataLoader, grad_required=True):
        """
        Run one epoch over dataloader

        :param data_loader: torch DataLoader object
        :param grad_required: bool flag for gradient updates needed. True used for training.
        :return: Average loss float
        """
        # Initialize average loss
        avg_loss = 0.0

        #  Iterate through data_loader to get batches of data
        for i, batch_data in enumerate(tqdm.tqdm(data_loader)):
            # Run forward over batch of data to get loss
            loss = self(batch_data)

            # If gradient updates required, run backward pass
            if grad_required:
                self.backward(loss)

            # Update average loss
            avg_loss += loss.item()

        # Compute final average loss
        avg_loss /= len(data_loader)

        return avg_loss

    def train_epoch(self, train_data_loader: DataLoader, val_data_loader: DataLoader = None) -> Tuple[float, float]:
        """
        Run one epoch cycle over training data and validation data. If val_data_loader not provided,
        will default to using train_data_loader as validation data.

        :param train_data_loader: torch DataLoader
        :param val_data_loader: torch DataLoader

        :return: train_loss (float), val_loss (float)
        """
        # If val_data_loader not provided, will default to using train_data_loader as validation data.
        val_data_loader = val_data_loader if val_data_loader is not None else train_data_loader

        # Compute train loss
        train_loss = self.run_one_epoch(train_data_loader)

        # Compute val loss
        with torch.no_grad():
            val_loss = self.run_one_epoch(val_data_loader, grad_required=False)

        return train_loss, val_loss

    def log_status(self, train_loss: float, val_loss: float, param_values: Dict, epoch_num: int) -> None:
        """
        Method to print training status to console

        :param train_loss: Training loss
        :param val_loss: Validation loss
        :param param_values: Training parameters
        :param epoch_num: Current epoch
        """
        pass

    def run(self, num_epochs: int) -> Tuple[List, List, Dict]:
        """
        Method to run entire training

        :param num_epochs: Number of epochs to train
        :return: List of train losses, List of val losses, dictionary of trainable params and list of losses
        """
        # Initialize by running evaluation over train and validation loss
        with torch.no_grad():
            init_train_loss = self.run_one_epoch(self.train_data_loader, grad_required=False)
            init_val_loss = self.run_one_epoch(self.val_data_loader, grad_required=False)

        # Initialize storing objects
        # train_losses, val_losses = [0], [0]
        train_losses, val_losses = [init_train_loss], [init_val_loss]
        param_values_dict = {k: [v.detach().item()] for k, v in self.trainable_params.items()}

        self.log_status(train_losses[-1], val_losses[-1], param_values_dict, 0)

        # Run training over num_epochs
        for n in range(num_epochs):
            # Run single epoch training and evaluation
            train_loss, val_loss = self.train_epoch(self.train_data_loader, self.val_data_loader)
            self.scheduler.step()

            # Store current epoch's values
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            for k, v in self.trainable_params.items():
                param_values_dict[k].append(v.detach().item())

            # Log training status
            self.log_status(train_loss, val_loss, param_values_dict, n + 1)

        return train_losses, val_losses, param_values_dict
