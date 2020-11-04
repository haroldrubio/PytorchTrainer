import torch
import numpy as np
import shutil
import os
import sys
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Hyperparameters import Hyperparameters as Hyp

# ###### TODO: README ######
# In order to work with this template, do the following steps:
# 1) Implement the GenericDataset class
# 2) Adjust the train() function to handle how you implemented the Dataset class
# Optional: Adjust the 'paths' declaration to change where you want the run statistics to be stored

class Trainer:
    """
    A wrapper class to simplify and make training of PyTorch models easier

    Variables:
        BATCH_SIZE: Size of each batch
        model: Inherits torch.nn.Module
        device: Determines which device the model and loaded data will be on
        dtype: The preferred data type for the model and data
        paths['type']: Directories to be accessed/written to
        train_set, val_set, test_set: PyTorch datasets
            train_loader, val_loader, test_loader: and their corresponding loaders
        hyperparams: Hyperparameter object to sample hyperparams
        optimizer: An instance of torch.optim.Optimizer
        criterion: An instance of torch.nn.Module corresponding to a loss function
        history['type']['set']: Dictionary for loss/acc histories across datasets
        device: The device that is being used
        activations: Dictionary containing layer activations (see single_pass)
    """
    def __init__(self, model, data_path, device, batch_size=1):
        """
        Create a Trainer object

        Args:
            model(torch.nn.Module): A PyTorch model that inherits torch.nn.Module
            data_path(str): Relative directory containing the data
                            Contains /train and /val directories
            device(torch.device): The device to use for the model and data
        """
        self.BATCH_SIZE = batch_size
        # Store the PyTorch Model
        self.model = model
        self.device = device
        self.dtype = torch.float32
        # Generate and store data paths
        self.paths = {}
        self.paths['train'] = data_path + '/train'
        self.paths['val'] = data_path + '/val'
        self.paths['test'] = data_path + '/test'
        self.paths['runs'] = 'runs'
        self.paths['hyp'] = 'hyp'
        self.paths['checkpoints'] = 'chk'
        # Create and store generic DataSet objects
        self.train_set = GenericDataset(self.paths['train'])
        self.train_loader = DataLoader(self.train_set, batch_size=self.BATCH_SIZE, shuffle=True)
        self.val_set = GenericDataset(self.paths['val'])
        self.val_loader = DataLoader(self.val_set, batch_size=self.BATCH_SIZE, shuffle=True)
        self.test_set = GenericDataset(self.paths['test'])
        self.test_loader = DataLoader(self.test_set, batch_size=self.BATCH_SIZE, shuffle=True)
        # Initialize a Hyperparameter object
        self.hyp = None
        # Define None objects for criterion and optimizer
        self.optimizer = None
        self.criterion = None
        # Define dictionary of empty lists for histories
        self.history = {}
        self.init_history()
        # Define empty activation dictionary
        self.activations = {}
    def init_history(self):
        """ 
        Helper function to re/initialize the history of the Trainer object
        Edit to allow capacity for other metrics to be kept track of and make
            corresponding changes in train() to update the history
        """
        self.history = {}
        self.history['loss'] = {}
        self.history['loss']['train'] = []
        self.history['loss']['val'] = []


    def hyp_opt(self, optim_name='Adam', epochs=1, iters=10):
        """
        Repeatedly call train for a short number of epochs to measure the effectiveness of each hyperparameter combination

        Args:
            optim_name(str): The name of the optimizer to use
            epochs(int): The number of epochs to train for per combination
            iters(int): Number of times to sample hyperparameters
        """
        # Clear hyperparameter directory
        hyp_dir = self.paths['hyp']
        if os.path.isdir(hyp_dir):
            shutil.rmtree(hyp_dir)
        os.mkdir(hyp_dir)
        # Store the local default model
        local_model = self.model
        # For iters
        for iter in range(iters):
            # Sample hyperparameters
            sample = self.hyp.sample()
            # Set the optimizer
            self.set_optimizer(optim_name, sample)
            # Train a model
            self.train(epochs=epochs, opt=sample)
            # Write to TensorBoard
            path = self.paths['hyp'] + '/' + str(iter)
            writer = SummaryWriter(path)
            writer.add_hparams(sample, {'train': self.history['loss']['train'][epochs-1], 'val': self.history['loss']['val'][epochs-1]})
            writer.close()
            # Reset histories and model
            self.model = local_model
            self.init_history()

    def train(self, epochs=1, update_every=0, save_every=0, opt=None):
        """
        Train the stored model and optionally save checkpoints and output model statistics
        every few epochs

        By default, no checkpoints or statistics are saved
        Args:
            epochs(int): The number of epochs to train for
            update_every(int): After update_every epochs, write out to TensorBoard
            save_every(int): After save_every epochs, save a checkpoint
            opt(dict): If not None, contains a dictionary of optimizer
                       hyperparameters to use
        """
        # Check for existing criterion
        if self.criterion is None:
            print('ERR: No criterion declared')
            sys.exit(1)
        # If not optimizing hyperparams, reset runs directory
        if opt is None:
            if os.path.isdir(self.paths['runs']):
                shutil.rmtree(self.paths['runs'])
            os.mkdir(self.paths['runs'])
        # Wipe/Create the checkpoints directory
        if os.path.isdir(self.paths['checkpoints']):
            shutil.rmtree(self.paths['checkpoints'])
        os.mkdir(self.paths['checkpoints'])
        # For epochs
        for e in range(epochs):
        #   Draw sample from train dataloader
        #   Set network to train
            self.model.train()
            train_loss = []
            for idx, batch in enumerate(self.train_loader):
        #       Perform forward and backward passes
        #       ### TODO: CHOOSE HOW TO REPRESENT DATA ###
                x, y = batch['x'], batch['y']
                x = x.to(self.device)
                y = y.to(self.device)
        #       ### TODO: CHOOSE HOW TO REPRESENT DATA ###                
                preds = self.model(x)
                loss = self.criterion(preds, y)
                train_loss.append(float(loss))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        #   Draw sample from val dataloader
        #   Set network to eval                
            self.model.eval()
            val_loss = []
        #   Get validation loss
            for idx, batch in enumerate(self.val_loader):
                x, y = batch['x'], batch['y']
        #       ### TODO: CHOOSE HOW TO REPRESENT DATA ###
                x = x.to(self.device)
                y = y.to(self.device)
        #       ### TODO: CHOOSE HOW TO REPRESENT DATA ###
                preds = self.model(x)
                loss = self.criterion(preds, y)
                val_loss.append(float(loss))
        #   Store histories
            self.history['loss']['train'].append(np.mean(train_loss))
            self.history['loss']['val'].append(np.mean(val_loss))
        #   If not optimizing hyperparams
            if opt is None:
        #       Check if updating
                if update_every > 0 and e % update_every == 0:
                    writer = SummaryWriter()
                    writer.add_scalars('Loss', {'train': np.mean(train_loss), 'val': np.mean(val_loss)}, e)
                    writer.close()
        #       Check if saving
                if save_every > 0 and e % save_every == 0:
                    self.store_checkpoint(e)
    def set_hyperparameters(self, hyp):
        """
        Set the local Hyperparameters object (see: Hyperparameters.py)

        Args:
            hyp(Hyperparameters): Redefine the local Hyperparameters object
        -Set the local Hyperparameters object
        """
        self.hyp = hyp
    def set_optimizer(self, name, params):
        """
        Given the name of the optimizer, and a dictionary of parameters, set the local optimizer as the desired optimizer

        Args:
            name(str): The name of the optimizer to be used
            params(dict): Dictionary of optimizer hyperparameters (hyp: value)
        Parse name into different optimizers and set to the class variable
        """
        self.model.to(device=self.device)
        if name == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), **params)
        elif name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), **params)
        elif name == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), **params)
        elif name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), **params)


    def set_criterion(self, name):
        """
        Set the local loss function as the desired loss function

        Args:
            name(str): The name of the loss function to be used
        Parse name into different losses and set to the class variable
        """
        if name == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        elif name == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss()
        elif name == 'MSE':
            self.criterion = nn.MSELoss()
    def get_model(self):
        """
        Get the internal PyTorch model

        Returns: model(nn.Module): The interally stored model
        """
        return self.model
    def validate(self):
        """
        Pass the validation set through the PyTorch model

        Returns: preds(torch.tensor): A tensor of the validation predictions of the model
        """
        pass
    def store_checkpoint(self, epoch_number):
        """
        Write out a file for the current weights and state of the optimizer

        Args:
            epoch_number(int): The current epoch number
        """
        checkpoint = {}
        checkpoint['epoch'] = epoch_number
        checkpoint['model_state_dict'] = self.model.state_dict()
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(checkpoint, self.paths['checkpoints'])

class GenericDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with the data (no sub-directories exist)
        """
        ### TODO: CHOOSE HOW TO IMPLEMENT FETCHING DATA ###
    def __len__(self):
        return 0
    def __getitem__(self, idx):
        ret_val = {}
        return ret_val