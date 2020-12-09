import torch
import numpy as np
import shutil
import os
import sys
import math
import Metrics
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Hyperparameters import Hyperparameters as Hyp
from tqdm import tqdm

# ###### TODO: README ######
# In order to work with this template, do the following steps:
# 0) Enter in the size of the training/validation/test set as constants
# 1) Implement the GenericDataset class
# 2) Adjust the train() function to handle how you implemented the Dataset class
# Optional: Adjust the 'paths' declaration to change where you want the run statistics to be stored

class Trainer:
    """
    A wrapper class to simplify and make training of PyTorch models easier
    
    Variables:
        BATCH_SIZE: Size of each batch
        model: An instance that inherits torch.nn.Module
        model: A class that inherits torch.nn.Module
        device: Determines which device the model and loaded data will be on
        dtype: The preferred data type for the model and data
        paths['type']: Directories to be accessed/written to
        train_set, val_set, test_set: PyTorch datasets
            train_loader, val_loader, test_loader: and their corresponding loaders
        hyperparams: Hyperparameter object to sample hyperparams
        optimizer: An instance of torch.optim.Optimizer
        criterion: An instance of torch.nn.Module corresponding to a loss function
        scheduler: An instance of torch.optim.lr_scheduler
        history['type']['set']: Dictionary for loss/acc histories across datasets
        device: The device that is being used
        activations: Dictionary containing layer activations (see single_pass)
    """
    def __init__(self, model_class, data_path, device, batch_size=1):
        """
        Create a Trainer object

        Args:
            model_class(torch.nn.Module): A PyTorch class that inherits torch.nn.Module
            data_path(str): Relative directory containing the data
                            Contains /train and /val directories
            device(torch.device): The device to use for the model and data
        """
        self.BATCH_SIZE = batch_size
        self.NUM_TRAIN = 10000
        self.NUM_VAL = 5000
        self.NUM_TEST = 5000
        # Store the PyTorch Model
        self.model_class = model_class
        self.model_args, self.model_kwargs = None, None
        self.model = None
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
        self.arch_hyp = None
        # Define None objects for criterion, optimizer, and scheduler
        self.optimizer_class = None
        self.optimizer_args, self.optimizer_kwargs = None, None
        self.optimizer = None
        self.scheduler_class = None
        self.scheduler_args, self.scheduler_kwargs = None, None
        self.scheduler = None
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
        self.history['loss']['test'] = []

    def hyp_opt(self, epochs=1, iters=10):
        """
        Repeatedly call train for a short number of epochs to measure the effectiveness of each hyperparameter combination

        Args:
            epochs(int): The number of epochs to train for per combination
            iters(int): Number of times to sample hyperparameters
        """
        # Clear hyperparameter directory
        hyp_dir = self.paths['hyp']
        if os.path.isdir(hyp_dir):
            shutil.rmtree(hyp_dir)
        os.mkdir(hyp_dir)
        # For iters
        for iter in range(iters):
            # Sample hyperparameters
            sample = self.hyp.sample()
            # Create the model and obtain the hyperparameter sample
            sample = self.create_model()
            # Train a model
            self.train(epochs=epochs, opt=sample)
            # Write to TensorBoard
            path = self.paths['hyp'] + '/' + str(iter)
            writer = SummaryWriter(path)
            writer.add_hparams(sample, {'train': self.history['loss']['train'][epochs-1], 'val': self.history['loss']['val'][epochs-1]})
            writer.close()
            # Reset histories
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
        # Check for hyperparameter optimization
        is_hyp_opt = opt is not None
        # Check for existing criterion
        if self.criterion is None:
            print('ERR: No criterion declared')
            sys.exit(1)
        # If checkpointing, clear the checkpoint directory
        if save_every > 0:
            if os.path.isdir(self.paths['checkpoints']):
                shutil.rmtree(self.paths['checkpoints'])
            os.mkdir(self.paths['checkpoints'])
        # If not optimizing hyperparams, reset runs directory
        if opt is None:
            if os.path.isdir(self.paths['runs']):
                shutil.rmtree(self.paths['runs'])
            os.mkdir(self.paths['runs'])
        # Wipe/Create the checkpoints directory
        if os.path.isdir(self.paths['checkpoints']):
            shutil.rmtree(self.paths['checkpoints'])
        os.mkdir(self.paths['checkpoints'])

        # Instantiate a model to train if not optimizing for hyperparameters
        # Otherwise, assume creation was done by the hyp_opt function
        if opt is None:
            self.create_model()
        for e in range(epochs):
        #   If first epoch, get initial evaluations
            if e == 0:
                train_loss = self.evaluate_step(e-1, self.train_loader, self.NUM_TRAIN)
                self.evaluation(e-1, is_hyp_opt)
        #   Perform network training
            self.training(e)
        #   Perform network evaluation
            self.evaluation(e, is_hyp_opt)
        #   Perform scheduler steps
            if self.scheduler is not None:
                self.scheduler.step()
        #   If not optimizing hyperparams
            if opt is None:
        #       Check if updating
                if update_every > 0 and e % update_every == 0:
                    train_loss, val_loss = self.history['loss']['train'][e], self.history['loss']['val'][e]
                    writer = SummaryWriter()
                    writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, e)
                    writer.close()
        #       Check if saving
                if save_every > 0 and e % save_every == 0:
                    self.store_checkpoint(e)
    def training(self, epoch):
        """
        Helper function to further abstract the training step
        Performs training step and necessary metric storage

        Args:
            epoch(int): The current epoch number
        """
        train_loss = self.train_step(epoch)
        self.history['loss']['train'].append(train_loss)
    def evaluation(self, epoch, hyp_opt=False):
        """
        Helper function to make the choice of whether to evaluate the validation set or the test set

        Args:
            epoch(int): The current epoch number
            hyp_opt(bool): An indicator of whether or not hyperparameters are being optimized
        """
        # If not optimizing hyperparameters, perform test step
        if hyp_opt == False:
            test_loss = self.test_step(epoch)
            self.history['loss']['test'].append(np.mean(test_loss))
        # Otherwise, perform validation step
        else:
            val_loss = self.val_step(epoch)
            self.history['loss']['val'].append(np.mean(val_loss))
    def train_step(self, epoch):
        """
        Helper function to facilitate the training steps of an epoch

        Args:
            epoch(int): The current epoch number
        Returns: train_loss(float): The average loss from an epoch of training
        """
        #   Scale epoch to start at 1
        epoch += 1
        #   Set network to train
        self.model.train()
        train_loss = []
        for idx, batch in tqdm(enumerate(self.train_loader), total=math.ceil(self.NUM_TRAIN/self.BATCH_SIZE),
                                    desc=f'Training Epoch {epoch}', unit=' Batches'):
        #   Perform forward and backward passes
            forward_pass = self.pass_batch(batch)
            loss, preds = forward_pass['loss'], forward_pass['preds']
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        #   Store metrics
            train_loss.append(float(loss))
        return np.mean(train_loss)
    def val_step(self, epoch):
        """
        Helper function to facilitate the validation steps of an epoch

        Args:
            epoch(int): The current epoch number
        Returns: val_loss(float): The average loss from an epoch of training
        """
        return self.evaluate_step(epoch, self.val_loader, self.NUM_VAL)
    def test_step(self, epoch):
        """
        Helper function to facilitate the testing steps of an epoch

        Args:
            epoch(int): The current epoch number
        Returns: loss(float): The average loss from an epoch of training
        """
        return self.evaluate_step(epoch, self.test_loader, self.NUM_TEST)
    def evaluate_step(self, epoch, data_loader, num_examples):
        """
        Helper function as a general way of evaluating a dataset

        Args:
            epoch(int): The current epoch number
            data_loader(torch.utils.dataset.DataLoader): A PyTorch dataloader
            num_examples(int): The number of examples covered by the dataloader
        Returns: loss(float): The average loss from an epoch of evaluation
        """
        #   Scale epoch to start at 1
        epoch += 1
        eval_loss = []
        #   Set network to eval                
        self.model.eval()
        with torch.no_grad():
        #   Get validation loss
            for idx, batch in tqdm(enumerate(data_loader), total=math.ceil(num_examples/self.BATCH_SIZE),
                                            desc=f'Evaluating Epoch {epoch}', unit=' Batches'):
        #       Perform forward and backward passes
                forward_pass = self.pass_batch(batch)
                loss, preds = forward_pass['loss'], forward_pass['preds']
        #       Store metrics
                eval_loss.append(float(loss))
        return np.mean(eval_loss)
    def pass_batch(self, batch):
        """
        Passes a batch through the model and returns the required statistics

        Args:
            batch: A custom-formatted object that contains the data necessary for the forward and backward passes
                   Usually consists of a dict\n
        Returns: ret_dict{dict}: A dictionary containing values to be returned from a forward pass
        """
        ret_dict = {}
        ### TODO: CHOOSE HOW TO REPRESENT DATA ###
        x, y = batch['x'], batch['y']
        ### TODO: CHOOSE HOW TO REPRESENT DATA ###
        # Move tensors to device and optionally enforce data types
        x, y = x.to(self.device).type(self.dtype), y.to(self.device).type(torch.long)
        # Pass through the model
        preds = self.model(x)
        loss = self.criterion(preds, y)
        # Define values of ret_dict
        ret_dict['preds'] = preds
        ret_dict['loss'] = loss
        return ret_dict

    def create_object(self, object_type=None):
        """
        Helper function to create objects within the trainer

        Returns: (object, sample): object: an instantiation of the object_type
                                 : sample: a sample from the hyperparameter object if it exists
                                           otherwise, returns None
        """
        # First parse the object type
        obj_class, args, kwargs, hyp = None, None, None, None
        if object_type == 'model':
            obj_class, args, kwargs, hyp = self.model_class, self.model_args, self.model_kwargs, self.arch_hyp
        elif object_type == 'optim':
            obj_class, optim_args, kwargs, hyp = self.optimizer_class, self.optimizer_args, self.optimizer_kwargs, self.hyp
        #   Prepend model parameters to arguments
            optim_args = list(optim_args)
            args = [self.model.parameters()]
            args.extend(optim_args)
            args = tuple(args)
        # Transform empty arguments into none
        if len(args) == 0:
            args = None
        if len(kwargs) == 0:
            kwargs = None
        
        # Perform object creation
        ret_obj = None
        # Obtain a hyperparameter sample
        param_sample = None
        if hyp is not None:
            param_sample = hyp.sample()
        #   Check for kwargs - if not none, merge the hyperparameter sample
            if kwargs is not None:
                kwargs.update(param_sample)

        # Choose which combination of parameters to instantiate with
        if hyp is None:
            if args is None and kwargs is not None:
                ret_obj = obj_class(**kwargs)
            elif args is not None and kwargs is None:
                ret_obj = obj_class(*args)
            elif args is None and kwargs is None:
                ret_obj = obj_class()
            else:
                ret_obj = obj_class(*args, **kwargs)
        else:
            if args is None and kwargs is None:
                ret_obj = obj_class(**param_sample)
            elif args is None and kwargs is not None:
                ret_obj = obj_class(**kwargs)
            elif args is not None and kwargs is None:
                ret_obj = obj_class(*args, **param_sample)
            else:
                ret_obj = obj_class(*args, **kwargs)
        return ret_obj, param_sample
    def create_model(self):
        """
        Creates an instantiation of the model using the given hyperparameters and the primed
        hyperparameters

        Returns: hyperparameters(dict{hyp:val}): dictionary of hyperparameters for the optimizer
                                                 and model
        """
        # If the model has not been primed and no hyperparameter object has been provided, throw error
        if self.model_args is None and self.model_kwargs is None and self.arch_hyp is None:
            print('ERR: Model not primed')
            sys.exit(1)
        
        # Create a model
        self.model, model_hyp = self.create_object(object_type='model')
        self.model.to(self.device)

        # Then, instantiate the optimizer
        optim_hyp = self.create_optimizer()

        # Optionally, if using a scheduler, instantiate a scheduler
        if self.scheduler_class is not None:
            self.create_scheduler()

        # Find non-none hyperparameter samples and update
        hyp = {}
        if model_hyp is not None:
            hyp.update(model_hyp)
        if optim_hyp is not None:
            hyp.update(optim_hyp)
        
        return hyp
    def create_optimizer(self):
        """
        Creates an instantiation of an optimizer using the given hyperparameters and the primed
        hyperparameters

        Returns: optim_hyp(dict{hyp:val}): dictionary of hyperparameters for the optimizer
        """
        # If the optimizer was not primed and no optimizer hyperparameters were provided, throw error
        if self.optimizer_args is None and self.optimizer_kwargs is None and self.hyp is None:
            print('ERR: Optimizer not primed')
            sys.exit(1)
        
        self.optimizer, optim_hyp = self.create_object(object_type='optim')
        return optim_hyp
    def create_scheduler(self):
        """
        Creates an instantiation of scheduler using the primed hyperparameters
        """
        # If the scheduler was not primed throw error
        if self.scheduler_args is None and self.scheduler_kwargs is None:
            print('ERR: Scheduler not primed')
            sys.exit(1)
        
        # For empty argument, set to none
        if len(self.scheduler_args) == 0:
            self.scheduler_args = None
        if len(self.scheduler_kwargs) == 0:
            self.scheduler_kwargs = None

        # Choose which combination of parameters to instantiate with
        if self.scheduler_args is not None and self.scheduler_kwargs is None:
            self.scheduler = self.scheduler_class(self.optimizer, *self.scheduler_args)
        elif self.scheduler_args is None and self.scheduler_kwargs is not None:
            self.scheduler = self.scheduler_class(self.optimizer, **self.scheduler_kwargs)
        elif self.scheduler_args is None and self.scheduler_kwargs is None:
            self.scheduler = self.scheduler_class(self.optimizer)
        else:
            self.scheduler = self.scheduler_class(self.optimizer, *self.scheduler_args, **self.scheduler_kwargs)
    def prime_model(self, *args, **kwargs):
        """
        Stores the model parameters to be used upon each initialization

        Args:
            args, kwargs: Static hyperparameters that will be used each time a model
                          is instantiated
        """
        self.model_args = args
        self.model_kwargs = kwargs
    def prime_optimizer(self, *args, **kwargs):
        """
        Stores the optimizer parameters to be used upon each initialization
        
        Args:
            args, kwargs: Static hyperparameters that will be used each time an optimizer
                          is instantiated
        """
        self.optimizer_args = args
        self.optimizer_kwargs = kwargs
    def prime_scheduler(self, *args, **kwargs):
        """
        Stores the scheduler parameters to be used upon each initialization
        
        Args:
            args, kwargs: Static hyperparameters that will be used each time a scheduler
                          is instantiated
        """
        self.scheduler_args = args
        self.scheduler_kwargs = kwargs
    def set_hyperparameters(self, hyp):
        """
        Set the local Hyperparameters object for the optimizer (see: Hyperparameters.py)

        Args:
            hyp(Hyperparameters): Redefine the local Hyperparameters object
        -Set the local Hyperparameters object
        """
        self.hyp = hyp
    def set_arch_hyperparameters(self, hyp):
        """
        Set the local Hyperparameters object for the model (see: Hyperparameters.py)

        Args:
            hyp(Hyperparameters): Redefine the local Hyperparameters object
        """
        self.arch_hyp = hyp
    def set_optimizer(self, optimizer_class):
        """
        Given the class of a PyTorch optimizer, set the local optimizer to create off of
        Args:
            optimizer_class(torch.optim): The class of a desired optimizer
        """
        self.optimizer_class = optimizer_class
    def set_model(self, model_class):
        """
        Given the class of a PyTorch model(from nn.Module), create models based off of this class
        Args:
            model_class(torch.nn.Module): The class of the model to train
        """
        self.model_class = model_class
    def set_scheduler(self, scheduler_class):
        """
        Sets a local scheduler class

        Args:
            scheduler_class(torch.optim.lr_scheduler): A scheduler class to be instantiated later
        """
        self.scheduler_class = scheduler_class
    def set_criterion(self, criterion_class):
        """
        Set the local loss function as the desired loss function

        Args:
            criterion_class(torch.nn): An nn.Module class to be instantiated as the target task loss
        """
        self.criterion = criterion_class()
    def get_model(self):
        """
        Get the internal PyTorch model

        Returns: model(nn.Module): The interally stored model
        """
        return self.model
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
        self.data = np.loadtxt(os.path.join(root_dir, 'data.txt'))
        self.N, self.D = self.data.shape
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        ret_val = {}
        example = self.data[idx]
        data_point = torch.tensor(example[:self.D-1])
        label = torch.tensor(example[self.D-1])
        ret_val['x'] = data_point
        ret_val['y'] = label
        return ret_val