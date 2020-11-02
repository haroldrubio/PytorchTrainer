import torch
class Trainer():
    """
    A wrapper 

    Variables:
        model: Inherits torch.nn.Module
        paths['type']: Directories to be accessed/written to
        train_set, val_set, test_set: PyTorch datasets
            train_loader, val_loader, test_loader: and their corresponding loaders
        hyperparams: Hyperparameter object to sample hyperparams
        optimizer: An instance of torch.optim.Optimizer
        criterion: An instance of torch.nn.Module corresponding to a loss function
        history['type']['set']: Dictionary for loss/acc histories across datasets
        activations: Dictionary containing layer activations (see single_pass)
    """
    def __init__(self, model, data_path, device):
        """
        Create a Trainer object

        Args:
            model(torch.nn.Module): A PyTorch model that inherits torch.nn.Module
            data_path(str): Relative directory containing the data
                            Contains /data/train and /data/val directories
            device(torch.device): The device to use for the model and data
        """
        # Store the PyTorch Model
        # Generate and store data paths
        # Create and store generic DataSet objects
        # Create and store corresponding DataLoader objects
        # Initialize a Hyperparameter object
        # Define None objects for criterion and optimizer
        # Define dictionary of empty lists for histories
        # Define empty activation dictionary
        pass

    def hyp_opt(self, epochs=1, iters=10):
        """
        Repeatedly call train for a short number of epochs to measure the effectiveness of each hyperparameter combination

        Args:
            epochs(int): The number of epochs to train for per combination
            iters(int): Number of times to sample hyperparameters
        """
        # Clear hyperparameter directroy
        # Store the local default model
        # For iters
            # Sample hyperparameters
            # Train a model
            # Reset histories
        pass

    def train(self, epochs=1, update_every=1, save_every=1, opt=None):
        """
        Train the stored model and optionally save checkpoints and output model statistics
        every few epochs

        Args:
            epochs(int): The number of epochs to train for
            update_every(int): After update_every epochs, write out to TensorBoard
            save_every(int): After save_every epochs, save a checkpoint
            opt(dict): If not None, contains a dictionary of optimizer
                       hyperparameters to use
        """
        # Define the optimizer
        # If not optimizing hyperparams
            # Define and/or clear training directory
            # Retrieve stored optimizer
        # If optimizing
            # Define and/or clear hyperparameter directory
            # Create optimizer from sampled hyperparameters
        # For epochs
        #     Draw sample from dataloader
        #     Perform forward and backward passes
        #     Store histories
        #     If optimizing hyperparams
        #     Check if updating
        #         Create a file string corresponding to hyperparams
        #         Save hyperparam
        #     If not optimizing hyperparams
        #     Check if updating
        #         Create a file string correpsonding to train/val loss
        #         Save scalars
        #     Check if saving
        #         Create file string to save to and save
        pass
    def set_hyperparameters(self, hyp):
        """
        Set the local Hyperparameters object (see: Hyperparameters.py)

        Args:
            hyp(Hyperparameters): Redefine the local Hyperparameters object
        -Set the local Hyperparameters object
        """
    def set_optimizer(self, name, params):
        """
        Given the name of the optimizer, and a dictionary of parameters, set the local optimizer as the desired optimizer

        Args:
            name(str): The name of the optimizer to be used
            params(dict): Dictionary of optimizer hyperparameters (hyp: value)
        Parse name into different optimizers and set to the class variable
        """
        pass
    def set_criterion(self, name):
        """
        Set the local loss function as the desired loss function

        Args:
            name(str): The name of the loss function to be used
        Parse name into different losses and set to the class variable
        """
        pass
    def get_model(self):
        """
        Get the internal PyTorch model

        Returns: model(nn.Module): The interally stored model
        """
        pass
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
        pass