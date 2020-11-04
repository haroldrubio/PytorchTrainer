from itertools import product
import numpy as np
class Hyperparameters:
    """
    A way to store an arbitrary set of hyperparameters and sample from ranges

    Variables:
        names(list[str]): A list of the names of hyperparameters
                          Note: these should match the keyword of the argument of the destination function
        active(list[int]): A list of indicators determining whether or not this hyperparameter
                           is being sampled stochastically
        values(list[floats]): A list of floats that will be used as the value of a hyperparameter
                              if the hyperparameter is inactive
        ranges(dict{name: (low, high)}): A dictionary mapping hyperparameter name to a tuple of ints
                                         defining the range of exponents to sample from
    """
    def __init__(self):
        self.names = []
        self.values = []
        self.active = []
        self.ranges = {}
    def register(self, name):
        """
        Registers one or more hyperparameters to sample/store
        Each hyperparameter is active by default
        Active hyperparameters have a value of -1
        The default search range is 10^(-1, 0)

        Args:
            name(str or list[str]): The name or names of hyperparameters to add
        """
        if type(name) is not str:
            for n in name:
                self.register_name(n)
        else:
            self.register_name(name)
    def register_name(self, name):
        """
        Helper function for register(name)
        Adds a single hyperparameter to the class

        Args:
            name(str): The name of hyperparameters to add
        """
        # Add name to list of names
        self.names.append(name)
        # Initialize other lists to default values
        self.values.append(-1)
        self.active.append(1)
        default_pair = (-1, 0)
        self.ranges[name] = default_pair
    def activate(self, name):
        """
        Activates one or more hyperparameters

        Args:
            name(str or list[str]): The name or names of hyperparameters to activate
        """
        idx = self.names.index(name)
        self.active[idx] = 1
    def deactivate(self, name):
        """
        De-activates one or more hyperparameters

        Args:
            name(str or list[str]): The name or names of hyperparameters to de-activate
        """
        idx = self.names.index(name)
        self.active[idx] = 0
    def set_range(self, name, low, high):
        """
        Sets the range of exponents to sample from for a single hyperparameter

        Args:
            name(str): The name of the hyperparameter
            low(int): The lower end of the exponent range
            high(int): The upper end of the exponent range
        """
        new_range = (low, high)
        self.ranges[name] = new_range
    def set_value(self, name, value):
        """
        Sets the raw value for a single hyperparameter and deactivates it
        To search over this hyperparameter again, activate() it

        Args:
            name(str): The name of the hyperparameter
            value(float): The value of the hyperparameter
        """
        idx = self.names.index(name)
        self.values[idx] = value
        self.deactivate(name)
    def sample(self):
        """
        Return a dictionary of hyperparameter values
        If a hyperparameter is active, sample from 2**(low, high)
        If a hyperparameter is not active, sample it from value
        """
        # Define the return dictionary
        sample = {}
        # For each hyperparameter
        for idx in range(len(self.names)):
            # Retrieve relevant attributes
            hyp_name = self.names[idx]
            active = self.active[idx]
            set_val = self.values[idx]
            # Check if this hyperparameter is active
            hyp_val = 0
            if active:
                # If active, sample stochastically
                low, high = self.ranges[hyp_name]
                hyp_val = 10**np.random.uniform(low, high)
            else:
                # If not active, sample the stored value
                hyp_val = set_val
            # Add to the return dictionary
            sample[hyp_name] = hyp_val
        #Return the dictionary
        return sample