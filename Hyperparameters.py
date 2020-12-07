from itertools import product
import numpy as np
import sys
class Hyperparameters:
    """
    A way to store an arbitrary set of hyperparameters and sample from ranges

    Variables:
        names(list[str]): A list of the names of hyperparameters
                          Note: these should match the keyword of the argument of the destination function
        active(list[int]): A list of indicators determining whether or not this hyperparameter
                           is being sampled stochastically
        values(list[float]): A list of floats that will be used as the value of a hyperparameter
                              if the hyperparameter is inactive
        ranges(dict{name: (low, high)}): A dictionary mapping hyperparameter name to a tuple of ints
                                         defining the range of exponents to sample from
        types(list[str]):  A list of types of hyperparameters (only allows 'integer' and 'decimal')
        bases(list[int]): A list of integers indicating the base of log space
    """
    def __init__(self, init_dict=None):
        self.names = []
        self.types = []
        self.values = []
        self.active = []
        self.bases = []
        self.ranges = {}

        # Handle the init_dict parameters to set
        if init_dict is not None:
            for key, val in init_dict.items():
        #       Obtain data type
                hyp_type = str(type(val))
                if 'float' in hyp_type:
                    hyp_type = 'decimal'
                elif 'int' in hyp_type:
                    hyp_type = 'integer'
        #       Register the hyperparameter
                self.register_name(key, hyp_type)
        #       Set the value for the hyperparameter
                self.set_value(key, val)
    def register(self, name, hyp_type=None):
        """
        Registers one or more hyperparameters to sample/store
        Each hyperparameter is active by default
        Active hyperparameters have a value of -1
        The type of each new hyperparameter is 'decimal'
        The default value for the base is 10
        The default search range is base^(-1, 0) for decimal and
            (-1, 0) for integer

        Args:
            name(str or list[str]): The name or names of hyperparameters to add
            hyp_type(str or list[str]): The type or types of hyperparameters to add
        """

        # Handle hyperparameters to sample
        if type(name) is not str:
            num_hyp = len(name)
            if hyp_type is None:
                hyp_type = ['decimal' for i in range(num_hyp)]
            for hyp_num in range(num_hyp):
                self.register_name(name[hyp_num], hyp_type[hyp_num])
        else:
        # If no type is provided, generate data types
            if hyp_type is None:
                hyp_type = 'decimal'
            self.register_name(name, hyp_type)
    def register_name(self, name, hyp_type):
        """
        Helper function for register(name)
        Adds a single hyperparameter to the class

        Args:
            name(str): The name of hyperparameters to add
            type(str): The type of the hyperparameter to add
        """
        # Add name to list of names
        self.names.append(name)
        # Add type to list of types
        self.types.append(hyp_type)
        # Initialize other lists to default values
        self.bases.append(10)
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
    def set_base(self, name, base):
        """
        Sets the base for the log space to search in for a given hyperparameter

        Args:
            name(str): The name of the hyperparameter
            base(int): The base for the log space
        """
        idx = self.names.index(name)
        self.bases[idx] = base
    def sample(self):
        """
        Return a dictionary of hyperparameter values
        If a hyperparameter is active, sample from 10**(low, high)
        If a hyperparameter is not active, sample it from value
        """
        # Define the return dictionary
        sample = {}
        # For each hyperparameter
        num_hyp = len(self.names)
        for idx in range(num_hyp):
        #   Retrieve relevant attributes
            hyp_name = self.names[idx]
            active = self.active[idx]
            hyp_base = self.bases[idx]
            set_val = self.values[idx]
            hyp_type = self.types[idx]
        #   Check if this hyperparameter is active
            hyp_val = 0
            if active:
        #       If active, sample stochastically
                low, high = self.ranges[hyp_name]
                if hyp_type == 'decimal':
                    hyp_val = self.sample_decimal(low, high,  hyp_base)
                elif hyp_type == 'integer':
                    hyp_val = self.sample_integer(low, high)
            else:
        #       If not active, sample the stored value
                hyp_val = set_val
        #   Add to the return dictionary
            sample[hyp_name] = hyp_val
        # Return the dictionary
        return sample
    def sample_decimal(self, low, high, base):
        '''
        Helper function for sample()
        Returns a sample from base**random(low, high)
        Constraint: low > high

        Args:
            low(int): The low end of the log-space sample
            high(int): The high end of the log-space sample
        '''
        return base**np.random.uniform(low, high)
    def sample_integer(self, low, high):
        '''
        Helper function for sample()
        Returns a sample from (low, high)
        Constraint: low > high

        Args:
            low(int): The low end of the integer sample space
            high(int): The high end of the integer sample space
        '''
        return np.random.randint(low, high)