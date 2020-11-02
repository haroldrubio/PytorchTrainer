class Hyperparameters():
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
        Each hyperparameter is inactive by default

        Args:
            name(str or list[str]): The name or names of hyperparameters to add
        """
        pass
    def activate(self, name):
        """
        Activates one or more hyperparameters

        Args:
            name(str or list[str]): The name or names of hyperparameters to activate
        """
        pass
    def deactivate(self, name):
        """
        De-activates one or more hyperparameters

        Args:
            name(str or list[str]): The name or names of hyperparameters to de-activate
        """
        pass
    def set_range(self, name, low, high):
        """
        Sets the range of exponents to sample from for a single hyperparameter

        Args:
            name(str): The name of the hyperparameter
            low(int): The lower end of the exponent range
            high(int): The upper end of the exponent range
        """
        pass
    def set_value(self, name, value):
        """
        Sets the raw value for a single hyperparameter

        Args:
            name(str): The name of the hyperparameter
            value(float): The value of the hyperparameter
        """
        pass
    def sample(self):
        """
        Return a dictionary of hyperparameter values
        If a hyperparameter is active, sample from 2**(low, high)
        If a hyperparameter is not active, sample it from value
        """
        # Define the return dictionary
        # For each hyperparameter
            # Check if this hyperparameter is active
            # If active, sample stochastically
            # If not active, sample the stored value
            # Add to the return dictionary
        #Return the dictionary
        pass
