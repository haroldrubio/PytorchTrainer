# PytorchTrainer
A Template For Hyperparameter Optimization and Training PyTorch Models

Across my various machine learning projects and classes, I found myself re-writing the same infrastructure to train some deep learning model. This repository abstracts
out the task and deals with the raw model. 
## Setting Up Trainer.py
0. **Obtain the number of examples in your train-dev-test splits and enter them in the** `__init__` **function of** `Trainer` **as shown here*:

```python
  self.BATCH_SIZE = batch_size
  self.NUM_TRAIN = 10000
  self.NUM_VAL = 5000
  self.NUM_TEST = 5000
```
1. **Choose how to represent your data**\
In order to use PyTorch dataloaders, you must implement a dataset class that is able to retrieve a single item using the `__getitem__` function.\
An example where the data is represented in a single text document and stored into memory all at once is given below.\
The data here consists of `N` rows with each row containing a 2D vector followed by a class label.\
This form of dataset works by being able to get the number of examples in the dataset, and returning an abstract item. Use a dictionary to include multiple objects for each item.\
*Note: The keys used in the* `__getitem__` *function are the same keys used when accessing a batch returned from an enumerated dataloader.*
```python
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
        data_point = torch.tensor(example[:self.D-1]) #<---- If needed, transform data into tensors
        label = torch.tensor(example[self.D-1])       #<---- 
        ret_val['x'] = data_point #<---- Tensors for a given key must always have the same dimension
        ret_val['y'] = label      #<---- See above
        return ret_val
```
2. **Use a PyTorch built-in dataset function** ***(optional)***\
PyTorch provides several tools in its built-in libraries for handling several well known datasets.\
The following example uses the `ImageFolder` class from `torchvision.datasets` to construct a dataset class. This modification would be placed in the `__init__` function of Trainer.
```python
  from torchvision import datasets
  from torch.utils.data import Dataloader

  self.train_set = datasets.ImageFolder(self.paths['train'], pre_process)
  self.train_loader = DataLoader(self.train_set, batch_size=self.BATCH_SIZE, shuffle=True)
```
3. **Adjust the forward pass**\
Your dataset may handle data differently or you may wish to include more things in each batch depending on the dataset implementation.\
In order to compensate for these changes, adjust the `pass_batch` function to parse the batch dict.
```python
def pass_batch(self, batch):
    x, y = batch['x'], batch['y'] # <--- This step may need changing
    # Move tensors to device and optionally enforce data types
    x, y = x.to(self.device).type(self.dtype), y.to(self.device).type(torch.long)
    # Pass through the model
    preds = self.model(x)
```
## Training Models
### Hyperparameter Optimization
So you have your model and your data defined and you're ready to start searching for hyperparameters. Use a `Hyperparameters` object to either store static valued hyperparameters or sample from a random distribution. For example, I have the following model with a single hidden layer:
```python
import torch.nn.functional as F
import torch.nn as nn
class Net(nn.Module):
    def __init__(self, input_size=2, hidden_size=1, target_length=1):
        super().__init__()
        self.h = nn.Linear(input_size, hidden_size)
        self.o = nn.Linear(hidden_size, target_length)

    def forward(self, x):
        x = self.h(x)
        x = F.relu(x)
        x = self.o(x)
        return x
```
and I want to search over learning rates between `1e-5` and `1e-1` using a momentum of 0.9 with SGD as my optimizer. I would do this by creating a `Hyperparameters` object, registering the hyperparameters and passing it to my trainer instance:\
**(NOTE: The names of your hyperparameters MUST match the keyword arguments for your selected optimizer)**
```python
from Trainer import Trainer
from Hyperparameters import Hyperparameters as Hyp
  tr = Trainer(net,...)
  optim_params = Hyp()
  optim_params.register(['lr', 'momentum'])
  optim_params.set_range('lr', -5, -1)
  optim_params.set_value('momentum', 0.9)
  tr.set_hyperparameters(optim_params)
```
Next, you want to set the loss function for your target task, and set and prime your optimizer and scheduler, if you are using a scheduler. Afterwards, prime your model. The `prime` functions exist to give you the ability to set hyperparameters that you would want to exist within every trained model.\
Setting static hyperparameters (those not sampled randomly) by setting the value in a `Hyperparameters`, like with momentum above, and passing the argument directly to the `prime` function are equivalent - but you **cannot** use both. In this example, momentum can also be set by `tr.prime_optimizer(momentum=0.9)`.\
**(NOTE: `prime` functions MUST be called, even without setting static hyperparameters)**
```python
  tr.set_criterion(CrossEntropyLoss)                           #<---- Using Cross Entropy Loss
  tr.set_optimizer(SGD)                                        #<---- Optimizing using SGD
  tr.set_scheduler(StepLR)                                     #<---- Scaling the learning rate every few epochs
  tr.prime_optimizer()                                         #<---- All optimizer params are being handled by optim_params
  tr.prime_scheduler(5, gamma=0.1)                             #<---- Every 5 epochs, scale the learning rate by 0.1
  tr.prime_model(input_size=2, hidden_size=5, target_length=2) #<---- Set the dimensionalities of the network
```
And now you're ready to search for hyperparameters! Just call `tr.hyp_opt(epochs=5, iters=20)` if you want to sample 20 different hyperparameters, with each model training for 5 epochs. To visualize your results, please refer to the TensorBoard results in the `hyp` directory by using the following command in the Anaconda prompt when in the project directory: `tensorboard --logdir=hyp`
### Training
For this example, suppose the best learning rate found is `1e-3`. You can choose to train using either a `Hyperparameters` object with set values, or pass the found parameters directly into the `prime` functions.
**(NOTE: If an optimizer/scheduler/model requires positional arguments, these MUST be passed into the `prime` function)**
```python
  tr.prime_optimizer(lr=1e-3, momentum=0.9)                                      
  tr.prime_scheduler(5, gamma=0.1)                             
  tr.prime_model(input_size=2, hidden_size=5, target_length=2) 
```
Now just train using `tr.train(epochs=50, update_every=1, save_every=5)`. This will train the model for 50 epochs, saving checkpoints every 5 epochs and writing TensorBoard files for the loss every epoch.
## Interpreting Results
By default, the trainer only keeps track of training and validation/testing loss as a function of epoch.\
This can be adjusted by adding new entries into the history dictionary in the `init_history` function and adding corresponding code into either `train_step` or `evaluate_step`:
```python
# Perform forward and backward passes
  forward_pass = self.pass_batch(batch)
  loss, preds = forward_pass['loss'], forward_pass['preds']
# Store metrics <-- additional measurements get appended after passing through the model
  eval_loss.append(float(loss))
```
For interpretation, extract the `history` variable and access the data using the keys in `init_history`.\
To make inferences, extract the model and perform a forward pass. Infrastructure for inference is not supported by this program
