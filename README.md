# PytorchTrainer
A Template For Hyperparameter Optimization and Training PyTorch Models

Across my various machine learning projects and classes, I found myself re-writing the same infrastructure to train some deep learning model. This repository abstracts
out the task and deals with the raw model. 
## How To Use
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
        data_point = torch.tensor(example[:self.D-1])
        label = torch.tensor(example[self.D-1])
        ret_val['x'] = data_point
        ret_val['y'] = label
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
