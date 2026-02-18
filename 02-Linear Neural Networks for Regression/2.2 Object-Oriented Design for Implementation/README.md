# Object-Oriented Design (OOD) for Deep Learning

## Part 1: Utilities

Before building the deep learning components, we need three specific Python tools to keep the code clean.

### 1. Dynamic Method Registration (`add_to_class`)
Concept: In Jupyter notebooks, classes often get too long. This decorator allows you to define a class first, and then add methods to it later in a separate cell.

**Code:**

```python
import time
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import inspect

def add_to_class(Class):  #@save
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```

**Example Usage**: Here, `A` is defined without `do`. We inject `do` into `A` later.

```python
class A:
    def __init__(self):
        self.b = 1

a = A()

@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)

a.do()  # Output: Class attribute "b" is 1
```

### 2. Automatic Hyperparameter Saving (HyperParameters)
**Concept**: Instead of manually writing `self.a = a`, `self.b = b` inside `__init__`, this utility saves all constructor arguments as class attributes automatically.

**Code:**

```python
class HyperParameters:  #@save
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        """Save constructor arguments as class attributes."""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in local_vars.items() 
                       if k != 'self' and k not in ignore}
        for k, v in self.hparams.items():
            setattr(self, k, v)
```

**Example Usage**: Notice `self.a` is available even though we never explicitly wrote `self.a = a`.

```python
# Using the implemented HyperParameters class
class B(HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)
```

### 3. Live Plotting (`ProgressBoard`)

**Concept**: A simplified version of TensorBoard. It allows you to plot loss and accuracy curves in real-time during training.

```python
class ProgressBoard(HyperParameters):  #@save
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()
        self.data = {}
        
    def draw(self, x, y, label, every_n=1):
        """Draw points in real-time animation."""
        if label not in self.data:
            self.data[label] = {'x': [], 'y': []}
        self.data[label]['x'].append(x)
        self.data[label]['y'].append(y)
        
        if len(self.data[label]['x']) % every_n == 0:
            plt.clf()
            for key, values in self.data.items():
                plt.plot(values['x'], values['y'], label=key)
            plt.xlabel(self.xlabel or 'x')
            plt.ylabel(self.ylabel or 'y')
            plt.legend()
            plt.pause(0.01)
```

**Example Usage:**

```python
board = ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
```

## Part 2: Core Architecture (The "Trinity")

The architecture divides the deep learning workflow into three distinct classes: **Module**, **DataModule**, and **Trainer**.

### 1. The Model (`Module`)
**Role**: Contains the neural network, loss function, and optimization logic. Inheritance: Inherits from `nn.Module` (PyTorch standard) and `HyperParameters`.

**Key Methods**:
- `forward`: The standard forward pass.
- `training_step`: Calculates loss for a batch (used by Trainer).
- `configure_optimizers`: Defines how the model updates weights (e.g., SGD, Adam).    

**Code:**

```python
class Module(nn.Module, HyperParameters):  #@save
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, value.cpu().detach().numpy(),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```

### 2. The Data (`DataModule`)
**Role**: Handles data loading, preprocessing, and splitting. Key Methods:
- `train_dataloader`: Returns a generator for training data batches.
- `val_dataloader`: Returns a generator for validation data batches.

**Code:**

```python
class DataModule(HyperParameters):  #@save
    """The base class of data."""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
```

### 3. The Trainer (`Trainer`)
**Role**: The "Conductor." It takes a Module and a DataModule and runs the training loop. It handles the epochs and batch iterations. **Key Methods**:
- `fit`: The main entry point. It links the model to the data and starts the loop.
- `fit_epoch`: Logic for running one full pass over the dataset (to be implemented in later chapters).

**Code:**

```python
class Trainer(HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
```

**Study Summary**

To implement a new Deep Learning project using this framework, you only need to:

1. Subclass `Module`: Define your network and loss.
2. Subclass `DataModule`: Define how to load your specific dataset.
3. Instantiate `Trainer`: Pass the model and data to Trainer.fit().

