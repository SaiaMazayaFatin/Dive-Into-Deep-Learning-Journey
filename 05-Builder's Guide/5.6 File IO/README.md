# File IO

## 1. The Concept: Saving Your Work

Training a deep neural network can take days or weeks. You don't want to lose all that progress if your computer crashes or if you want to use the model later.

- **Persistence**: We need a way to save the learnable parameters (weights and biases) to a file on your hard drive.
- **Loading**: Later, we can load these numbers back into a new network architecture to resume training or make predictions.
- **Analogy**: Think of this like saving a game. The "Architecture" is the game cartridge (code), and the "Parameters" are your save file (the specific state you reached).

## 2. Saving and Loading Tensors

The most basic unit of data in PyTorch is a Tensor. We can save individual tensors or lists/dictionaries of tensors directly.

- **The Function**: `torch.save` and `torch.load`
- **Under the hood**: PyTorch uses Python's pickle utility to serialize objects.

### The Code (PyTorch):

```Python
import torch
from torch import nn

# 1. Create some tensors (simulating data or weights)
x = torch.arange(4)
y = torch.zeros(4)

# 2. Save them to a file named 'x-file'
torch.save([x, y], 'x-file')

# 3. Load them back
x2, y2 = torch.load('x-file')
print(x2)
print(y2)

# Dictionary Example
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
dict2 = torch.load('mydict')
print(dict2)
```

## 3. Saving and Loading Models (The Important Part)

When saving a whole neural network, we usually save the **Parameter Dictionary** (called `state_dict`), not the entire model object itself.

**Why?** The model object contains code definition, which might change. The `state_dict` just contains the numbers (weights/biases) mapped to layer names.

### The Math (State Dictionary Mapping):

A model consists of layers $L_1, L_2, \dots, L_n$.

Each layer $L_i$ has parameters $\theta_i = \{\mathbf{W}_i, \mathbf{b}_i\}$.

The `state_dict` is a mapping function $M$:

$$M: \text{"layer\_name"} \rightarrow \theta_{val}$$

**Example mapping:**
- "hidden.weight" $\rightarrow$ Matrix of shape $(256, 20)$
- "output.bias" $\rightarrow$ Vector of shape $(10)$

### The Code (Saving Model Parameters):

```Python
# 1. Define a simple MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.relu = nn.ReLU()
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(self.relu(self.hidden(x)))

net = MLP()
X = torch.randn(2, 20)
Y = net(X)

# 2. Save the Parameters (State Dict)
# We save ONLY the weights, not the class definition
torch.save(net.state_dict(), 'mlp.params')

# 3. Load the Parameters
# First, we must create a FRESH instance of the model (The "Game Cartridge")
clone = MLP()

# Then, we load the saved weights (The "Save File")
clone.load_state_dict(torch.load('mlp.params'))

# 4. Verify
# Both networks should produce the exact same output
Y_clone = clone(X)
print(Y == Y_clone) # Should be all True
```

## 4. Summary: Best Practices

- **Always save the state_dict**: It is safer and more flexible than saving the entire model object.
- **Architecture Match**: To load parameters successfully, the code for the model architecture (class MLP) must match exactly the architecture used when the parameters were saved. If you change layer sizes in the code, the saved weights won't fit!
- **File Extension**: It is common convention to use `.pt` or `.pth` for PyTorch files (e.g., `model.pth`).