# Parameter Initialization

## 1. Why Initialization Matters? (The Goldilocks Principle)

Initializing a neural network means setting the starting values for all the weights ($\mathbf{W}$) and biases ($\mathbf{b}$) before training begins.

- **Too Small**: If weights are too small, the signal shrinks as it passes through layers until it vanishes (Vanishing Gradient).
- **Too Large**: If weights are too large, the signal grows uncontrollably until it explodes (Exploding Gradient).
- **Just Right**: We want the signal's variance (spread) to remain roughly the same from the first layer to the last.

## 2. The Goal: Variance Preservation

We want the variance of the outputs of a layer to equal the variance of its inputs.

### Mathematical Theory:

Consider a single linear neuron output $o_i$:

$$o_i = \sum_{j=1}^{n_{in}} w_{ij} x_j$$

where $n_{in}$ is the number of inputs. If we assume inputs $x$ and weights $w$ are independent and have a mean of 0, the variance of the output is:

$$Var(o_i) = \sum_{j=1}^{n_{in}} Var(w_{ij} x_j)$$

$$Var(o_i) = \sum_{j=1}^{n_{in}} Var(w_{ij}) Var(x_j)$$

Since all inputs and weights are identically distributed:

$$Var(o_i) = n_{in} \cdot Var(W) \cdot Var(x)$$

### The Condition:

To keep the signal stable ($Var(o) = Var(x)$), we need:

$$n_{in} \cdot Var(W) = 1 \implies Var(W) = \frac{1}{n_{in}}$$

## 3. Xavier (Glorot) Initialization

This method is designed for Sigmoid or Tanh activation functions. It considers both the number of inputs ($n_{in}$) and the number of outputs ($n_{out}$).

### The Formula:

To balance the variance during both forward pass (using $n_{in}$) and backward pass (using $n_{out}$), Xavier initialization sets the variance of weights to:

$$Var(W) = \frac{2}{n_{in} + n_{out}}$$

If we sample weights from a Uniform Distribution $U[-a, a]$, the boundary $a$ is calculated as:

$$a = \sqrt{\frac{6}{n_{in} + n_{out}}}$$

### The Code (PyTorch):

```Python
import torch
from torch import nn

def init_xavier(m):
    if type(m) == nn.Linear:
        # Fills the input Tensor with values according to Xavier Uniform
        nn.init.xavier_uniform_(m.weight)

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
net.apply(init_xavier)

print("Xavier Initialized Weights:", net[0].weight.data[0])
```

## 4. He (Kaiming) Initialization

This method is designed specifically for ReLU activation functions.

- **The Issue**: ReLU sets half of the activations to zero (for negative inputs). This halves the variance of the signal.
- **The Fix**: We need to double the variance of the weights to compensate.

### The Formula:

To keep the variance stable with ReLU, we need:

$$Var(W) = \frac{2}{n_{in}}$$

If we sample from a Normal Distribution $N(0, \sigma^2)$, the standard deviation $\sigma$ is:

$$\sigma = \sqrt{\frac{2}{n_{in}}}$$

### The Code (PyTorch):

```Python
def init_kaiming(m):
    if type(m) == nn.Linear:
        # Fills the input Tensor with values according to Kaiming (He) Normal
        # mode='fan_in' preserves variance in forward pass
        # nonlinearity='relu' adjusts for the ReLU zeroing effect
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
net.apply(init_kaiming)

print("Kaiming Initialized Weights:", net[0].weight.data[0])
```

## 5. Summary Table

| Initialization | Best For | Mathematical Variance Goal |
|---|---|---|
| Xavier (Glorot) | Sigmoid, Tanh | $Var(W) = \frac{2}{n_{in} + n_{out}}$ |
| He (Kaiming) | ReLU, LeakyReLU | $Var(W) = \frac{2}{n_{in}}$ |
| Normal (Standard) | Simple networks | $Var(W) = 0.01$ (Fixed small number) |