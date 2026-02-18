# Parameter Management

## 1. What are Parameters? (The "Knobs")

A deep neural network is essentially a mathematical function with millions of adjustable "knobs." These knobs are called Parameters.

- **Weights ($\mathbf{W}$)**: Determine the strength of the connection between neurons.
- **Biases ($\mathbf{b}$)**: Determine the activation threshold (shift) of a neuron.

When we "train" a network, we are simply adjusting these numbers to minimize error.

[PLACE IMAGE HERE: A diagram of a single neuron showing inputs $x_1, x_2$ connecting to weights $w_1, w_2$ and a bias $b$, producing an output $y$. Highlight $w$ and $b$ as the "learnable parameters."]

## 2. Accessing Parameters (The Inventory)

In PyTorch, parameters are stored as tensors. You can access them by indexing the layers of your network.

### The Code (Accessing Weights):

```Python
import torch
from torch import nn

# Let's build a simple network with one hidden layer
net = nn.Sequential(
    nn.Linear(4, 8),  # Layer 0: Input size 4, Output size 8
    nn.ReLU(),        # Layer 1: Activation
    nn.Linear(8, 1)   # Layer 2: Output layer
)

# Access the weight of the first layer (Layer 0)
# 'state_dict()' gives us a dictionary of all parameters
print(net[0].state_dict()) 

# Access specifically the Weight tensor
print(net[0].weight.data) 

# Access specifically the Bias tensor
print(net[0].bias.data)
```

### The Math (Shapes and Dimensions):

For a Linear layer taking an input vector $\mathbf{x}$ of dimension $d$ and producing an output vector $\mathbf{h}$ of dimension $q$:

- **Input ($\mathbf{x}$)**: $\mathbf{x} \in \mathbb{R}^{1 \times d}$ (Row vector)
- **Weight Matrix ($\mathbf{W}$)**: $\mathbf{W} \in \mathbb{R}^{d \times q}$
- **Bias Vector ($\mathbf{b}$)**: $\mathbf{b} \in \mathbb{R}^{1 \times q}$

The formula for the linear transformation is:

$$\mathbf{h} = \mathbf{x}\mathbf{W} + \mathbf{b}$$

If you look at the code above (`nn.Linear(4, 8)`):

- $\mathbf{W}$ has shape $(8, 4)$ in PyTorch implementation (transposed for efficiency).
- $\mathbf{b}$ has shape $(8)$.

## 3. Parameter Initialization (Starting Point)

By default, PyTorch initializes weights randomly. However, sometimes we need to set them to specific values (like all zeros, or a specific statistical distribution) to help the network learn better.

### Types of Initialization:

- **Gaussian (Normal) Init**: Random numbers from a bell curve.
- **Constant Init**: Setting all weights to a specific number (e.g., 1).
- **Xavier/Kaiming Init**: Specialized math to prevent gradients from exploding/vanishing (advanced).

### The Code (Custom Initialization):

```Python
# Define a function to initialize weights
def init_normal(m):
    if type(m) == nn.Linear:
        # Fill weights with random numbers from Normal Distribution
        # Mean=0, Std=0.01
        nn.init.normal_(m.weight, mean=0, std=0.01)
        # Set bias to 0
        nn.init.zeros_(m.bias)

# Apply this function to every layer in the network
net.apply(init_normal)
print("Weights after Normal Init:", net[0].weight.data[0])

# --- Another Example: Constant Initialization ---
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1) # Set all weights to 1

net.apply(init_constant)
print("Weights after Constant Init:", net[0].weight.data[0])
```

### The Math (Distributions):

**Gaussian Initialization:**
We draw weights $w$ from a Normal distribution with mean $\mu$ and standard deviation $\sigma$:

$$w \sim \mathcal{N}(\mu, \sigma^2)$$

**Constant Initialization:**
We simply set every element $w_{ij}$ to a constant $c$:

$$w_{ij} = c$$

[PLACE IMAGE HERE: A visualization of a Bell Curve (Normal Distribution). Show that most weights are initialized near 0, with very few large numbers.]

## 4. Tied Parameters (Shared Memory)

Sometimes, you want two different layers in a network to share the exact same weights. If you update the weight in Layer A, Layer B updates automatically because they point to the same memory location.

This is common in models like RNNs or Transformers (embeddings).

### The Code (Sharing Weights):

```Python
# Create a standard linear layer
shared = nn.Linear(8, 8)

# We use the 'shared' object twice!
net_shared = nn.Sequential(
    nn.Linear(4, 8),    # Input layer
    nn.ReLU(),
    shared,             # Hidden Layer 1 (Uses 'shared' weights)
    nn.ReLU(),
    shared,             # Hidden Layer 2 (Uses SAME 'shared' weights)
    nn.Linear(8, 1)     # Output
)

# Proof: If we change weights in Layer 2, Layer 4 changes too.
net_shared[2].weight.data[0,0] = 100
print(net_shared[2].weight.data[0,0] == net_shared[4].weight.data[0,0]) 
# Output will be True
```

### The Math (Gradient Accumulation):

If a parameter $\mathbf{W}$ is used in two places in the calculation (let's call them function $g$ and function $h$), the total gradient is the sum of the gradients from both places.

If Loss $L = f(g(\mathbf{x}, \mathbf{W}), h(\mathbf{x}, \mathbf{W}))$, then by the Chain Rule:

$$\frac{\partial L}{\partial \mathbf{W}} = \underbrace{\frac{\partial L}{\partial g} \frac{\partial g}{\partial \mathbf{W}}}_{\text{Gradient from first use}} + \underbrace{\frac{\partial L}{\partial h} \frac{\partial h}{\partial \mathbf{W}}}_{\text{Gradient from second use}}$$

This ensures that the shared weight learns from both parts of the network simultaneously.

[PLACE IMAGE HERE: A diagram of a network where two distinct layers are connected by a dotted line to a single "Weight Storage" box, indicating they read/write to the same location.]