# Lazy Initialization

## 1. The Problem: "Guessing" Shapes

When we define a neural network, we usually have to specify the input and output sizes for every single layer manually.

- **The Annoyance**: If you change the input size of the first layer, you have to do math to figure out how that changes the input size of the second layer, third layer, and so on. It's tedious and prone to errors.
- **The Solution (Lazy Initialization)**: We tell the framework: "I don't know the input shape yet. Just wait until I pass the first batch of data, then figure it out automatically."

## 2. How It Works (Deferred Initialization)

In PyTorch, there is a specific module called `LazyLinear` (and `LazyConv2d`, etc.) that handles this.

1. **Definition Phase**: You define the layer, but you only specify the Output Size. You leave the Input Size blank (or let the framework handle it).
2. **Dry Run**: The framework creates a placeholder for the weights but doesn't initialize them yet (because it doesn't know how big the weight matrix should be).
3. **First Forward Pass**: When you pass the first batch of data $\mathbf{X}$ through the network:
   - The framework looks at the shape of $\mathbf{X}$.
   - It infers the correct input size $n_{in}$.
   - It initializes the weights $\mathbf{W}$ and bias $\mathbf{b}$ using that size.
   - It runs the calculation.

## 3. The Mathematical Theory (Shape Inference)

Let's look at the math behind determining the shapes.

### Variable Definitions:

- $\mathbf{X}$: The input batch matrix.
- $n_{in}$: The number of input features (unknown initially).
- $n_{out}$: The number of output features (defined by us).
- $\mathbf{W}$: The weight matrix.

### The Transformation:

For a standard linear layer, the operation is:

$$\mathbf{H} = \mathbf{X}\mathbf{W} + \mathbf{b}$$

- **Input Shape**: $\mathbf{X} \in \mathbb{R}^{batch \times n_{in}}$
- **Output Shape**: $\mathbf{H} \in \mathbb{R}^{batch \times n_{out}}$
- **Weight Shape**: $\mathbf{W} \in \mathbb{R}^{n_{in} \times n_{out}}$

### The "Lazy" Logic:

1. We define $n_{out} = 256$.
2. We receive data $\mathbf{X}$ with shape $(32, 20)$.
3. The framework sees that the second dimension of $\mathbf{X}$ is 20.
4. Therefore, it sets $n_{in} = 20$.
5. Now it creates a weight matrix $\mathbf{W}$ of shape $(20, 256)$.

## 4. The Code (PyTorch Implementation)

PyTorch provides `nn.LazyLinear` for fully connected layers and `nn.LazyConv2d` for convolutional layers.

**Note**: You must pass a sample input through the network to trigger the initialization.

```Python
import torch
from torch import nn

# 1. Define the Network using Lazy Modules
# Notice we ONLY specify the output size (256), not the input size.
net = nn.Sequential(
    nn.LazyLinear(256), 
    nn.ReLU(),
    nn.LazyLinear(10)   
)

# 2. Check Parameters BEFORE initialization
# At this point, the weights are "Uninitialized" or empty placeholders.
# (In older PyTorch versions, accessing them might raise an error or show shape 0)
print("Network defined, but input shape unknown.")

# 3. Create dummy input data
# Batch size 2, Input features 20
X = torch.rand(2, 20)

# 4. The "Dry Run" (Trigger Initialization)
# The moment we pass X, PyTorch infers the input size is 20.
out = net(X)

print("\n--- After First Forward Pass ---")
print(f"Output shape: {out.shape}") # Should be [2, 10]

# 5. Verify the inferred shapes
# The first layer should now have weights of shape [256, 20]
# (PyTorch stores weights as [out_features, in_features])
print(f"Layer 1 Weight Shape: {net[0].weight.shape}") 
```

## 5. Summary: Why use it?

### Pros:

- **Convenience**: You don't need to calculate intermediate shapes manually (especially useful for Convolutional Networks where shapes shrink after pooling).
- **Flexibility**: The same network code can handle different input sizes (e.g., images of $28 \times 28$ or $64 \times 64$) without rewriting the class.

### Cons:

- **Overhead**: The first pass is slightly slower because it has to allocate memory and initialize weights on the fly.
- **Debugging**: Errors related to shape mismatch might only appear at runtime when data is passed, rather than when the model is defined.