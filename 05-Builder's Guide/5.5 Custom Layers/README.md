# Custom Layers

## 1. The Concept: Building Your Own LEGO Bricks

We've already learned how to build a "Block" (a network) by stacking existing layers like `nn.Linear` or `nn.ReLU`. But what if you need a specific mathematical operation that doesn't exist in PyTorch?

You need to build a **Custom Layer**.

- **Standard Layer**: Provided by PyTorch (e.g., Linear, Conv2d).
- **Custom Layer**: A class you write that inherits from `nn.Module`. It can have parameters (weights) or just be a fixed mathematical function.

## 2. Type A: Layers Without Parameters (Functional)

These are layers that perform a fixed calculation. They don't "learn" anything; they just transform data. An example is a layer that centers the data (subtracts the mean).

### The Math (Centering):

For an input vector $\mathbf{x}$ containing $N$ elements, we calculate the mean $\mu$:

$$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$$

The output $\mathbf{y}$ is simply the input shifted by the mean:

$$\mathbf{y} = \mathbf{x} - \mu$$

### The Code (PyTorch):

```Python
import torch
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters to define here!

    def forward(self, X):
        # Calculate the mean of the input data
        # X.mean() gives a single scalar value
        return X - X.mean()

# Let's test it
layer = CenteredLayer()
data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(layer(data)) 
# Output: [-2., -1., 0., 1., 2.] (Mean was 3.0)
```

## 3. Type B: Layers With Parameters (Learnable)

This is the more powerful type. These layers have internal variables (Weights and Biases) that update during training.

**Crucial Step**: You cannot just use a standard tensor (e.g., `torch.rand`) for weights. You must wrap it in `nn.Parameter`.

- **Regular Tensor**: PyTorch treats it as static data.
- **nn.Parameter**: PyTorch adds it to the list of "things to learn" (gradients will be calculated for it).

### The Math (Linear Transformation):

Let's rebuild a standard Linear layer from scratch to see how it works.

Given an input $\mathbf{X}$ (batch size $\times$ input features) and Weights $\mathbf{W}$ (input features $\times$ output features):

$$\mathbf{H} = \mathbf{X}\mathbf{W} + \mathbf{b}$$

- **Matrix Multiplication**: $\mathbf{X}\mathbf{W}$
- **Broadcasting**: The bias $\mathbf{b}$ is added to every row.

### The Code (PyTorch):

```Python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        # 1. Wrap weights in nn.Parameter so PyTorch knows to train them
        # Shape: [in_units, units]
        self.weight = nn.Parameter(torch.randn(in_units, units))
        
        # 2. Wrap bias in nn.Parameter
        # Shape: [units] (one bias per output neuron)
        self.bias = nn.Parameter(torch.zeros(units))

    def forward(self, X):
        # Apply the linear formula: y = xW + b
        linear_map = torch.matmul(X, self.weight) + self.bias
        
        # We can even add a custom activation here if we want!
        return torch.relu(linear_map)

# Usage
custom_dense = MyLinear(in_units=3, units=2)
input_data = torch.rand(5, 3) # Batch of 5, 3 features each

output = custom_dense(input_data)
print("Output Shape:", output.shape) # Should be [5, 2]

# Check the parameters
print("\nWeights (Learned Parameter):\n", custom_dense.weight)
```

## 4. Summary: When to use which?

| Feature | Standard nn.Linear | Custom Layer (nn.Module) |
|---|---|---|
| **Simplicity** | Ready to use, one line of code. | Requires writing a class. |
| **Flexibility** | Limited to standard matrix multiplication. | Can do anything: complex formulas, custom logic, weird reshaping. |
| **Parameters** | Handled automatically. | You must wrap variables in nn.Parameter. |

### Key Takeaway:

If you just need a standard neural network layer, use the built-in ones. If you are inventing a new mathematical operation (like a new type of Attention mechanism or normalization), build a Custom Layer.