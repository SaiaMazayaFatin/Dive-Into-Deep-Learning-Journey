# Concise Linear Regression

## 1. Introduction

In the previous "From Scratch" implementation, we manually defined parameters (`w`, `b`) and math operations (`matmul`). Modern frameworks like PyTorch automate this using **Layers** and Modules.

**Key Advantages**:
- **Automation**: No need to manually calculate gradients or shapes.
- **Optimization**: Framework layers are highly optimized C++ implementations.
- **Simplicity**: Focus on architecture rather than arithmetic.

## 2. Defining the Model (The Lazy Way)
**Concept**: Instead of manually creating weights `w` and bias `b`, we use a **Fully Connected Layer** (also known as a `Linear` layer).
- `nn.Linear(in, out)`: You must specify input and output dimensions.
- `nn.LazyLinear(out)`: You specify only the output dimension. The framework infers the input dimension automatically when data is first passed through.

**Code Implementation**: We define a single layer with 1 output unit. We also initialize the weights using a normal distribution ($\sigma=0.01$) and bias to $0$.

```python
import numpy as np
import torch
from torch import nn

# Import our OOD framework components
from .."2.2 Object-Oriented Design for Implementation".code import Module, add_to_class

class LinearRegression(Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        # LazyLinear infers input shape automatically
        self.net = nn.LazyLinear(1)
        # Initialize weight and bias
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

@add_to_class(LinearRegression)  #@save
def forward(self, X):
    # Just pass input X into the predefined net
    return self.net(X)
```

## 3. Defining the Loss Function

**Concept**: We replace our manual squared error calculation with PyTorch's built-in `MSELoss`.

**Formula:**
$$L = \text{mean}((y - \hat{y})^2)$$

Note: unlike our scratch implementation, standard `MSELoss`typically does not include the $\frac{1}{2}$ factor, but since it is a constant, it only scales the gradient and does not change the location of the minimum.

**Code Implementation:**

```python
@add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    fn = nn.MSELoss()
    return fn(y_hat, y)
```

## 4. Defining the Optimizer

**Concept**: We use the framework's built-in Stochastic **Gradient Descent (SGD)**. This handles the parameter updates ($w \leftarrow w - \eta \cdot g$) automatically.
- `self.parameters()`: A helper method in `nn.Module` that returns a list of all learnable weights and biases in the model.
- `lr`: The learning rate.

**Code Implementation:**

```python
@add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr)
```

## 5. Training and Validation
**Concept**: We reuse the `Trainer` class (from the Object-Oriented Design module). The logic remains identical: iterate through epochs, calculate loss, and update weights.

**Code Implementation:** We generate the same synthetic data as before (Ground Truth: $\mathbf{w}=[2, -3.4]$, $b=4.2$) to verify the concise model works.

```python
# Import our framework components
from .."2.2 Object-Oriented Design for Implementation".code import Trainer
from .."2.3 Synthetic Regression Data" import SyntheticRegressionData

# 1. Initialize Model
model = LinearRegression(lr=0.03)

# 2. Get Data
data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)

# 3. Setup Trainer
trainer = Trainer(max_epochs=3)

# 4. Train
trainer.fit(model, data)
```

**Verification**: We extract the learned weights from model.`net.weight.data` and compare them to the true values. They should be nearly identical.

```python
@add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return (self.net.weight.data, self.net.bias.data)
w, b = model.get_w_b()

print(f'error in estimating w: {data.w - w.reshape(data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
```

**Study Summary**:
1. `nn.LazyLinear`: Automates shape inference and weight creation.
2. `nn.MSELoss`: Standardized loss function calculation.
3. `torch.optim`:` Standardized optimization algorithms.
4. **Abstraction**: The code is significantly shorter and less prone to math errors compared to the "from scratch" version.
