# Linear Regression from Scratch

## 1. The Model
**Concept**: Linear regression posits that the output $y$ is a weighted sum of the inputs $\mathbf{X}$ plus a bias $b$.**Formula:**

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$$

where:
- $\mathbf{X} \in \mathbb{R}^{n \times d}$ is the input data
- $\mathbf{w} \in \mathbb{R}^{d \times 1}$ are the learnable weights.
- $b \in \mathbb{R}$ is the learnable bias (offset).

**Code Implementation:**


We define the class and initialize parameters $\mathbf{w}$ and $b$. We sample $\mathbf{w}$ from a normal distribution ($\sigma=0.01$) and initialize $b$ to zero. `requires_grad=True` is crucial to allow PyTorch to calculate gradients automatically.

```python
import torch

# Import our OOD framework components
from ..."2.2 Object-Oriented Design for Implementation".code import Module, HyperParameters, add_to_class

class LinearRegressionScratch(Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
```

**The Forward Pass**: This method calculates the output based on current parameters.

```python
@add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    return torch.matmul(X, self.w) + self.b
```

## 2. The Loss Function
**Concept**: We need to measure how "wrong" the model is. We use the Squared Error loss. **Formula**:

$$L(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n \frac{1}{2} (\hat{y}^{(i)} - y^{(i)})^2$$

We use $\frac{1}{2}$ to simplify the derivative (since $\frac{d}{dx} x^2 = 2x$, the $2$ and $1/2$ cancel out).

**Code Implementation**: Note that we average the loss (`l.mean()`) over the batch.

```python
@add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return l.mean()
```

## 3. The Optimization Algorithm (Minibatch SGD)
**Concept**: To minimize loss, we adjust parameters in the opposite direction of the gradient. This is **Stochastic Gradient Descent (SGD)**. **Formula**:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \nabla_{\mathbf{w}} L$$

$$b \leftarrow b - \eta \cdot \nabla_{b} L$$

where $\eta$ (eta) is the **learning rate** (`lr`).

**Code Implementation**:The `step` function applies the update. `zero_grad` clears old gradients before calculating new ones.

```python
class SGD(HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            # Update param: param = param - lr * gradient
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

@add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    return SGD([self.w, self.b], self.lr)
```

## 4. The Training Loop
**Concept**: Training is an iterative loop. For every Epoch (full pass through data), we iterate through **Batches** (chunks of data).

**Sequence per Batch**:

1. **Forward**: Compute predictions $\hat{y}$.
2. **Loss**: Calculate error $L(\hat{y}, y)$.
3. **Backward**: Calculate gradients $\nabla L$ via backpropagation (`loss.backward()`).
4. **Update**: Adjust weights using SGD (`self.optim.step()`).

**Code Implementation**: The `Trainer` class (from the Object-Oriented Design module) is extended here with `fit_epoch`.

```python
@add_to_class(Trainer)  #@save
def prepare_batch(self, batch):
    return batch

@add_to_class(Trainer)  #@save
def fit_epoch(self):
    self.model.train()
    for batch in self.train_dataloader:
        # 1. Forward & 2. Loss
        loss = self.model.training_step(self.prepare_batch(batch))
        
        # 3. Backward
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            # 4. Update
            self.optim.step()
        self.train_batch_idx += 1
        
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

## 5. Execution and Validation
**Concept**: We generate synthetic data where we know the true weights ($\mathbf{w} = [2, -3.4]$, $b=4.2$) to verify if our model learns correctly.

**Code Implementation**: We instantiate the components and call `trainer.fit()`.

```python
# Import our framework components  
from ..."2.2 Object-Oriented Design for Implementation".code import Trainer
from ..."2.3 Synthetic Regression Data" import SyntheticRegressionData

# 1. Define Model
model = LinearRegressionScratch(2, lr=0.03)

# 2. Generate Data (Ground truth: w=[2, -3.4], b=4.2)
data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)

# 3. Setup Trainer
trainer = Trainer(max_epochs=3)

# 4. Start Training
trainer.fit(model, data)
```

**Verification**: After training, we compare the learned parameters (`model.w`, `model.b`) with the true parameters (`data.w`, `data.b`). The error should be near zero.

```python
with torch.no_grad():
    print(f'error in estimating w: {data.w - model.w.reshape(data.w.shape)}')
    print(f'error in estimating b: {data.b - model.b}')
```

