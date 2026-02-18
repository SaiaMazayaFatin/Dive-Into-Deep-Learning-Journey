# Concise Implementation of Softmax Regression

## 1. The Model Architecture (High-Level API)
In the "concise" approach, we stop manually initializing parameters ($\mathbf{W}, \mathbf{b}$). We use PyTorch's `nn` (Neural Network) module to define the layers.

**Mathematical Operation**:The model performs a linear transformation on the flattened input:

$$\mathbf{o} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

- $\mathbf{x}$: The input vector (flattened from $28 \times 28$ to $784$).
- $\mathbf{o}$: The output vector (logits) of size $10$.

**The Code Implementation**:
- `nn.Flatten()`: Reshapes input tensor $\mathbf{X}$ from $(N, 1, 28, 28)$ to $(N, 784)$.
- `nn.LazyLinear(num_outputs)`: Automatically infers the input dimension ($784$) from the data during the first forward pass and initializes $\mathbf{W}$ and $\mathbf{b}$.

```Python
import torch
from torch import nn
from torch.nn import functional as F
from base_classes import Classifier

class SoftmaxRegression(Classifier):
    """The softmax regression model."""
    def __init__(self, num_outputs, lr=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_outputs)
        )

    def forward(self, X):
        """Forward pass - returns logits (not probabilities)."""
        return self.net(X)
```

## 2. The Numerical Stability Problem (Overflow & Underflow)
You will notice the `forward` method above returns raw logits ($\mathbf{o}$), not probabilities ($\hat{\mathbf{y}}$). We intentionally skip the softmax activation here.

**The Problem**:The standard softmax formula involves exponentiation:

$$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$$

- **Overflow**: If $o_j$ is large (e.g., $100$), then $\exp(100) \approx 2.6 \times 10^{43}$, which is too large for standard floating-point numbers.
- **Underflow**: If $o_j$ is very negative (e.g., $-100$), then $\exp(-100) \approx 0$. If the denominator becomes $0$, division fails.

**The Mathematical Fix (Shift Invariance)**:We can subtract the maximum value $\bar{o} = \max_k o_k$ from every logit. This shifts the largest exponent to $\exp(0) = 1$, guaranteeing stability without changing the result.

$$\hat y_j = \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})}$$

## 3. The Loss Function (LogSumExp Trick)
To compute the Cross-Entropy Loss efficiently and stably, we combine the Softmax and Logarithm steps into a single mathematical operation.

**Standard Cross-Entropy Formula:**

$$l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_j y_j \log \hat{y}_j$$

**Stable Derivation (The "LogSumExp" Trick)**: When we plug the "stable softmax" formula into the log term, the exponentials and logs partially cancel out:

$$\begin{aligned}
\log \hat{y}_j &= \log \left( \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})} \right) \\
&= \underbrace{(o_j - \bar{o})}_{\text{Linear Term}} - \underbrace{\log \left( \sum_k \exp (o_k - \bar{o}) \right)}_{\text{LogSumExp Term}}
\end{aligned}$$

By calculating this directly, we avoid ever creating the massive number $\exp(o_j)$.

**The Code**: PyTorch's `F.cross_entropy` implements exactly the formula derived above. It takes raw logits (`Y_hat`) and integer labels (`Y`).

```Python
# Add this method to the Classifier base class
def loss(self, predictions, targets, averaged=True):
    """Cross-entropy loss using PyTorch's built-in function."""
    predictions = predictions.reshape((-1, predictions.shape[-1]))
    targets = targets.reshape((-1,))
    return F.cross_entropy(
        predictions, targets, reduction='mean' if averaged else 'none'
    )
```

## 4. Training Loop
The training process remains the same as in previous sections. We use a standard training loop that handles the gradient descent (forward pass → loss calculation → backward pass → optimizer update).

```Python
from base_classes import FashionMNIST, Trainer

data = FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = Trainer(max_epochs=10)
trainer.fit(model, data)
```

## Summary
1. **Architecture**: We use `nn.Sequential` containing `Flatten` and `LazyLinear` to define $\mathbf{o} = \mathbf{W}\mathbf{x} + \mathbf{b}$.
2. **Stability**: We do not calculate Softmax explicitly in the forward pass to avoid calculating $\exp(\text{large number})$
3. **Optimization**: We use F.cross_entropy, which uses the formula $\log \hat{y}_j = o_j - \bar{o} - \log \sum \exp(o_k - \bar{o})$ to safely compute loss from raw logits.