# Implementing Softmax Regression from Scratch

## 1. The Softmax Function
To turn raw model outputs (logits) into probabilities, we use the Softmax function. It performs three steps:
1. **Exponentiate**: $exp(x)$ makes all values positive.
2. **Sum**: Calculate the sum of each row (partition function).
3. **Divide**: Normalize each row so it sums to 1.

**The Formula:**

$$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}$$
​
 
**The Code**: We implement this using PyTorch's broadcasting mechanism. ``keepdims=True`` ensures the shape remains compatible for division.

```Python
import torch
from base_classes import Classifier

def softmax(X):
    """Compute softmax values for X."""
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition  # Broadcasting is applied here
``` 

**Test Case**: If we create a random matrix, the result should sum to 1 along the rows.

```Python
X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(1))
```

## 2. The Model Architecture
We define the class ``SoftmaxRegressionScratch``.
- **Input**: We flatten the $28×28$ images into vectors of length **784**.
- **Output**: We have **10 classes**, so we need 10 outputs.
- **Parameters**:
    - **Weights** $W: 784×10$ (Initialized with Gaussian noise).
    - **Bias** $b: 1×10$ (Initialized to zeros).

**The Code:**

```Python
class SoftmaxRegressionScratch(Classifier):
    """Softmax regression implementation from scratch."""
    def __init__(self, num_inputs, num_outputs, lr=0.1, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize weights with Gaussian noise
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        # Initialize bias with zeros
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        """Return model parameters."""
        return [self.W, self.b]

    def forward(self, X):
        """Forward pass through the model."""
        # Flatten the image (batch_size, 784)
        X = X.reshape((-1, self.W.shape[0]))
        # Linear mapping + Softmax
        return softmax(torch.matmul(X, self.W) + self.b)
```

## 3. The Cross-Entropy Loss
We use Cross-Entropy (Negative Log-Likelihood) to measure error.
- **Goal**: Maximize the probability assigned to the correct label.
- **Implementation Trick**: Instead of a loop, we use array slicing to pick the specific probability corresponding to the correct class for each example.

**The Code:**

```Python
def cross_entropy(y_hat, y):
    """Cross-entropy loss function."""
    # Select the predicted probability for the correct class label
    return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()

# Add this method to SoftmaxRegressionScratch class
def loss(self, y_hat, y):
    """Compute cross-entropy loss."""
    return cross_entropy(y_hat, y)
```

**Example of the Slicing Trick:**

```Python
y = torch.tensor([0, 2]) # True labels
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]]) # Predictions
# This selects 0.1 (class 0) for the 1st example and 0.5 (class 2) for the 2nd
print(y_hat[[0, 1], y]) 
print(cross_entropy(y_hat, y))
```

## 4. Training
We reuse the training loop (SGD) defined in the previous Linear Regression section.

**Hyperparameters:**
- `num_inputs`: 784
- `num_outputs`: 10
- `lr` (Learning Rate): 0.1
- `max_epochs`: 10
- `batch_size`: 256

**The Code:**

```Python
# 1. Load Data
from base_classes import FashionMNIST, Trainer

data = FashionMNIST(batch_size=256)

# 2. Initialize Model
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)

# 3. Setup Trainer
trainer = Trainer(max_epochs=10)

# 4. Run Training
trainer.fit(model, data)
```

## 5. Prediction & Visualization
After training, we use `argmax` to find the class with the highest probability. We are specifically interested in visualizing the incorrect predictions (where the model failed).

**The Code:**

```Python
# Get a batch of validation data
val_loader = data.get_dataloader(train=False)
batch = next(iter(val_loader))
X, y = batch

# Make predictions (argmax gets the index of the highest probability)
with torch.no_grad():
    preds = model(X).argmax(dim=1)

# Identify wrong predictions
wrong_mask = (preds != y)
X_wrong, y_wrong, preds_wrong = X[wrong_mask], y[wrong_mask], preds[wrong_mask]

# Create labels for the plot (True Label vs Predicted Label)
labels = [f'True: {true}\nPred: {pred}' for true, pred in zip(
    data.get_labels(y_wrong), data.get_labels(preds_wrong))]

# Visualize using matplotlib
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(min(8, len(X_wrong))):
    ax = axes[i // 4, i % 4]
    ax.imshow(X_wrong[i].squeeze(), cmap='gray')
    ax.set_title(labels[i])
    ax.axis('off')

plt.tight_layout()
plt.show()
```

## Summary
- We implemented **Softmax** manually to convert outputs to probabilities.
- We used **Cross-Entropy** with an indexing trick for the loss function.
- We flattened images into vectors (784 dimensions) to fit a linear model.
- We reused the standard **SGD** training loop to optimize the weights.