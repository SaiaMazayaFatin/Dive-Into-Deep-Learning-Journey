# Synthetic Regression Data

## 1. Why Synthetic Data?

Before working with real-world messy data, we use synthetic (fake) data to verify our algorithms.

- **Goal**: Evaluate learning algorithms and confirm implementation correctness.
- **Advantage**: We know the ground truth parameters ($w$ and $b$) a priori, so we can check if our model recovers them accurately.

## 2. The Mathematical Model

We generate data where the relationship between inputs (features) and output (label) is linear, with some added noise.

**Formula:**

$$\mathbf{y} = \mathbf{X} \mathbf{w} + b + \epsilon$$

Where: 
- $\mathbf{X}$: The input features matrix (randomly sampled from a normal distribution).
- $\mathbf{w}$: The true weights (coefficients).
- $b$: The true bias (intercept).
- $\epsilon$: Additive noise (allows us to simulate real-world imperfection), drawn from a normal distribution with mean $0$ and standard deviation $0.01$.

## 3. Generating the Dataset (Code Implementation)

We create a class `SyntheticRegressionData` that inherits from the `DataModule` (defined in the previous Object-Oriented Design module).

Key Steps in `__init__`:

1. Generate $\mathbf{X}$ (features) using torch.randn.
2. Calculate $\mathbf{y}$ using matrix multiplication (torch.matmul) plus the bias and noise.

```python
import random
import torch
import torch.utils.data

# Import our OOD framework components
from .."2.2 Object-Oriented Design for Implementation".code import DataModule, add_to_class

class SyntheticRegressionData(DataModule):  #@save
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        # Generate X: Matrix of shape (n, len(w))
        self.X = torch.randn(n, len(w))
        # Generate Noise
        noise = torch.randn(n, 1) * noise
        # Generate y: y = Xw + b + noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise
```
**Testing the Generation**: We create an instance with specific "Ground Truth" parameters: $\mathbf{w} = [2, -3.4]$ and $b = 4.2$.

```python
data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)

# Inspect the first example
print('features:', data.X[0],'\nlabel:', data.y[0])
```

## 4. Reading Data (The "From Scratch" Way)

Training requires iterating over the dataset in small groups called **minibatches**. This function shuffles the data (for training) and yields chunks of size `batch_size`.

- **Training Mode**: Shuffles indices to read data in random order.

- **Validation Mode**: Reads data sequentially (no shuffle).

```python
@add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    
    # Yield batches
    for i in range(0, len(indices), self.batch_size):
        batch_indices = torch.tensor(indices[i: i+self.batch_size])
        yield self.X[batch_indices], self.y[batch_indices]
```

**Inspection:**
```python
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

## 5. Reading Data (The Concise / Efficient Way)
Writing custom iterators (as done above) is useful for learning but inefficient. Deep learning frameworks provide optimized DataLoaders that handle memory and shuffling better.

We replace the previous method with PyTorch's `DataLoader` and `TensorDataset`.

```python
@add_to_class(DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    # Slice the tensors (X and y) based on indices (train vs val)
    tensors = tuple(a[indices] for a in tensors)
    # Wrap them in a TensorDataset
    dataset = torch.utils.data.TensorDataset(*tensors)
    # Return a PyTorch DataLoader
    return torch.utils.data.DataLoader(dataset, self.batch_size,
                                       shuffle=train)

@add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    # Define the slice for training vs validation rows
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)
```

**Verification:** The behavior remains the same, but the backend is now more robust.

```python
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
print('Number of batches:', len(data.train_dataloader()))
```

**Study Summary**
1. **Synthetic Data** is critical for debugging because you know the correct answer ($w, b$).
2. **Generation** relies on the linear algebra formula $\mathbf{y} = \mathbf{X} \mathbf{w} + b + \epsilon$.
3. **Data Loaders** abstract the complexity of shuffling and batching, allowing the model training loop to remain clean and agnostic to the data source.

