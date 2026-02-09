# Image Classification Dataset (Fashion-MNIST)

## 1. Why Fashion-MNIST?
Historically, the **MNIST** dataset (handwritten digits) was the standard benchmark. However, it is now considered too simple because even weak models achieve >95% accuracy.

We use **Fashion-MNIST** as a modern alternative.
- **Similarities**: Same size ($28 \times 28$ pixels), same number of categories (10), same split (60k training, 10k test).
- **Difference**: It contains images of clothing (T-shirts, coats, sneakers) which are harder to classify than digits.

## 2. The ``FashionMNIST`` Class
We build a class to handle downloading, resizing, and storing the data. This class inherits from a base ``DataModule`` class that provides common functionality.

**Features**:
- **Transforms**: We resize images to $32 \times 32$ (to match typical modern architectures) and convert them to Tensors.
- **Storage**: We load ``self.train`` and ``self.val`` using the built-in ``torchvision.datasets``.
 
```Python
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from base_classes import DataModule

class FashionMNIST(DataModule):
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])
        
        # Download Training Data
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=transform, download=True
        )
        # Download Validation Data
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=transform, download=True
        )
```

**Checking Data Shapes**:The images are grayscale (1 channel). By convention, PyTorch stores images as $(c \times h \times w)$.

```Python
data = FashionMNIST(resize=(32, 32))
# Result: 60,000 training, 10,000 validation
print(len(data.train), len(data.val)) 

# Shape of one image: torch.Size([1, 32, 32])
print(data.train[0][0].shape)
```

## 3. converting Labels to Names
The dataset uses numbers (0-9) to represent categories. We need a helper function to convert these numbers into human-readable text labels (e.g., 0 $\rightarrow$ "t-shirt").

```Python
# Add this method to the FashionMNIST class
def get_labels(self, indices):
    """Return text labels for Fashion-MNIST."""
    text_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return [text_labels[int(i)] for i in indices]
```
## 4. Reading Minibatches (``DataLoader``)
Deep learning models don't process one image at a time (too slow) or all images at once (too much memory). We use a **DataLoader** to read small batches (e.g., 64 images) at a time.
- **Shuffle**: We shuffle the training data so the model doesn't memorize the order.
- **Workers**: We can use multiple processes (**num_workers**) to load data faster.

```Python
# Add this method to the FashionMNIST class
def get_dataloader(self, train=True):
    """Get a data iterator."""
    dataset = self.train if train else self.val
    return DataLoader(
        dataset, 
        batch_size=self.batch_size,
        shuffle=train,
        num_workers=0  # Set to 0 for Windows compatibility
    )
```

**Testing the Loader**:We can extract one batch to verify the shapes
- **X (Images)**: $(64, 1, 32, 32)$ — Batch of 64, 1 channel, 32 height, 32 width.
- **y (Labels)**: $(64)$ — A vector of 64 integers (the correct class for each image).

```Python
X, y = next(iter(data.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)
```
## 5. Visualization
It is critical to visualize the data to ensure it loaded correctly. We use a helper method ``visualize`` which calls the ``d2l.show_images`` function (not implemented here, but part of the library).

```Python
# Add this method to the FashionMNIST class
def visualize(self, batch_size=18, nrows=3, ncols=6):
    """Visualize sample images from the dataset."""
    import matplotlib.pyplot as plt
    
    # Get a batch of data
    dataloader = self.get_dataloader(train=True)
    batch = next(iter(dataloader))
    images, labels = batch
    
    # Select subset for visualization
    indices = torch.randperm(len(images))[:batch_size]
    selected_images = images[indices]
    selected_labels = labels[indices]
    
    # Convert to numpy and get label names
    image_array = selected_images.squeeze().numpy()
    label_names = self.get_labels(selected_labels)
    
    # Create visualization
    show_images(image_array, nrows, ncols, titles=label_names)

def show_images(images, nrows, ncols, titles=None):
    """Display a grid of images using matplotlib."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]
    
    for i in range(min(len(images), nrows * ncols)):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].axis('off')
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
    
    # Hide unused subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example Usage:
data = FashionMNIST(batch_size=256)
data.visualize()
```

## Summary
1. **Dataset**: We use FashionMNIST because it is a drop-in replacement for MNIST but offers a more meaningful challenge
2. **Structure**: The class handles downloading, resizing, and batching.
3. **Process**: Initialize $\rightarrow$ Load Data $\rightarrow$ Create Minibatches $\rightarrow$ Visualize.
4. **Performance**: Using built-in DataLoaders ensures training is not slowed down by reading files from the disk.