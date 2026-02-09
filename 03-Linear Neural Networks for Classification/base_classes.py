# Base Classes for Linear Neural Networks for Classification
# Standard PyTorch implementations replacing d2l framework

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple, List


class HyperParameters:
    """Simple hyperparameter management."""
    
    def save_hyperparameters(self, ignore: Optional[List[str]] = None):
        """Save hyperparameters to self."""
        import inspect
        frame = inspect.currentframe().f_back
        args_info = inspect.getargvalues(frame)
        
        for arg_name, arg_value in args_info.locals.items():
            if arg_name == 'self':
                continue
            if ignore and arg_name in ignore: 
                continue
            setattr(self, arg_name, arg_value)


class Module(nn.Module, HyperParameters):
    """Base module class combining PyTorch Module with utilities."""
    
    def __init__(self):
        super().__init__()
        self.training_losses = []
        self.validation_losses = []
        self.validation_accuracies = []
        
    def plot(self, key: str, value: float, train: bool = True):
        """Store metrics for plotting.""" 
        if key == 'loss':
            if train:
                self.training_losses.append(value)
            else:
                self.validation_losses.append(value) 
        elif key == 'acc' and not train:
            self.validation_accuracies.append(value)
    
    def configure_optimizers(self):
        """Default optimizer configuration."""
        return optim.SGD(self.parameters(), lr=getattr(self, 'lr', 0.01))


class DataModule(HyperParameters):
    """Base data module for handling datasets."""
    
    def __init__(self, root: str = './data'):
        self.root = root
        
    @abstractmethod
    def get_dataloader(self, train: bool = True) -> DataLoader:
        """Return dataloader for training or validation."""
        pass


class Classifier(Module):
    """Base class for classification models."""
    
    def validation_step(self, batch):
        """Standard validation step."""
        *inputs, targets = batch
        predictions = self(*inputs)
        
        # Calculate loss and accuracy
        loss = self.loss(predictions, targets)
        acc = self.accuracy(predictions, targets)
        
        # Store for plotting
        self.plot('loss', loss.item(), train=False)
        self.plot('acc', acc.item(), train=False)
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def accuracy(self, predictions: torch.Tensor, targets: torch.Tensor, averaged: bool = True):
        """Calculate classification accuracy."""
        # Reshape predictions to ensure it's a matrix
        predictions = predictions.reshape((-1, predictions.shape[-1])) 
        
        # Get predicted classes
        predicted_classes = predictions.argmax(dim=1).type(targets.dtype)
        
        # Compare with targets
        correct = (predicted_classes == targets.reshape(-1)).type(torch.float32)
        
        return correct.mean() if averaged else correct
    
    def loss(self, predictions: torch.Tensor, targets: torch.Tensor, averaged: bool = True):
        """Cross-entropy loss function."""
        predictions = predictions.reshape((-1, predictions.shape[-1]))
        targets = targets.reshape((-1,))
        return nn.functional.cross_entropy(
            predictions, targets, reduction='mean' if averaged else 'none'
        )


class FashionMNIST(DataModule):
    """Fashion-MNIST dataset handler."""
    
    def __init__(self, batch_size: int = 64, resize: Tuple[int, int] = (28, 28)):
        super().__init__()
        self.save_hyperparameters()
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])
        
        # Load datasets
        self.train_dataset = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=transform, download=True
        )
        self.val_dataset = torchvision.datasets.FashionMNIST( 
            root=self.root, train=False, transform=transform, download=True
        )
        
        # Class names
        self.text_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    def get_dataloader(self, train: bool = True) -> DataLoader:
        """Return appropriate dataloader."""
        dataset = self.train_dataset if train else self.val_dataset
        return DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=0  # Set to 0 for Windows compatibility
        )
    
    @property
    def train(self):
        """Training dataset for compatibility.""" 
        return self.train_dataset
    
    @property  
    def val(self):
        """Validation dataset for compatibility."""
        return self.val_dataset
    
    def get_labels(self, labels: torch.Tensor) -> List[str]:
        """Convert label indices to text names.""" 
        return [self.text_labels[int(i)] for i in labels]
    
    def visualize(self, batch_size: int = 18, nrows: int = 3, ncols: int = 6):
        """Visualize sample images from the dataset."""
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


def show_images(images: np.ndarray, nrows: int, ncols: int, titles: Optional[List[str]] = None):
    """Display a grid of images using matplotlib."""
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


class Trainer:
    """Simple training loop implementation."""
    
    def __init__(self, max_epochs: int = 10, logging_interval: int = 100):
        self.max_epochs = max_epochs
        self.logging_interval = logging_interval
    
    def fit(self, model: Classifier, data: DataModule):
        """Train the model using the provided data module."""
        # Get data loaders
        train_loader = data.get_dataloader(train=True)
        val_loader = data.get_dataloader(train=False)
        
        # Get optimizer
        optimizer = model.configure_optimizers()
        
        model.train()
        
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Training loop
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward pass
                *inputs, targets = batch
                predictions = model(*inputs)
                loss = model.loss(predictions, targets) 
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                num_batches += 1
                model.plot('loss', loss.item(), train=True)
                
                # Logging
                if batch_idx % self.logging_interval == 0:
                    print(f'Epoch {epoch+1}/{self.max_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Validation
            model.eval()
            val_metrics = []
            with torch.no_grad():
                for batch in val_loader:
                    metrics = model.validation_step(batch)
                    val_metrics.append(metrics)
            
            # Calculate average validation metrics
            avg_val_loss = torch.stack([m['val_loss'] for m in val_metrics]).mean()
            avg_val_acc = torch.stack([m['val_acc'] for m in val_metrics]).mean()
            
            print(f'Epoch {epoch+1}/{self.max_epochs} - '
                  f'Train Loss: {epoch_loss/num_batches:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Val Acc: {avg_val_acc:.4f}')
            
            model.train()
        
        print("Training completed!")
        
    def plot_metrics(self, model: Classifier):
        """Plot training metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        if model.training_losses:
            ax1.plot(model.training_losses, label='Training Loss')
        if model.validation_losses:
            ax1.plot(model.validation_losses, label='Validation Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        
        # Plot accuracy
        if model.validation_accuracies:
            ax2.plot(model.validation_accuracies, label='Validation Accuracy')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Accuracy')  
            ax2.set_title('Validation Accuracy')
            ax2.legend()
        
        plt.tight_layout()
        plt.show()