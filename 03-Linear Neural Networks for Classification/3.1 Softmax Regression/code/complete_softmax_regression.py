# Complete Softmax Regression Implementation
# Full implementation combining all concepts from Basic.md

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time


class SoftmaxRegressionScratch:
    """
    Softmax regression implementation from scratch using pure NumPy.
    Demonstrates all mathematical concepts from Basic.md.
    """
    
    def __init__(self, num_inputs: int, num_outputs: int, learning_rate: float = 0.1):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.lr = learning_rate
        
        # Initialize parameters as described in Basic.md
        # Weights: Gaussian noise with small std
        self.W = np.random.normal(0, 0.01, size=(num_inputs, num_outputs))
        # Bias: Initialize to zeros
        self.b = np.zeros(num_outputs)
        
        # Track training history
        self.train_losses = []
        self.train_accuracies = []
        
    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """
        Numerically stable softmax implementation.
        Formula: ŷᵢ = exp(oᵢ) / Σⱼ exp(oⱼ)
        """
        # Subtract max for numerical stability
        logits_max = np.max(logits, axis=-1, keepdims=True)
        logits_shifted = logits - logits_max
        exp_logits = np.exp(logits_shifted)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the model.
        
        Args:
            X: Input batch (batch_size, num_inputs)
        Returns:
            logits: Raw outputs (batch_size, num_outputs)
            probabilities: Softmax outputs (batch_size, num_outputs)
        """
        # Linear transformation: O = XW + b
        logits = X @ self.W + self.b
        
        # Apply softmax
        probabilities = self.softmax(logits)
        
        return logits, probabilities
    
    def cross_entropy_loss(self, probabilities: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        Formula: L = -Σ yⱼ log(ŷⱼ)
        """
        # Clip probabilities to prevent log(0)
        epsilon = 1e-15
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        
        # Cross-entropy for the batch
        batch_size = probabilities.shape[0]
        loss = -np.sum(targets * np.log(probabilities)) / batch_size
        return loss
    
    def compute_gradients(self, X: np.ndarray, probabilities: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients using the derived formula.
        Key insight from Basic.md: ∂L/∂oⱼ = ŷⱼ - yⱼ
        """
        batch_size = X.shape[0]
        
        # Gradient w.r.t. logits: ŷ - y
        grad_logits = probabilities - targets
        
        # Gradient w.r.t. weights: X^T (ŷ - y)
        grad_W = X.T @ grad_logits / batch_size
        
        # Gradient w.r.t. bias: mean(ŷ - y)
        grad_b = np.mean(grad_logits, axis=0)
        
        return grad_W, grad_b
    
    def accuracy(self, probabilities: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute classification accuracy.
        """
        predictions = np.argmax(probabilities, axis=1)
        true_labels = np.argmax(targets, axis=1)
        return np.mean(predictions == true_labels)
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Single training step.
        """
        # Forward pass
        logits, probabilities = self.forward(X)
        
        # Compute loss
        loss = self.cross_entropy_loss(probabilities, y)
        
        # Compute accuracy
        acc = self.accuracy(probabilities, y)
        
        # Compute gradients
        grad_W, grad_b = self.compute_gradients(X, probabilities, y)
        
        # Update parameters
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
        
        return {"loss": loss, "accuracy": acc}
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, verbose: bool = True):
        """
        Train the model.
        """
        print("🚀 Training Softmax Regression from Scratch")
        print("-" * 50)
        
        for epoch in range(epochs):
            # Training step
            metrics = self.train_step(X, y)
            
            # Store metrics
            self.train_losses.append(metrics["loss"])
            self.train_accuracies.append(metrics["accuracy"])  
            
            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Loss = {metrics['loss']:.4f}, "
                      f"Accuracy = {metrics['accuracy']:.4f}")
        
        if verbose:
            print(f"\n✅ Training completed!")
            print(f"Final Loss: {self.train_losses[-1]:.4f}")
            print(f"Final Accuracy: {self.train_accuracies[-1]:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        """
        _, probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        """
        _, probabilities = self.forward(X)
        return probabilities


class SoftmaxRegressionPyTorch(nn.Module):
    """
    PyTorch implementation for comparison.
    """
    
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        
        # Initialize with same distribution as scratch version
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        return self.linear(x)


class SoftmaxRegressionDemo:
    """
    Complete demonstration of softmax regression concepts.
    """
    
    def __init__(self):
        self.results = {}
    
    def generate_synthetic_data(self, n_samples: int = 1000, n_features: int = 4, n_classes: int = 3, noise: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic classification data.
        """
        np.random.seed(seed)
        
        # Create class centers
        class_centers = np.random.randn(n_classes, n_features) * 2
        
        # Generate samples
        X = []
        y = []
        
        samples_per_class = n_samples // n_classes
        
        for class_idx in range(n_classes):
            # Generate samples around class center
            class_samples = np.random.randn(samples_per_class, n_features) * noise + class_centers[class_idx]
            class_labels = np.zeros((samples_per_class, n_classes))
            class_labels[:, class_idx] = 1  # One-hot encoding
            
            X.append(class_samples)
            y.append(class_labels)
        
        X = np.vstack(X)
        y = np.vstack(y)
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def comparison_demo(self):
        """
        Compare scratch implementation with PyTorch.
        """
        print("="*60)
        print("SCRATCH vs PYTORCH IMPLEMENTATION COMPARISON")
        print("="*60)
        
        # Generate data
        X, y = self.generate_synthetic_data(n_samples=1000, n_features=4, n_classes=3)
        
        print(f"📊 Dataset Information:")
        print(f"Samples: {X.shape[0]}")
        print(f"Features: {X.shape[1]}")
        print(f"Classes: {y.shape[1]}")
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 1. Train scratch implementation
        print(f"\n🔨 TRAINING FROM SCRATCH:")
        scratch_model = SoftmaxRegressionScratch(X.shape[1], y.shape[1], learning_rate=0.1)
        
        start_time = time.time()
        scratch_model.fit(X_train, y_train, epochs=100, verbose=False)
        scratch_time = time.time() - start_time
        
        # Test scratch model
        scratch_train_acc = scratch_model.accuracy(scratch_model.predict_proba(X_train), y_train)
        scratch_test_acc = scratch_model.accuracy(scratch_model.predict_proba(X_test), y_test)
        
        print(f"Training time: {scratch_time:.3f} seconds")
        print(f"Train accuracy: {scratch_train_acc:.4f}")
        print(f"Test accuracy: {scratch_test_acc:.4f}")
        
        # 2. Train PyTorch implementation
        print(f"\n🚀 TRAINING WITH PYTORCH:")
        
        # Convert to PyTorch tensors
        X_train_torch = torch.FloatTensor(X_train)
        y_train_torch = torch.LongTensor(np.argmax(y_train, axis=1))
        X_test_torch = torch.FloatTensor(X_test)
        y_test_torch = torch.LongTensor(np.argmax(y_test, axis=1))
        
        # Create model
        pytorch_model = SoftmaxRegressionPyTorch(X.shape[1], y.shape[1])
        optimizer = optim.SGD(pytorch_model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        start_time = time.time()
        pytorch_train_losses = []
        pytorch_train_accs = []
        
        pytorch_model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            
            # Forward pass
            logits = pytorch_model(X_train_torch)
            loss = criterion(logits, y_train_torch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            with torch.no_grad():
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                accuracy = (predictions == y_train_torch).float().mean()
                
                pytorch_train_losses.append(loss.item())
                pytorch_train_accs.append(accuracy.item())
        
        pytorch_time = time.time() - start_time
        
        # Test PyTorch model
        pytorch_model.eval()
        with torch.no_grad():
            train_logits = pytorch_model(X_train_torch)
            train_predictions = torch.argmax(train_logits, dim=1)
            pytorch_train_acc = (train_predictions == y_train_torch).float().mean().item()
            
            test_logits = pytorch_model(X_test_torch)
            test_predictions = torch.argmax(test_logits, dim=1)
            pytorch_test_acc = (test_predictions == y_test_torch).float().mean().item()
        
        print(f"Training time: {pytorch_time:.3f} seconds")
        print(f"Train accuracy: {pytorch_train_acc:.4f}")
        print(f"Test accuracy: {pytorch_test_acc:.4f}")
        
        # Comparison
        print(f"\n📊 COMPARISON:")
        print(f"Speed difference: {scratch_time/pytorch_time:.1f}x (scratch/pytorch)")
        print(f"Train accuracy difference: {abs(scratch_train_acc - pytorch_train_acc):.4f}")
        print(f"Test accuracy difference: {abs(scratch_test_acc - pytorch_test_acc):.4f}")
        
        return scratch_model, pytorch_model, (X_train, y_train), (X_test, y_test)
    
    def architecture_visualization(self):
        """
        Visualize the softmax regression architecture.
        """
        print("\n" + "="*50)
        print("SOFTMAX REGRESSION ARCHITECTURE")
        print("="*50)
        
        print("🏗️  ARCHITECTURE OVERVIEW:")
        print("Input → Linear Layer → Softmax → Probabilities")
        print()
        print("Mathematical Flow:")
        print("1. Input: x ∈ ℝᵈ")
        print("2. Linear: o = Wx + b")  
        print("3. Softmax: ŷ = softmax(o)")
        print("4. Loss: L = -Σ y log(ŷ)")
        print("5. Gradient: ∇L = ŷ - y")
        
        # Demonstrate with actual numbers
        print(f"\n🧮 NUMERICAL EXAMPLE:")
        
        # Example dimensions
        batch_size = 2
        n_features = 3
        n_classes = 4
        
        # Generate example data
        np.random.seed(42)
        X = np.random.randn(batch_size, n_features)
        W = np.random.randn(n_features, n_classes) * 0.1
        b = np.zeros(n_classes)
        y = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])  # One-hot labels
        
        print(f"Dimensions:")
        print(f"  X (input): {X.shape}")
        print(f"  W (weights): {W.shape}")
        print(f"  b (bias): {b.shape}")
        print(f"  y (labels): {y.shape}")
        
        print(f"\nStep 1 - Input:")
        print(f"X =\n{X}")
        
        print(f"\nStep 2 - Linear transformation:")
        O = X @ W + b
        print(f"O = X @ W + b =\n{O}")
        
        print(f"\nStep 3 - Softmax:")
        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        probs = softmax(O)
        print(f"ŷ = softmax(O) =\n{probs}")
        print(f"Row sums: {probs.sum(axis=1)}")
        
        print(f"\nStep 4 - Loss:")
        epsilon = 1e-15
        probs_clipped = np.clip(probs, epsilon, 1-epsilon)
        loss = -np.mean(np.sum(y * np.log(probs_clipped), axis=1))
        print(f"L = -mean(Σ y log(ŷ)) = {loss:.4f}")
        
        print(f"\nStep 5 - Gradients:")
        grad_O = probs - y
        print(f"∂L/∂O = ŷ - y =\n{grad_O}")
        
        grad_W = X.T @ grad_O / batch_size
        grad_b = np.mean(grad_O, axis=0)
        print(f"∂L/∂W shape: {grad_W.shape}")
        print(f"∂L/∂b shape: {grad_b.shape}")
        
    def visualization_demo(self):
        """
        Create comprehensive visualizations.
        """
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        # Generate comparison data
        scratch_model, pytorch_model, (X_train, y_train), (X_test, y_test) = self.comparison_demo()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Training curves comparison
        ax1 = axes[0, 0]
        
        epochs = range(1, len(scratch_model.train_losses) + 1)
        ax1.plot(epochs, scratch_model.train_losses, 'b-', linewidth=2, label='From Scratch')
        
        # Add PyTorch training curve (if available)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cross-Entropy Loss')
        ax1.set_title('Training Loss Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy curves
        ax2 = axes[0, 1]
        
        ax2.plot(epochs, scratch_model.train_accuracies, 'r-', linewidth=2, label='From Scratch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Decision boundaries (2D projection)
        ax3 = axes[0, 2]
        
        if X_train.shape[1] >= 2:
            # Use first 2 features for visualization
            X_2d = X_train[:, :2]
            y_classes = np.argmax(y_train, axis=1)
            
            # Create scatter plot
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for class_idx in range(y_train.shape[1]):
                mask = y_classes == class_idx
                ax3.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                           c=colors[class_idx % len(colors)], 
                           label=f'Class {class_idx}', alpha=0.7)
            
            ax3.set_xlabel('Feature 1')
            ax3.set_ylabel('Feature 2')
            ax3.set_title('Data Distribution (2D Projection)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Weight visualization
        ax4 = axes[1, 0]
        
        im = ax4.imshow(scratch_model.W, cmap='RdBu', aspect='auto')
        ax4.set_xlabel('Output Class')
        ax4.set_ylabel('Input Feature') 
        ax4.set_title('Learned Weights Matrix')
        plt.colorbar(im, ax=ax4)
        
        # 5. Prediction confidence  
        ax5 = axes[1, 1]
        
        # Get prediction probabilities for test set
        test_probs = scratch_model.predict_proba(X_test)
        max_probs = np.max(test_probs, axis=1)
        
        ax5.hist(max_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.set_xlabel('Maximum Prediction Probability')
        ax5.set_ylabel('Number of Samples')
        ax5.set_title('Prediction Confidence Distribution')
        ax5.grid(True, alpha=0.3)
        
        # Add confidence statistics
        mean_conf = np.mean(max_probs)
        ax5.axvline(mean_conf, color='red', linestyle='--', 
                   label=f'Mean: {mean_conf:.3f}')
        ax5.legend()
        
        # 6. Confusion matrix
        ax6 = axes[1, 2]
        
        y_test_classes = np.argmax(y_test, axis=1)
        predictions = scratch_model.predict(X_test)
        
        # Create confusion matrix
        n_classes = y_train.shape[1]
        confusion_matrix = np.zeros((n_classes, n_classes))
        
        for true_class, pred_class in zip(y_test_classes, predictions):
            confusion_matrix[true_class, pred_class] += 1
        
        # Normalize
        confusion_matrix = confusion_matrix / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-10)
        
        im = ax6.imshow(confusion_matrix, cmap='Blues', aspect='auto')
        ax6.set_xlabel('Predicted Class')
        ax6.set_ylabel('True Class')
        ax6.set_title('Confusion Matrix (Normalized)')
        
        # Add text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                text = ax6.text(j, i, f'{confusion_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im, ax=ax6)
        
        plt.tight_layout()
        plt.savefig('c:/File/AI/Dive Into Deep Learning Journey/03-Linear Neural Networks for Classification/3.1 Softmax Regression/img/complete_softmax_demo.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Complete softmax regression visualizations created!")
        print("📁 Saved as: complete_softmax_demo.png")


def main():
    """Run complete softmax regression demonstration."""
    print("🎯 COMPLETE SOFTMAX REGRESSION IMPLEMENTATION")
    print("="*60)
    
    demo = SoftmaxRegressionDemo()
    
    # Architecture explanation
    demo.architecture_visualization()
    
    # Implementation comparison and visualization
    demo.visualization_demo()
    
    print("\n🎉 COMPLETE SOFTMAX REGRESSION DEMO FINISHED!")
    print("="*60)
    print("📚 Concepts Implemented:")
    print("  • Complete from-scratch implementation")
    print("  • PyTorch comparison")
    print("  • Synthetic data generation") 
    print("  • Training loop with gradient descent")
    print("  • Numerical stability techniques")
    print("  • Performance evaluation")
    print("  • Comprehensive visualizations")
    print("\n✅ All concepts from Basic.md successfully implemented!")
    print("🚀 Ready for real-world applications!")


if __name__ == "__main__":
    main()