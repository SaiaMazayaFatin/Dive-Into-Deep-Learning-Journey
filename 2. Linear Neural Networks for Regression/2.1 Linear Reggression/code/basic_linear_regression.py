"""
Basic Linear Regression Implementation
=======================================

This module implements linear regression from scratch using the mathematical concepts
described in Basic.md. It includes both the analytic solution and gradient descent
approaches.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class LinearRegressionAnalytic:
    """Linear Regression using the analytical (closed-form) solution."""
    
    def __init__(self):
        self.w = None
        self.b = None
        
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fit the linear regression model using the analytical solution.
        
        Formula: w* = (X^T X)^-1 X^T y
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples, 1)
        """
        # Add bias column (column of 1s) to X
        n_samples = X.shape[0]
        X_with_bias = torch.cat([X, torch.ones(n_samples, 1)], dim=1)
        
        # Analytical solution: w* = (X^T X)^-1 X^T y
        XtX = torch.mm(X_with_bias.T, X_with_bias)
        Xty = torch.mm(X_with_bias.T, y)
        
        # Solve the normal equation
        w_with_bias = torch.linalg.solve(XtX, Xty)
        
        # Separate weights and bias
        self.w = w_with_bias[:-1]  # All except last element
        self.b = w_with_bias[-1]   # Last element
        
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions using the fitted model."""
        if self.w is None or self.b is None:
            raise ValueError("Model must be fitted before making predictions")
        return torch.mm(X, self.w) + self.b
    
    def loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate the mean squared error loss."""
        y_pred = self.predict(X)
        return torch.mean((y_pred - y) ** 2)


class LinearRegressionSGD:
    """Linear Regression using Stochastic Gradient Descent."""
    
    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 1000):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.w = None
        self.b = None
        self.loss_history = []
        
    def fit(self, X: torch.Tensor, y: torch.Tensor, 
            batch_size: int = 32, verbose: bool = True) -> None:
        """
        Fit the linear regression model using minibatch SGD.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples, 1)
            batch_size: Size of minibatches for SGD
            verbose: Whether to print training progress
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.w = torch.randn(n_features, 1, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        
        # Training loop
        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = torch.randperm(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            n_batches = 0
            
            # Minibatch SGD
            for i in range(0, n_samples, batch_size):
                # Get minibatch
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred = torch.mm(X_batch, self.w) + self.b
                
                # Calculate loss
                loss = torch.mean((y_pred - y_batch) ** 2)
                epoch_loss += loss.item()
                n_batches += 1
                
                # Backward pass
                if self.w.grad is not None:
                    self.w.grad.zero_()
                if self.b.grad is not None:
                    self.b.grad.zero_()
                    
                loss.backward()
                
                # Update parameters
                with torch.no_grad():
                    self.w -= self.lr * self.w.grad
                    self.b -= self.lr * self.b.grad
            
            # Record average loss for this epoch
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {avg_loss:.6f}")
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions using the fitted model."""
        if self.w is None or self.b is None:
            raise ValueError("Model must be fitted before making predictions")
        return torch.mm(X, self.w) + self.b
    
    def loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate the mean squared error loss."""
        y_pred = self.predict(X)
        return torch.mean((y_pred - y) ** 2)
    
    def plot_loss_curve(self):
        """Plot the training loss curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training Loss Curve')
        plt.grid(True)
        plt.show()


def generate_synthetic_data(n_samples: int = 1000, n_features: int = 2, 
                          noise_std: float = 0.1, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic linear regression data.
    
    Formula: y = X * w_true + b_true + noise
    
    Args:
        n_samples: Number of data points
        n_features: Number of features
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility
        
    Returns:
        X: Feature matrix
        y: Target values
        w_true: True weights used to generate data
        b_true: True bias used to generate data
    """
    torch.manual_seed(seed)
    
    # Generate random features
    X = torch.randn(n_samples, n_features)
    
    # Generate true parameters
    w_true = torch.randn(n_features, 1)
    b_true = torch.randn(1)
    
    # Generate noise
    noise = torch.randn(n_samples, 1) * noise_std
    
    # Generate target values
    y = torch.mm(X, w_true) + b_true + noise
    
    return X, y, w_true, b_true


def compare_methods_demo():
    """Demonstrate and compare analytical vs SGD methods."""
    print("Linear Regression Implementation Comparison")
    print("=" * 50)
    
    # Generate synthetic data
    X, y, w_true, b_true = generate_synthetic_data(n_samples=1000, n_features=2)
    
    print(f"Generated data with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"True weights: {w_true.flatten()}")
    print(f"True bias: {b_true.item():.4f}")
    print()
    
    # Method 1: Analytical Solution
    print("1. Analytical Solution:")
    model_analytic = LinearRegressionAnalytic()
    model_analytic.fit(X, y)
    
    loss_analytic = model_analytic.loss(X, y)
    print(f"   Learned weights: {model_analytic.w.flatten()}")
    print(f"   Learned bias: {model_analytic.b.item():.4f}")
    print(f"   Final loss: {loss_analytic.item():.6f}")
    print()
    
    # Method 2: SGD
    print("2. Stochastic Gradient Descent:")
    model_sgd = LinearRegressionSGD(learning_rate=0.01, n_epochs=500)
    model_sgd.fit(X, y, batch_size=32, verbose=False)
    
    loss_sgd = model_sgd.loss(X, y)
    print(f"   Learned weights: {model_sgd.w.detach().flatten()}")
    print(f"   Learned bias: {model_sgd.b.detach().item():.4f}")
    print(f"   Final loss: {loss_sgd.item():.6f}")
    print()
    
    # Compare accuracy
    w_error_analytic = torch.norm(model_analytic.w - w_true).item()
    b_error_analytic = abs(model_analytic.b - b_true).item()
    
    w_error_sgd = torch.norm(model_sgd.w.detach() - w_true).item()
    b_error_sgd = abs(model_sgd.b.detach() - b_true).item()
    
    print("Parameter Estimation Errors:")
    print(f"   Analytical - Weight error: {w_error_analytic:.6f}, Bias error: {b_error_analytic:.6f}")
    print(f"   SGD        - Weight error: {w_error_sgd:.6f}, Bias error: {b_error_sgd:.6f}")
    
    # Plot SGD training curve
    model_sgd.plot_loss_curve()


if __name__ == "__main__":
    compare_methods_demo()