"""
Basic Linear Regression Implementation
=====================================

This module implements linear regression from scratch, demonstrating:
1. Analytic solution (closed-form)
2. Minibatch Stochastic Gradient Descent (SGD)
3. Loss calculation
4. Predictions (inference)

Based on the materials in 1.Basic.md
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class LinearRegression:
    """
    Linear Regression implementation with both analytic and SGD solutions.
    
    The model implements: y = w^T * x + b
    """
    
    def __init__(self):
        self.w = None  # Weights
        self.b = None  # Bias
        self.trained = False
        
    def _add_bias_column(self, X: torch.Tensor) -> torch.Tensor:
        """Add a column of ones to X for bias term."""
        ones = torch.ones(X.shape[0], 1)
        return torch.cat([X, ones], dim=1)
    
    def analytic_solution(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Solve linear regression using the analytic (closed-form) solution.
        
        Formula: w* = (X^T X)^(-1) X^T y
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
        """
        # Add bias column to X
        X_with_bias = self._add_bias_column(X)
        
        # Compute the analytic solution: w* = (X^T X)^(-1) X^T y
        XtX = X_with_bias.T @ X_with_bias
        Xty = X_with_bias.T @ y
        
        # Check if XtX is invertible
        try:
            w_with_bias = torch.inverse(XtX) @ Xty
        except RuntimeError:
            # Use pseudo-inverse if matrix is not invertible
            w_with_bias = torch.pinverse(XtX) @ Xty
        
        # Split weights and bias
        self.w = w_with_bias[:-1]
        self.b = w_with_bias[-1]
        self.trained = True
        
    def sgd_solution(self, X: torch.Tensor, y: torch.Tensor, 
                     learning_rate: float = 0.01, batch_size: int = 32, 
                     epochs: int = 100, verbose: bool = False) -> list:
        """
        Solve linear regression using Minibatch Stochastic Gradient Descent.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
            learning_rate: Learning rate (eta)
            batch_size: Size of minibatches
            epochs: Number of training epochs
            verbose: Whether to print progress
            
        Returns:
            List of losses for each epoch
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.w = torch.randn(n_features, requires_grad=False) * 0.01
        self.b = torch.zeros(1, requires_grad=False)
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Shuffle the data
            indices = torch.randperm(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process minibatches
            for i in range(0, n_samples, batch_size):
                # Get minibatch
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                # Forward pass: compute predictions
                y_pred = X_batch @ self.w + self.b
                
                # Compute loss (mean squared error)
                loss = torch.mean(0.5 * (y_pred - y_batch) ** 2)
                epoch_loss += loss.item()
                
                # Compute gradients
                error = y_pred - y_batch
                grad_w = torch.mean(X_batch * error.unsqueeze(1), dim=0)
                grad_b = torch.mean(error)
                
                # Update parameters
                self.w -= learning_rate * grad_w
                self.b -= learning_rate * grad_b
            
            # Average loss for the epoch
            avg_loss = epoch_loss / (n_samples // batch_size + 1)
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}')
        
        self.trained = True
        return losses
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        return X @ self.w + self.b
    
    def loss(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute the mean squared error loss.
        
        Args:
            X: Feature matrix
            y: True targets
            
        Returns:
            Mean squared error
        """
        if not self.trained:
            raise ValueError("Model must be trained before computing loss")
        
        y_pred = self.predict(X)
        return torch.mean(0.5 * (y_pred - y) ** 2).item()


def generate_synthetic_data(n_samples: int = 100, n_features: int = 2, 
                          noise_std: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data for linear regression.
    
    Args:
        n_samples: Number of data points
        n_features: Number of features
        noise_std: Standard deviation of noise
        
    Returns:
        Tuple of (features, targets)
    """
    # Generate random features
    X = torch.randn(n_samples, n_features)
    
    # True weights and bias
    true_w = torch.randn(n_features)
    true_b = torch.randn(1)
    
    # Generate targets with noise
    y = X @ true_w + true_b + torch.randn(n_samples) * noise_std
    
    return X, y


def demo_comparison():
    """
    Demonstrate both analytic and SGD solutions and compare their results.
    """
    print("=" * 60)
    print("Linear Regression: Analytic vs SGD Comparison")
    print("=" * 60)
    
    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=200, n_features=3, noise_std=0.1)
    
    # Split into train and test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Analytic solution
    print("\n1. Analytic Solution")
    print("-" * 30)
    model_analytic = LinearRegression()
    model_analytic.analytic_solution(X_train, y_train)
    
    analytic_train_loss = model_analytic.loss(X_train, y_train)
    analytic_test_loss = model_analytic.loss(X_test, y_test)
    
    print(f"Weights: {model_analytic.w}")
    print(f"Bias: {model_analytic.b}")
    print(f"Train Loss: {analytic_train_loss:.6f}")
    print(f"Test Loss: {analytic_test_loss:.6f}")
    
    # SGD solution
    print("\n2. SGD Solution")
    print("-" * 30)
    model_sgd = LinearRegression()
    losses = model_sgd.sgd_solution(X_train, y_train, 
                                   learning_rate=0.1, 
                                   batch_size=16, 
                                   epochs=100, 
                                   verbose=True)
    
    sgd_train_loss = model_sgd.loss(X_train, y_train)
    sgd_test_loss = model_sgd.loss(X_test, y_test)
    
    print(f"\nWeights: {model_sgd.w}")
    print(f"Bias: {model_sgd.b}")
    print(f"Train Loss: {sgd_train_loss:.6f}")
    print(f"Test Loss: {sgd_test_loss:.6f}")
    
    # Compare results
    print("\n3. Comparison")
    print("-" * 30)
    weight_diff = torch.norm(model_analytic.w - model_sgd.w)
    bias_diff = torch.abs(model_analytic.b - model_sgd.b)
    
    print(f"Weight difference (L2 norm): {weight_diff:.6f}")
    print(f"Bias difference: {bias_diff:.6f}")
    print(f"Train loss difference: {abs(analytic_train_loss - sgd_train_loss):.6f}")
    print(f"Test loss difference: {abs(analytic_test_loss - sgd_test_loss):.6f}")
    
    # Plot loss curve for SGD
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('SGD Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run the demonstration
    demo_comparison()