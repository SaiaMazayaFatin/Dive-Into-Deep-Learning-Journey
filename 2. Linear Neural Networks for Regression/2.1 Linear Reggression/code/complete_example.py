"""
Complete Linear Regression Example
=================================

This comprehensive example brings together all concepts from the linear regression
materials to create a complete, practical implementation. It demonstrates:

1. Data generation and preprocessing
2. Both analytic and iterative solutions
3. Vectorized operations for efficiency
4. Connection to normal distribution and MLE
5. Neural network perspective
6. Model evaluation and visualization

This serves as a complete reference implementation combining all the concepts.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import time
import pandas as pd
from pathlib import Path


class ComprehensiveLinearRegression:
    """
    A complete linear regression implementation showcasing all concepts
    from the study materials.
    """
    
    def __init__(self, method: str = 'sgd'):
        """
        Initialize the linear regression model.
        
        Args:
            method: 'analytic' for closed-form solution or 'sgd' for gradient descent
        """
        self.method = method
        self.w = None
        self.b = None
        self.training_history = {}
        self.fitted = False
        
    def _add_bias_column(self, X: torch.Tensor) -> torch.Tensor:
        """Add bias column for analytic solution."""
        return torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, **kwargs) -> Dict:
        """
        Fit the model to training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            **kwargs: Additional parameters for SGD
            
        Returns:
            Training history dictionary
        """
        if self.method == 'analytic':
            return self._fit_analytic(X, y)
        else:
            return self._fit_sgd(X, y, **kwargs)
    
    def _fit_analytic(self, X: torch.Tensor, y: torch.Tensor) -> Dict:
        """Fit using the analytic (closed-form) solution."""
        start_time = time.time()
        
        # Add bias column and solve: w = (X^T X)^{-1} X^T y
        X_with_bias = self._add_bias_column(X)
        
        try:
            # Use Cholesky decomposition for numerical stability
            XtX = X_with_bias.T @ X_with_bias
            Xty = X_with_bias.T @ y
            
            # Check condition number for numerical stability
            cond_num = torch.linalg.cond(XtX)
            if cond_num > 1e12:
                print(f"Warning: Matrix is ill-conditioned (cond={cond_num:.2e})")
                w_with_bias = torch.linalg.pinv(XtX) @ Xty
            else:
                w_with_bias = torch.linalg.solve(XtX, Xty)
                
        except Exception as e:
            print(f"Using pseudo-inverse due to: {e}")
            w_with_bias = torch.linalg.pinv(X_with_bias) @ y
        
        self.w = w_with_bias[:-1]
        self.b = w_with_bias[-1]
        self.fitted = True
        
        training_time = time.time() - start_time
        final_loss = self.compute_loss(X, y)
        
        self.training_history = {
            'method': 'analytic',
            'training_time': training_time,
            'final_loss': final_loss,
            'condition_number': cond_num.item() if 'cond_num' in locals() else None
        }
        
        return self.training_history
    
    def _fit_sgd(self, X: torch.Tensor, y: torch.Tensor,
                 learning_rate: float = 0.01,
                 batch_size: int = 32,
                 max_epochs: int = 1000,
                 tolerance: float = 1e-6,
                 verbose: bool = False) -> Dict:
        """Fit using Stochastic Gradient Descent."""
        start_time = time.time()
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.w = torch.randn(n_features) * 0.01
        self.b = torch.zeros(1)
        
        # Training history
        losses = []
        epochs_run = 0
        
        # Adaptive learning rate
        initial_lr = learning_rate
        lr = learning_rate
        
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Shuffle data
            indices = torch.randperm(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process mini-batches
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                # Forward pass (vectorized)
                predictions = X_batch @ self.w + self.b
                
                # Compute loss and gradients (vectorized)
                errors = predictions - y_batch
                loss = torch.mean(0.5 * errors ** 2)
                epoch_loss += loss.item()
                n_batches += 1
                
                # Compute gradients (vectorized)
                grad_w = torch.mean(X_batch * errors.unsqueeze(1), dim=0)
                grad_b = torch.mean(errors)
                
                # Update parameters
                self.w -= lr * grad_w
                self.b -= lr * grad_b
            
            # Average loss for epoch
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            epochs_run = epoch + 1
            
            # Learning rate decay
            if epoch > 100 and epoch % 100 == 0:
                lr *= 0.95
            
            # Early stopping
            if len(losses) > 10 and abs(losses[-1] - losses[-10]) < tolerance:
                if verbose:
                    print(f"Converged at epoch {epoch + 1}")
                break
            
            # Verbose output
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1:4d}: Loss = {avg_loss:.8f}, LR = {lr:.6f}")
        
        self.fitted = True
        training_time = time.time() - start_time
        
        self.training_history = {
            'method': 'sgd',
            'training_time': training_time,
            'losses': losses,
            'epochs': epochs_run,
            'final_loss': losses[-1],
            'initial_lr': initial_lr,
            'final_lr': lr,
            'batch_size': batch_size
        }
        
        return self.training_history
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions on new data."""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        return X @ self.w + self.b
    
    def compute_loss(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Compute mean squared error loss."""
        if not self.fitted:
            raise ValueError("Model must be fitted before computing loss")
        predictions = self.predict(X)
        return torch.mean(0.5 * (predictions - y) ** 2).item()
    
    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Compute R-squared score."""
        if not self.fitted:
            raise ValueError("Model must be fitted before scoring")
        
        predictions = self.predict(X)
        ss_res = torch.sum((y - predictions) ** 2)
        ss_tot = torch.sum((y - torch.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2.item()


def generate_regression_data(n_samples: int = 200, 
                           n_features: int = 5,
                           noise_std: float = 0.1,
                           random_state: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic regression data with known ground truth.
    
    Args:
        n_samples: Number of data points
        n_features: Number of features
        noise_std: Standard deviation of Gaussian noise
        random_state: Random seed
        
    Returns:
        Tuple of (X, y, true_weights, true_bias)
    """
    torch.manual_seed(random_state)
    
    # Generate features from various distributions
    X = torch.randn(n_samples, n_features)
    
    # Create interesting feature relationships
    if n_features > 1:
        X[:, 1] = X[:, 0] * 0.5 + torch.randn(n_samples) * 0.3  # Correlated feature
    if n_features > 2:
        X[:, 2] = torch.uniform(-2, 2, (n_samples,))  # Uniform feature
    
    # True parameters with some structure
    true_w = torch.tensor([2.0, -1.5, 0.8, 0.0, -0.3][:n_features])
    true_b = torch.tensor([1.0])
    
    # Generate targets with Gaussian noise (justifying MSE loss)
    y_clean = X @ true_w + true_b
    noise = torch.normal(0, noise_std, (n_samples,))
    y = y_clean + noise
    
    return X, y, true_w, true_b


def create_comprehensive_evaluation(model: ComprehensiveLinearRegression,
                                  X_train: torch.Tensor, y_train: torch.Tensor,
                                  X_test: torch.Tensor, y_test: torch.Tensor,
                                  true_w: torch.Tensor, true_b: torch.Tensor) -> Dict:
    """
    Create a comprehensive evaluation of the model.
    """
    evaluation = {
        'training': {
            'mse': model.compute_loss(X_train, y_train),
            'r2': model.score(X_train, y_train)
        },
        'testing': {
            'mse': model.compute_loss(X_test, y_test),
            'r2': model.score(X_test, y_test)
        },
        'parameter_recovery': {
            'weight_error': torch.norm(model.w - true_w).item(),
            'bias_error': torch.abs(model.b - true_b).item(),
            'learned_weights': model.w.numpy(),
            'true_weights': true_w.numpy(),
            'learned_bias': model.b.item(),
            'true_bias': true_b.item()
        },
        'training_info': model.training_history
    }
    
    return evaluation


def visualize_results(evaluations: Dict[str, Dict], 
                     X_test: torch.Tensor, y_test: torch.Tensor,
                     models: Dict[str, ComprehensiveLinearRegression]):
    """
    Create comprehensive visualizations of the results.
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Training curves comparison (if SGD was used)
    plt.subplot(2, 4, 1)
    for name, eval_dict in evaluations.items():
        if 'losses' in eval_dict['training_info']:
            losses = eval_dict['training_info']['losses']
            plt.semilogy(losses, label=name, linewidth=2)
    
    plt.title('Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Parameter recovery comparison
    plt.subplot(2, 4, 2)
    methods = list(evaluations.keys())
    weight_errors = [evaluations[m]['parameter_recovery']['weight_error'] for m in methods]
    bias_errors = [evaluations[m]['parameter_recovery']['bias_error'] for m in methods]
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    plt.bar(x_pos - width/2, weight_errors, width, label='Weight Error', alpha=0.8)
    plt.bar(x_pos + width/2, bias_errors, width, label='Bias Error', alpha=0.8)
    
    plt.title('Parameter Recovery Error')
    plt.xlabel('Method')
    plt.ylabel('Error')
    plt.xticks(x_pos, methods)
    plt.legend()
    plt.yscale('log')
    
    # 3. Performance comparison
    plt.subplot(2, 4, 3)
    train_mse = [evaluations[m]['training']['mse'] for m in methods]
    test_mse = [evaluations[m]['testing']['mse'] for m in methods]
    
    plt.bar(x_pos - width/2, train_mse, width, label='Train MSE', alpha=0.8)
    plt.bar(x_pos + width/2, test_mse, width, label='Test MSE', alpha=0.8)
    
    plt.title('MSE Comparison')
    plt.xlabel('Method')
    plt.ylabel('Mean Squared Error')
    plt.xticks(x_pos, methods)
    plt.legend()
    
    # 4. R² scores
    plt.subplot(2, 4, 4)
    train_r2 = [evaluations[m]['training']['r2'] for m in methods]
    test_r2 = [evaluations[m]['testing']['r2'] for m in methods]
    
    plt.bar(x_pos - width/2, train_r2, width, label='Train R²', alpha=0.8)
    plt.bar(x_pos + width/2, test_r2, width, label='Test R²', alpha=0.8)
    
    plt.title('R² Scores')
    plt.xlabel('Method')
    plt.ylabel('R² Score')
    plt.xticks(x_pos, methods)
    plt.legend()
    plt.ylim(0, 1.1)
    
    # 5. Predictions vs Actual (using first model)
    plt.subplot(2, 4, 5)
    first_model = list(models.values())[0]
    predictions = first_model.predict(X_test)
    
    plt.scatter(y_test.numpy(), predictions.numpy(), alpha=0.6)
    
    # Perfect prediction line
    min_val = min(torch.min(y_test), torch.min(predictions))
    max_val = max(torch.max(y_test), torch.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.title('Predictions vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Residuals plot
    plt.subplot(2, 4, 6)
    residuals = predictions - y_test
    plt.scatter(predictions.numpy(), residuals.numpy(), alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)
    
    # 7. Weight comparison
    plt.subplot(2, 4, 7)
    first_eval = list(evaluations.values())[0]
    true_weights = first_eval['parameter_recovery']['true_weights']
    
    n_features = len(true_weights)
    feature_names = [f'w_{i+1}' for i in range(n_features)]
    
    x_pos = np.arange(n_features)
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        learned_weights = evaluations[method]['parameter_recovery']['learned_weights']
        offset = (i - len(methods)/2 + 0.5) * width
        plt.bar(x_pos + offset, learned_weights, width, 
               label=f'{method} (learned)', alpha=0.7)
    
    plt.bar(x_pos, true_weights, width, label='True', alpha=0.9, color='black', linestyle='--')
    
    plt.title('Weight Comparison')
    plt.xlabel('Feature')
    plt.ylabel('Weight Value')
    plt.xticks(x_pos, feature_names)
    plt.legend()
    
    # 8. Training time comparison
    plt.subplot(2, 4, 8)
    train_times = [evaluations[m]['training_info']['training_time'] for m in methods]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    bars = plt.bar(methods, train_times, color=colors, alpha=0.8)
    
    plt.title('Training Time Comparison')
    plt.xlabel('Method')
    plt.ylabel('Time (seconds)')
    plt.yscale('log')
    
    # Add values on top of bars
    for bar, time_val in zip(bars, train_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.4f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def demonstrate_normal_distribution_connection(X: torch.Tensor, y: torch.Tensor, 
                                            model: ComprehensiveLinearRegression):
    """
    Demonstrate the connection to normal distribution and MLE.
    """
    print("\n" + "=" * 70)
    print("NORMAL DISTRIBUTION CONNECTION")
    print("=" * 70)
    
    # Calculate residuals
    predictions = model.predict(X)
    residuals = y - predictions
    
    # Test normality of residuals
    print("Testing the assumption that residuals follow a normal distribution:")
    print(f"Residual mean: {torch.mean(residuals):.6f}")
    print(f"Residual std:  {torch.std(residuals):.6f}")
    
    # Visual test for normality
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram with normal overlay
    ax1.hist(residuals.numpy(), bins=30, density=True, alpha=0.7, color='skyblue')
    
    # Overlay theoretical normal distribution
    x_range = torch.linspace(torch.min(residuals), torch.max(residuals), 100)
    normal_curve = (1 / (torch.sqrt(2 * torch.pi) * torch.std(residuals))) * \
                   torch.exp(-0.5 * ((x_range - torch.mean(residuals)) / torch.std(residuals)) ** 2)
    
    ax1.plot(x_range.numpy(), normal_curve.numpy(), 'r-', linewidth=2, label='Theoretical Normal')
    ax1.set_title('Residual Distribution')
    ax1.set_xlabel('Residual Value')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals.numpy(), dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normal Distribution)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate likelihood
    sigma_mle = torch.std(residuals)
    log_likelihood = -0.5 * len(residuals) * torch.log(2 * torch.pi * sigma_mle**2) - \
                     torch.sum(residuals**2) / (2 * sigma_mle**2)
    
    print(f"\nMaximum Likelihood Estimation:")
    print(f"Estimated noise σ: {sigma_mle:.6f}")
    print(f"Log-likelihood: {log_likelihood:.3f}")
    print(f"MSE loss: {model.compute_loss(X, y):.6f}")
    
    print(f"\nThis demonstrates that minimizing MSE ≡ maximizing likelihood")
    print(f"under the assumption of Gaussian noise!")


def main():
    """
    Main comprehensive demonstration.
    """
    print("=" * 80)
    print("COMPREHENSIVE LINEAR REGRESSION DEMONSTRATION")
    print("=" * 80)
    print("This example combines all concepts from the study materials:")
    print("• Basic linear regression (analytic vs SGD)")
    print("• Vectorization for speed")
    print("• Normal distribution connection")
    print("• Neural network perspective")
    print("• Comprehensive evaluation\n")
    
    # 1. Generate comprehensive dataset
    print("1. Generating Dataset")
    print("-" * 50)
    
    X, y, true_w, true_b = generate_regression_data(
        n_samples=1000, n_features=5, noise_std=0.2, random_state=42
    )
    
    # Split into train/test
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Training: {len(X_train)} samples")
    print(f"Testing: {len(X_test)} samples")
    print(f"True weights: {true_w.numpy()}")
    print(f"True bias: {true_b.item():.3f}")
    print(f"Noise std: 0.2")
    
    # 2. Train models with different methods
    print(f"\n2. Training Models")
    print("-" * 50)
    
    models = {}
    evaluations = {}
    
    # Analytic solution
    print("Training with analytic solution...")
    model_analytic = ComprehensiveLinearRegression(method='analytic')
    model_analytic.fit(X_train, y_train)
    models['Analytic'] = model_analytic
    evaluations['Analytic'] = create_comprehensive_evaluation(
        model_analytic, X_train, y_train, X_test, y_test, true_w, true_b
    )
    
    # SGD with different configurations
    print("Training with SGD (small batches)...")
    model_sgd_small = ComprehensiveLinearRegression(method='sgd')
    model_sgd_small.fit(X_train, y_train, learning_rate=0.01, batch_size=16, 
                       max_epochs=500, verbose=False)
    models['SGD (small batch)'] = model_sgd_small
    evaluations['SGD (small batch)'] = create_comprehensive_evaluation(
        model_sgd_small, X_train, y_train, X_test, y_test, true_w, true_b
    )
    
    print("Training with SGD (large batches)...")
    model_sgd_large = ComprehensiveLinearRegression(method='sgd')
    model_sgd_large.fit(X_train, y_train, learning_rate=0.05, batch_size=128, 
                       max_epochs=200, verbose=False)
    models['SGD (large batch)'] = model_sgd_large
    evaluations['SGD (large batch)'] = create_comprehensive_evaluation(
        model_sgd_large, X_train, y_train, X_test, y_test, true_w, true_b
    )
    
    # 3. Print comparison results
    print(f"\n3. Results Comparison")
    print("-" * 50)
    
    results_df = []
    for name, eval_dict in evaluations.items():
        results_df.append({
            'Method': name,
            'Train MSE': eval_dict['training']['mse'],
            'Test MSE': eval_dict['testing']['mse'],
            'Train R²': eval_dict['training']['r2'],
            'Test R²': eval_dict['testing']['r2'],
            'Weight Error': eval_dict['parameter_recovery']['weight_error'],
            'Bias Error': eval_dict['parameter_recovery']['bias_error'],
            'Training Time': eval_dict['training_info']['training_time']
        })
    
    df = pd.DataFrame(results_df)
    print(df.to_string(index=False, float_format='%.6f'))
    
    # 4. Visualize all results
    print(f"\n4. Comprehensive Visualization")
    print("-" * 50)
    visualize_results(evaluations, X_test, y_test, models)
    
    # 5. Demonstrate normal distribution connection
    print(f"\n5. Normal Distribution Connection")
    print("-" * 50)
    demonstrate_normal_distribution_connection(X_test, y_test, models['Analytic'])
    
    # 6. Neural network perspective
    print(f"\n6. Neural Network Perspective")
    print("-" * 50)
    print("Linear regression can be viewed as a single-layer neural network:")
    print(f"Input dimension: {X.shape[1]}")
    print(f"Output dimension: 1")
    print(f"Parameters: {X.shape[1]} weights + 1 bias = {X.shape[1] + 1} total")
    print(f"Activation function: None (linear)")
    print(f"Loss function: Mean Squared Error")
    
    # Compare with PyTorch implementation
    pytorch_model = nn.Linear(X.shape[1], 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(pytorch_model.parameters(), lr=0.01)
    
    print(f"\nComparing with PyTorch nn.Linear:")
    
    # Set same initial weights for fair comparison
    with torch.no_grad():
        pytorch_model.weight.copy_(models['Analytic'].w.unsqueeze(0))
        pytorch_model.bias.copy_(models['Analytic'].b)
    
    # Test forward pass
    our_pred = models['Analytic'].predict(X_test[:5])
    pytorch_pred = pytorch_model(X_test[:5]).squeeze()
    
    print(f"Our prediction:     {our_pred.numpy()}")
    print(f"PyTorch prediction: {pytorch_pred.detach().numpy()}")
    print(f"Difference:         {torch.max(torch.abs(our_pred - pytorch_pred)).item():.10f}")
    
    print(f"\n" + "=" * 80)
    print("SUMMARY AND KEY INSIGHTS")
    print("=" * 80)
    print("1. ANALYTIC vs SGD:")
    print("   • Analytic solution is exact and fast for small problems")
    print("   • SGD is more flexible and scales to large datasets")
    print("   • Both converge to the same solution when implemented correctly")
    print()
    print("2. VECTORIZATION:")
    print("   • All operations use vectorized PyTorch operations")
    print("   • Dramatic speedup compared to Python loops")
    print("   • Essential for handling realistic dataset sizes")
    print()
    print("3. NORMAL DISTRIBUTION CONNECTION:")
    print("   • MSE loss emerges naturally from Gaussian noise assumption")
    print("   • Minimizing MSE ≡ Maximizing likelihood")
    print("   • Residual analysis validates model assumptions")
    print()
    print("4. NEURAL NETWORK PERSPECTIVE:")
    print("   • Linear regression IS a neural network (single layer)")
    print("   • Foundation for understanding deep learning")
    print("   • Same mathematical operations as modern architectures")
    print()
    print("5. PRACTICAL CONSIDERATIONS:")
    print("   • Model selection depends on data size and computational resources")
    print("   • Regularization may be needed for high-dimensional data")
    print("   • Understanding fundamentals enables better deep learning")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Run the comprehensive demonstration
    main()