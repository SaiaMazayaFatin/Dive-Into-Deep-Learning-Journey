"""
Normal Distribution and Squared Loss Connection
=============================================

This module demonstrates the mathematical connection between the normal distribution
and squared error loss in linear regression, as explained in Basic.md.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
import math
from typing import Tuple, List


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Calculate the probability density function of a normal distribution.
    
    Formula: p(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
    
    Args:
        x: Input values
        mu: Mean of the distribution
        sigma: Standard deviation of the distribution
        
    Returns:
        Probability densities
    """
    # Calculate the constant part
    constant = 1 / math.sqrt(2 * math.pi * sigma**2)
    
    # Calculate the exponential part
    exponent = -0.5 * ((x - mu) ** 2) / (sigma ** 2)
    
    return constant * np.exp(exponent)


def visualize_normal_distributions():
    """Visualize normal distributions with different parameters."""
    print("Normal Distribution Visualization")
    print("=" * 40)
    
    # Create x values
    x = np.linspace(-7, 7, 1000)
    
    # Different parameter combinations (mean, std)
    params = [(0, 1), (0, 2), (3, 1), (-2, 0.5)]
    colors = ['blue', 'red', 'green', 'orange']
    
    plt.figure(figsize=(12, 8))
    
    # Plot different normal distributions
    for i, (mu, sigma) in enumerate(params):
        y = normal_pdf(x, mu, sigma)
        plt.plot(x, y, color=colors[i], linewidth=2, 
                label=f'μ={mu}, σ={sigma}')
    
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title('Normal Distributions with Different Parameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Notice how:")
    print("- Changing μ (mean) shifts the curve left/right")
    print("- Changing σ (std) makes the curve wider/narrower")
    print("- All curves are bell-shaped and symmetric")


def demonstrate_mle_connection():
    """Demonstrate the connection between MLE and squared error."""
    print("\nMaximum Likelihood Estimation Connection")
    print("=" * 45)
    
    # Generate synthetic data with known parameters
    torch.manual_seed(42)
    n_samples = 100
    true_w = torch.tensor([[2.0], [-1.5]])
    true_b = torch.tensor([0.5])
    noise_std = 0.2
    
    # Generate features and true target
    X = torch.randn(n_samples, 2)
    y_true = torch.mm(X, true_w) + true_b
    
    # Add Gaussian noise
    noise = torch.randn(n_samples, 1) * noise_std
    y_observed = y_true + noise
    
    print(f"Generated {n_samples} samples with noise std = {noise_std}")
    print(f"True parameters: w = {true_w.flatten().tolist()}, b = {true_b.item()}")
    
    def negative_log_likelihood(w: torch.Tensor, b: torch.Tensor, 
                              X: torch.Tensor, y: torch.Tensor, 
                              sigma: float = 0.2) -> torch.Tensor:
        """
        Calculate negative log-likelihood for linear regression.
        
        Formula: -log P(y|X) = sum[1/2 * log(2πσ²) + (y_i - X_i*w - b)²/(2σ²)]
        """
        predictions = torch.mm(X, w) + b
        residuals = y - predictions
        
        # Constant term (can be ignored for optimization)
        constant_term = 0.5 * math.log(2 * math.pi * sigma**2)
        
        # Variable term (this is what we minimize)
        variable_term = torch.sum(residuals**2) / (2 * sigma**2)
        
        return n_samples * constant_term + variable_term
    
    def squared_error_loss(w: torch.Tensor, b: torch.Tensor, 
                          X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate standard squared error loss."""
        predictions = torch.mm(X, w) + b
        return torch.sum((y - predictions)**2) / (2 * len(y))
    
    # Test different parameter values around the true values
    w_test = torch.tensor([[1.8], [-1.3]])
    b_test = torch.tensor([0.3])
    
    nll = negative_log_likelihood(w_test, b_test, X, y_observed, noise_std)
    mse = squared_error_loss(w_test, b_test, X, y_observed)
    
    print(f"\nFor test parameters w={w_test.flatten().tolist()}, b={b_test.item()}:")
    print(f"Negative Log-Likelihood: {nll.item():.4f}")
    print(f"Squared Error Loss: {mse.item():.4f}")
    
    # Show that minimizing NLL is equivalent to minimizing squared error
    print(f"\nKey insight: When σ is constant, minimizing NLL is equivalent")
    print(f"to minimizing squared error (ignoring constant terms).")


def likelihood_surface_visualization():
    """Visualize the likelihood surface for a simple 1D case."""
    print("\nLikelihood Surface Visualization")
    print("=" * 35)
    
    # Simple 1D case for visualization
    torch.manual_seed(42)
    n_points = 20
    x = torch.linspace(-2, 2, n_points).reshape(-1, 1)
    true_w = 1.5
    true_b = 0.3
    noise_std = 0.2
    
    # Generate data
    y_true = true_w * x + true_b
    noise = torch.randn_like(y_true) * noise_std
    y = y_true + noise
    
    # Create parameter grid for visualization
    w_range = np.linspace(0.5, 2.5, 50)
    b_range = np.linspace(-0.5, 1.0, 50)
    W, B = np.meshgrid(w_range, b_range)
    
    # Calculate likelihood for each parameter combination
    likelihood_surface = np.zeros_like(W)
    mse_surface = np.zeros_like(W)
    
    for i in range(len(w_range)):
        for j in range(len(b_range)):
            w_test = torch.tensor([[w_range[i]]])
            b_test = torch.tensor([b_range[j]])
            
            # Calculate likelihood (log scale)
            predictions = torch.mm(x, w_test) + b_test
            residuals = y - predictions
            log_likelihood = -0.5 * torch.sum(residuals**2) / (noise_std**2)
            likelihood_surface[j, i] = log_likelihood.item()
            
            # Calculate MSE
            mse = torch.mean(residuals**2)
            mse_surface[j, i] = mse.item()
    
    # Plot both surfaces
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Log-likelihood surface
    contour1 = ax1.contour(W, B, likelihood_surface, levels=20)
    ax1.clabel(contour1, inline=True, fontsize=8)
    ax1.plot(true_w, true_b, 'r*', markersize=15, label='True parameters')
    ax1.set_xlabel('Weight (w)')
    ax1.set_ylabel('Bias (b)')
    ax1.set_title('Log-Likelihood Surface')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MSE surface
    contour2 = ax2.contour(W, B, mse_surface, levels=20)
    ax2.clabel(contour2, inline=True, fontsize=8)
    ax2.plot(true_w, true_b, 'r*', markersize=15, label='True parameters')
    ax2.set_xlabel('Weight (w)')
    ax2.set_ylabel('Bias (b)')
    ax2.set_title('MSE Surface')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Notice that both surfaces have their minimum at the same location!")
    print("This confirms that maximizing likelihood = minimizing squared error.")


def noise_distribution_analysis():
    """Analyze the distribution of residuals in linear regression."""
    print("\nNoise Distribution Analysis")
    print("=" * 30)
    
    # Generate data with known Gaussian noise
    torch.manual_seed(42)
    n_samples = 1000
    X = torch.randn(n_samples, 2)
    true_w = torch.tensor([[1.5], [-0.8]])
    true_b = torch.tensor([0.2])
    noise_std = 0.3
    
    # Generate clean target and add noise
    y_clean = torch.mm(X, true_w) + true_b
    noise = torch.randn(n_samples, 1) * noise_std
    y_noisy = y_clean + noise
    
    # Fit linear regression
    from basic_linear_regression import LinearRegressionAnalytic
    model = LinearRegressionAnalytic()
    model.fit(X, y_noisy)
    
    # Calculate residuals
    predictions = model.predict(X)
    residuals = y_noisy - predictions
    
    # Analyze residual distribution
    residuals_np = residuals.flatten().numpy()
    
    # Plot residual distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(residuals_np, bins=30, density=True, alpha=0.7, color='skyblue', 
             label='Observed residuals')
    
    # Overlay theoretical normal distribution
    x_theory = np.linspace(residuals_np.min(), residuals_np.max(), 100)
    y_theory = normal_pdf(x_theory, 0, noise_std)
    plt.plot(x_theory, y_theory, 'r-', linewidth=2, label=f'N(0, {noise_std}²)')
    
    plt.xlabel('Residual Value')
    plt.ylabel('Density')
    plt.title('Distribution of Residuals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot
    plt.subplot(1, 2, 2)
    stats.probplot(residuals_np, dist="norm", plot=plt)
    plt.title('Q-Q Plot: Residuals vs Normal Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical test for normality
    from scipy.stats import shapiro
    statistic, p_value = shapiro(residuals_np)
    
    print(f"Residual statistics:")
    print(f"- Mean: {residuals_np.mean():.4f} (should be ≈ 0)")
    print(f"- Std:  {residuals_np.std():.4f} (should be ≈ {noise_std})")
    print(f"- Shapiro-Wilk test p-value: {p_value:.4f}")
    print(f"  {'✓' if p_value > 0.05 else '✗'} Residuals appear normally distributed" + 
          f" (p > 0.05: {p_value > 0.05})")


def why_squared_loss_demo():
    """Demonstrate why squared loss makes sense from probability perspective."""
    print("\nWhy Squared Loss? - Probabilistic Justification")
    print("=" * 50)
    
    # Generate data
    torch.manual_seed(42)
    n_samples = 50
    x = torch.linspace(-2, 2, n_samples).reshape(-1, 1)
    true_w = 1.2
    true_b = 0.5
    noise_std = 0.4
    
    y_clean = true_w * x + true_b
    noise = torch.randn_like(y_clean) * noise_std
    y = y_clean + noise
    
    # Compare different loss functions
    def compare_loss_functions(w_test: float, b_test: float):
        """Compare different loss functions for given parameters."""
        predictions = w_test * x + b_test
        residuals = y - predictions
        
        # Different loss functions
        mse_loss = torch.mean(residuals**2)
        mae_loss = torch.mean(torch.abs(residuals))
        neg_log_likelihood = -torch.sum(stats.norm.logpdf(residuals.numpy(), 0, noise_std))
        
        return mse_loss.item(), mae_loss.item(), neg_log_likelihood
    
    # Test around true parameters
    w_values = np.linspace(0.8, 1.6, 100)
    mse_losses = []
    mae_losses = []
    nll_losses = []
    
    for w in w_values:
        mse, mae, nll = compare_loss_functions(w, true_b)
        mse_losses.append(mse)
        mae_losses.append(mae)
        nll_losses.append(nll)
    
    # Normalize for comparison
    mse_losses = np.array(mse_losses)
    mae_losses = np.array(mae_losses)
    nll_losses = np.array(nll_losses)
    
    # Normalize to [0, 1] for comparison
    mse_norm = (mse_losses - mse_losses.min()) / (mse_losses.max() - mse_losses.min())
    mae_norm = (mae_losses - mae_losses.min()) / (mae_losses.max() - mae_losses.min())
    nll_norm = (nll_losses - nll_losses.min()) / (nll_losses.max() - nll_losses.min())
    
    plt.figure(figsize=(10, 6))
    plt.plot(w_values, mse_norm, 'b-', linewidth=2, label='MSE Loss')
    plt.plot(w_values, mae_norm, 'g--', linewidth=2, label='MAE Loss')
    plt.plot(w_values, nll_norm, 'r:', linewidth=3, label='Negative Log-Likelihood')
    
    plt.axvline(x=true_w, color='black', linestyle='-', alpha=0.7, label='True weight')
    plt.xlabel('Weight Parameter (w)')
    plt.ylabel('Normalized Loss')
    plt.title('Comparison of Loss Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Key observations:")
    print("1. MSE and Negative Log-Likelihood have very similar shapes")
    print("2. Both have their minimum at the true parameter value")
    print("3. MAE loss behaves differently (less smooth)")
    print("4. This justifies using squared loss for Gaussian noise assumptions")


if __name__ == "__main__":
    visualize_normal_distributions()
    demonstrate_mle_connection()
    likelihood_surface_visualization()
    noise_distribution_analysis()
    why_squared_loss_demo()