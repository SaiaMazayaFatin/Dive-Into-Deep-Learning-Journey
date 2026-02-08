"""
Normal Distribution and Squared Loss Connection
==============================================

This module demonstrates the deep connection between the Normal Distribution
and the Squared Loss function used in Linear Regression. It shows why Maximum
Likelihood Estimation naturally leads to minimizing squared error.

Based on the materials in 3.TheNormalDistributionandSquaredLoss.md
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Tuple, List
from scipy import stats


def normal_pdf(x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """
    Calculate the probability density function of a normal distribution.
    
    Formula: p(x) = (1/√(2πσ²)) * exp(-1/(2σ²) * (x-μ)²)
    
    Args:
        x: Input values
        mu: Mean of the distribution
        sigma: Standard deviation
        
    Returns:
        Probability densities
    """
    # Calculate the constant part
    normalization = 1 / math.sqrt(2 * math.pi * sigma**2)
    
    # Calculate the exponential part
    exponent = -0.5 * ((x - mu) ** 2) / (sigma ** 2)
    
    return normalization * torch.exp(exponent)


def plot_normal_distributions():
    """
    Visualize normal distributions with different parameters.
    """
    x = torch.linspace(-7, 7, 1000)
    
    # Different parameter combinations (mean, std)
    params = [(0, 1), (0, 2), (3, 1), (-2, 0.5)]
    colors = ['blue', 'red', 'green', 'orange']
    labels = ['μ=0, σ=1', 'μ=0, σ=2', 'μ=3, σ=1', 'μ=-2, σ=0.5']
    
    plt.figure(figsize=(12, 8))
    
    for i, (mu, sigma) in enumerate(params):
        y = normal_pdf(x, mu, sigma)
        plt.plot(x.numpy(), y.numpy(), color=colors[i], linewidth=2, label=labels[i])
    
    plt.title('Normal Distribution with Different Parameters', fontsize=16)
    plt.xlabel('x')
    plt.ylabel('Probability Density p(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def demonstrate_mle_connection():
    """
    Demonstrate how Maximum Likelihood Estimation leads to squared loss.
    """
    print("=" * 70)
    print("MAXIMUM LIKELIHOOD ESTIMATION → SQUARED LOSS")
    print("=" * 70)
    
    # Generate synthetic data with known parameters
    true_w = torch.tensor([2.0, -1.5])  # True weights
    true_b = torch.tensor([1.0])        # True bias
    sigma = 0.5                         # Noise standard deviation
    n_samples = 100
    
    # Generate features
    X = torch.randn(n_samples, 2)
    
    # Generate targets with Gaussian noise
    y_true = X @ true_w + true_b
    noise = torch.normal(0, sigma, (n_samples,))
    y = y_true + noise
    
    print(f"True weights: {true_w}")
    print(f"True bias: {true_b}")
    print(f"Noise std: {sigma}")
    print(f"Generated {n_samples} samples")
    
    # Define a range of possible weight values to test
    w1_range = torch.linspace(1.0, 3.0, 50)
    w2_range = torch.linspace(-2.5, -0.5, 50)
    
    # Calculate likelihood and squared loss for different weight combinations
    max_likelihood = -float('inf')
    min_squared_loss = float('inf')
    best_w1_mle = None
    best_w2_mle = None
    best_w1_mse = None
    best_w2_mse = None
    
    likelihood_surface = torch.zeros(len(w1_range), len(w2_range))
    mse_surface = torch.zeros(len(w1_range), len(w2_range))
    
    for i, w1 in enumerate(w1_range):
        for j, w2 in enumerate(w2_range):
            w_test = torch.tensor([w1, w2])
            
            # Predictions with current weights
            y_pred = X @ w_test + true_b  # Use true bias for simplicity
            
            # Calculate log-likelihood (assuming known sigma)
            residuals = y - y_pred
            log_likelihood = torch.sum(-0.5 * torch.log(torch.tensor(2 * math.pi * sigma**2)) 
                                     - 0.5 * (residuals**2) / (sigma**2))
            
            # Calculate mean squared error
            mse = torch.mean(0.5 * (residuals**2))
            
            likelihood_surface[i, j] = log_likelihood
            mse_surface[i, j] = mse
            
            if log_likelihood > max_likelihood:
                max_likelihood = log_likelihood
                best_w1_mle, best_w2_mle = w1, w2
            
            if mse < min_squared_loss:
                min_squared_loss = mse
                best_w1_mse, best_w2_mse = w1, w2
    
    print(f"\nMLE Solution:")
    print(f"  Best w1: {best_w1_mle:.3f}, Best w2: {best_w2_mle:.3f}")
    print(f"  Max log-likelihood: {max_likelihood:.3f}")
    
    print(f"\nMSE Solution:")
    print(f"  Best w1: {best_w1_mse:.3f}, Best w2: {best_w2_mse:.3f}")
    print(f"  Min squared loss: {min_squared_loss:.3f}")
    
    print(f"\nDifference between solutions:")
    print(f"  |w1_mle - w1_mse|: {abs(best_w1_mle - best_w1_mse):.6f}")
    print(f"  |w2_mle - w2_mse|: {abs(best_w2_mle - best_w2_mse):.6f}")
    
    # Plot the surfaces
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Likelihood surface
    W1, W2 = torch.meshgrid(w1_range, w2_range, indexing='ij')
    contour1 = ax1.contour(W1.numpy(), W2.numpy(), likelihood_surface.numpy(), levels=20)
    ax1.plot(best_w1_mle, best_w2_mle, 'r*', markersize=15, label='MLE Solution')
    ax1.plot(true_w[0], true_w[1], 'go', markersize=10, label='True Values')
    ax1.set_xlabel('w1')
    ax1.set_ylabel('w2')
    ax1.set_title('Log-Likelihood Surface')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MSE surface  
    contour2 = ax2.contour(W1.numpy(), W2.numpy(), mse_surface.numpy(), levels=20)
    ax2.plot(best_w1_mse, best_w2_mse, 'r*', markersize=15, label='MSE Solution')
    ax2.plot(true_w[0], true_w[1], 'go', markersize=10, label='True Values')
    ax2.set_xlabel('w1')
    ax2.set_ylabel('w2')
    ax2.set_title('Mean Squared Error Surface')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return best_w1_mle, best_w2_mle, best_w1_mse, best_w2_mse


def negative_log_likelihood_derivation():
    """
    Show the mathematical derivation from likelihood to squared loss.
    """
    print("\n" + "=" * 70)
    print("MATHEMATICAL DERIVATION")
    print("=" * 70)
    
    print("Starting from the assumption that noise ε ~ N(0, σ²):")
    print("y = w^T * x + b + ε")
    print()
    print("The probability of observing y given x is:")
    print("P(y|x) = (1/√(2πσ²)) * exp(-(y - w^T*x - b)²/(2σ²))")
    print()
    print("For the entire dataset, the likelihood is:")
    print("P(y₁,...,yₙ|x₁,...,xₙ) = ∏ⁱ P(yᵢ|xᵢ)")
    print()
    print("Taking the negative log-likelihood:")
    print("-log P = Σᵢ [1/2 * log(2πσ²) + (yᵢ - w^T*xᵢ - b)²/(2σ²)]")
    print()
    print("The terms that depend on w and b are:")
    print("Σᵢ (yᵢ - w^T*xᵢ - b)²")
    print()
    print("This is exactly the sum of squared errors!")
    print("Therefore: Minimizing MSE ≡ Maximizing Likelihood")


def demonstrate_noise_assumption():
    """
    Show how different noise assumptions lead to different loss functions.
    """
    print("\n" + "=" * 70)
    print("DIFFERENT NOISE ASSUMPTIONS → DIFFERENT LOSS FUNCTIONS")
    print("=" * 70)
    
    # Generate data with different types of noise
    n_samples = 1000
    x = torch.linspace(-3, 3, n_samples)
    true_y = 2 * x + 1  # Simple linear relationship
    
    # Different noise types
    gaussian_noise = torch.normal(0, 0.5, (n_samples,))
    laplace_noise = torch.tensor(np.random.laplace(0, 0.3, n_samples)).float()
    uniform_noise = torch.uniform(-0.8, 0.8, (n_samples,))
    
    y_gaussian = true_y + gaussian_noise
    y_laplace = true_y + laplace_noise  
    y_uniform = true_y + uniform_noise
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Top row: Data with different noise types
    noise_data = [
        (y_gaussian, "Gaussian Noise → MSE Loss", "blue"),
        (y_laplace, "Laplace Noise → MAE Loss", "red"), 
        (y_uniform, "Uniform Noise → Uniform Loss", "green")
    ]
    
    for i, (y_noisy, title, color) in enumerate(noise_data):
        axes[0, i].scatter(x[:100], y_noisy[:100], alpha=0.6, color=color, s=20)
        axes[0, i].plot(x, true_y, 'k-', linewidth=2, label='True function')
        axes[0, i].set_title(title)
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('y')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
    
    # Bottom row: Noise distributions
    noise_samples = [gaussian_noise, laplace_noise, uniform_noise]
    noise_names = ["Gaussian N(0,0.5²)", "Laplace(0,0.3)", "Uniform(-0.8,0.8)"]
    
    for i, (noise, name) in enumerate(zip(noise_samples, noise_names)):
        axes[1, i].hist(noise.numpy(), bins=50, density=True, alpha=0.7, 
                       color=noise_data[i][2], edgecolor='black')
        axes[1, i].set_title(f'{name} Distribution')
        axes[1, i].set_xlabel('Noise Value')
        axes[1, i].set_ylabel('Probability Density')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Key Insights:")
    print("1. Gaussian noise assumption → MSE loss is optimal")
    print("2. Laplace noise assumption → MAE loss is optimal") 
    print("3. Different noise types require different loss functions")
    print("4. MSE is popular because Gaussian noise is common in nature")


def interactive_parameter_exploration():
    """
    Allow exploration of how changing σ affects the likelihood surface.
    """
    print("\n" + "=" * 70)
    print("EFFECT OF NOISE LEVEL (σ) ON LIKELIHOOD")
    print("=" * 70)
    
    # Generate sample data
    X = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
    y = torch.tensor([5.0, 4.0, 8.0])
    true_w = torch.tensor([1.0, 2.0])
    
    # Test different noise levels
    sigma_values = [0.1, 0.5, 1.0, 2.0]
    w_test = torch.tensor([1.2, 1.8])  # Slightly off from true value
    
    print("True weights:", true_w.numpy())
    print("Test weights:", w_test.numpy())
    print()
    
    for sigma in sigma_values:
        y_pred = X @ w_test
        residuals = y - y_pred
        
        # Calculate log-likelihood
        log_likelihood = torch.sum(
            -0.5 * torch.log(torch.tensor(2 * math.pi * sigma**2)) 
            - 0.5 * (residuals**2) / (sigma**2)
        )
        
        # Calculate MSE
        mse = torch.mean(0.5 * residuals**2)
        
        print(f"σ = {sigma}:")
        print(f"  Log-likelihood: {log_likelihood:.3f}")
        print(f"  MSE: {mse:.3f}")
        print(f"  Residuals: {residuals.numpy()}")
        print()
    
    print("Observations:")
    print("1. Lower σ (less noise) → More peaked likelihood")
    print("2. Higher σ (more noise) → Flatter likelihood")
    print("3. MSE is independent of σ assumption")
    print("4. But MLE solution depends on σ for model selection")


def main():
    """
    Main demonstration function.
    """
    print("=" * 80)
    print("NORMAL DISTRIBUTION AND SQUARED LOSS CONNECTION")
    print("=" * 80)
    print("This demonstration shows why we use squared error loss in linear regression")
    print("and how it connects to the Maximum Likelihood Principle.\n")
    
    # 1. Visualize normal distributions
    print("1. Visualizing Normal Distributions")
    print("-" * 50)
    plot_normal_distributions()
    
    # 2. Show MLE connection
    print("\n2. Maximum Likelihood Estimation Connection")
    print("-" * 50)
    demonstrate_mle_connection()
    
    # 3. Mathematical derivation
    negative_log_likelihood_derivation()
    
    # 4. Different noise assumptions
    print("\n4. Different Noise Assumptions")
    print("-" * 50)
    demonstrate_noise_assumption()
    
    # 5. Parameter exploration
    print("\n5. Parameter Exploration")
    print("-" * 50)
    interactive_parameter_exploration()
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("1. Squared loss is not arbitrary - it comes from Gaussian noise assumption")
    print("2. Minimizing MSE ≡ Maximizing likelihood under Gaussian noise")
    print("3. Different noise distributions lead to different optimal loss functions")
    print("4. This connection provides theoretical justification for our choice")
    print("5. Understanding this helps in choosing appropriate loss functions")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the demonstration
    main()