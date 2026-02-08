"""
Linear Regression as a Neural Network
===================================

This module demonstrates how linear regression can be viewed as the simplest
neural network - a single-layer fully connected network, as explained in Basic.md.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List


class SingleNeuronNetwork(nn.Module):
    """
    A single neuron neural network for linear regression.
    
    This demonstrates the neural network perspective of linear regression:
    - Input layer: Features (x₁, x₂, ..., xₐ)
    - Single neuron: Computes weighted sum + bias
    - Output: Single prediction value
    """
    
    def __init__(self, input_dim: int):
        super(SingleNeuronNetwork, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
        # Initialize weights similar to our manual implementation
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: output = w₁x₁ + w₂x₂ + ... + wₐxₐ + b
        """
        return self.linear(x)
    
    def get_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the weights and bias of the neuron."""
        return self.linear.weight.data, self.linear.bias.data


class LinearRegressionNN:
    """Linear regression using PyTorch's neural network framework."""
    
    def __init__(self, input_dim: int, learning_rate: float = 0.01):
        self.model = SingleNeuronNetwork(input_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.loss_history = []
    
    def train(self, X: torch.Tensor, y: torch.Tensor, 
              epochs: int = 1000, batch_size: int = 32, verbose: bool = True) -> None:
        """Train the neural network model."""
        
        # Create data loader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_X, batch_y in dataloader:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch_X)
                
                # Compute loss
                loss = self.criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            # Record average loss
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions."""
        self.model.eval()
        with torch.no_grad():
            return self.model(X)
    
    def get_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get learned parameters."""
        return self.model.get_parameters()


def visualize_neuron_computation():
    """Visualize how a single neuron computes its output."""
    print("Single Neuron Computation Visualization")
    print("=" * 45)
    
    # Simple 2D example for visualization
    torch.manual_seed(42)
    
    # Sample input
    x = torch.tensor([[2.0, 3.0]])  # Single sample with 2 features
    w = torch.tensor([[0.5], [1.2]])  # Weights
    b = torch.tensor([0.3])  # Bias
    
    print(f"Input features: x = {x.numpy()}")
    print(f"Weights: w = {w.flatten().numpy()}")
    print(f"Bias: b = {b.numpy()}")
    print()
    
    # Step-by-step computation
    print("Step-by-step computation:")
    print("1. Weighted inputs:")
    weighted_inputs = x * w.T
    for i, (xi, wi, weighted) in enumerate(zip(x.flatten(), w.flatten(), weighted_inputs.flatten())):
        print(f"   w{i+1} × x{i+1} = {wi:.1f} × {xi:.1f} = {weighted:.1f}")
    
    print("\n2. Sum of weighted inputs:")
    sum_weighted = torch.sum(weighted_inputs)
    print(f"   Σ(wi × xi) = {' + '.join([f'{w:.1f}' for w in weighted_inputs.flatten()])} = {sum_weighted:.1f}")
    
    print("\n3. Add bias:")
    output = sum_weighted + b
    print(f"   output = {sum_weighted:.1f} + {b.item():.1f} = {output.item():.1f}")
    
    # Verify with PyTorch
    model = SingleNeuronNetwork(2)
    model.linear.weight.data = w.T
    model.linear.bias.data = b
    
    torch_output = model(x)
    print(f"\n4. PyTorch verification: {torch_output.item():.1f}")
    
    # Visualize the neuron
    visualize_single_neuron(x.numpy(), w.numpy(), b.numpy(), output.item())


def visualize_single_neuron(x: np.ndarray, w: np.ndarray, b: np.ndarray, output: float):
    """Create a visual representation of the neuron computation."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Input nodes
    input_positions = [(0, 0.7), (0, 0.3)]
    input_labels = [f'x₁ = {x[0,0]:.1f}', f'x₂ = {x[0,1]:.1f}']
    
    # Output node
    output_position = (2, 0.5)
    
    # Draw input nodes
    for i, ((x_pos, y_pos), label) in enumerate(zip(input_positions, input_labels)):
        circle = plt.Circle((x_pos, y_pos), 0.1, color='lightblue', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x_pos, y_pos, f'x{i+1}', ha='center', va='center', fontweight='bold')
        ax.text(x_pos - 0.3, y_pos, label, ha='center', va='center')
    
    # Draw output node
    circle = plt.Circle(output_position, 0.1, color='lightcoral', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(output_position[0], output_position[1], 'ŷ', ha='center', va='center', fontweight='bold')
    ax.text(output_position[0] + 0.3, output_position[1], f'ŷ = {output:.1f}', ha='left', va='center')
    
    # Draw connections with weights
    for i, (x_pos, y_pos) in enumerate(input_positions):
        # Draw arrow
        ax.annotate('', xy=output_position, xytext=(x_pos + 0.1, y_pos),
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))
        
        # Add weight label
        mid_x = (x_pos + 0.1 + output_position[0]) / 2
        mid_y = (y_pos + output_position[1]) / 2
        ax.text(mid_x, mid_y + 0.05, f'w{i+1} = {w[i,0]:.1f}', 
               ha='center', va='center', fontweight='bold', 
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='darkgreen'))
    
    # Draw bias
    ax.text(output_position[0], output_position[1] - 0.25, f'+ b = {b[0]:.1f}', 
           ha='center', va='center', fontweight='bold', 
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', edgecolor='orange'))
    
    # Add computation formula
    formula = f'ŷ = w₁x₁ + w₂x₂ + b = {w[0,0]:.1f}×{x[0,0]:.1f} + {w[1,0]:.1f}×{x[0,1]:.1f} + {b[0]:.1f} = {output:.1f}'
    ax.text(1, 0.05, formula, ha='center', va='center', fontsize=12, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Single Neuron Linear Regression Network', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def biological_inspiration_demo():
    """Demonstrate the biological inspiration behind artificial neurons."""
    print("\nBiological vs Artificial Neuron Comparison")
    print("=" * 45)
    
    biological_artificial_mapping = {
        'Dendrites': 'Input features (x₁, x₂, ...)',
        'Synapses': 'Weights (w₁, w₂, ...)',
        'Cell Nucleus': 'Summation + Bias (Σwixi + b)',
        'Axon': 'Output (ŷ)',
    }
    
    print("Biological Component → Artificial Equivalent:")
    print("-" * 45)
    for bio, art in biological_artificial_mapping.items():
        print(f"{bio:15} → {art}")
    
    print(f"\nKey differences:")
    print("- Biological neurons use electrochemical signals")
    print("- Artificial neurons use mathematical operations")
    print("- Both perform weighted aggregation of inputs")
    print("- Both have activation thresholds (bias in artificial neurons)")


def compare_implementations():
    """Compare manual implementation vs neural network framework."""
    print("\nImplementation Comparison: Manual vs Neural Network Framework")
    print("=" * 65)
    
    # Generate synthetic data
    torch.manual_seed(42)
    n_samples, n_features = 1000, 3
    X = torch.randn(n_samples, n_features)
    true_w = torch.tensor([[2.0], [-1.5], [0.8]])
    true_b = torch.tensor([0.3])
    noise = torch.randn(n_samples, 1) * 0.1
    y = torch.mm(X, true_w) + true_b + noise
    
    print(f"Generated data: {n_samples} samples, {n_features} features")
    print(f"True parameters: w = {true_w.flatten().tolist()}, b = {true_b.item()}")
    
    # Method 1: Manual implementation (from basic_linear_regression.py)
    from basic_linear_regression import LinearRegressionSGD
    model_manual = LinearRegressionSGD(learning_rate=0.01, n_epochs=500)
    model_manual.fit(X, y, batch_size=32, verbose=False)
    
    # Method 2: Neural network framework
    model_nn = LinearRegressionNN(input_dim=n_features, learning_rate=0.01)
    model_nn.train(X, y, epochs=500, batch_size=32, verbose=False)
    
    # Compare results
    w_manual = model_manual.w.detach()
    b_manual = model_manual.b.detach()
    w_nn, b_nn = model_nn.get_parameters()
    
    print(f"\nResults comparison:")
    print(f"Manual implementation: w = {w_manual.flatten().tolist()}, b = {b_manual.item():.4f}")
    print(f"Neural network:        w = {w_nn.flatten().tolist()}, b = {b_nn.item():.4f}")
    print(f"True parameters:       w = {true_w.flatten().tolist()}, b = {true_b.item():.4f}")
    
    # Calculate errors
    w_error_manual = torch.norm(w_manual - true_w).item()
    w_error_nn = torch.norm(w_nn.T - true_w).item()
    b_error_manual = abs(b_manual - true_b).item()
    b_error_nn = abs(b_nn - true_b).item()
    
    print(f"\nParameter errors:")
    print(f"Manual: w_error = {w_error_manual:.6f}, b_error = {b_error_manual:.6f}")
    print(f"NN:     w_error = {w_error_nn:.6f}, b_error = {b_error_nn:.6f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(model_manual.loss_history, label='Manual Implementation', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss - Manual Implementation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(model_nn.loss_history, label='Neural Network Framework', linewidth=2, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss - Neural Network Framework')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def network_architecture_evolution():
    """Show how linear regression fits into the broader neural network landscape."""
    print("\nNeural Network Architecture Evolution")
    print("=" * 40)
    
    architectures = {
        "Linear Regression": {
            "layers": "Input → Output",
            "neurons": "1 output neuron",
            "activation": "None (linear)",
            "use_case": "Simple regression"
        },
        "Multilayer Perceptron": {
            "layers": "Input → Hidden → Output",
            "neurons": "Multiple neurons per layer",
            "activation": "Non-linear (ReLU, sigmoid)",
            "use_case": "Complex patterns"
        },
        "Deep Neural Network": {
            "layers": "Input → Hidden₁ → ... → Hiddenₙ → Output",
            "neurons": "Many neurons, many layers",
            "activation": "Non-linear functions",
            "use_case": "Very complex patterns"
        }
    }
    
    print("Architecture Evolution:")
    print("-" * 25)
    for name, specs in architectures.items():
        print(f"\n{name}:")
        for key, value in specs.items():
            print(f"  {key:12}: {value}")
    
    print(f"\nKey insight: Linear regression is the foundational building block.")
    print(f"More complex networks are built by:")
    print(f"1. Adding more neurons (width)")
    print(f"2. Adding more layers (depth)")
    print(f"3. Adding non-linear activation functions")


def universal_approximation_demo():
    """Demonstrate how even simple networks can approximate complex functions."""
    print("\nFrom Linear to Non-linear: Adding Complexity")
    print("=" * 50)
    
    # Generate non-linear data
    torch.manual_seed(42)
    x = torch.linspace(-2, 2, 100).reshape(-1, 1)
    y_true = x**3 + 0.5*x**2 - 2*x + torch.randn_like(x) * 0.1
    
    # 1. Linear model (single neuron)
    linear_model = SingleNeuronNetwork(1)
    linear_optimizer = optim.SGD(linear_model.parameters(), lr=0.01)
    linear_criterion = nn.MSELoss()
    
    # 2. Non-linear model (multiple neurons with activation)
    class NonLinearNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(1, 10)
            self.output = nn.Linear(10, 1)
            
        def forward(self, x):
            x = torch.relu(self.hidden(x))  # Non-linear activation
            return self.output(x)
    
    nonlinear_model = NonLinearNetwork()
    nonlinear_optimizer = optim.SGD(nonlinear_model.parameters(), lr=0.01)
    nonlinear_criterion = nn.MSELoss()
    
    # Train both models
    epochs = 1000
    for epoch in range(epochs):
        # Linear model
        linear_optimizer.zero_grad()
        linear_pred = linear_model(x)
        linear_loss = linear_criterion(linear_pred, y_true)
        linear_loss.backward()
        linear_optimizer.step()
        
        # Non-linear model
        nonlinear_optimizer.zero_grad()
        nonlinear_pred = nonlinear_model(x)
        nonlinear_loss = nonlinear_criterion(nonlinear_pred, y_true)
        nonlinear_loss.backward()
        nonlinear_optimizer.step()
    
    # Plot results
    with torch.no_grad():
        linear_predictions = linear_model(x)
        nonlinear_predictions = nonlinear_model(x)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(x.numpy(), y_true.numpy(), alpha=0.6, label='True data')
    plt.plot(x.numpy(), linear_predictions.numpy(), 'r-', linewidth=2, label='Linear model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Model (Single Neuron)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(x.numpy(), y_true.numpy(), alpha=0.6, label='True data')
    plt.plot(x.numpy(), nonlinear_predictions.numpy(), 'g-', linewidth=2, label='Non-linear model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Non-linear Model (Multiple Neurons + Activation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Observe:")
    print("- Linear model can only fit straight lines")
    print("- Non-linear model can capture curved patterns")
    print("- Both use the same basic neuron building blocks")


if __name__ == "__main__":
    visualize_neuron_computation()
    biological_inspiration_demo()
    compare_implementations()
    network_architecture_evolution()
    universal_approximation_demo()