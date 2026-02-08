"""
Linear Regression as a Neural Network
====================================

This module demonstrates how Linear Regression can be viewed as the simplest
possible neural network: a single-layer fully connected network with no
activation function.

Based on the materials in 4.LinearRegressionasaNeuralNetwork.md
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import networkx as nx


class LinearNeuron:
    """
    A single linear neuron that implements: output = w^T * x + b
    
    This demonstrates the basic building block of neural networks.
    """
    
    def __init__(self, input_size: int):
        """
        Initialize a linear neuron.
        
        Args:
            input_size: Number of input features (dendrites)
        """
        self.input_size = input_size
        self.weights = torch.randn(input_size) * 0.1  # Synaptic weights
        self.bias = torch.randn(1) * 0.1              # Internal bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the neuron's output (forward propagation).
        
        Args:
            x: Input vector of size (input_size,) or batch (batch_size, input_size)
            
        Returns:
            Neuron output
        """
        if len(x.shape) == 1:
            # Single input
            return torch.dot(self.weights, x) + self.bias
        else:
            # Batch of inputs
            return x @ self.weights + self.bias
    
    def update_weights(self, grad_w: torch.Tensor, grad_b: torch.Tensor, learning_rate: float):
        """
        Update neuron parameters using gradients (learning).
        
        Args:
            grad_w: Gradient with respect to weights
            grad_b: Gradient with respect to bias
            learning_rate: Step size for updates
        """
        self.weights -= learning_rate * grad_w
        self.bias -= learning_rate * grad_b
    
    def __repr__(self):
        return f"LinearNeuron(inputs={self.input_size}, w={self.weights.numpy()}, b={self.bias.item():.3f})"


class LinearRegressionNetwork(nn.Module):
    """
    Linear Regression implemented as a PyTorch neural network.
    
    This is a single-layer fully connected network with no activation.
    """
    
    def __init__(self, input_size: int):
        """
        Initialize the network.
        
        Args:
            input_size: Number of input features
        """
        super(LinearRegressionNetwork, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # Single output neuron
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output predictions of shape (batch_size, 1)
        """
        return self.linear(x)
    
    def get_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the learned weights and bias.
        
        Returns:
            Tuple of (weights, bias)
        """
        return self.linear.weight.squeeze(), self.linear.bias


def visualize_network_structure(input_size: int = 3):
    """
    Create a visual representation of the linear regression neural network.
    
    Args:
        input_size: Number of input features to show
    """
    print("=" * 60)
    print("NEURAL NETWORK STRUCTURE VISUALIZATION")
    print("=" * 60)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add input nodes
    input_nodes = [f"x_{i+1}" for i in range(input_size)]
    for node in input_nodes:
        G.add_node(node, layer='input')
    
    # Add output node
    G.add_node("o_1", layer='output')
    
    # Add edges (connections) from inputs to output
    for input_node in input_nodes:
        G.add_edge(input_node, "o_1")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define positions for nodes
    pos = {}
    
    # Input layer positions
    for i, node in enumerate(input_nodes):
        pos[node] = (0, i - (input_size - 1) / 2)
    
    # Output layer position
    pos["o_1"] = (2, 0)
    
    # Draw the network
    # Input nodes
    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, 
                          node_color='lightblue', node_size=1500, 
                          node_shape='o')
    
    # Output node
    nx.draw_networkx_nodes(G, pos, nodelist=["o_1"], 
                          node_color='lightcoral', node_size=1500, 
                          node_shape='o')
    
    # Edges (connections)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=20, arrowstyle='->')
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Add layer labels
    plt.text(0, (input_size) / 2 + 0.5, 'Input Layer\n(Features)', 
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.text(2, (input_size) / 2 + 0.5, 'Output Layer\n(Prediction)', 
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    # Add connection labels (weights)
    for i, input_node in enumerate(input_nodes):
        plt.text(1, i - (input_size - 1) / 2 - 0.2, f'w_{i+1}', 
                ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))
    
    # Add bias
    plt.text(1.8, -0.3, 'b', ha='center', va='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.2", facecolor="orange", alpha=0.7))
    
    plt.title(f'Linear Regression as a Neural Network\n(Single Neuron with {input_size} Inputs)', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Show the mathematical formula
    print("\nMathematical Formula:")
    print("o₁ = w₁x₁ + w₂x₂ + ... + wₐxₐ + b")
    print("\nIn vector notation:")
    print("o₁ = w^T · x + b")
    
    # Biological analogy
    print("\nBiological Analogy:")
    print("┌─────────────────┬─────────────────┬─────────────────────────┐")
    print("│ Artificial      │ Biological      │ Function                │")
    print("├─────────────────┼─────────────────┼─────────────────────────┤")
    print("│ Input (xᵢ)      │ Dendrites       │ Receives signals        │")
    print("│ Weights (wᵢ)    │ Synapses        │ Signal strength         │")
    print("│ Summation + Bias│ Cell Nucleus    │ Aggregates signals      │")
    print("│ Output (o₁)     │ Axon            │ Sends processed signal  │")
    print("└─────────────────┴─────────────────┴─────────────────────────┘")


def demonstrate_single_neuron():
    """
    Demonstrate how a single linear neuron works.
    """
    print("\n" + "=" * 60)
    print("SINGLE LINEAR NEURON DEMONSTRATION")
    print("=" * 60)
    
    # Create a neuron with 3 inputs
    neuron = LinearNeuron(input_size=3)
    print(f"Created neuron: {neuron}")
    
    # Test with different inputs
    inputs = [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([0.5, -1.0, 2.5]),
        torch.tensor([-1.0, 0.0, 1.0])
    ]
    
    print("\nTesting with different inputs:")
    print("Input\t\t\tOutput")
    print("-" * 40)
    
    for i, x in enumerate(inputs):
        output = neuron.forward(x)
        print(f"{x.numpy()}\t{output.item():.4f}")
    
    # Batch processing
    print("\nBatch processing:")
    X_batch = torch.stack(inputs)
    outputs_batch = neuron.forward(X_batch)
    print(f"Batch input shape: {X_batch.shape}")
    print(f"Batch output: {outputs_batch.numpy()}")
    
    return neuron


def train_single_neuron(n_epochs: int = 100):
    """
    Train a single neuron using gradient descent.
    
    Args:
        n_epochs: Number of training epochs
        
    Returns:
        Trained neuron and loss history
    """
    print("\n" + "=" * 60)
    print("TRAINING A SINGLE LINEAR NEURON")
    print("=" * 60)
    
    # Generate synthetic data
    torch.manual_seed(42)
    n_samples = 100
    input_size = 2
    
    # True parameters
    true_w = torch.tensor([2.0, -1.5])
    true_b = torch.tensor([0.5])
    
    # Generate data
    X = torch.randn(n_samples, input_size)
    y = X @ true_w + true_b + torch.randn(n_samples) * 0.1
    
    print(f"Generated {n_samples} samples with {input_size} features")
    print(f"True weights: {true_w.numpy()}")
    print(f"True bias: {true_b.item()}")
    
    # Create and train neuron
    neuron = LinearNeuron(input_size)
    learning_rate = 0.01
    losses = []
    
    print(f"\nInitial neuron: {neuron}")
    print(f"Learning rate: {learning_rate}")
    print("\nTraining progress:")
    
    for epoch in range(n_epochs):
        # Forward pass
        predictions = neuron.forward(X)
        
        # Compute loss
        loss = torch.mean(0.5 * (predictions - y) ** 2)
        losses.append(loss.item())
        
        # Compute gradients
        errors = predictions - y
        grad_w = torch.mean(X * errors.unsqueeze(1), dim=0)
        grad_b = torch.mean(errors)
        
        # Update parameters
        neuron.update_weights(grad_w, grad_b, learning_rate)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d}: Loss = {loss.item():.6f}")
    
    print(f"\nFinal neuron: {neuron}")
    print(f"Weight error: {torch.norm(neuron.weights - true_w).item():.6f}")
    print(f"Bias error: {torch.abs(neuron.bias - true_b).item():.6f}")
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Progress: Single Linear Neuron')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return neuron, losses


def compare_implementations():
    """
    Compare our custom neuron with PyTorch's built-in implementation.
    """
    print("\n" + "=" * 60)
    print("COMPARING IMPLEMENTATIONS")
    print("=" * 60)
    
    # Generate test data
    torch.manual_seed(42)
    X = torch.randn(50, 3)
    y = torch.randn(50)
    
    # Our custom neuron
    custom_neuron = LinearNeuron(input_size=3)
    
    # PyTorch neural network
    pytorch_net = LinearRegressionNetwork(input_size=3)
    
    # Set same initial parameters for fair comparison
    with torch.no_grad():
        pytorch_net.linear.weight.copy_(custom_neuron.weights.unsqueeze(0))
        pytorch_net.linear.bias.copy_(custom_neuron.bias)
    
    print("Initial parameters:")
    print(f"Custom neuron: {custom_neuron}")
    w_pt, b_pt = pytorch_net.get_parameters()
    print(f"PyTorch net:   weights={w_pt.numpy()}, bias={b_pt.item():.3f}")
    
    # Test forward pass
    pred_custom = custom_neuron.forward(X)
    pred_pytorch = pytorch_net(X).squeeze()
    
    print(f"\nForward pass comparison:")
    print(f"Max difference: {torch.max(torch.abs(pred_custom - pred_pytorch)).item():.8f}")
    print(f"Outputs are equal: {torch.allclose(pred_custom, pred_pytorch)}")
    
    # Training comparison
    print(f"\nTraining both implementations...")
    
    # Train custom neuron
    custom_losses = []
    learning_rate = 0.01
    
    for epoch in range(100):
        pred = custom_neuron.forward(X)
        loss = torch.mean(0.5 * (pred - y) ** 2)
        custom_losses.append(loss.item())
        
        errors = pred - y
        grad_w = torch.mean(X * errors.unsqueeze(1), dim=0)
        grad_b = torch.mean(errors)
        custom_neuron.update_weights(grad_w, grad_b, learning_rate)
    
    # Train PyTorch network
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(pytorch_net.parameters(), lr=learning_rate)
    pytorch_losses = []
    
    for epoch in range(100):
        optimizer.zero_grad()
        pred = pytorch_net(X).squeeze()
        loss = criterion(pred, y)
        pytorch_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(custom_losses, label='Custom Neuron', linewidth=2)
    plt.plot(pytorch_losses, label='PyTorch Network', linewidth=2, linestyle='--')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    diff = np.array(custom_losses) - np.array(pytorch_losses)
    plt.plot(diff)
    plt.title('Loss Difference (Custom - PyTorch)')
    plt.xlabel('Epoch')
    plt.ylabel('Difference')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Final loss - Custom: {custom_losses[-1]:.6f}")
    print(f"Final loss - PyTorch: {pytorch_losses[-1]:.6f}")
    print(f"Max loss difference: {max(abs(d) for d in diff):.8f}")


def demonstrate_biological_inspiration():
    """
    Show the biological inspiration behind artificial neurons.
    """
    print("\n" + "=" * 60)
    print("BIOLOGICAL INSPIRATION")
    print("=" * 60)
    
    print("The artificial neuron is inspired by biological neurons:")
    print()
    
    # ASCII art representation
    print("Biological Neuron:")
    print("                    Dendrites")
    print("                       |")
    print("    Signal 1 ────────┐ │ ┌──────── Synapses")
    print("    Signal 2 ────────┤ ▼ ├──────── (Weights)")
    print("    Signal 3 ────────┼─●─┼──────── Cell Body")
    print("       ...           │   │        (Summation)")
    print("    Signal n ────────┘   └──────► Axon (Output)")
    print()
    
    print("Artificial Neuron:")
    print("    x₁ ─────── w₁ ───┐")
    print("    x₂ ─────── w₂ ───┤")
    print("    x₃ ─────── w₃ ───┼─── Σ + b ─────► o₁")
    print("    ...              │")
    print("    xₙ ─────── wₙ ───┘")
    print()
    
    # Simulate neuron "firing"
    print("Simulating neuron activation:")
    neuron = LinearNeuron(input_size=4)
    
    # Weak input (below threshold)
    weak_input = torch.tensor([0.1, 0.1, 0.1, 0.1])
    weak_output = neuron.forward(weak_input)
    
    # Strong input (above threshold)  
    strong_input = torch.tensor([2.0, 1.5, 3.0, 2.5])
    strong_output = neuron.forward(strong_input)
    
    print(f"Weak stimulation:   input={weak_input.numpy()}")
    print(f"                   output={weak_output.item():.3f}")
    print(f"Strong stimulation: input={strong_input.numpy()}")
    print(f"                   output={strong_output.item():.3f}")
    
    print("\nKey similarities:")
    print("• Both receive multiple inputs")
    print("• Both weight the importance of each input")
    print("• Both sum the weighted inputs")
    print("• Both can have an internal bias/threshold")
    print("• Both produce a single output")
    
    print("\nKey differences:")
    print("• Biological neurons use spikes, artificial use continuous values")
    print("• Biological learning is more complex than simple gradient descent")
    print("• Artificial neurons are much simpler approximations")


def main():
    """
    Main demonstration function.
    """
    print("=" * 80)
    print("LINEAR REGRESSION AS A NEURAL NETWORK")
    print("=" * 80)
    print("This demonstration shows how linear regression can be viewed as")
    print("the simplest possible neural network: a single linear neuron.\n")
    
    # 1. Visualize network structure
    print("1. Network Structure Visualization")
    print("-" * 50)
    visualize_network_structure(input_size=3)
    
    # 2. Single neuron demonstration
    print("\n2. Single Neuron Operations")
    print("-" * 50)
    neuron = demonstrate_single_neuron()
    
    # 3. Training demonstration
    print("\n3. Training Process")
    print("-" * 50)
    trained_neuron, losses = train_single_neuron(n_epochs=100)
    
    # 4. Implementation comparison
    print("\n4. Implementation Comparison")
    print("-" * 50)
    compare_implementations()
    
    # 5. Biological inspiration
    print("\n5. Biological Inspiration")
    print("-" * 50)
    demonstrate_biological_inspiration()
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("1. Linear regression IS a neural network (the simplest one)")
    print("2. A single neuron computes: output = weights^T * inputs + bias")
    print("3. Training adjusts weights and bias to minimize prediction error")
    print("4. This is the foundation for understanding deep learning")
    print("5. Modern neural networks are just many of these units connected")
    print("6. The biological inspiration provides intuition for the math")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run the demonstration
    main()