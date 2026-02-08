"""
Vectorization for Speed Demonstration
====================================

This module demonstrates the speed advantages of vectorized operations over
Python loops, as discussed in the Basic.md file.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple


def timing_decorator(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__}: {execution_time:.6f} seconds")
        return result, execution_time
    return wrapper


@timing_decorator
def vector_addition_loop(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Add two vectors using a Python for-loop (slow method)."""
    n = len(a)
    c = torch.zeros(n)
    for i in range(n):
        c[i] = a[i] + b[i]
    return c


@timing_decorator
def vector_addition_vectorized(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Add two vectors using vectorized operation (fast method)."""
    return a + b


@timing_decorator
def matrix_multiplication_loop(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication using nested loops (very slow)."""
    n, k = A.shape
    k2, m = B.shape
    assert k == k2, "Matrix dimensions don't match"
    
    C = torch.zeros(n, m)
    for i in range(n):
        for j in range(m):
            for l in range(k):
                C[i, j] += A[i, l] * B[l, j]
    return C


@timing_decorator
def matrix_multiplication_vectorized(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication using vectorized operation (fast)."""
    return torch.mm(A, B)


@timing_decorator
def linear_regression_prediction_loop(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Linear regression predictions using loops (slow)."""
    n_samples, n_features = X.shape
    predictions = torch.zeros(n_samples, 1)
    
    for i in range(n_samples):
        pred = b.clone()
        for j in range(n_features):
            pred += X[i, j] * w[j]
        predictions[i] = pred
    
    return predictions


@timing_decorator
def linear_regression_prediction_vectorized(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Linear regression predictions using vectorized operations (fast)."""
    return torch.mm(X, w) + b


def gradient_computation_comparison():
    """Compare gradient computation methods for linear regression."""
    print("\nGradient Computation Comparison")
    print("=" * 40)
    
    # Generate sample data
    n_samples, n_features = 1000, 10
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, 1)
    w = torch.randn(n_features, 1)
    b = torch.randn(1)
    
    @timing_decorator
    def compute_gradients_loop(X, y, w, b):
        """Compute gradients using loops."""
        n_samples = X.shape[0]
        w_grad = torch.zeros_like(w)
        b_grad = torch.zeros_like(b)
        
        for i in range(n_samples):
            # Forward pass for sample i
            pred = b.clone()
            for j in range(len(w)):
                pred += X[i, j] * w[j]
            
            # Error for sample i
            error = pred - y[i]
            
            # Accumulate gradients
            b_grad += error
            for j in range(len(w)):
                w_grad[j] += error * X[i, j]
        
        return w_grad / n_samples, b_grad / n_samples
    
    @timing_decorator
    def compute_gradients_vectorized(X, y, w, b):
        """Compute gradients using vectorized operations."""
        # Forward pass (vectorized)
        predictions = torch.mm(X, w) + b
        
        # Compute errors (vectorized)
        errors = predictions - y
        
        # Compute gradients (vectorized)
        w_grad = torch.mm(X.T, errors) / X.shape[0]
        b_grad = torch.mean(errors)
        
        return w_grad, b_grad
    
    # Test both methods
    w_grad_loop, b_grad_loop, time_loop = compute_gradients_loop(X, y, w, b)
    w_grad_vec, b_grad_vec, time_vec = compute_gradients_vectorized(X, y, w, b)
    
    # Check if results are close
    w_close = torch.allclose(w_grad_loop, w_grad_vec, atol=1e-6)
    b_close = torch.allclose(b_grad_loop, b_grad_vec, atol=1e-6)
    
    print(f"Gradient results match: w={w_close}, b={b_close}")
    print(f"Speedup: {time_loop / time_vec:.1f}x faster with vectorization")


def speed_comparison_demo():
    """Comprehensive speed comparison demonstration."""
    print("Vectorization Speed Comparison")
    print("=" * 50)
    
    # Test different vector sizes
    sizes = [1000, 5000, 10000, 50000, 100000]
    loop_times = []
    vectorized_times = []
    
    print("\n1. Vector Addition Comparison:")
    for n in sizes:
        print(f"\nVector size: {n}")
        a = torch.ones(n)
        b = torch.ones(n)
        
        # Time both methods
        _, loop_time = vector_addition_loop(a, b)
        _, vec_time = vector_addition_vectorized(a, b)
        
        loop_times.append(loop_time)
        vectorized_times.append(vec_time)
        
        speedup = loop_time / vec_time
        print(f"Speedup: {speedup:.1f}x")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(sizes, loop_times, 'o-', label='For-loop', linewidth=2)
    plt.plot(sizes, vectorized_times, 's-', label='Vectorized', linewidth=2)
    plt.xlabel('Vector Size')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    speedups = [l/v for l, v in zip(loop_times, vectorized_times)]
    plt.plot(sizes, speedups, 'g^-', linewidth=2, markersize=8)
    plt.xlabel('Vector Size')
    plt.ylabel('Speedup Factor')
    plt.title('Vectorization Speedup')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nAverage speedup: {np.mean(speedups):.1f}x")


def matrix_operations_demo():
    """Demonstrate vectorization for matrix operations."""
    print("\n\n2. Matrix Operations Comparison:")
    
    # Test matrix multiplication
    sizes = [(100, 100), (200, 200), (500, 500)]
    
    for n, m in sizes:
        print(f"\nMatrix size: {n}×{n} × {n}×{m}")
        A = torch.randn(n, n)
        B = torch.randn(n, m)
        
        # Only test small matrices with loops (too slow otherwise)
        if n <= 200:
            _, loop_time = matrix_multiplication_loop(A, B)
        else:
            print("matrix_multiplication_loop: Skipped (too slow)")
            loop_time = float('inf')
            
        _, vec_time = matrix_multiplication_vectorized(A, B)
        
        if loop_time != float('inf'):
            speedup = loop_time / vec_time
            print(f"Speedup: {speedup:.1f}x")


def linear_regression_demo():
    """Demonstrate vectorization in linear regression context."""
    print("\n\n3. Linear Regression Predictions:")
    
    # Test with different data sizes
    test_sizes = [(1000, 10), (5000, 20), (10000, 50)]
    
    for n_samples, n_features in test_sizes:
        print(f"\nData size: {n_samples} samples, {n_features} features")
        
        X = torch.randn(n_samples, n_features)
        w = torch.randn(n_features, 1)
        b = torch.randn(1)
        
        # Compare prediction methods
        _, loop_time = linear_regression_prediction_loop(X, w, b)
        _, vec_time = linear_regression_prediction_vectorized(X, w, b)
        
        speedup = loop_time / vec_time
        print(f"Speedup: {speedup:.1f}x")


def memory_efficiency_demo():
    """Demonstrate memory efficiency of vectorized operations."""
    print("\n\n4. Memory Efficiency Comparison:")
    
    n = 10000
    X = torch.randn(n, 5)
    w = torch.randn(5, 1)
    
    # Simulate memory usage by creating temporary variables
    def memory_inefficient_prediction(X, w):
        """Simulate memory-inefficient approach."""
        results = []
        for i in range(len(X)):
            temp_result = torch.zeros(1)
            for j in range(len(w)):
                temp_result += X[i, j] * w[j]
            results.append(temp_result)
        return torch.stack(results)
    
    def memory_efficient_prediction(X, w):
        """Memory-efficient vectorized approach."""
        return torch.mm(X, w)
    
    print("Memory usage patterns:")
    print("- Loop method: Creates many temporary variables")
    print("- Vectorized method: Single operation, minimal temporaries")
    print("- Vectorized operations are optimized at the C/CUDA level")


def practical_tips():
    """Print practical tips for vectorization."""
    print("\n\nPractical Vectorization Tips:")
    print("=" * 40)
    print("1. Replace explicit loops with tensor operations")
    print("2. Use torch.mm() for matrix multiplication")
    print("3. Use element-wise operations (+, -, *, /) directly on tensors")
    print("4. Utilize broadcasting for operations on different-sized tensors")
    print("5. Use torch.sum(), torch.mean() instead of manual loops")
    print("6. Batch operations when possible (process multiple samples together)")
    print("7. Profile your code to identify bottlenecks")


if __name__ == "__main__":
    speed_comparison_demo()
    matrix_operations_demo()
    linear_regression_demo()
    gradient_computation_comparison()
    memory_efficiency_demo()
    practical_tips()