"""
Vectorization for Speed Demonstration
====================================

This module demonstrates the importance of vectorization in machine learning.
It compares for-loop implementations vs vectorized operations to show 
the dramatic speed improvements possible.

Based on the materials in 2.VectorizationForSpeed.md
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Tuple


def time_function(func, *args, **kwargs) -> Tuple[float, any]:
    """
    Time the execution of a function and return both time and result.
    
    Args:
        func: Function to time
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Tuple of (execution_time, result)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result


def vector_addition_loop(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Add two vectors using a for-loop (SLOW approach).
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Sum of vectors computed element-wise in a loop
    """
    n = len(a)
    result = torch.zeros(n)
    for i in range(n):
        result[i] = a[i] + b[i]
    return result


def vector_addition_vectorized(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Add two vectors using vectorized operations (FAST approach).
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Sum of vectors computed using vectorized operations
    """
    return a + b


def matrix_multiplication_loops(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Multiply two matrices using nested for-loops (VERY SLOW approach).
    
    Args:
        A: First matrix of shape (m, k)
        B: Second matrix of shape (k, n)
        
    Returns:
        Product matrix of shape (m, n)
    """
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, "Matrix dimensions must match for multiplication"
    
    result = torch.zeros(m, n)
    for i in range(m):
        for j in range(n):
            for l in range(k):
                result[i, j] += A[i, l] * B[l, j]
    return result


def matrix_multiplication_vectorized(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Multiply two matrices using vectorized operations (FAST approach).
    
    Args:
        A: First matrix
        B: Second matrix
        
    Returns:
        Product matrix computed using optimized linear algebra
    """
    return torch.mm(A, B)


def linear_regression_prediction_loop(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Make predictions using a for-loop (SLOW approach).
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        w: Weight vector of shape (n_features,)
        b: Bias scalar
        
    Returns:
        Predictions computed element-wise in loops
    """
    n_samples, n_features = X.shape
    predictions = torch.zeros(n_samples)
    
    for i in range(n_samples):
        prediction = b
        for j in range(n_features):
            prediction += X[i, j] * w[j]
        predictions[i] = prediction
    
    return predictions


def linear_regression_prediction_vectorized(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Make predictions using vectorized operations (FAST approach).
    
    Args:
        X: Feature matrix
        w: Weight vector
        b: Bias scalar
        
    Returns:
        Predictions computed using matrix multiplication
    """
    return X @ w + b


def benchmark_operations(sizes: List[int]) -> dict:
    """
    Benchmark different operations across various sizes.
    
    Args:
        sizes: List of vector/matrix sizes to test
        
    Returns:
        Dictionary containing benchmark results
    """
    results = {
        'sizes': sizes,
        'vector_add_loop': [],
        'vector_add_vectorized': [],
        'matrix_mult_loop': [],
        'matrix_mult_vectorized': [],
        'linreg_loop': [],
        'linreg_vectorized': []
    }
    
    print("Benchmarking vectorization vs for-loops...")
    print("Size\t\tVec Add\t\tMatrix Mult\t\tLinear Reg")
    print("-" * 70)
    
    for size in sizes:
        # Create test data
        a = torch.ones(size)
        b = torch.ones(size)
        A = torch.randn(size // 10, size // 10)  # Smaller matrices for multiplication
        B = torch.randn(size // 10, size // 10)
        X = torch.randn(size, min(size // 100, 10))  # Reasonable feature count
        w = torch.randn(X.shape[1])
        bias = torch.randn(1)
        
        # Vector addition benchmark
        time_loop, _ = time_function(vector_addition_loop, a, b)
        time_vec, _ = time_function(vector_addition_vectorized, a, b)
        results['vector_add_loop'].append(time_loop)
        results['vector_add_vectorized'].append(time_vec)
        
        # Matrix multiplication benchmark (only for smaller sizes)
        if size <= 1000:  # Avoid extremely slow nested loops
            time_loop, _ = time_function(matrix_multiplication_loops, A, B)
            time_vec, _ = time_function(matrix_multiplication_vectorized, A, B)
            results['matrix_mult_loop'].append(time_loop)
            results['matrix_mult_vectorized'].append(time_vec)
        else:
            results['matrix_mult_loop'].append(None)
            results['matrix_mult_vectorized'].append(None)
        
        # Linear regression prediction benchmark
        time_loop, _ = time_function(linear_regression_prediction_loop, X, w, bias)
        time_vec, _ = time_function(linear_regression_prediction_vectorized, X, w, bias)
        results['linreg_loop'].append(time_loop)
        results['linreg_vectorized'].append(time_vec)
        
        # Print results
        vec_speedup = f"{time_loop/time_vec:.1f}x" if time_vec > 0 else "N/A"
        
        if size <= 1000:
            mat_speedup = f"{results['matrix_mult_loop'][-1]/results['matrix_mult_vectorized'][-1]:.1f}x"
        else:
            mat_speedup = "Skip"
        
        linreg_speedup = f"{time_loop/time_vec:.1f}x" if time_vec > 0 else "N/A"
        
        print(f"{size}\t\t{vec_speedup}\t\t{mat_speedup}\t\t{linreg_speedup}")
    
    return results


def plot_benchmark_results(results: dict):
    """
    Plot the benchmark results to visualize the speedup.
    
    Args:
        results: Dictionary containing benchmark results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Vectorization vs For-Loop Performance Comparison', fontsize=16)
    
    sizes = results['sizes']
    
    # Vector Addition Plot
    axes[0, 0].loglog(sizes, results['vector_add_loop'], 'r-o', label='For-loop', markersize=4)
    axes[0, 0].loglog(sizes, results['vector_add_vectorized'], 'b-s', label='Vectorized', markersize=4)
    axes[0, 0].set_title('Vector Addition')
    axes[0, 0].set_xlabel('Vector Size')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Matrix Multiplication Plot (filter out None values)
    valid_indices = [i for i, x in enumerate(results['matrix_mult_loop']) if x is not None]
    if valid_indices:
        valid_sizes = [sizes[i] for i in valid_indices]
        valid_loop = [results['matrix_mult_loop'][i] for i in valid_indices]
        valid_vec = [results['matrix_mult_vectorized'][i] for i in valid_indices]
        
        axes[0, 1].loglog(valid_sizes, valid_loop, 'r-o', label='For-loop', markersize=4)
        axes[0, 1].loglog(valid_sizes, valid_vec, 'b-s', label='Vectorized', markersize=4)
    
    axes[0, 1].set_title('Matrix Multiplication')
    axes[0, 1].set_xlabel('Matrix Size')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Linear Regression Plot
    axes[1, 0].loglog(sizes, results['linreg_loop'], 'r-o', label='For-loop', markersize=4)
    axes[1, 0].loglog(sizes, results['linreg_vectorized'], 'b-s', label='Vectorized', markersize=4)
    axes[1, 0].set_title('Linear Regression Predictions')
    axes[1, 0].set_xlabel('Number of Samples')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Speedup Plot
    speedups_vec = [l/v for l, v in zip(results['vector_add_loop'], results['vector_add_vectorized']) if v > 0]
    speedups_linreg = [l/v for l, v in zip(results['linreg_loop'], results['linreg_vectorized']) if v > 0]
    
    axes[1, 1].semilogx(sizes, speedups_vec, 'g-o', label='Vector Addition', markersize=4)
    axes[1, 1].semilogx(sizes, speedups_linreg, 'purple', marker='s', linestyle='-', 
                       label='Linear Regression', markersize=4)
    axes[1, 1].axhline(y=1, color='k', linestyle='--', alpha=0.5, label='No speedup')
    axes[1, 1].set_title('Speedup (Loop time / Vectorized time)')
    axes[1, 1].set_xlabel('Size')
    axes[1, 1].set_ylabel('Speedup Factor')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demonstrate_memory_efficiency():
    """
    Demonstrate that vectorized operations are also more memory efficient.
    """
    print("\n" + "=" * 60)
    print("Memory Efficiency Demonstration")
    print("=" * 60)
    
    size = 10000
    
    # Create test vectors
    a = torch.randn(size)
    b = torch.randn(size)
    
    print(f"Vector size: {size}")
    print(f"Memory per vector: {a.nbytes / 1024:.2f} KB")
    
    # Measure memory usage for loop version
    import tracemalloc
    
    tracemalloc.start()
    result_loop = vector_addition_loop(a, b)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Loop version peak memory: {peak / 1024:.2f} KB")
    
    # Measure memory usage for vectorized version
    tracemalloc.start()
    result_vec = vector_addition_vectorized(a, b)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Vectorized version peak memory: {peak / 1024:.2f} KB")
    
    # Verify results are the same
    print(f"Results are equal: {torch.allclose(result_loop, result_vec)}")


def main():
    """
    Main demonstration function showing the power of vectorization.
    """
    print("=" * 70)
    print("VECTORIZATION FOR SPEED - DEMONSTRATION")
    print("=" * 70)
    print("This demo shows why we use vectorized operations instead of for-loops")
    print("in machine learning computations.\n")
    
    # Simple demonstration from the materials
    print("1. Simple Vector Addition (from materials)")
    print("-" * 50)
    
    n = 10000
    a = torch.ones(n)
    b = torch.ones(n)
    
    # For-loop version
    time_loop, result_loop = time_function(vector_addition_loop, a, b)
    print(f'For-loop: {time_loop:.5f} sec')
    
    # Vectorized version
    time_vec, result_vec = time_function(vector_addition_vectorized, a, b)
    print(f'Vectorized: {time_vec:.5f} sec')
    
    speedup = time_loop / time_vec if time_vec > 0 else float('inf')
    print(f'Speedup: {speedup:.1f}x faster')
    print(f'Results equal: {torch.allclose(result_loop, result_vec)}')
    
    # Comprehensive benchmark
    print(f"\n2. Comprehensive Benchmark")
    print("-" * 50)
    
    sizes = [100, 500, 1000, 5000, 10000]
    results = benchmark_operations(sizes)
    
    # Plot results
    print(f"\n3. Plotting Results...")
    plot_benchmark_results(results)
    
    # Memory efficiency demo
    demonstrate_memory_efficiency()
    
    print(f"\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("1. Vectorized operations are typically 10-100x faster")
    print("2. The speedup increases with data size")
    print("3. Vectorized code is more readable and less error-prone")
    print("4. Libraries like PyTorch/NumPy use optimized C/CUDA code")
    print("5. Always prefer vectorized operations in machine learning!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the demonstration
    main()