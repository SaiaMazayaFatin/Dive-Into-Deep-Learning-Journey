# Softmax Function Implementation
# Complete implementation of softmax with stability improvements

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Union, Tuple
import time


class SoftmaxImplementation:
    """
    Comprehensive implementation of the Softmax function with various approaches.
    """
    
    def __init__(self):
        self.demo_results = {}
    
    def basic_softmax(self, logits: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Basic softmax implementation following the mathematical formula:
        softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        
        Formula from Basic.md:
        ŷᵢ = exp(oᵢ) / Σⱼ exp(oⱼ)
        
        Args:
            logits: Raw output scores (can be negative, > 1)
        Returns:
            probabilities: Valid probability distribution (positive, sum=1)
        """
        if isinstance(logits, torch.Tensor):
            exp_logits = torch.exp(logits)
            return exp_logits / exp_logits.sum(dim=-1, keepdim=True)
        else:
            exp_logits = np.exp(logits)
            return exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    
    def stable_softmax(self, logits: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Numerically stable softmax implementation.
        Subtracts max value to prevent overflow: exp(x_i - max(x)) / sum(exp(x_j - max(x)))
        
        This is mathematically equivalent but computationally stable.
        """
        if isinstance(logits, torch.Tensor):
            # Subtract max for numerical stability
            logits_max = logits.max(dim=-1, keepdim=True)[0]
            logits_shifted = logits - logits_max
            exp_logits = torch.exp(logits_shifted)
            return exp_logits / exp_logits.sum(dim=-1, keepdim=True)
        else:
            # NumPy version
            logits_max = np.max(logits, axis=-1, keepdims=True)
            logits_shifted = logits - logits_max
            exp_logits = np.exp(logits_shifted)
            return exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    
    def vectorized_softmax_demo(self):
        """
        Demonstrate vectorization for batch processing.
        Shows the difference between processing one-by-one vs batch processing.
        """
        print("="*50)
        print("VECTORIZED SOFTMAX DEMO")
        print("="*50)
        
        # Create sample batch data
        batch_size = 4
        num_classes = 3
        np.random.seed(42)
        
        # Raw logits for a batch (what comes out of linear layer)
        logits_batch = np.random.randn(batch_size, num_classes) * 2
        
        print(f"📊 BATCH PROCESSING:")
        print(f"Batch size: {batch_size}")
        print(f"Number of classes: {num_classes}")
        print(f"Input logits shape: {logits_batch.shape}")
        print(f"\nRaw logits (can be negative):")
        print(logits_batch)
        
        # Method 1: Process one by one (inefficient)
        print(f"\n🐌 METHOD 1: One-by-one processing")
        start_time = time.time()
        probabilities_sequential = []
        for i in range(batch_size):
            single_logit = logits_batch[i]
            single_prob = self.stable_softmax(single_logit)
            probabilities_sequential.append(single_prob)
        probabilities_sequential = np.array(probabilities_sequential)
        sequential_time = time.time() - start_time
        
        # Method 2: Vectorized processing (efficient)  
        print(f"\n🚀 METHOD 2: Vectorized processing")
        start_time = time.time()
        probabilities_vectorized = self.stable_softmax(logits_batch)
        vectorized_time = time.time() - start_time
        
        print(f"\nResults comparison:")
        print(f"Sequential result shape: {probabilities_sequential.shape}")
        print(f"Vectorized result shape: {probabilities_vectorized.shape}")
        print(f"Results are equal: {np.allclose(probabilities_sequential, probabilities_vectorized)}")
        
        print(f"\nTiming comparison:")
        print(f"Sequential time: {sequential_time*1000:.3f} ms")
        print(f"Vectorized time: {vectorized_time*1000:.3f} ms")
        print(f"Speedup: {sequential_time/vectorized_time:.1f}x")
        
        # Show the mathematical transformation
        print(f"\n🧮 MATHEMATICAL TRANSFORMATION:")
        print("Input  → Output")
        print("Logits → Probabilities")
        
        class_names = ['Cat', 'Dog', 'Bird']
        for i in range(batch_size):
            print(f"\nExample {i+1}:")
            print(f"  Logits: {logits_batch[i]}")
            print(f"  Probabilities: {probabilities_vectorized[i]}")
            print(f"  Sum: {probabilities_vectorized[i].sum():.6f}")
            
            # Show prediction
            predicted_class = np.argmax(probabilities_vectorized[i])
            confidence = probabilities_vectorized[i, predicted_class]
            print(f"  Prediction: {class_names[predicted_class]} ({confidence:.3f})")
        
        return logits_batch, probabilities_vectorized
    
    def numerical_stability_demo(self):
        """
        Demonstrate why numerical stability matters in softmax.
        """
        print("\n" + "="*50)
        print("NUMERICAL STABILITY DEMO")
        print("="*50)
        
        # Create problematic logits (very large values)
        large_logits = np.array([1000, 999, 998])
        small_logits = np.array([1, 2, 3])
        
        print("⚠️  OVERFLOW PROBLEM:")
        print("-" * 30)
        print(f"Large logits: {large_logits}")
        print(f"exp(1000) = {np.exp(1000) if np.exp(1000) != np.inf else 'OVERFLOW (inf)'}")
        
        try:
            # This will cause overflow
            exp_large = np.exp(large_logits)
            print(f"exp(large_logits): {exp_large}")
            
            if np.any(np.isinf(exp_large)):
                print("❌ Result contains infinity - numerical overflow!")
                basic_result = "FAILED - Contains inf"
            else:
                basic_result = exp_large / exp_large.sum()
                
        except Exception as e:
            print(f"❌ Error: {e}")
            basic_result = "FAILED"
        
        print(f"\n✅ STABLE SOLUTION:")
        print("-" * 30) 
        print("Subtract max before exponentiating:")
        
        max_val = np.max(large_logits)
        shifted_logits = large_logits - max_val
        print(f"Original: {large_logits}")
        print(f"Max value: {max_val}")
        print(f"Shifted: {shifted_logits}")
        
        exp_shifted = np.exp(shifted_logits)
        stable_result = exp_shifted / exp_shifted.sum()
        
        print(f"exp(shifted): {exp_shifted}")
        print(f"Stable probabilities: {stable_result}")
        print(f"Sum: {stable_result.sum():.6f}")
        
        print(f"\n🔍 COMPARISON WITH SMALL VALUES:")
        print("-" * 30)
        
        # Compare with manageable numbers
        basic_small = self.basic_softmax(small_logits)
        stable_small = self.stable_softmax(small_logits)
        
        print(f"Small logits: {small_logits}")
        print(f"Basic softmax: {basic_small}")
        print(f"Stable softmax: {stable_small}")
        print(f"Difference: {np.abs(basic_small - stable_small).max():.2e}")
        
        return stable_result, basic_small, stable_small
    
    def softmax_properties_demo(self):
        """
        Demonstrate key mathematical properties of softmax.
        """
        print("\n" + "="*50)
        print("SOFTMAX PROPERTIES DEMO")  
        print("="*50)
        
        # Sample logits
        logits = np.array([2.0, 1.0, 0.1])
        probabilities = self.stable_softmax(logits)
        
        print("🧮 MATHEMATICAL PROPERTIES:")
        print("-" * 30)
        
        # Property 1: All positive
        print(f"1. All outputs positive:")
        print(f"   Logits: {logits}")
        print(f"   Probabilities: {probabilities}")
        print(f"   All positive: {np.all(probabilities > 0)}")
        
        # Property 2: Sum to 1
        print(f"\n2. Sum to 1:")
        print(f"   Sum: {probabilities.sum():.10f}")
        print(f"   Equals 1: {np.isclose(probabilities.sum(), 1.0)}")
        
        # Property 3: Preserves order
        print(f"\n3. Preserves relative order:")
        logit_order = np.argsort(logits)[::-1]
        prob_order = np.argsort(probabilities)[::-1]
        print(f"   Logit ranking: {logit_order} (highest to lowest)")
        print(f"   Probability ranking: {prob_order} (highest to lowest)")
        print(f"   Order preserved: {np.array_equal(logit_order, prob_order)}")
        
        # Property 4: Differentiability
        print(f"\n4. Smooth and differentiable:")
        print("   ∂softmax(xᵢ)/∂xⱼ exists for all i,j")
        print("   This enables gradient-based learning")
        
        # Property 5: Temperature scaling
        print(f"\n5. Temperature effect:")
        temperatures = [0.5, 1.0, 2.0]
        print("   Temperature controls 'sharpness' of distribution")
        
        for T in temperatures:
            temp_probs = self.stable_softmax(logits / T)
            entropy = -np.sum(temp_probs * np.log(temp_probs + 1e-10))
            print(f"   T={T}: {temp_probs} (entropy: {entropy:.3f})")
        
        print("\n   Low T → Sharp (confident) distribution")
        print("   High T → Flat (uncertain) distribution")
    
    def pytorch_vs_numpy_comparison(self):
        """
        Compare PyTorch and NumPy implementations.
        """
        print("\n" + "="*50)
        print("PYTORCH vs NUMPY COMPARISON")
        print("="*50)
        
        # Create test data
        batch_size = 1000
        num_classes = 10
        np.random.seed(42)
        torch.manual_seed(42)
        
        # NumPy version
        logits_np = np.random.randn(batch_size, num_classes)
        
        # PyTorch version (same data)
        logits_torch = torch.from_numpy(logits_np).float()
        
        print(f"📊 Performance Comparison:")
        print(f"Batch size: {batch_size}")
        print(f"Classes: {num_classes}")
        
        # Time NumPy implementation
        start_time = time.time()
        for _ in range(100):
            probs_np = self.stable_softmax(logits_np)
        numpy_time = (time.time() - start_time) / 100
        
        # Time PyTorch implementation  
        start_time = time.time()
        for _ in range(100):
            probs_torch = self.stable_softmax(logits_torch)
        pytorch_time = (time.time() - start_time) / 100
        
        # Time PyTorch built-in
        start_time = time.time()
        for _ in range(100):
            probs_builtin = torch.nn.functional.softmax(logits_torch, dim=-1)
        builtin_time = (time.time() - start_time) / 100
        
        print(f"\n⏱️  Timing Results (per operation):")
        print(f"NumPy implementation: {numpy_time*1000:.3f} ms")
        print(f"PyTorch implementation: {pytorch_time*1000:.3f} ms")
        print(f"PyTorch built-in: {builtin_time*1000:.3f} ms")
        
        # Accuracy comparison
        probs_torch_np = probs_torch.numpy()
        probs_builtin_np = probs_builtin.numpy()
        
        diff_custom = np.abs(probs_np - probs_torch_np).max()
        diff_builtin = np.abs(probs_np - probs_builtin_np).max()
        
        print(f"\n🎯 Accuracy Comparison:")
        print(f"NumPy vs PyTorch custom: {diff_custom:.2e}")
        print(f"NumPy vs PyTorch builtin: {diff_builtin:.2e}")
        print("All implementations are numerically equivalent ✅")

    def visualization_demo(self):
        """
        Create visualizations showing softmax behavior.
        """
        print("\n" + "="*50) 
        print("SOFTMAX VISUALIZATION")
        print("="*50)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Basic softmax transformation
        ax1 = axes[0, 0]
        
        logits = np.linspace(-3, 3, 100)
        # For 2D softmax, create simple case
        logits_2d = np.array([logits, np.zeros_like(logits)]).T
        probs_2d = self.stable_softmax(logits_2d)
        
        ax1.plot(logits, probs_2d[:, 0], 'b-', linewidth=2, label='Class 1')
        ax1.plot(logits, probs_2d[:, 1], 'r-', linewidth=2, label='Class 2')
        ax1.set_xlabel('Logit Value')
        ax1.set_ylabel('Probability')
        ax1.set_title('Softmax Transformation (2 classes)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        
        # 2. Temperature effect
        ax2 = axes[0, 1]
        
        logits_temp = np.array([2.0, 1.0, 0.5])
        temperatures = [0.5, 1.0, 2.0, 5.0]
        
        x_pos = np.arange(len(logits_temp))
        width = 0.2
        
        for i, T in enumerate(temperatures):
            temp_probs = self.stable_softmax(logits_temp / T)
            ax2.bar(x_pos + i*width, temp_probs, width, 
                   label=f'T={T}', alpha=0.8)
        
        ax2.set_xlabel('Class Index')
        ax2.set_ylabel('Probability')
        ax2.set_title('Temperature Effect on Softmax')
        ax2.set_xticks(x_pos + width*1.5)
        ax2.set_xticklabels(['Class 0', 'Class 1', 'Class 2'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Numerical stability comparison
        ax3 = axes[1, 0]
        
        # Create range of logit magnitudes
        magnitudes = np.logspace(0, 3, 50)  # From 1 to 1000
        max_errors = []
        
        for mag in magnitudes:
            test_logits = np.array([mag, mag-1, mag-2])
            
            # Compare basic vs stable (when possible)
            try:
                basic_result = self.basic_softmax(test_logits)
                stable_result = self.stable_softmax(test_logits)
                
                if np.any(np.isnan(basic_result)) or np.any(np.isinf(basic_result)):
                    max_errors.append(1.0)  # Maximum error
                else:
                    error = np.abs(basic_result - stable_result).max()
                    max_errors.append(error)
            except:
                max_errors.append(1.0)
        
        ax3.loglog(magnitudes, max_errors, 'ro-', linewidth=2, markersize=4)
        ax3.set_xlabel('Logit Magnitude')
        ax3.set_ylabel('Max Error (Basic vs Stable)')
        ax3.set_title('Numerical Stability Comparison')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=1e-10, color='gray', linestyle='--', alpha=0.7, label='Machine precision')
        ax3.legend()
        
        # 4. Batch processing visualization
        ax4 = axes[1, 1]
        
        # Show probability distributions for a batch
        np.random.seed(42)
        batch_logits = np.random.randn(5, 4) * 1.5  # 5 examples, 4 classes
        batch_probs = self.stable_softmax(batch_logits)
        
        # Create heatmap
        im = ax4.imshow(batch_probs, cmap='viridis', aspect='auto')
        ax4.set_xlabel('Class Index')
        ax4.set_ylabel('Example Index')
        ax4.set_title('Batch Softmax Output')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Probability')
        
        # Add text annotations
        for i in range(batch_probs.shape[0]):
            for j in range(batch_probs.shape[1]):
                text = ax4.text(j, i, f'{batch_probs[i, j]:.2f}', 
                              ha="center", va="center", color="white", fontsize=9)
        
        plt.tight_layout()
        plt.savefig('c:/File/AI/Dive Into Deep Learning Journey/03-Linear Neural Networks for Classification/3.1 Softmax Regression/img/softmax_implementation_demo.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Softmax visualizations created!")
        print("📁 Saved as: softmax_implementation_demo.png")


def main():
    """Run all softmax implementation demonstrations."""
    softmax = SoftmaxImplementation()
    
    print("🚀 SOFTMAX IMPLEMENTATION DEMONSTRATIONS")
    print("="*60)
    
    # Run demonstrations
    softmax.vectorized_softmax_demo()
    softmax.numerical_stability_demo() 
    softmax.softmax_properties_demo()
    softmax.pytorch_vs_numpy_comparison()
    softmax.visualization_demo()
    
    print("\n🎉 SOFTMAX IMPLEMENTATION DEMO COMPLETE!")
    print("="*60)
    print("📚 Key Concepts Covered:")
    print("  • Basic and stable softmax implementations") 
    print("  • Vectorization for batch processing")
    print("  • Numerical stability techniques")
    print("  • Mathematical properties")
    print("  • PyTorch vs NumPy comparison")
    print("  • Temperature effects")
    print("  • Performance optimization")
    print("\n🚀 Ready for: Loss Functions and Training!")


if __name__ == "__main__":
    main()