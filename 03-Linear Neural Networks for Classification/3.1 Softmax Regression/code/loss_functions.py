# Loss Functions and Information Theory
# Complete implementation of cross-entropy, gradients, and information theory concepts

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Union, Tuple, List
import seaborn as sns


class LossAndInfoTheory:
    """
    Implementation of loss functions and information theory concepts for classification.
    """
    
    def __init__(self):
        self.results = {}
    
    def cross_entropy_from_scratch(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute cross-entropy loss from scratch using the mathematical formula.
        
        Formula from Basic.md:
        L = -Σ yⱼ log(ŷⱼ)
        
        Args:
            predictions: Predicted probabilities (output of softmax)
            targets: True labels (one-hot encoded)
        Returns:
            Cross-entropy loss value
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Cross-entropy formula: -sum(y * log(y_hat))
        return -np.sum(targets * np.log(predictions))
    
    def stable_cross_entropy(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Numerically stable cross-entropy implementation using log-sum-exp trick.
        
        This combines softmax + cross-entropy into a single stable operation.
        Derivation from Basic.md Section 3:
        L = log(Σ exp(oₖ)) - Σ yⱼoⱼ
        """
        # Log-sum-exp trick for numerical stability
        max_logits = np.max(logits)
        log_sum_exp = max_logits + np.log(np.sum(np.exp(logits - max_logits)))
        
        # Linear term: sum of y_j * o_j
        linear_term = np.sum(targets * logits)
        
        # Combined formula
        return log_sum_exp - linear_term
    
    def gradient_computation_demo(self):
        """
        Demonstrate gradient computation for softmax + cross-entropy.
        """
        print("="*50)
        print("GRADIENT COMPUTATION DEMO")
        print("="*50)
        
        # Sample data
        logits = np.array([2.0, 1.0, 0.1])
        targets = np.array([1, 0, 0])  # True class is index 0
        
        print("📊 INPUT DATA:")
        print(f"Logits: {logits}")
        print(f"True class: {targets} (one-hot)")
        print(f"True class index: {np.argmax(targets)}")
        
        # Compute softmax probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Stable version
        probabilities = exp_logits / np.sum(exp_logits)
        
        print(f"\n🧮 SOFTMAX COMPUTATION:")
        print(f"Probabilities: {probabilities}")
        print(f"Sum: {probabilities.sum():.6f}")
        print(f"Predicted class: {np.argmax(probabilities)}")
        
        # Compute loss
        loss = self.cross_entropy_from_scratch(probabilities, targets)
        print(f"\n💸 LOSS COMPUTATION:")
        print(f"Cross-entropy loss: {loss:.4f}")
        
        # Compute gradients - the key insight!
        gradients = probabilities - targets
        
        print(f"\n🎯 GRADIENT COMPUTATION:")
        print("Formula: ∂L/∂oⱼ = ŷⱼ - yⱼ")
        print(f"Gradients: {gradients}")
        
        print(f"\n🔍 GRADIENT INTERPRETATION:")
        class_names = ['Cat', 'Dog', 'Bird']
        for i, (pred, true, grad) in enumerate(zip(probabilities, targets, gradients)):
            print(f"{class_names[i]}:")
            print(f"  Predicted: {pred:.3f}")
            print(f"  True: {true}")
            print(f"  Gradient: {grad:+.3f}")
            
            if true == 1:
                print(f"  → Correct class: negative gradient (increase logits)")
            elif pred > 0.1:
                print(f"  → Wrong class with high confidence: positive gradient (decrease logits)")
            else:
                print(f"  → Wrong class with low confidence: small positive gradient")
        
        return logits, targets, probabilities, gradients
    
    def information_theory_demo(self):
        """
        Demonstrate information theory concepts: entropy, surprisal, cross-entropy.
        """
        print("\n" + "="*50)
        print("INFORMATION THEORY DEMO")
        print("="*50)
        
        print("📚 INFORMATION THEORY CONCEPTS:")
        print("• Surprisal: -log(p) - How 'surprised' we are by an event")
        print("• Entropy: Expected surprisal - Uncertainty in a distribution") 
        print("• Cross-entropy: Expected surprisal using wrong distribution")
        
        # Surprisal examples
        print(f"\n🎯 SURPRISAL EXAMPLES:")
        events = [
            ("Sun rises tomorrow", 0.9999),
            ("Coin flip heads", 0.5),
            ("Lottery win", 0.0001)
        ]
        
        for event, prob in events:
            surprisal = -np.log(prob)
            print(f"{event:20s}: P={prob:6.4f} → Surprisal={surprisal:6.2f} nats")
        
        print("\nInterpretation:")
        print("• High probability → Low surprisal (not surprising)")
        print("• Low probability → High surprisal (very surprising)")
        
        # Entropy calculation
        print(f"\n📊 ENTROPY EXAMPLES:")
        
        # Fair coin (maximum uncertainty)
        fair_coin = np.array([0.5, 0.5])
        entropy_fair = -np.sum(fair_coin * np.log(fair_coin))
        
        # Biased coin (less uncertainty)
        biased_coin = np.array([0.9, 0.1])
        entropy_biased = -np.sum(biased_coin * np.log(biased_coin))
        
        # Certain outcome (no uncertainty)
        certain = np.array([1.0, 0.0])
        entropy_certain = -np.sum(certain[certain > 0] * np.log(certain[certain > 0]))
        
        print(f"Fair coin [0.5, 0.5]: Entropy = {entropy_fair:.3f} nats")
        print(f"Biased coin [0.9, 0.1]: Entropy = {entropy_biased:.3f} nats")
        print(f"Certain [1.0, 0.0]: Entropy = {entropy_certain:.3f} nats")
        
        print("\nInterpretation:")
        print("• More uncertainty → Higher entropy")
        print("• Maximum entropy = log(n) for n equally likely outcomes")
        
        # Cross-entropy vs entropy
        print(f"\n🔄 CROSS-ENTROPY vs ENTROPY:")
        
        # True distribution (what we want to predict)
        true_dist = np.array([0.7, 0.2, 0.1])
        
        # Model predictions (what our model outputs)
        good_model = np.array([0.75, 0.15, 0.1])   # Close to truth
        bad_model = np.array([0.1, 0.1, 0.8])      # Far from truth
        
        # Calculate entropies
        true_entropy = -np.sum(true_dist * np.log(true_dist))
        cross_ent_good = -np.sum(true_dist * np.log(good_model))
        cross_ent_bad = -np.sum(true_dist * np.log(bad_model))
        
        print(f"True distribution: {true_dist}")
        print(f"True entropy (best possible): {true_entropy:.3f}")
        print(f"Good model prediction: {good_model}")
        print(f"Cross-entropy (good model): {cross_ent_good:.3f}")
        print(f"Bad model prediction: {bad_model}")
        print(f"Cross-entropy (bad model): {cross_ent_bad:.3f}")
        
        print(f"\nKey insight: Cross-entropy ≥ Entropy")
        print(f"• When model = truth: Cross-entropy = Entropy")
        print(f"• When model ≠ truth: Cross-entropy > Entropy")
        
        return true_entropy, cross_ent_good, cross_ent_bad
    
    def likelihood_principle_demo(self):
        """
        Demonstrate the connection between maximum likelihood and cross-entropy.
        """
        print("\n" + "="*50)
        print("LIKELIHOOD PRINCIPLE DEMO") 
        print("="*50)
        
        print("🎯 MAXIMUM LIKELIHOOD ESTIMATION:")
        print("Goal: Find model parameters that make observed data most likely")
        
        # Sample dataset
        true_labels = np.array([[1, 0, 0],   # Cat
                               [0, 1, 0],    # Dog  
                               [1, 0, 0],    # Cat
                               [0, 0, 1]])   # Bird
        
        # Two different models
        model_A_logits = np.array([[2.0, 0.5, 0.1],   # Confident in cats
                                  [0.1, 1.8, 0.2],    # Confident in dogs
                                  [1.5, 0.3, 0.1],    # Somewhat confident in cats  
                                  [0.1, 0.2, 1.6]])   # Confident in birds
        
        model_B_logits = np.array([[0.5, 1.2, 0.8],   # Confused
                                  [1.0, 1.0, 1.0],    # Very confused
                                  [0.8, 0.9, 0.7],    # Confused
                                  [0.6, 0.8, 0.9]])   # Confused
        
        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        model_A_probs = softmax(model_A_logits)
        model_B_probs = softmax(model_B_logits)
        
        print(f"\n📊 DATASET:")
        class_names = ['Cat', 'Dog', 'Bird']
        for i, label in enumerate(true_labels):
            true_class = class_names[np.argmax(label)]
            print(f"Example {i+1}: {true_class}")
        
        # Calculate likelihoods
        print(f"\n🔍 MODEL COMPARISON:")
        
        def calculate_likelihood_and_loss(probs, labels):
            # Likelihood: product of probabilities for true classes
            individual_probs = []
            total_log_likelihood = 0
            total_cross_entropy = 0
            
            for prob, label in zip(probs, labels):
                true_class_prob = prob[np.argmax(label)]
                individual_probs.append(true_class_prob)
                total_log_likelihood += np.log(true_class_prob)
                total_cross_entropy += self.cross_entropy_from_scratch(prob, label)
            
            likelihood = np.prod(individual_probs)
            return likelihood, total_log_likelihood, total_cross_entropy, individual_probs
        
        # Model A analysis
        likelihood_A, log_lik_A, ce_A, probs_A = calculate_likelihood_and_loss(model_A_probs, true_labels)
        
        # Model B analysis  
        likelihood_B, log_lik_B, ce_B, probs_B = calculate_likelihood_and_loss(model_B_probs, true_labels)
        
        print(f"Model A (confident):")
        print(f"  Individual probabilities: {[f'{p:.3f}' for p in probs_A]}")
        print(f"  Total likelihood: {likelihood_A:.6f}")
        print(f"  Log-likelihood: {log_lik_A:.3f}")  
        print(f"  Cross-entropy loss: {ce_A:.3f}")
        
        print(f"\nModel B (confused):")
        print(f"  Individual probabilities: {[f'{p:.3f}' for p in probs_B]}")
        print(f"  Total likelihood: {likelihood_B:.6f}")
        print(f"  Log-likelihood: {log_lik_B:.3f}")
        print(f"  Cross-entropy loss: {ce_B:.3f}")
        
        print(f"\n🏆 RESULTS:")
        print(f"• Higher likelihood → Better model")
        print(f"• Model A likelihood: {likelihood_A:.6f}")
        print(f"• Model B likelihood: {likelihood_B:.6f}")
        print(f"• Winner: {'Model A' if likelihood_A > likelihood_B else 'Model B'}")
        
        print(f"\n🔗 KEY CONNECTION:")
        print(f"• Maximizing likelihood ≡ Minimizing negative log-likelihood")
        print(f"• Negative log-likelihood ≡ Cross-entropy loss")
        print(f"• Therefore: Minimizing cross-entropy ≡ Maximum likelihood!")
        
        return likelihood_A, likelihood_B, ce_A, ce_B
    
    def pytorch_implementation_demo(self):
        """
        Compare manual implementation with PyTorch built-ins.
        """
        print("\n" + "="*50)
        print("PYTORCH IMPLEMENTATION COMPARISON")
        print("="*50)
        
        # Sample data
        batch_size = 4
        num_classes = 3
        torch.manual_seed(42)
        
        # Create sample tensors
        logits = torch.randn(batch_size, num_classes, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 0])  # Class indices
        
        print(f"📊 SAMPLE DATA:")
        print(f"Logits shape: {logits.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Logits:\n{logits}")
        print(f"Targets: {targets}")
        
        # Method 1: Manual implementation
        print(f"\n🔨 MANUAL IMPLEMENTATION:")
        
        # Convert targets to one-hot
        targets_onehot = F.one_hot(targets, num_classes).float()
        
        # Manual softmax
        logits_shifted = logits - logits.max(dim=1, keepdim=True)[0]
        exp_logits = torch.exp(logits_shifted)
        probs_manual = exp_logits / exp_logits.sum(dim=1, keepdim=True)
        
        # Manual cross-entropy
        epsilon = 1e-15
        probs_clipped = torch.clamp(probs_manual, epsilon, 1-epsilon)
        loss_manual = -torch.sum(targets_onehot * torch.log(probs_clipped)) / batch_size
        
        print(f"Manual probabilities:\n{probs_manual}")
        print(f"Manual loss: {loss_manual:.4f}")
        
        # Method 2: PyTorch F.softmax + manual cross-entropy
        print(f"\n🚀 PYTORCH F.SOFTMAX:")
        probs_pytorch = F.softmax(logits, dim=1)
        loss_mixed = F.cross_entropy(logits, targets)
        
        print(f"PyTorch probabilities:\n{probs_pytorch}")
        print(f"Mixed loss: {loss_mixed:.4f}")
        
        # Method 3: PyTorch F.cross_entropy (recommended)
        print(f"\n✅ PYTORCH F.CROSS_ENTROPY (RECOMMENDED):")
        loss_builtin = F.cross_entropy(logits, targets)
        
        print(f"Built-in loss: {loss_builtin:.4f}")
        print(f"This combines softmax + cross-entropy in one stable operation!")
        
        # Gradient comparison
        print(f"\n🎯 GRADIENT COMPARISON:")
        
        # Manual gradients
        loss_manual.backward(retain_graph=True)
        grad_manual = logits.grad.clone()
        logits.grad.zero_()
        
        # PyTorch gradients  
        loss_builtin.backward()
        grad_builtin = logits.grad.clone()
        
        print(f"Manual gradients:\n{grad_manual}")
        print(f"PyTorch gradients:\n{grad_builtin}")
        print(f"Max difference: {torch.abs(grad_manual - grad_builtin).max():.2e}")
        
        # Show the simple gradient formula
        expected_grad = probs_pytorch - targets_onehot
        print(f"\nExpected (ŷ - y):\n{expected_grad}")
        print(f"Matches PyTorch: {torch.allclose(grad_builtin, expected_grad, atol=1e-6)}")
        
        return loss_manual, loss_builtin, grad_manual, grad_builtin
    
    def visualization_demo(self):
        """
        Create visualizations for loss functions and information theory.
        """
        print("\n" + "="*50)
        print("LOSS FUNCTIONS VISUALIZATION")
        print("="*50)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Cross-entropy loss curve
        ax1 = axes[0, 0]
        
        # Show how cross-entropy changes with predicted probability
        true_prob = 1.0  # True class
        pred_probs = np.linspace(0.001, 0.999, 1000)
        cross_entropies = -np.log(pred_probs)
        
        ax1.plot(pred_probs, cross_entropies, 'b-', linewidth=2)
        ax1.set_xlabel('Predicted Probability (for true class)')
        ax1.set_ylabel('Cross-Entropy Loss')
        ax1.set_title('Cross-Entropy Loss vs Prediction Quality')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Add annotations
        ax1.axvline(x=0.5, color='r', linestyle='--', alpha=0.7, label='50% confidence')
        ax1.axvline(x=0.9, color='g', linestyle='--', alpha=0.7, label='90% confidence')
        ax1.legend()
        
        # 2. Entropy vs number of classes
        ax2 = axes[0, 1]
        
        num_classes_range = range(2, 11)
        max_entropies = [np.log(n) for n in num_classes_range]
        
        ax2.plot(num_classes_range, max_entropies, 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('Number of Classes')
        ax2.set_ylabel('Maximum Entropy (nats)')
        ax2.set_title('Maximum Entropy vs Number of Classes')
        ax2.grid(True, alpha=0.3)
        
        # Add formula annotation
        ax2.text(6, 1.5, 'H_max = log(n)', fontsize=12, 
                bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # 3. Gradient visualization  
        ax3 = axes[1, 0]
        
        # Show gradients for different prediction scenarios
        true_class = 0  # First class is correct
        scenarios = {
            'Perfect': [0.99, 0.005, 0.005],
            'Good': [0.7, 0.2, 0.1], 
            'Poor': [0.4, 0.3, 0.3],
            'Wrong': [0.1, 0.8, 0.1]
        }
        
        x_pos = np.arange(3)
        width = 0.2
        
        for i, (name, probs) in enumerate(scenarios.items()):
            # True labels (one-hot)
            true_labels = [1, 0, 0] 
            # Gradients = predictions - true  
            gradients = np.array(probs) - np.array(true_labels)
            
            ax3.bar(x_pos + i*width, gradients, width, label=name, alpha=0.8)
        
        ax3.set_xlabel('Class Index')
        ax3.set_ylabel('Gradient Value')
        ax3.set_title('Gradients for Different Prediction Quality')
        ax3.set_xticks(x_pos + width*1.5)
        ax3.set_xticklabels(['True Class', 'False Class 1', 'False Class 2'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 4. Information theory comparison
        ax4 = axes[1, 1]
        
        # Compare different distributions  
        distributions = {
            'Uniform': [1/3, 1/3, 1/3],
            'Peaked': [0.8, 0.1, 0.1],
            'Very Peaked': [0.95, 0.025, 0.025],
            'Bimodal': [0.45, 0.1, 0.45]
        }
        
        names = list(distributions.keys())
        entropies = []
        
        for dist in distributions.values():
            dist = np.array(dist)
            entropy = -np.sum(dist * np.log(dist))
            entropies.append(entropy)
        
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'plum']
        bars = ax4.bar(names, entropies, color=colors, alpha=0.8)
        ax4.set_ylabel('Entropy (nats)')
        ax4.set_title('Entropy of Different Distributions')
        ax4.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, entropy in zip(bars, entropies):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{entropy:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('c:/File/AI/Dive Into Deep Learning Journey/03-Linear Neural Networks for Classification/3.1 Softmax Regression/img/loss_functions_demo.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Loss function visualizations created!")
        print("📁 Saved as: loss_functions_demo.png")


def main():
    """Run all loss function and information theory demonstrations.""" 
    demo = LossAndInfoTheory()
    
    print("🚀 LOSS FUNCTIONS & INFORMATION THEORY DEMONSTRATIONS")
    print("="*70)
    
    # Run all demonstrations
    demo.gradient_computation_demo()
    demo.information_theory_demo()
    demo.likelihood_principle_demo()
    demo.pytorch_implementation_demo()
    demo.visualization_demo()
    
    print("\n🎉 LOSS FUNCTIONS DEMO COMPLETE!")
    print("="*70)
    print("📚 Key Concepts Covered:")
    print("  • Cross-entropy loss derivation and computation")
    print("  • Gradient computation (ŷ - y)")  
    print("  • Information theory: entropy, surprisal, cross-entropy")
    print("  • Maximum likelihood principle")
    print("  • Numerical stability techniques")
    print("  • PyTorch implementation patterns")
    print("\n🚀 Ready for: Complete Softmax Regression Model!")


if __name__ == "__main__":
    main()