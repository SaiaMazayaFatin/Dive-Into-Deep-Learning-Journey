# Basic Classification Concepts
# Implementation examples for classification fundamentals

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Dict, Any
import seaborn as sns


class ClassificationBasics:
    """Demonstrate fundamental concepts in classification."""
    
    def __init__(self):
        self.examples = {}
        
    def regression_vs_classification_demo(self):
        """
        Demonstrate the difference between regression and classification problems.
        """
        print("="*50)
        print("REGRESSION vs CLASSIFICATION DEMO")
        print("="*50)
        
        # Regression Example: House Price Prediction
        print("\n📈 REGRESSION EXAMPLE: House Price Prediction")
        print("-" * 30)
        
        # Sample data
        sizes = np.array([1200, 1500, 1800, 2000, 2200])  # sq ft
        prices = np.array([200000, 250000, 300000, 350000, 400000])  # dollars
        
        print("Input (House Size sq ft):", sizes)
        print("Output (Price $):", prices)
        print("Type: Continuous values")
        print("Goal: Predict exact numerical value")
        print("Example Question: 'How much will a 1700 sq ft house cost?'")
        
        # Classification Example: Email Spam Detection  
        print("\n📧 CLASSIFICATION EXAMPLE: Email Spam Detection")
        print("-" * 30)
        
        emails = ["Buy now! Limited time offer!", "Meeting at 3pm tomorrow", "WINNER! Claim prize now!", "Project update attached"]
        labels = ["Spam", "Not Spam", "Spam", "Not Spam"]
        
        for email, label in zip(emails, labels):
            print(f"Email: '{email[:30]}...' → Label: {label}")
            
        print("Type: Discrete categories")
        print("Goal: Predict which category")
        print("Example Question: 'Is this email spam or not spam?'")
        
        # Multi-class Classification
        print("\n🖼️  MULTI-CLASS CLASSIFICATION: Image Recognition")
        print("-" * 30)
        
        images = ["cat.jpg", "dog.jpg", "bird.jpg", "fish.jpg"]
        categories = ["Cat", "Dog", "Bird", "Fish"]
        
        for img, cat in zip(images, categories):
            print(f"Image: {img} → Category: {cat}")
            
        print("Type: Multiple discrete categories")  
        print("Goal: Choose one category from many")
        print("Example Question: 'What animal is in this image?'")

    def one_hot_encoding_demo(self):
        """
        Demonstrate one-hot encoding with visual examples.
        """
        print("\n" + "="*50)
        print("ONE-HOT ENCODING DEMO")
        print("="*50)
        
        # Categories
        categories = ['Cat', 'Dog', 'Bird']
        print(f"Categories: {categories}")
        print(f"Number of categories: {len(categories)}")
        
        # Create one-hot encodings
        print("\n🔢 ONE-HOT ENCODING REPRESENTATION:")
        print("-" * 30)
        
        for i, category in enumerate(categories):
            # Create one-hot vector
            one_hot = np.zeros(len(categories))
            one_hot[i] = 1
            print(f"{category:6s} = {one_hot} (index {i})")
            
        print("\n❌ WHY NOT USE NUMBERS DIRECTLY?")
        print("-" * 30)
        print("Don't use: Cat=1, Dog=2, Bird=3")
        print("Problem: Implies order (Dog > Cat) and distance (Bird-Dog = Dog-Cat)")
        print("Reality: Categories have no natural ordering or distance")
        
        # Practical example
        print("\n🔧 PRACTICAL IMPLEMENTATION:")
        print("-" * 30)
        
        # Sample batch of labels  
        label_indices = [0, 2, 1, 0, 1]  # Cat, Bird, Dog, Cat, Dog
        batch_size = len(label_indices)
        num_classes = len(categories)
        
        # Convert to one-hot
        one_hot_batch = np.zeros((batch_size, num_classes))
        one_hot_batch[np.arange(batch_size), label_indices] = 1
        
        print("Label indices:", label_indices)
        print("Corresponding names:", [categories[i] for i in label_indices])
        print("One-hot matrix:")
        print(one_hot_batch)
        print("Shape:", one_hot_batch.shape, "(batch_size, num_classes)")
        
        return one_hot_batch

    def linear_model_architecture_demo(self):
        """
        Demonstrate the linear model architecture for classification.
        """  
        print("\n" + "="*50)
        print("LINEAR MODEL ARCHITECTURE DEMO")
        print("="*50)
        
        # Example dimensions
        batch_size = 3
        num_features = 4  # Input features  
        num_classes = 3   # Output classes
        
        print(f"📊 PROBLEM SETUP:")
        print(f"- Input features: {num_features}")
        print(f"- Output classes: {num_classes}")
        print(f"- Batch size: {batch_size}")
        
        # Create sample data
        np.random.seed(42)
        X = np.random.randn(batch_size, num_features)
        W = np.random.randn(num_features, num_classes) * 0.1
        b = np.zeros(num_classes)
        
        print(f"\n🎯 MATRIX DIMENSIONS:")
        print(f"X (inputs): {X.shape}")  
        print(f"W (weights): {W.shape}")
        print(f"b (bias): {b.shape}")
        
        # Show the computation
        print(f"\n⚡ LINEAR TRANSFORMATION:")
        print("Formula: O = X @ W + b")
        print("Where:")
        print("- X: Input batch")
        print("- W: Weight matrix") 
        print("- b: Bias vector")
        print("- O: Raw outputs (logits)")
        
        # Compute outputs
        O = X @ W + b
        print(f"\nComputed outputs O:")
        print(O)
        print(f"Output shape: {O.shape}")
        
        print(f"\n🔍 INTERPRETATION:")
        print("- Each row is one example")
        print("- Each column is one class score")
        print("- Values can be negative or > 1")
        print("- Higher values = higher confidence for that class")
        
        # Show per-example breakdown
        class_names = ['Cat', 'Dog', 'Bird']
        print(f"\n📋 PER-EXAMPLE BREAKDOWN:")
        for i in range(batch_size):
            print(f"Example {i+1}: {O[i]}")
            best_class = np.argmax(O[i])
            print(f"  → Highest score: {class_names[best_class]} ({O[i, best_class]:.3f})")
            
        return X, W, b, O

    def multi_label_vs_multi_class_demo(self):
        """
        Show the difference between multi-class and multi-label classification.
        """
        print("\n" + "="*50) 
        print("MULTI-CLASS vs MULTI-LABEL DEMO")
        print("="*50)
        
        print("🎯 MULTI-CLASS CLASSIFICATION:")
        print("-" * 30)
        print("• One and only one class per example")
        print("• Classes are mutually exclusive") 
        print("• Probabilities sum to 1.0")
        print("• Use: Softmax + Cross-entropy")
        
        # Multi-class example
        image_classes = ['Cat', 'Dog', 'Bird'] 
        probabilities = np.array([0.7, 0.2, 0.1])  # Sum = 1.0
        
        print("\nExample: Image Classification")
        for class_name, prob in zip(image_classes, probabilities):
            print(f"  {class_name}: {prob:.1f}")
        print(f"  Total: {probabilities.sum():.1f}")
        print("  Prediction: Cat (highest probability)")
        
        print("\n🏷️  MULTI-LABEL CLASSIFICATION:")  
        print("-" * 30)
        print("• Multiple classes can be true simultaneously")
        print("• Classes are not mutually exclusive")
        print("• Each class has independent probability")
        print("• Use: Sigmoid + Binary cross-entropy per class")
        
        # Multi-label example
        article_tags = ['Politics', 'Technology', 'Business', 'Science']
        tag_probabilities = np.array([0.1, 0.8, 0.6, 0.2])  # Independent
        threshold = 0.5
        
        print("\nExample: News Article Tagging")
        print("Article: 'Tech Company IPO Announcement'")
        for tag, prob in zip(article_tags, tag_probabilities):
            is_relevant = "✓" if prob > threshold else "✗"
            print(f"  {tag}: {prob:.1f} {is_relevant}")
        
        active_tags = [tag for tag, prob in zip(article_tags, tag_probabilities) if prob > threshold]
        print(f"  Predicted tags: {active_tags}")
        print("  Notice: Multiple tags can be active!")

    def visualization_demo(self):
        """
        Create visualizations to illustrate classification concepts.
        """
        print("\n" + "="*50)
        print("VISUALIZATION DEMO") 
        print("="*50)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Regression vs Classification scatter plot
        ax1 = axes[0, 0]
        
        # Regression data
        x_reg = np.array([1, 2, 3, 4, 5])
        y_reg = x_reg * 2 + np.random.normal(0, 0.5, len(x_reg))
        ax1.scatter(x_reg, y_reg, color='blue', s=100, alpha=0.7)
        ax1.plot(x_reg, x_reg * 2, 'r--', linewidth=2)
        ax1.set_title("Regression: Continuous Output", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Input Feature")
        ax1.set_ylabel("Continuous Output")
        ax1.grid(True, alpha=0.3)
        
        # 2. Classification data  
        ax2 = axes[0, 1]
        
        # Generate 2D classification data
        np.random.seed(42)
        n_points = 50
        
        # Class 1 (blue)
        x1 = np.random.normal(2, 0.8, n_points)
        y1 = np.random.normal(2, 0.8, n_points)
        
        # Class 2 (red)  
        x2 = np.random.normal(5, 0.8, n_points)
        y2 = np.random.normal(5, 0.8, n_points)
        
        # Class 3 (green)
        x3 = np.random.normal(3.5, 0.8, n_points)
        y3 = np.random.normal(6, 0.8, n_points)
        
        ax2.scatter(x1, y1, c='blue', label='Class 1', s=60, alpha=0.7)
        ax2.scatter(x2, y2, c='red', label='Class 2', s=60, alpha=0.7)  
        ax2.scatter(x3, y3, c='green', label='Class 3', s=60, alpha=0.7)
        ax2.set_title("Classification: Discrete Categories", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Feature 1")
        ax2.set_ylabel("Feature 2")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. One-hot encoding visualization
        ax3 = axes[1, 0]
        
        categories = ['Cat', 'Dog', 'Bird']
        one_hot_matrix = np.array([[1, 0, 0],  # Cat
                                  [0, 1, 0],   # Dog  
                                  [0, 0, 1]])  # Bird
        
        im = ax3.imshow(one_hot_matrix, cmap='RdYlBu_r', aspect='auto')
        ax3.set_xticks(range(len(categories)))
        ax3.set_xticklabels(categories)
        ax3.set_yticks(range(len(categories)))
        ax3.set_yticklabels(categories)
        ax3.set_title("One-Hot Encoding Matrix", fontsize=14, fontweight='bold')
        ax3.set_ylabel("True Class")
        ax3.set_xlabel("One-Hot Vector")
        
        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(categories)):
                text = ax3.text(j, i, int(one_hot_matrix[i, j]), 
                              ha="center", va="center", color="black", fontsize=14, fontweight='bold')
        
        # 4. Linear model architecture
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Draw network architecture
        input_nodes = 4
        output_nodes = 3
        
        # Input layer
        input_y = np.linspace(0.2, 0.8, input_nodes)
        for i, y in enumerate(input_y):
            circle = plt.Circle((0.2, y), 0.05, color='lightblue', ec='black')
            ax4.add_patch(circle)
            ax4.text(0.1, y, f'x{i+1}', ha='center', va='center', fontsize=10)
        
        # Output layer  
        output_y = np.linspace(0.3, 0.7, output_nodes)
        output_labels = ['Cat', 'Dog', 'Bird']
        for i, (y, label) in enumerate(zip(output_y, output_labels)):
            circle = plt.Circle((0.8, y), 0.05, color='lightcoral', ec='black')
            ax4.add_patch(circle)
            ax4.text(0.9, y, label, ha='left', va='center', fontsize=10)
        
        # Connections
        for i, y_in in enumerate(input_y):
            for j, y_out in enumerate(output_y):
                ax4.plot([0.25, 0.75], [y_in, y_out], 'gray', alpha=0.3, linewidth=1)
        
        ax4.text(0.5, 0.9, 'Linear Model Architecture\n(Fully Connected)', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        ax4.text(0.2, 0.05, 'Inputs\n(Features)', ha='center', va='center', fontsize=10)  
        ax4.text(0.8, 0.05, 'Outputs\n(Classes)', ha='center', va='center', fontsize=10)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1) 
        
        plt.tight_layout()
        plt.savefig('c:/File/AI/Dive Into Deep Learning Journey/03-Linear Neural Networks for Classification/3.1 Softmax Regression/img/classification_basics_demo.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations created and saved!")
        print("📁 Saved as: classification_basics_demo.png")


def main():
    """Run all classification basics demonstrations."""
    demo = ClassificationBasics()
    
    # Run all demonstrations
    demo.regression_vs_classification_demo()
    demo.one_hot_encoding_demo() 
    demo.linear_model_architecture_demo()
    demo.multi_label_vs_multi_class_demo()
    demo.visualization_demo()
    
    print("\n" + "="*60)
    print("🎉 CLASSIFICATION BASICS DEMO COMPLETE!")
    print("="*60)
    print("📚 Key Concepts Covered:")
    print("  • Regression vs Classification")
    print("  • One-Hot Encoding")  
    print("  • Linear Model Architecture")
    print("  • Multi-Class vs Multi-Label")
    print("  • Visual Understanding")
    print("\n🚀 Ready for: Softmax Functions and Loss Functions!")


if __name__ == "__main__":
    main()