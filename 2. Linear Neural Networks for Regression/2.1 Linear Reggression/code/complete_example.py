"""
Complete Linear Regression Example
=================================

This module provides a comprehensive example that demonstrates all concepts
from Basic.md in a practical, real-world scenario.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import time

# Import our implementations
from basic_linear_regression import LinearRegressionAnalytic, LinearRegressionSGD
from neural_network_perspective import LinearRegressionNN


class CompleteLinearRegressionDemo:
    """
    A comprehensive demonstration of linear regression covering all concepts
    from Basic.md: theory, implementation methods, and practical applications.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        self.data = None
        self.results = {}
        
    def generate_realistic_dataset(self) -> Dict:
        """
        Generate a realistic dataset simulating house price prediction.
        Features: area, age, bedrooms, location_score
        Target: price
        """
        print("Step 1: Generating Realistic Dataset")
        print("=" * 40)
        
        n_samples = 1000
        
        # Generate correlated features (more realistic)
        # Area (sq ft): 800-3000
        area = np.random.normal(1500, 400, n_samples)
        area = np.clip(area, 800, 3000)
        
        # Age (years): 0-50
        age = np.random.exponential(15, n_samples)
        age = np.clip(age, 0, 50)
        
        # Bedrooms: 1-5 (correlated with area)
        bedrooms = np.round(area / 600 + np.random.normal(0, 0.5, n_samples))
        bedrooms = np.clip(bedrooms, 1, 5)
        
        # Location score: 1-10
        location_score = np.random.normal(6, 2, n_samples)
        location_score = np.clip(location_score, 1, 10)
        
        # Price formula (with realistic coefficients)
        # Base price + area effect - age depreciation + bedroom premium + location premium + noise
        base_price = 100000
        area_coef = 120  # $120 per sq ft
        age_coef = -2000  # -$2000 per year
        bedroom_coef = 15000  # +$15000 per bedroom
        location_coef = 8000  # +$8000 per location point
        
        price = (base_price + 
                area * area_coef + 
                age * age_coef + 
                bedrooms * bedroom_coef + 
                location_score * location_coef +
                np.random.normal(0, 20000, n_samples))  # $20k noise
        
        # Create feature matrix
        X = np.column_stack([area, age, bedrooms, location_score])
        y = price.reshape(-1, 1)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Store true coefficients for comparison
        true_coefficients = {
            'intercept': base_price,
            'area': area_coef,
            'age': age_coef, 
            'bedrooms': bedroom_coef,
            'location_score': location_coef
        }
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=self.random_state
        )
        
        self.data = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'feature_names': ['area', 'age', 'bedrooms', 'location_score'],
            'true_coefficients': true_coefficients
        }
        
        print(f"Dataset created:")
        print(f"- {n_samples} total samples ({len(X_train)} train, {len(X_test)} test)")
        print(f"- 4 features: {', '.join(self.data['feature_names'])}")
        print(f"- Target: house price")
        print(f"- True coefficients: {true_coefficients}")
        
        return self.data
    
    def demonstrate_data_exploration(self):
        """Explore and visualize the dataset."""
        print(f"\nStep 2: Data Exploration")
        print("=" * 25)
        
        X_train = self.data['X_train']
        y_train = self.data['y_train']
        feature_names = self.data['feature_names']
        
        # Basic statistics
        print("Dataset Statistics:")
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {X_train.shape[1]}")
        
        # Create DataFrame for easier analysis
        df = pd.DataFrame(X_train.numpy(), columns=feature_names)
        df['price'] = y_train.numpy().flatten()
        
        print(f"\nFeature statistics:")
        print(df.describe())
        
        # Visualize data
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Feature distributions
        for i, feature in enumerate(feature_names):
            row, col = i // 2, i % 2
            axes[row, col].hist(df[feature], bins=30, alpha=0.7, color=f'C{i}')
            axes[row, col].set_title(f'Distribution of {feature}')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].grid(True, alpha=0.3)
        
        # Price distribution
        axes[1, 2].hist(df['price'], bins=30, alpha=0.7, color='red')
        axes[1, 2].set_title('Distribution of Price')
        axes[1, 2].set_xlabel('Price ($)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Remove empty subplot
        fig.delaxes(axes[1, 2])
        
        # Correlation plot
        axes[1, 2] = fig.add_subplot(2, 3, 6)
        correlation_matrix = df.corr()
        im = axes[1, 2].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 2].set_xticks(range(len(correlation_matrix.columns)))
        axes[1, 2].set_yticks(range(len(correlation_matrix.columns)))
        axes[1, 2].set_xticklabels(correlation_matrix.columns, rotation=45)
        axes[1, 2].set_yticklabels(correlation_matrix.columns)
        axes[1, 2].set_title('Feature Correlation Matrix')
        
        # Add correlation values
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix)):
                axes[1, 2].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                               ha='center', va='center', fontsize=9)
        
        plt.colorbar(im, ax=axes[1, 2])
        plt.tight_layout()
        plt.show()
    
    def compare_all_methods(self):
        """Compare all implementation methods side by side."""
        print(f"\nStep 3: Method Comparison")
        print("=" * 30)
        
        X_train = self.data['X_train']
        X_test = self.data['X_test']
        y_train = self.data['y_train'] 
        y_test = self.data['y_test']
        
        methods = {}
        
        # Method 1: Analytical Solution
        print("Training Analytical Solution...")
        start_time = time.time()
        model_analytic = LinearRegressionAnalytic()
        model_analytic.fit(X_train, y_train)
        analytic_time = time.time() - start_time
        
        # Predictions and metrics
        train_pred_analytic = model_analytic.predict(X_train)
        test_pred_analytic = model_analytic.predict(X_test)
        
        methods['Analytical'] = {
            'model': model_analytic,
            'train_time': analytic_time,
            'train_pred': train_pred_analytic,
            'test_pred': test_pred_analytic,
            'weights': model_analytic.w,
            'bias': model_analytic.b
        }
        
        # Method 2: SGD (Manual)
        print("Training Manual SGD...")
        start_time = time.time()
        model_sgd = LinearRegressionSGD(learning_rate=0.001, n_epochs=1000)
        model_sgd.fit(X_train, y_train, batch_size=32, verbose=False)
        sgd_time = time.time() - start_time
        
        train_pred_sgd = model_sgd.predict(X_train)
        test_pred_sgd = model_sgd.predict(X_test)
        
        methods['Manual SGD'] = {
            'model': model_sgd,
            'train_time': sgd_time,
            'train_pred': train_pred_sgd,
            'test_pred': test_pred_sgd,
            'weights': model_sgd.w.detach(),
            'bias': model_sgd.b.detach()
        }
        
        # Method 3: Neural Network
        print("Training Neural Network...")
        start_time = time.time()
        model_nn = LinearRegressionNN(input_dim=X_train.shape[1], learning_rate=0.001)
        model_nn.train(X_train, y_train, epochs=1000, batch_size=32, verbose=False)
        nn_time = time.time() - start_time
        
        train_pred_nn = model_nn.predict(X_train)
        test_pred_nn = model_nn.predict(X_test)
        
        w_nn, b_nn = model_nn.get_parameters()
        methods['Neural Network'] = {
            'model': model_nn,
            'train_time': nn_time,
            'train_pred': train_pred_nn,
            'test_pred': test_pred_nn,
            'weights': w_nn.T,
            'bias': b_nn
        }
        
        # Calculate metrics for all methods
        self.results = {}
        
        for method_name, method_data in methods.items():
            train_mse = torch.mean((method_data['train_pred'] - y_train) ** 2).item()
            test_mse = torch.mean((method_data['test_pred'] - y_test) ** 2).item()
            
            self.results[method_name] = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_rmse': np.sqrt(train_mse),
                'test_rmse': np.sqrt(test_mse),
                'train_time': method_data['train_time'],
                'weights': method_data['weights'],
                'bias': method_data['bias']
            }
        
        # Display results
        print(f"\nResults Comparison:")
        print(f"{'Method':<15} {'Train RMSE':<12} {'Test RMSE':<12} {'Time (s)':<10}")
        print("-" * 55)
        
        for method, metrics in self.results.items():
            print(f"{method:<15} {metrics['train_rmse']:<12.0f} {metrics['test_rmse']:<12.0f} {metrics['train_time']:<10.4f}")
        
        return methods
    
    def analyze_learned_parameters(self):
        """Analyze and compare learned parameters with true values."""
        print(f"\nStep 4: Parameter Analysis") 
        print("=" * 30)
        
        true_coef = self.data['true_coefficients']
        feature_names = self.data['feature_names']
        
        # Create comparison table
        print(f"Parameter Comparison:")
        print(f"{'Parameter':<15} {'True Value':<12} {'Analytical':<12} {'Manual SGD':<12} {'Neural Net':<12}")
        print("-" * 75)
        
        # Bias comparison
        print(f"{'bias':<15} {true_coef['intercept']:<12.0f} {self.results['Analytical']['bias'].item():<12.0f} "
              f"{self.results['Manual SGD']['bias'].item():<12.0f} {self.results['Neural Network']['bias'].item():<12.0f}")
        
        # Weight comparison
        for i, feature in enumerate(feature_names):
            true_val = true_coef[feature]
            analytical_val = self.results['Analytical']['weights'][i].item()
            sgd_val = self.results['Manual SGD']['weights'][i].item()
            nn_val = self.results['Neural Network']['weights'][i].item()
            
            print(f"{feature:<15} {true_val:<12.0f} {analytical_val:<12.0f} {sgd_val:<12.0f} {nn_val:<12.0f}")
        
        # Calculate parameter errors
        print(f"\nParameter Errors (vs True Values):")
        print(f"{'Method':<15} {'Bias Error':<12} {'Weight RMSE':<12}")
        print("-" * 40)
        
        for method_name in self.results.keys():
            bias_error = abs(self.results[method_name]['bias'].item() - true_coef['intercept'])
            
            # Calculate weight RMSE
            learned_weights = self.results[method_name]['weights'].flatten()
            true_weights = torch.tensor([true_coef[name] for name in feature_names], dtype=torch.float32)
            weight_rmse = torch.sqrt(torch.mean((learned_weights - true_weights) ** 2)).item()
            
            print(f"{method_name:<15} {bias_error:<12.0f} {weight_rmse:<12.0f}")
    
    def visualize_predictions(self, methods):
        """Visualize model predictions vs actual values."""
        print(f"\nStep 5: Prediction Visualization")
        print("=" * 35)
        
        y_test = self.data['y_test']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        method_names = list(methods.keys())
        colors = ['blue', 'red', 'green']
        
        for i, (method_name, color) in enumerate(zip(method_names, colors)):
            ax = axes[i]
            
            test_pred = methods[method_name]['test_pred']
            
            # Scatter plot: predicted vs actual
            ax.scatter(y_test.numpy(), test_pred.detach().numpy(), 
                      alpha=0.6, color=color, s=20)
            
            # Perfect prediction line
            min_val = min(y_test.min(), test_pred.min())
            max_val = max(y_test.max(), test_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
            
            # Calculate R²
            ss_res = torch.sum((y_test - test_pred) ** 2)
            ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            ax.set_xlabel('Actual Price ($)')
            ax.set_ylabel('Predicted Price ($)')
            ax.set_title(f'{method_name}\nR² = {r2.item():.3f}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Make axes equal
            ax.set_aspect('equal', 'box')
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_overfitting_concept(self):
        """Demonstrate the concept of overfitting with polynomial features."""
        print(f"\nStep 6: Overfitting Demonstration")
        print("=" * 35)
        
        # Create simple 1D dataset for clear visualization
        torch.manual_seed(42)
        n_train = 20
        n_test = 100
        
        # Generate training data
        x_train = torch.linspace(-1, 1, n_train).reshape(-1, 1)
        y_train = 1.5 * x_train + 0.3 * x_train**2 + torch.randn(n_train, 1) * 0.1
        
        # Generate test data (more points)
        x_test = torch.linspace(-1, 1, n_test).reshape(-1, 1)
        y_test = 1.5 * x_test + 0.3 * x_test**2 + torch.randn(n_test, 1) * 0.1
        
        def create_polynomial_features(x, degree):
            """Create polynomial features up to given degree."""
            features = torch.ones(len(x), 1)  # bias term
            for d in range(1, degree + 1):
                features = torch.cat([features, x**d], dim=1)
            return features
        
        degrees = [1, 3, 10]
        fig, axes = plt.subplots(1, len(degrees), figsize=(15, 5))
        
        print("Polynomial Regression Results:")
        print(f"{'Degree':<8} {'Train RMSE':<12} {'Test RMSE':<12} {'Status':<15}")
        print("-" * 50)
        
        for i, degree in enumerate(degrees):
            # Create polynomial features
            X_train_poly = create_polynomial_features(x_train, degree)
            X_test_poly = create_polynomial_features(x_test, degree)
            
            # Fit model
            model = LinearRegressionAnalytic()
            model.fit(X_train_poly, y_train)
            
            # Predictions
            train_pred = model.predict(X_train_poly)
            test_pred = model.predict(X_test_poly)
            
            # Metrics
            train_rmse = torch.sqrt(torch.mean((train_pred - y_train) ** 2)).item()
            test_rmse = torch.sqrt(torch.mean((test_pred - y_test) ** 2)).item()
            
            # Determine status
            if test_rmse > train_rmse * 1.5:
                status = "Overfitting"
            elif abs(test_rmse - train_rmse) < 0.1:
                status = "Good fit"
            else:
                status = "Underfitting"
            
            print(f"{degree:<8} {train_rmse:<12.4f} {test_rmse:<12.4f} {status:<15}")
            
            # Plot
            ax = axes[i]
            ax.scatter(x_train.numpy(), y_train.numpy(), color='blue', alpha=0.7, label='Training data')
            ax.scatter(x_test.numpy(), y_test.numpy(), color='red', alpha=0.3, s=10, label='Test data')
            
            # Plot prediction curve
            x_smooth = torch.linspace(-1, 1, 200).reshape(-1, 1)
            X_smooth_poly = create_polynomial_features(x_smooth, degree)
            y_smooth_pred = model.predict(X_smooth_poly)
            
            ax.plot(x_smooth.numpy(), y_smooth_pred.detach().numpy(), 'g-', linewidth=2, label=f'Degree {degree}')
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'Polynomial Degree {degree}\n{status}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-2, 2)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nKey Insights:")
        print(f"- Degree 1: May underfit (too simple)")
        print(f"- Degree 3: Usually good balance")
        print(f"- Degree 10: Likely overfits (too complex for small dataset)")
        print(f"- Overfitting: Low training error, high test error")
    
    def practical_tips_and_insights(self):
        """Share practical tips and insights from the analysis."""
        print(f"\nStep 7: Practical Tips & Insights")
        print("=" * 40)
        
        print("🎯 Key Takeaways from this Analysis:")
        print()
        
        print("1. Method Selection:")
        print("   • Analytical solution: Fast, exact (when feasible)")
        print("   • SGD: Flexible, works with large datasets")
        print("   • Neural networks: Scalable to complex models")
        
        print(f"\n2. Performance Insights:")
        analytical_time = self.results['Analytical']['train_time']
        sgd_time = self.results['Manual SGD']['train_time']
        nn_time = self.results['Neural Network']['train_time']
        
        print(f"   • Analytical solution: {analytical_time:.4f}s (fastest)")
        print(f"   • Manual SGD: {sgd_time:.4f}s ({sgd_time/analytical_time:.1f}x slower)")
        print(f"   • Neural Network: {nn_time:.4f}s ({nn_time/analytical_time:.1f}x slower)")
        
        print(f"\n3. Accuracy Comparison:")
        for method in ['Analytical', 'Manual SGD', 'Neural Network']:
            rmse = self.results[method]['test_rmse']
            print(f"   • {method}: {rmse:.0f} RMSE")
        
        print(f"\n4. When to Use Each Method:")
        print("   • Small datasets (<10k samples): Analytical solution")
        print("   • Large datasets (>100k samples): SGD or Neural Networks")
        print("   • Non-linear patterns: Neural Networks with activations")
        print("   • Need interpretability: Analytical solution")
        
        print(f"\n5. Data Quality Matters:")
        print("   • Standardize features for better convergence") 
        print("   • Remove outliers for more robust results")
        print("   • More data generally leads to better generalization")
        
        print(f"\n6. Hyperparameter Tuning:")
        print("   • Learning rate: Start with 0.01, adjust based on convergence")
        print("   • Batch size: 32-256 usually works well")
        print("   • Epochs: Monitor validation loss to prevent overfitting")


def main():
    """Run the complete linear regression demonstration."""
    print("Complete Linear Regression Demonstration")
    print("🏠 House Price Prediction Example")
    print("="*50)
    
    # Initialize demo
    demo = CompleteLinearRegressionDemo()
    
    # Run all steps
    demo.generate_realistic_dataset()
    demo.demonstrate_data_exploration()
    methods = demo.compare_all_methods()
    demo.analyze_learned_parameters()
    demo.visualize_predictions(methods)
    demo.demonstrate_overfitting_concept()
    demo.practical_tips_and_insights()
    
    print(f"\n🎉 Demo Complete!")
    print("You've seen linear regression from theory to practice!")


if __name__ == "__main__":
    main()