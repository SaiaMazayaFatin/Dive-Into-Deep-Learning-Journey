# Linear Regression Implementation Examples

This folder contains comprehensive implementations of Linear Regression based on the study materials from "Dive Into Deep Learning Journey". Each file demonstrates specific concepts and provides hands-on examples.

## Files Overview

### 1. `basic_linear_regression.py`
**Concepts**: Core linear regression algorithms, analytic solution, SGD
**Based on**: `1.Basic.md`

**Key Features**:
- `LinearRegression` class with both analytic and SGD solutions
- Demonstrates the closed-form solution: `w* = (X^T X)^(-1) X^T y`
- Minibatch SGD with vectorized operations
- Comprehensive comparison between methods
- Loss visualization and convergence analysis

**Usage**:
```python
python basic_linear_regression.py
```

**What you'll see**:
- Side-by-side comparison of analytic vs SGD solutions
- Training loss curves
- Parameter recovery analysis
- Speed and accuracy comparisons

### 2. `vectorization_demo.py`
**Concepts**: Vectorization importance, speed optimization
**Based on**: `2.VectorizationForSpeed.md`

**Key Features**:
- Direct comparison of for-loop vs vectorized operations
- Benchmarking across different data sizes
- Memory efficiency analysis
- Real-world examples (vector addition, matrix multiplication, linear regression)

**Usage**:
```python
python vectorization_demo.py
```

**What you'll see**:
- Dramatic speed differences (10x-100x faster with vectorization)
- Performance scaling with data size
- Memory usage comparisons
- Interactive benchmark results

### 3. `normal_distribution_demo.py`
**Concepts**: Maximum Likelihood Estimation, connection to squared loss
**Based on**: `3.TheNormalDistributionandSquaredLoss.md`

**Key Features**:
- Visual demonstration of normal distributions with different parameters
- Mathematical derivation from likelihood to MSE loss
- MLE vs MSE solution comparison
- Residual analysis for assumption validation
- Different noise assumptions and their optimal loss functions

**Usage**:
```python
python normal_distribution_demo.py
```

**What you'll see**:
- Normal distribution visualizations
- Proof that MSE = MLE under Gaussian noise
- Likelihood surfaces vs MSE surfaces
- Q-Q plots for normality testing

### 4. `neural_network_perspective.py`
**Concepts**: Linear regression as neural network, biological inspiration
**Based on**: `4.LinearRegressionasaNeuralNetwork.md`

**Key Features**:
- `LinearNeuron` class implementing single neuron behavior
- Network structure visualization
- Biological vs artificial neuron comparison
- PyTorch integration and compatibility
- Training visualization from neural network perspective

**Usage**:
```python
python neural_network_perspective.py
```

**What you'll see**:
- Neural network diagram generation
- Single neuron training demonstration
- Comparison with PyTorch `nn.Linear`
- Biological inspiration explanations

### 5. `complete_example.py`
**Concepts**: All concepts integrated into comprehensive example
**Based on**: All study materials combined

**Key Features**:
- `ComprehensiveLinearRegression` class with all methods
- Complete evaluation pipeline
- Advanced visualizations
- Performance benchmarking
- Real-world considerations
- Statistical validation

**Usage**:
```python
python complete_example.py
```

**What you'll see**:
- Complete training and evaluation pipeline
- Comprehensive result visualizations
- Method comparison across multiple metrics
- Professional-level analysis and reporting

## Dependencies

Make sure you have the following packages installed:

```bash
pip install torch numpy matplotlib seaborn pandas networkx scipy
```

For plotting capabilities:
```bash
pip install matplotlib seaborn
```

For the neural network visualizations:
```bash
pip install networkx
```

## Running the Examples

### Quick Start
Run all examples in order to see the progression:

```bash
# Start with basic concepts
python basic_linear_regression.py

# Learn about vectorization importance  
python vectorization_demo.py

# Understand the theoretical foundation
python normal_distribution_demo.py

# See the neural network connection
python neural_network_perspective.py

# Experience the complete implementation
python complete_example.py
```

### Individual Concepts
Each file can be run independently to focus on specific concepts:

```python
# Just want to see basic linear regression?
python basic_linear_regression.py

# Need to understand vectorization benefits?
python vectorization_demo.py

# Want to explore the MLE connection?
python normal_distribution_demo.py
```

## Expected Output

### Console Output
Each script provides detailed console output including:
- Step-by-step explanations
- Numerical comparisons
- Performance metrics
- Key insights and takeaways

### Visualizations
The scripts generate various plots:
- Training loss curves
- Parameter recovery analysis
- Performance comparisons
- Statistical validation plots
- Network structure diagrams

### Learning Outcomes
After running these examples, you should understand:

1. **Mathematical Foundation**:
   - Why we use squared loss (MLE connection)
   - Analytic vs iterative solutions
   - Gradient descent mechanics

2. **Implementation Details**:
   - Vectorization importance
   - Numerical stability considerations
   - Parameter initialization
   - Learning rate selection

3. **Neural Network Connection**:
   - How linear regression relates to deep learning
   - Single neuron computation
   - Biological inspiration

4. **Practical Considerations**:
   - When to use analytic vs SGD
   - Batch size effects
   - Convergence criteria
   - Model evaluation

## Customization

### Modifying Parameters
You can easily modify parameters in any script:

```python
# In basic_linear_regression.py
X, y = generate_synthetic_data(
    n_samples=500,      # Change dataset size
    n_features=10,      # Change dimensionality  
    noise_std=0.05      # Change noise level
)

# In vectorization_demo.py
sizes = [100, 1000, 5000, 10000]  # Change benchmark sizes

# In complete_example.py  
model.fit(X_train, y_train,
         learning_rate=0.02,    # Adjust learning rate
         batch_size=64,         # Change batch size
         max_epochs=1000)       # Change training duration
```

### Adding New Features
The implementations are designed to be extensible:

```python
# Add regularization
class RegularizedLinearRegression(LinearRegression):
    def __init__(self, lambda_reg=0.01):
        super().__init__()
        self.lambda_reg = lambda_reg
    
    def _compute_loss_with_regularization(self, X, y):
        # Add L2 regularization term
        base_loss = super().loss(X, y)
        reg_term = self.lambda_reg * torch.norm(self.w) ** 2
        return base_loss + reg_term
```

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce dataset size or batch size
2. **Slow Performance**: Ensure you have vectorized operations
3. **Numerical Instability**: Check condition number of X^T X
4. **Poor Convergence**: Adjust learning rate or increase epochs

### Performance Tips

1. **Use appropriate data types**: `torch.float32` is usually sufficient
2. **Batch size selection**: Start with 32-128 for most problems
3. **Learning rate**: Start with 0.01 and adjust based on convergence
4. **Early stopping**: Monitor validation loss to prevent overfitting

## Further Extensions

These implementations provide a solid foundation for exploring:

- **Regularization**: L1/L2 penalty terms
- **Feature Engineering**: Polynomial features, interactions
- **Optimization**: Advanced optimizers (Adam, RMSprop)
- **Validation**: Cross-validation, learning curves
- **Diagnostics**: Residual analysis, influence functions

## Connection to Study Materials

Each implementation directly corresponds to concepts in the markdown files:

| File | Study Material | Key Concepts |
|------|---------------|--------------|
| `basic_linear_regression.py` | `1.Basic.md` | Core algorithms, mathematical foundations |
| `vectorization_demo.py` | `2.VectorizationForSpeed.md` | Performance optimization |
| `normal_distribution_demo.py` | `3.TheNormalDistributionandSquaredLoss.md` | Theoretical justification |
| `neural_network_perspective.py` | `4.LinearRegressionasaNeuralNetwork.md` | Deep learning connection |
| `complete_example.py` | All materials | Comprehensive integration |

This provides a complete learning journey from basic concepts to advanced implementations, preparing you for more complex machine learning topics!