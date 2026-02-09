# Linear Neural Networks for Classification  

## 🎯 Overview

**Switching to Classification**  

The core training loop (loading data, calculating gradients, and updating weights) stays exactly the same. You only need to change three specific things:
1. **The Targets**: Using categories (labels) instead of continuous numbers.
2. **The Output Layer**: Changing how the network delivers results.
3. **The Loss Function**: Changing how we measure error.

In short: The "plumbing" is the same; only the endpoints change.

## 📚 Learning Path

This folder contains a complete classification tutorial using **standard PyTorch** implementations (no external dependencies like d2l):

### **Prerequisites**
- [2. Linear Neural Networks for Regression](../2.%20Linear%20Neural%20Networks%20for%20Regression/) - Complete this first
- Basic PyTorch knowledge
- Understanding of gradient descent

### **Recommended Order**

1. **[3.1 Softmax Regression/Basic.md](3.1%20Softmax%20Regression/Basic.md)**
   - Mathematical foundations of classification
   - Softmax function and cross-entropy loss
   - One-hot encoding and probability interpretation

2. **[3.2 The Image Classification Dataset](3.2%20The%20Image%20Classification%20Dataset/)**  
   - Fashion-MNIST dataset introduction
   - Data loading and preprocessing with PyTorch
   - Visualization and exploration techniques

3. **[3.3 The Base Classification Model](3.3%20The%20Base%20Classification%20Model/)**
   - Reusable classification framework
   - Accuracy metrics and validation
   - Standard PyTorch training patterns

4. **[3.4 Softmax Regression Implementation from Scratch](3.4%20Softmax%20Regression%20Implementation%20from%20Scratch/)**
   - Manual implementation of softmax regression
   - Understanding the internals
   - Custom gradient computation

5. **[3.5 Concise Implementation of Softmax Regression](3.5%20Concise%20Implementation%20of%20Softmax%20Regression/)**
   - PyTorch high-level API approach
   - Numerical stability considerations
   - Production-ready implementation

6. **[3.6 Generalization in Classification](3.6%20Generalization%20in%20Classification/)**
   - Statistical learning theory
   - Overfitting and generalization gaps
   - Test set usage best practices

7. **[3.7 Environment and Distribution Shift](3.7%20Environment%20and%20Distribution%20Shift/)**
   - Real-world deployment challenges
   - Types of distribution shift
   - Practical considerations

## 🛠️ Framework: Standard PyTorch Implementation

**Key Features:**
- **No External Dependencies**: Pure PyTorch + standard libraries
- **Modular Design**: Reusable base classes and utilities  
- **Educational Focus**: Clear, well-documented implementations
- **Production Ready**: Following PyTorch best practices

**Core Components:** ([base_classes.py](base_classes.py))
- `Module`: Base class extending `nn.Module` with utilities
- `Classifier`: Classification-specific base class  
- `FashionMNIST`: Dataset handler with built-in visualization
- `Trainer`: Simple training loop implementation

## 📊 What You'll Learn

### **Mathematical Foundations**
- **Softmax Function**: Converting logits to probabilities
- **Cross-Entropy Loss**: Optimal loss function for classification
- **One-Hot Encoding**: Representing categorical targets
- **Argmax Prediction**: Converting probabilities to class predictions

### **Implementation Skills**
- **PyTorch Datasets**: Using `torchvision.datasets` effectively
- **DataLoaders**: Efficient batch processing and shuffling
- **Neural Network Modules**: Building models with `nn.Module`
- **Training Loops**: Standard supervised learning patterns
- **Metrics**: Implementing and tracking accuracy

### **Advanced Topics**
- **Numerical Stability**: LogSumExp trick and overflow prevention
- **Generalization Theory**: VC dimension and statistical bounds
- **Distribution Shift**: Covariate shift, label shift, concept shift
- **Real-World Deployment**: Common failure modes and solutions

## 🚀 Quick Start

### **Setup**
```bash
# Install dependencies
pip install torch torchvision matplotlib numpy pandas

# Navigate to the classification folder
cd "03-Linear Neural Networks for Classification"
```

### **Run Complete Example**
```python
# Import framework
from base_classes import FashionMNIST, Classifier, Trainer
import torch
import torch.nn as nn

# Load data
data = FashionMNIST(batch_size=256)

# Create model (concise version)
class SoftmaxRegression(Classifier):
    def __init__(self, num_outputs=10, lr=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_outputs)
        )
    
    def forward(self, X):
        return self.net(X)

# Train model
model = SoftmaxRegression()
trainer = Trainer(max_epochs=10)
trainer.fit(model, data)

# Visualize results
data.visualize()
```

## 💡 Key Differences from Regression

| Aspect | Regression | Classification |
|--------|------------|----------------|
| **Targets** | Continuous numbers | Discrete categories |
| **Output** | Single value | Probability distribution |
| **Loss** | Mean Squared Error | Cross-Entropy |
| **Activation** | None (linear) | Softmax |
| **Metrics** | RMSE, MAE | Accuracy, F1-score |
| **Example** | House price: $250,000 | Image class: "Cat" |

## 🎯 Expected Outcomes

After completing this module:

✅ **Understand classification fundamentals** - Softmax, cross-entropy, one-hot encoding  
✅ **Handle image datasets** - Fashion-MNIST loading, preprocessing, visualization  
✅ **Implement from scratch** - Manual softmax regression with gradient computation  
✅ **Use PyTorch effectively** - High-level API, numerical stability, best practices  
✅ **Evaluate models properly** - Accuracy metrics, generalization assessment  
✅ **Recognize real-world challenges** - Distribution shift, deployment considerations  

**Ready for next steps**: Multilayer perceptrons, deep classification networks, and advanced architectures!

## 🔧 Troubleshooting

### **Common Issues**

**Import Errors**
```python
# If you get import errors, ensure base_classes.py is in your Python path
import sys
sys.path.append('/path/to/classification/folder')
from base_classes import *
```

**CUDA Issues**  
```python
# Force CPU usage if needed
device = torch.device('cpu')
model = model.to(device)
```

**Visualization Problems**
```python
# If matplotlib doesn't display
import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg'
```

### **Performance Tips**
- Use `num_workers=0` in DataLoader for Windows
- Reduce batch size if memory issues occur
- Monitor training with built-in plotting utilities

## 📈 Extensions

**Next Learning Steps:**
- Regularization techniques (dropout, weight decay)
- Different optimizers (Adam, RMSprop) 
- Advanced architectures (CNNs for images)
- Multi-class vs. multi-label classification
- Imbalanced dataset handling

**Project Ideas:**
- Custom dataset classification
- Transfer learning experiments  
- Real-world distribution shift analysis
- Comparative study of optimizers