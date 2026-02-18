# Base Classification Model

## 1. The ``Classifier`` Base Class
Since classification is such a common task, we create a reusable base class to handle the repetitive parts of training and validation.

**What it does**:
- It inherits from a base ``Module`` class (a wrapper around ``nn.Module``).
- **Validation Step**: It calculates both Loss and Accuracy for the validation set and plots them.
- **Optimizer**: By default, it uses Stochastic Gradient Descent (SGD).

```Python
import torch
import torch.nn as nn
import torch.optim as optim
from base_classes import Module, Classifier

class Classifier(Module):
    """The base class of classification models."""
    def validation_step(self, batch):
        *inputs, targets = batch
        predictions = self(*inputs)
        
        # Calculate loss and accuracy
        loss = self.loss(predictions, targets)
        acc = self.accuracy(predictions, targets)
        
        # Store for plotting
        self.plot('loss', loss.item(), train=False)
        self.plot('acc', acc.item(), train=False)
        
        return {'val_loss': loss, 'val_acc': acc}
        
    def configure_optimizers(self):
        """Default optimizer configuration."""
        return optim.SGD(self.parameters(), lr=getattr(self, 'lr', 0.01))
```

## 2. Accuracy (The Metric)
While we train models using differentiable Loss functions (like Cross-Entropy), humans care about Accuracy: "What percentage of predictions were correct?".

**The Logic**:
1. **Get Predictions**: The model outputs probabilities (scores) for every class. We take the class with the highest score using **argmax**.
    - **Formula**: $\hat{y} = \operatorname*{argmax}_j P(y=j \mid x)$
2. **Compare**: We check if the predicted class equals the true label $y$.
3. **Calculate Mean**: We convert the True/False results into 1s and 0s and take the average.

**The Code:**

```Python
# Add this method to the Classifier class
def accuracy(self, predictions, targets, averaged=True):
    """Compute the number of correct predictions."""
    # 1. Reshape predictions to ensure it's a matrix of shape (samples, classes)
    predictions = predictions.reshape((-1, predictions.shape[-1]))
    
    # 2. Find the class with the highest score (argmax)
    # We cast to the same type as targets (labels) for comparison
    predicted_classes = predictions.argmax(dim=1).type(targets.dtype)
    
    # 3. Compare predictions to truth
    correct = (predicted_classes == targets.reshape(-1)).type(torch.float32)
    
    # 4. Return the mean (accuracy percentage) or the raw comparison vector
    return correct.mean() if averaged else correct
```

**Why do we need this?**
- Loss functions (like Cross Entropy) are essentially "soft" errors used for calculus.
- Accuracy is a "hard" metric used for benchmarking performance
- Because ``argmax`` is not differentiable, we cannot use Accuracy directly as our Loss function for training.

## 3. Summary
- ``Classifier`` **Class:** A standardized template that automatically handles validation plotting and SGD configuration
- ``accuracy`` **Method: A** utility to convert raw model probabilities into a human-readable percentage of correct guesses
- **Workflow**:
    1. Model outputs raw scores (logits).
    2. argmax finds the predicted class.
    3. Compare with ground truth.
    4. Average the results to get Accuracy.