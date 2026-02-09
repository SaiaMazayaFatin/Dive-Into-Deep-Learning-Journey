# Generalization in Classification

## 1. The Core Problem: Generalization
Our goal is not just to do well on the data we have (Training Set), but to learn patterns that apply to new, unseen data (Test Set).- **Memorization vs. Learning**: A model could get 100% accuracy by simply memorizing every training image. This is useless for new images.
- **Generalization Gap**: The difference between the model's performance on training data vs. new test data.
- **The Risk**: A model can achieve 100% training accuracy by memorizing labels, but this strategy fails completely on new inputs.

## 2. Empirical vs. Population Error
In classification, we want to know how often our model makes mistakes. We distinguish between the error we can measure (Empirical) and the error we care about (Population)

**A. Empirical Error** ($\epsilon_\mathcal{D}$) This is the error calculated on a specific dataset $\mathcal{D}$ of size $n$. It is the average number of disagreements between the model's prediction $f(\mathbf{x}^{(i)})$ and the true label $y^{(i)}$.

$$\epsilon_\mathcal{D}(f) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(f(\mathbf{x}^{(i)}) \neq y^{(i)})$$

- $n$: The total number of examples in dataset $\mathcal{D}$.
- $\mathbf{1}(\cdot)$: The Indicator Function. It returns $1$ if the condition inside is true (error), and $0$ if false (correct).

**B. Population Error** ($\epsilon$) This is the expected error over the entire real-world distribution $P(X,Y)$. It represents the probability that the model will make a mistake on a randomly drawn example from the infinite population.

$$\epsilon(f) =  E_{(\mathbf{x}, y) \sim P} \mathbf{1}(f(\mathbf{x}) \neq y) = \int\int \mathbf{1}(f(\mathbf{x}) \neq y) p(\mathbf{x}, y) \;d\mathbf{x} dy$$

- $E$: Expectation (weighted average by probability).
- $p(\mathbf{x}, y)$: The probability density function of the data.

## 3. How Much Test Data Do We Need?
We use statistical theory to determine how large our test set ($n$) needs to be to trust our Empirical Error.

### The Central Limit Theorem (CLT) & Asymptotic Rate
We use the test set $\mathcal{D}$ to estimate the population error $\epsilon(f)$. Since $\epsilon_\mathcal{D}(f)$ is just a sample average, the Central Limit Theorem applies.

**The Theorem**:As $n \to \infty$, the distribution of the sample error approaches a Normal Distribution centered at the true error $\epsilon(f)$.

**The Standard Deviation of the Error**:The error is a Bernoulli random variable (0 or 1). Its variance is maximal when the error rate is $0.5$.

$$\sigma = \sqrt{\frac{\epsilon(f)(1-\epsilon(f))}{n}} \leq \sqrt{\frac{0.25}{n}} = \frac{0.5}{\sqrt{n}}$$

**The Convergence Rate**:The error in our estimate shrinks at a rate of:

$$\text{Error Rate} \approx \mathcal{O}\left(\frac{1}{\sqrt{n}}\right)$$

- To halve the uncertainty, you need $4 \times$ data.
- To reduce uncertainty by $10 \times$, you need $100 \times$ data.

### Hoeffding's Inequality (Finite Sample Guarantees)
While CLT tells us what happens at infinity, **Hoeffding's Inequality** gives us a guarantee for finite datasets. It bounds the probability that our estimated error is far from the true error.

**The Formula**:

$$P(\epsilon_\mathcal{D}(f) - \epsilon(f) \geq t) < \exp\left( - 2n t^2 \right)$$

- $t$: The "tolerance" or margin of error (e.g., $0.01$).
- $n$: The number of test samples.
- **Interpretation**: The probability that the estimated error is off by more than $t$ drops exponentially as $n$ increases.

**Example Calculation**:If we want to be 95% confident (probability of failure $< 0.05$) that our estimate is within $t=0.01$:

$$0.05 = \exp(-2n(0.01)^2) \implies n \approx 15,000$$

This explains why standard benchmarks (like MNIST/Fashion-MNIST) often have test sets of 10,000 images.

## 4. Statistical Learning Theory: VC Dimension
What if we want to guarantee generalization before seeing the test set? We use **Uniform Convergence** to bound the error for all possible models in a class $\mathcal{F}$ simultaneously

**Vapnik-Chervonenkis (VC) Dimension** This measures the complexity (flexibility) of a model class.
- **Definition**: The maximum number of points that the model class can perfectly classify (shatter) for any random labeling.
- **Linear Models**: For inputs of dimension $d$, the VC dimension is $d+1$.

**The Generalization Bound**:With probability $1-\delta$, the gap between training error and true error is bounded by:

$$R[p, f] - R_\textrm{emp}[\mathbf{X}, \mathbf{Y}, f] < c \sqrt{\frac{\text{VC} - \log \delta}{n}}$$

- **Insight**: The gap $\alpha$ grows if the model is too complex (high VC dimension) but shrinks if we have more data (high $n$).
- **Deep Learning Paradox**: Deep networks have massive VC dimensions but still generalize well, contradicting this classical pessimistic bound.

## 5. Practical Pitfalls: Test Set Reuse
A true test set must be "fresh" and unseen.

- **Adaptive Overfitting**: If you evaluate a model on the test set, tweak it, and evaluate again, information "leaks" from the test set to the model. The test set is no longer unbiased.
- **Multiple Hypothesis Testing**: If you test 20 different models, one might perform well on the test set purely by chance (False Discovery).
- **Solution**: Use a separate Validation Set for tuning and only use the Test Set once for the final evaluation.

## 6. Conceptual Code

```Python
import torch

def empirical_error(net, data_iter):
    """
    Computes the Empirical Error (1 - Accuracy) on a dataset.
    Formula: (1/n) * Sum(Indicator(prediction != label))
    """
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set model to evaluation mode
        
    mistakes = 0
    total = 0
    
    with torch.no_grad():
        for X, y in data_iter:
            # Get predictions
            y_hat = net(X)
            preds = y_hat.argmax(axis=1)
            
            # Count how many predictions did NOT match the label
            # This is the Indicator Function 1(f(x) != y)
            mistakes += (preds != y).sum().item()
            total += y.numel()
            
    return mistakes / total

# Example: If mistakes=5 and total=100, Empirical Error = 0.05
```