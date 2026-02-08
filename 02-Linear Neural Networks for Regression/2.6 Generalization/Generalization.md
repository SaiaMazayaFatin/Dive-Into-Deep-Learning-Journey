# Generalization

## 1. The Core Problem: Generalization

The fundamental goal of machine learning is to discover **patterns** that apply to new, unseen data, rather than simply **memorizing** the training data.

- **The Student Analogy**:
    - **"Ellie" (Memorization)**: Memorizes answers to past exams. She gets 100% on old questions but fails new ones.
    - **"Irene" (Generalization)**: Learns patterns. She might get 90% on old exams but maintains that 90% accuracy on new, unseen questions.
- **Goal**: We want "Irene"—a model that performs well on previously undiagnosed ailments in previously unseen patients.

## 2. The Statistical Assumption (IID)
To assume that patterns learned from training data will apply to test data, we rely on the **IID Assumption**:
- Independent: Samples are not related to each other
- Identically Distributed: Training and test data are drawn from the exact same underlying distribution $P(X,Y)$.

Without this assumption, we cannot justify applying our model to new data.

## 3. Training vs. Generalization Error
We must distinguish between how the model performs on data it knows versus data it doesn't know.

### A. Training Error ($R_\textrm{emp}$)

This is a **statistic** calculated on the specific data we have. It is the average loss over the training set. **Formula:**

$$R_\textrm{emp}[\mathbf{X}, \mathbf{y}, f] = \frac{1}{n} \sum_{i=1}^n l(\mathbf{x}^{(i)}, y^{(i)}, f(\mathbf{x}^{(i)}))$$

## B. Generalization Error ($R$)

This is an **expectation** (probability) of how the model performs on an infinite stream of new data. We can never calculate this exactly; we can only estimate it using a Test Set.**Formula:**

$$R[p, f] = E_{(\mathbf{x}, y) \sim P} [l(\mathbf{x}, y, f(\mathbf{x}))] = \int \int l(\mathbf{x}, y, f(\mathbf{x})) p(\mathbf{x}, y) \;d\mathbf{x} dy$$

## 4. Model Complexity & Selection
A model's ability to fit data is called its complexity (or capacity).

**Polynomial Curve Fitting Example**

Consider predicting a label $y$ using a polynomial of degree $d$.**Formula:**

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

- Low degree ($d$): Simple model. Cannot fit complex curves
- High degree ($d$): Complex model. Can fit (or memorize) noise in the training data.

**Underfitting vs. Overfitting**

We evaluate models by comparing Training Error vs. Validation (Holdout) Error.

![Underfitting vs Overfitting](img/1.png)

1. **Underfitting**:
    - **Signs**: High Training Error + High Validation Error
    - **Gap**: Small gap between the two errors.
    - **Cause**: Model is too simple (insufficiently expressive) to capture the pattern.
2. **Overfitting**:
    - **Signs**: Low Training Error + High Validation Error
    - **Gap**: Large gap (Training error is significantly lower than Validation error).
    - **Cause**: Model is too complex; it has memorized the training data (including noise) rather than learning the rule.
## 5. Managing Data and Model Selection

Since we cannot verify Generalization Error directly, we use dataset splitting strategies.

**The Three-Way Split**

1. **Training Set**: Used to fit the model parameters
2. **Validation Set**: Used to select the best model (hyperparameters). Crucially, we must not use the Test Set for this, or we risk overfitting the test data.
3. **Test Set**: Used only once at the very end to estimate the final generalization error.

**K-Fold Cross-Validation**

When data is scarce (e.g., medical data with only hundreds of samples), we cannot afford to throw away a chunk for a validation set.

- **Method**: Split data into $K$ parts.
- **Process**: Train on $K-1$ parts, validate on the remaining 1 part. Repeat $K$ times.
- **Result**: Average the $K$ validation errors to get a stable estimate.

**Summary Rules of Thumb**
1. **Use Validation Sets**: Always keep a holdout set (or use Cross-Validation) to select your model.
2. **More Data is Better**: Increasing $n$ (dataset size) almost always improves generalization.
3. **Balance Complexity**:
    - Simple Model + Complex Data $\rightarrow$ Underfitting
    - Complex Model + Scant Data $\rightarrow$ Overfitting 
4. **IID Matters**: If the future doesn't look like the past (distribution shift), these guarantees fail.

