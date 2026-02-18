# Numerical Stability and Initialization


## 1. The Problem: Numerical Instability
When training deep neural networks, we often face two major mathematical disasters: **Vanishing Gradients** and **Exploding Gradients**.
- **The "Telephone Game" Analogy**: Imagine you are whispering a message through a line of 100 people (layers).
    - **Vanishing Gradient**: If everyone whispers slightly quieter than the person before, the message becomes silent (zero) before it reaches the end. The network stops learning because the error signal disappears.
    - **Exploding Gradient**: If everyone shouts slightly louder than the person before, the message becomes a deafening scream (infinity/NaN) and breaks the system.
    
![A diagram comparing "Normal Signal Flow" vs "Vanishing (arrow gets thinner)" vs "Exploding (arrow gets huge)" through layers](img/1.png)

## 2. Vanishing Gradients (The "Disappearing" Error)
This happens when the gradients (the signals telling the network how to fix itself) become so small that the weights stop updating.
- **The Culprit**: Activation functions like Sigmoid.
- **Why?** The Sigmoid function squashes large numbers into the range [0, 1]. Its derivative (slope) is even smaller—at best, it is 0.25.
- **The Result**: If you multiply many small numbers (e.g., $0.25 \times 0.25 \times 0.25 \dots$), the result quickly becomes almost zero.

### Mathematical Detail:
Consider the chain rule for a network with $L$ layers. To find the gradient for the first layer, we multiply the gradients of all subsequent layers:

$$\frac{\partial Loss}{\partial \mathbf{W}^{(1)}} \approx \underbrace{\frac{\partial \mathbf{h}^{(L)}}{\partial \mathbf{h}^{(L-1)}} \times \dots \times \frac{\partial \mathbf{h}^{(2)}}{\partial \mathbf{h}^{(1)}}}_{\text{Product of } L-1 \text{ matrices}} \times \frac{\partial \mathbf{h}^{(1)}}{\partial \mathbf{W}^{(1)}}$$

If we use a Sigmoid activation $\sigma(x)$, its derivative is:

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

The maximum value of this derivative is **0.25**.

Therefore, if the weight matrices $\mathbf{W}$ are not large enough to counteract this, the gradient decreases exponentially with depth $L$:

$$\text{Gradient} \propto (0.25)^{L} \approx 0$$

![A graph of the Sigmoid function and its derivative. Highlight that the derivative curve (bell shape) peaks at 0.25 and is almost 0 everywhere else.](img/2.png)

## 3. Exploding Gradients (The "Snowball" Effect)
This is the opposite problem. The gradients become too large, causing the weights to oscillate wildly or hit "NaN" (Not a Number), crashing the training.
- **The Cause**: Often happens when initialization weights are too large.
- **The Result**: If you multiply many numbers greater than 1 (e.g., $2 \times 2 \times 2 \dots$), the result exponentially grows to infinity.

### Mathematical Detail:
Imagine a deep network with no activation function (linear). The output $\mathbf{h}^{(L)}$ is essentially a product of weight matrices acting on the input $\mathbf{x}$:

$$\mathbf{h}^{(L)} = \mathbf{W}^{(L)} \times \mathbf{W}^{(L-1)} \times \dots \times \mathbf{W}^{(1)} \mathbf{x}$$

If the standard deviation (spread) of the weights in each matrix $\mathbf{W}$ is even slightly larger than 1 (say, 1.5), and we have 50 layers:

$$1.5^{50} \approx 637,621,500$$

The signal explodes, making stable learning impossible.

## 4. The Solution: Proper Initialization
To fix this, we need to initialize the random weights just right—not too small, not too big. This is the "Goldilocks" zone.

The goal is to preserve the **Variance** (spread of data) across layers.
1. **Forward Pass Goal**: The variance of the output should equal the variance of the input
2. **Backward Pass Goal**: The variance of the gradients should stay consistent effectivel

![Visualization of "Variance Preservation". Show a distribution curve (bell curve) entering a layer and leaving with the same width, rather than getting flatter or sharper.](img/3.png)

## 5. Xavier (Glorot) Initialization
Designed for **Sigmoid** or **Tanh** activation functions.
### The Theory (Simplified Math):
Let's look at a single neuron output $o_i$ which is a weighted sum of inputs $x_j$:

$$o_i = \sum_{j=1}^{n_{in}} w_{ij} x_j$$

We assume the inputs and weights have a mean of 0. We want the variance of the output $Var(o_i)$ to equal the variance of the input $Var(x)$.

Using probability theory, the variance of a sum of independent variables is:

$$Var(o_i) = \sum_{j=1}^{n_{in}} Var(w_{ij} x_j)$$

$$Var(o_i) = \sum_{j=1}^{n_{in}} Var(w_{ij}) Var(x_j)$$

$$Var(o_i) = n_{in} \cdot Var(W) \cdot Var(x)$$

To keep the variance the same ($Var(o_i) = Var(x)$), we must enforce:

$$n_{in} \cdot Var(W) = 1 \implies Var(W) = \frac{1}{n_{in}}$$

To satisfy this for both forward pass ($n_{in}$) and backward pass ($n_{out}$), Xavier initialization suggests taking the harmonic mean:

$$Var(W) = \frac{2}{n_{in} + n_{out}}$$
### The Formula to use:
Draw weights from a uniform distribution $U[-a, a]$ where:

$$a = \sqrt{\frac{6}{n_{in} + n_{out}}}$$

## 6. He Initialization (Kaiming Init)
Designed specifically for ReLU activation functions.
- **Why specific?** ReLU kills half of the neurons (sets negative values to zero). This halves the variance.
- **The Fix**: We need to double the variance of the weights to compensate for the "dead" neurons.

### The Formula to use:

$$Var(W) = \frac{2}{n_{in}}$$

Draw weights from a Normal distribution $N(0, \sigma^2)$ where:

$$\sigma = \sqrt{\frac{2}{n_{in}}}$$