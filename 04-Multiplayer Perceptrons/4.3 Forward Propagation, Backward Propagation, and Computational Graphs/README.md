# Forward Propagation, Backward Propagation, and Computational Graphs

## 1. The Big Picture: Forward vs. Backward
Training a neural network is a two-way street.
- **Forward Propagation (The Guess)**: Data flows from the input to the output to make a prediction and calculate how wrong it is (the loss).
- **Backward Propagation (The Blame)**: We retrace our steps from the loss back to the start. We calculate "gradients" to figure out which weights contributed most to the error so we can fix them.

![A simple diagram showing two arrows: one pointing right labeled "Forward Propagation (Compute Output)" and one pointing left labeled "Backpropagation (Compute Gradients)"](img/1.png)

## 2. Forward Propagation (Computing the Output)
This is the process of calculating the neural network's output layer by layer. Let's look at a network with one **hidden layer**.
### The Steps:
1. **Input**: We start with our input data, $\mathbf{x}$.
2. **Hidden Layer (Intermediate Step)**: We calculate an intermediate value $\mathbf{z}$ (linear part) and then apply the activation function $\phi$ to get $\mathbf{h}$ (non-linear part).
3. **Output Layer (Final Prediction)**: We take $\mathbf{h}$ and use it to calculate the final output $\mathbf{o}$.
4. **Loss Calculation**: We compare the prediction $\mathbf{o}$ with the true label $y$ to get the loss $L$. We usually add a "regularization" term $s$ (like L2 penalty) to prevent overfitting.
### The Detailed Formulas:
- **Step A (Hidden Layer)**:

    Multiply input by weights $\mathbf{W}^{(1)}$ and add bias.
    
    $$\mathbf{z} = \mathbf{W}^{(1)} \mathbf{x}$$
    
    Apply the activation function (like ReLU or Sigmoid):
    
    $$\mathbf{h} = \phi(\mathbf{z})$$
    
- **Step B (Output Layer)**:

    Multiply the hidden output $\mathbf{h}$ by the second set of weights $\mathbf{W}^{(2)}$:
    
    $$\mathbf{o} = \mathbf{W}^{(2)} \mathbf{h}$$
    
- **Step C (Total Loss)**:

    Calculate the error using a loss function $l$ (like Mean Squared Error) and add the regularization term $s$:
    
    $$L = l(\mathbf{o}, y) + \frac{\lambda}{2} (\|\mathbf{W}^{(1)}\|^2 + \|\mathbf{W}^{(2)}\|^2)$$
    
    ![A computational graph diagram showing nodes connected by arrows: x $\to$ z $\to$ h $\to$ o $\to$ L](img/2.png)

## 3. Computational Graph
To perform backpropagation efficiently, deep learning frameworks (like PyTorch or TensorFlow) build a **Computational Graph**.
- **What is it?** It is a visual map of the math operations we just did.
- **Why use it?** It keeps track of which variable created which result. This mapping is essential because when we calculate derivatives (gradients), we need to follow these distinct paths in reverse.

## 4. The Chain Rule (The Key Mathematical Concept)
Before explaining Backpropagation, you must understand the **Chain Rule** of calculus.
- **The Concept**: If variable $A$ affects $B$, and variable $B$ affects $C$, then the effect of $A$ on $C$ is the product of their individual effects.
- **The Formula**:

$$\frac{\partial C}{\partial A} = \frac{\partial C}{\partial B} \times \frac{\partial B}{\partial A}$$

This allows us to calculate how a weight in the first layer affects the Loss in the last layer by multiplying the links in between.

## 5. Backward Propagation (Computing the Gradients)
Now we traverse the graph in reverse (from Loss $L$ back to Input $\mathbf{x}$) to calculate the gradients. The goal is to find $\frac{\partial L}{\partial \mathbf{W}}$ (how much the Loss changes if we tweak the weights).
### The Steps & Detailed Formulas:
**Step 1: Gradient of the Output Layer**

First, we check how the Loss changes with respect to the output $\mathbf{o}$.

$$\frac{\partial L}{\partial \mathbf{o}} = \frac{\partial l(\mathbf{o}, y)}{\partial \mathbf{o}}$$

(For example, if the loss is Mean Squared Error, this is just $(\mathbf{o} - y)$).

**Step 2: Gradient of the Output Weights ($\mathbf{W}^{(2)}$)**

We use the chain rule. The weights $\mathbf{W}^{(2)}$ affected the output $\mathbf{o}$, which affected the Loss.

$$\frac{\partial L}{\partial \mathbf{W}^{(2)}} = \frac{\partial L}{\partial \mathbf{o}} \cdot \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}$$

(Note: The term $\lambda \mathbf{W}^{(2)}$ comes from the regularization penalty).

**Step 3: Gradient of the Hidden Layer ($\mathbf{h}$)**

Now we move back to the hidden layer. How much did the hidden layer values $\mathbf{h}$ contribute to the error?

$$\frac{\partial L}{\partial \mathbf{h}} = (\mathbf{W}^{(2)})^\top \cdot \frac{\partial L}{\partial \mathbf{o}}$$

**Step 4: Gradient of the Hidden Weights ($\mathbf{W}^{(1)}$)**

This is the trickiest part. The weights $\mathbf{W}^{(1)}$ affected $\mathbf{z}$, which affected $\mathbf{h}$ (through the activation function $\phi$). We must multiply by the derivative of the activation function, $\phi'(\mathbf{z})$.

$$\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \left( \frac{\partial L}{\partial \mathbf{h}} \odot \phi'(\mathbf{z}) \right) \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}$$

- Symbol Note: The symbol $\odot$ means element-wise multiplication. We are multiplying the error coming back from the top layers by the slope of the activation function at the current point.

![A visualization of the Chain Rule flow. Show the error signal originating at 'L' and splitting/flowing back through 'o', then 'h', then 'z', updating W2 and W1 along the way.](img/3.png)