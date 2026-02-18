# Multilayer Perceptrons

## 1. What is a Multilayer Perceptron (MLP)?
An MLP is one of the simplest and most fundamental types of deep neural networks. Think of it as a team of decision-makers working in stages.
- **The Problem with Simple Models**: A simple "linear" model (like a single perceptron) draws a straight line to separate data. It works for simple things (like separating apples from oranges by size) but fails at complex patterns (like the "XOR" problem, where data points are mixed in a checkerboard pattern).
- **The Solution (Hidden Layers)**: To solve this, we stack layers of neurons on top of each other. The layers in the middle are called Hidden Layers.
- **The "Secret Sauce" (Activation Functions)**: Just stacking layers isn't enough (stacking straight lines just makes a new straight line). We need a mathematical "spark" called an Activation Function to bend the lines and capture complex shapes.

![Comparison of a Linear Model (straight line cut) vs. an MLP (curved/complex cut) solving the XOR problem](img/1.png)

## 2. How It Works (The Architecture)
An MLP consists of three main stages. Here is the flow of data:
- **Input Layer**:
    - This is where the raw data enters (e.g., the pixels of an image)
    - It does not do any math; it just passes the numbers to the next layer.
- **Hidden Layer(s)**:
    - This is where the magic happens.
    - It takes the input, multiplies it by a set of "weights" (importance scores), and adds a "bias" (threshold).
    - Crucially, it applies an **Activation Function** to the result. This allows the network to learn non-linear relationships.
    - You can have multiple hidden layers stacked to learn deeper, more abstract patterns.
- **Output Layer**:
    - This layer produces the final prediction (e.g., "Cat" or "Dog").
    - It takes the processed information from the last hidden layer and transforms it into the desired output format.
    
![ standard MLP architecture diagram showing Input nodes $\rightarrow$ Hidden nodes $\rightarrow$ Output nodes, connected by arrows](img/2.png)

## 3. Mathematical Theory (Detailed)
This section details exactly what happens inside the network mathematically. We assume we have a "minibatch" of samples (processing multiple items at once).

### Variable Definitions:
- $\mathbf{X}$: The input data (a matrix where rows are samples, columns are features).
- $\mathbf{W}^{(1)}$: The weight matrix connecting the Input to the Hidden layer.
- $\mathbf{b}^{(1)}$: The bias vector for the Hidden layer.
- $\mathbf{H}$: The output of the Hidden layer (before the next layer uses it).
- $\sigma$ (sigma): The activation function.
- $\mathbf{W}^{(2)}$: The weight matrix connecting the Hidden to the Output layer.
- $\mathbf{b}^{(2)}$: The bias vector for the Output layer.
- $\mathbf{O}$: The final Output.

### The Step-by-Step Formulas:

**Step 1: Calculating the Hidden Layer ($\mathbf{H}$)**
First, we calculate the weighted sum of the inputs and add the bias. Then, we wrap the result in the activation function $\sigma$.

$$\mathbf{H} = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$$

- **Explanation**: The matrix multiplication $\mathbf{X}\mathbf{W}^{(1)}$ combines every input feature with every hidden neuron. The bias $\mathbf{b}^{(1)}$ shifts the result. The function $\sigma(\cdot)$ transforms the linear math into a non-linear signal.

**Step 2: Calculating the Output Layer ($\mathbf{O}$)**

Next, we take the hidden layer's output $\mathbf{H}$ and pass it to the output layer.

$$\mathbf{O} = \mathbf{H} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$$

- **Explanation**: The hidden features are now combined (weighted by $\mathbf{W}^{(2)}$) to form the final prediction. Note that we usually do not apply an activation function here if we are doing regression, or we apply a specific one like Softmax if we are doing classification.

## 4. Activation Functions (The Non-Linearity)
The activation function is the decision-maker. It decides if a neuron should "fire" or stay inactive. Without these, the MLP would just be a big Linear Regression model.

![Graphs showing the shapes of ReLU, Sigmoid, and Tanh functions side-by-side](img/3.png)

Here are the three most common ones mentioned in the D2L chapter:

### A. ReLU (Rectified Linear Unit)
This is the most popular choice today because it is simple and fast to compute.

- **Concept**: "If the number is positive, keep it. If it's negative, make it zero."
- **Formula**:

$$\text{ReLU}(x) = \max(x, 0)$$

- **Why use it**: It solves the "vanishing gradient" problem better than Sigmoid and calculates very quickly.

### B. Sigmoid
This squeezes numbers into a range between 0 and 1.
- **Concept**: "Squash any number, no matter how big or small, into a probability-like score."
- **Formula**:

$$\text{sigmoid}(x) = \frac{1}{1 + \exp(-x)}$$

- **Why use it**: Good for binary (Yes/No) outputs, but rarely used in hidden layers nowadays because it can slow down learning (vanishing gradients).

### C. Tanh (Hyperbolic Tangent)
Similar to Sigmoid, but squeezes numbers between -1 and 1.
- **Concept**: "Squash the number so it is centered around zero."
- **Formula**:

$$\text{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}$$

**Why use it**: It centers data at 0, which can help optimization in some cases, but still suffers from similar issues as Sigmoid in very deep networks.