# Layers and Modules

## 1. The Core Concept: Blocks (LEGO Bricks)

Think of a deep neural network as a complex structure built out of LEGO bricks.

- **The "Brick" (Block)**: In deep learning, a single layer (like a Linear layer) is a block.
- **The "Structure" (Network)**: The entire network (like ResNet or an MLP) is also a block.

- **The Power**: You can put blocks inside other blocks. A block takes input, does some math, and produces output.

In PyTorch, this "Block" is called an `nn.Module`. Every network you build must inherit from this class.

[PLACE IMAGE HERE: A diagram showing a hierarchy. A large box labeled "Network" contains smaller boxes labeled "Layers". Arrows show data flowing input $\rightarrow$ Layer 1 $\rightarrow$ Layer 2 $\rightarrow$ Output.]

## 2. Building a Custom Block (The Code)

To create your own block, you need to define two main things:

- `__init__`: What parts (layers/parameters) do I need? (The Inventory).
- `forward`: How does data flow through these parts? (The Instructions).

Here is how you build a standard Multilayer Perceptron (MLP) from scratch using torch:

```Python
import torch
from torch import nn

class MLP(nn.Module):
    # 1. Define the parts (Inventory)
    def __init__(self):
        super().__init__()
        # Hidden layer: Takes input, keeps it hidden
        self.hidden = nn.Linear(20, 256)
        # Activation function: The non-linear "spark"
        self.relu = nn.ReLU()
        # Output layer: Produces the final result
        self.output = nn.Linear(256, 10)

    # 2. Define the flow (Instructions)
    def forward(self, x):
        # Step A: Pass input through hidden layer
        h = self.hidden(x)
        # Step B: Apply activation
        h_activated = self.relu(h)
        # Step C: Pass to output layer
        out = self.output(h_activated)
        return out

# How to use it:
net = MLP()
X = torch.rand(2, 20) # Random input
print(net(X))
```

## 3. Mathematical Theory (Detailed)

What exactly happens inside `self.hidden(x)`? It performs an **Affine Transformation**.

### Variable Definitions:

- $\mathbf{X}$: The input batch matrix (shape: $batch\_size \times n_{in}$).
- $\mathbf{W}$: The weight matrix (shape: $n_{in} \times n_{out}$).
- $\mathbf{b}$: The bias vector (shape: $1 \times n_{out}$).
- $\mathbf{H}$: The output matrix (shape: $batch\_size \times n_{out}$).
- $\phi$: The activation function (e.g., ReLU).

### The Forward Formula:

The mathematical operation for a single fully connected layer is:

$$\mathbf{H} = \mathbf{X}\mathbf{W} + \mathbf{b}$$

If we include the activation function (like in the forward method above), the full formula for the hidden layer becomes:

$$\mathbf{H}_{activated} = \phi(\mathbf{X}\mathbf{W} + \mathbf{b})$$

### Deep Dive into the Math:

**Matrix Multiplication ($\mathbf{X}\mathbf{W}$):**
Each output neuron calculates a weighted sum of all input features. If $\mathbf{X}$ has dimensions $(2, 20)$ and $\mathbf{W}$ has dimensions $(20, 256)$, the result is $(2, 256)$.

**Broadcasting Bias ($+ \mathbf{b}$):**
The bias $\mathbf{b}$ is a single vector of size $256$. However, we have a batch of 2 samples. The math "broadcasts" (copies) this bias to every row in the batch so it can be added to the matrix product.

$$\mathbf{H}_{i,j} = (\sum_{k} \mathbf{X}_{i,k} \mathbf{W}_{k,j}) + \mathbf{b}_j$$

[PLACE IMAGE HERE: A visual representation of Matrix Multiplication. Show a Row from Matrix X multiplying with a Column from Matrix W to produce a single cell in the Output Matrix.]

## 4. Sequential Blocks (The Shortcut)

Often, we just want to stack layers in a straight line without writing a custom class every time. PyTorch provides `nn.Sequential` for this. It is a container that essentially says: "Run the data through the first item, then the second, then the third..."

### The Math of Sequencing:

If you have three layers $f_1, f_2, f_3$, a Sequential block performs function composition:

$$\text{Output} = f_3(f_2(f_1(\mathbf{x})))$$

### The Code:

```Python
# This does the exact same thing as the "class MLP" above
net_seq = nn.Sequential(
    nn.Linear(20, 256),  # f1
    nn.ReLU(),           # f2
    nn.Linear(256, 10)   # f3
)

print(net_seq(X))
```

[PLACE IMAGE HERE: A diagram of a Conveyor Belt. Raw material (Input) sits on the belt and passes through Machine 1, then Machine 2, then Machine 3, emerging as a finished Product (Output).]

## 5. Flexible Forward Pass (Control Flow)

The main reason we write custom classes (Concept #2) instead of always using Sequential (Concept #4) is **Control Flow**. In a custom block, you can use Python logic (loops, if-statements) inside the neural network.

For example, a "Fixed Hidden MLP" that doesn't learn weights but simply calculates a constant math operation:

```Python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # A constant weight that doesn't change (not trained)
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        # Matrix multiplication with constant weights
        x = torch.mm(x, self.rand_weight) + 1
        # Custom Logic: Reuse the layer multiple times!
        x = self.linear(x)
        # Custom Loop: Keep halving the value until it's small
        while x.abs().sum() > 1:
            x /= 2
        return x.sum()

net_flexible = FixedHiddenMLP()
print(net_flexible(X))
```

### Mathematical Insight:

This demonstrates that a Neural Network is not just a static equation. It is a computational program. The formula changes dynamically based on the data values (e.g., the while loop changes the effective divisor based on the magnitude of $x$).