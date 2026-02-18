# GPUs

## 1. The Core Concept: Why GPU? (Parallelism)

- **A CPU is like a Ferrari**: It is incredibly fast at doing one complex thing at a time (sequential processing).
- **A GPU is like a Traffic Jam of 5,000 Vespas**: Individually, they are slower than the Ferrari, but they can move 5,000 people at once (parallel processing).
- **Deep Learning = Matrix Multiplication**: Neural networks don't need complex logic; they need to multiply billions of numbers simultaneously.
- **The Advantage**: A GPU has thousands of tiny cores designed specifically to do the same simple math operation on different data at the same time. This makes training 50x to 100x faster.

## 2. Device Management in PyTorch

In PyTorch, every tensor has a "home" (Device). By default, everything lives on the CPU. You must explicitly move data to the GPU (often called `cuda` for NVIDIA cards or `mps` for Mac).

### The Code (Checking for GPU):

```Python
import torch
from torch import nn

# 1. Check if GPU is available
# 'cuda' = NVIDIA GPU
# 'mps' = Mac M1/M2/M3 GPU
# 'cpu' = Standard Processor
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(try_gpu())
print(try_gpu(10)) # Should return cpu if you don't have 11 GPUs
```

## 3. Tensors on GPU (The "Same Device" Rule)

This is the most common error in Deep Learning: `RuntimeError: Expected all tensors to be on the same device`.

- **The Rule**: You cannot add a CPU tensor to a GPU tensor. They live in different physical memory slots. You must move them to the same place first.
- **The Bottleneck**: Moving data between CPU (RAM) and GPU (VRAM) is slow (it goes through the PCIe bus). Do it once, keep it there, and do all your math there.

### The Math (Parallel Matrix Multiplication):

Let's calculate $\mathbf{C} = \mathbf{A} \times \mathbf{B}$.

If $\mathbf{A}$ is $(N \times N)$ and $\mathbf{B}$ is $(N \times N)$.

The formula for a single element $C_{ij}$ is:

$$C_{ij} = \sum_{k=1}^{N} A_{ik} \cdot B_{kj}$$

- **On CPU**: The processor calculates $C_{1,1}$, then $C_{1,2}$, then $C_{1,3}$... sequentially (or with limited vectorization).
- **On GPU**: The GPU assigns Thread (i, j) to calculate $C_{ij}$. Since there are thousands of threads, the entire matrix $\mathbf{C}$ is computed almost instantly.

### The Code (Moving Tensors):

```Python
# 1. Create a tensor (Defaults to CPU)
X = torch.ones(2, 3)
print(X.device) # device(type='cpu')

# 2. Move it to GPU 0 (if available)
device = try_gpu()
Y = X.to(device)
print(Y.device) # device(type='cuda', index=0)

# 3. Operations on GPU
# This happens FAST because it's parallelized
Z = Y + Y 
print(Z) 

# 4. The Crash (Mixed Devices)
try:
    # This will fail! X is on CPU, Y is on GPU.
    print(X + Y)
except Exception as e:
    print(f"\nError: {e}")
```

## 4. Models on GPU

Just like tensors, the entire Neural Network (its layers, weights, and biases) must be moved to the GPU.

**What happens**: When you call `net.to(device)`, PyTorch iterates through every parameter ($\mathbf{W}, \mathbf{b}$) in the model and copies it to the GPU memory.

### The Code (Moving Models):

```Python
# 1. Define a simple model
net = nn.Sequential(nn.Linear(3, 1))

# 2. Check where the weights are initially (CPU)
print("Weight before:", net[0].weight.data.device)

# 3. Move the WHOLE model to GPU
net = net.to(device)

# 4. Verify
print("Weight after:", net[0].weight.data.device)

# 5. Forward Pass
# Remember: The Input (X) MUST also be on the GPU!
# Y (on GPU) passed into net (on GPU) works perfectly.
pred = net(Y) 
print("Prediction:", pred)
```

## 5. Summary Table

| Concept | CPU | GPU |
|---|---|---|
| **Best For** | Complex logic, small data, sequential tasks. | Massive parallel math, big matrices, Deep Learning. |
| **PyTorch Name** | `torch.device('cpu')` | `torch.device('cuda')` |
| **Math Speed** | Vectorized (SIMD), slower for huge matrices. | Massively Parallel (SIMT), incredibly fast. |
| **Key Rule** | Can only compute with other CPU tensors. | Can only compute with other GPU tensors. |