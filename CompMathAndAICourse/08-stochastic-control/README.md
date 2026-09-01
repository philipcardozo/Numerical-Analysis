# Lecture 08: High-Dimensional Stochastic Optimal Control

**Course:** Computational Mathematics and AI

This directory contains Jupyter notebooks demonstrating neural network methods for solving high-dimensional stochastic optimal control problems, specifically the 100D benchmark from the Deep BSDE literature.

## Key Insight

> **"The sampling strategy, not just the neural network, determines success!"**

Neural networks + FBSDEs alone don't break the curse of dimensionality. Three components must work together:
1. **Neural Networks** - Function approximation with polynomial parameters
2. **Monte Carlo Integration** - Dimension-free convergence O(N^{-1/2})
3. **Smart Sampling** - PMP-informed trajectories (THE KEY!)

## Notebooks

| Notebook | Topic | Open in Colab |
|----------|-------|---------------|
| `stochastic-control.ipynb` | Complete comparison of PINN, FBSNN, and NeuralSOC methods | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lruthotto/CompMathAndAICourse/blob/main/08-stochastic-control/stochastic-control.ipynb) |

## Problem: 100D Stochastic Optimal Control

**Objective:**
$$\min_u J = \mathbb{E}\left[\int_0^T \|u\|^2 \, dt + g(X_T)\right]$$

**Dynamics:**
$$dX = 2u \, dt + \sqrt{2} \, dW$$

**Terminal cost options:**
- Log cost: $g(x) = \log\left(\frac{1 + \|x - x_{\text{target}}\|^2}{2}\right)$
- Quadratic cost: $g(x) = \|x - x_{\text{target}}\|^2$

**HJB Equation:**
$$-\frac{\partial \Phi}{\partial t} + \Delta \Phi - \|\nabla \Phi\|^2 = 0, \quad \Phi(T,x) = g(x)$$

**Optimal Control (from Pontryagin Maximum Principle):**
$$u^* = -\nabla_x \Phi$$

## Method Comparison

| Method | Sampling | Loss Function | Works on Shifted Target? |
|--------|----------|---------------|--------------------------|
| **PINN** | Random walk (trajectory-based) | HJB residual + terminal BC | ✓ Yes (with modifications) |
| **FBSNN** | Random walk (pure) | BSDE residual + terminal | ✗ FAILS! |
| **NeuralSOC** | PMP-guided | Control objective + terminal | ✓ Yes |

### Why FBSNN Fails on Shifted Targets

When the target is at $(3, 3, \ldots, 3)$ instead of the origin:
- Random walk trajectories stay near the origin
- They never explore the region around the target
- The network has no training signal for the relevant state space!

NeuralSOC succeeds because it uses the learned optimal control to guide sampling toward the target.

## Directory Structure

```
08-stochastic-control/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── problem.py                   # Problem definition, SDE sampling, evaluation
├── networks.py                  # Neural network architectures (MLP, ResNet)
├── solvers.py                   # PINN, FBSNN, NeuralSOC solvers
├── trainer.py                   # Training utilities
├── visualization.py             # Plotting functions
├── stochastic-control.ipynb     # Main experiment notebook
└── figures/                     # Generated plots and results
```

## Running Locally

### Prerequisites

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Notebooks

```bash
jupyter lab
# Open stochastic-control.ipynb and run all cells
```

## Configuration

The notebook includes a configuration section where you can modify:

```python
# Problem settings
D = 100                    # State dimension
T = 1.0                    # Terminal time
SHIFTED_TARGET = False     # If True, target at (3,3,...,3)
TERMINAL_COST_TYPE = "log" # "log" or "quadratic"

# Network architecture
ARCHITECTURE = "resnet"    # "resnet" or "mlp"
HIDDEN_WIDTH = 128         # Width of hidden layers
DEPTH = 4                  # Number of layers/blocks

# Training settings
MAX_ITERATIONS = 3000      # Training iterations
BATCH_SIZE = 64            # Batch size
LEARNING_RATE = 1e-3       # Initial learning rate
```

## Expected Results

### Default Target (origin)

All three methods should achieve similar performance:
- Control objective J ≈ 4.5-4.6
- Relative suboptimality < 5%

### Shifted Target (x = 3)

| Method | Control Objective | Relative Suboptimality |
|--------|-------------------|------------------------|
| PINN | ~350-360 | ~5-10% |
| **FBSNN** | ~400+ | **30%+** (FAILS) |
| NeuralSOC | ~340-350 | ~2-5% |

## Dependencies

- JAX >= 0.4.0
- Equinox >= 0.11.0
- Diffrax >= 0.4.0
- Optax >= 0.1.0
- Matplotlib
- Pandas

## References

1. **Deep BSDE (baseline):**
   - Han, Jentzen, E (2018): "Solving high-dimensional PDEs using deep learning" (PNAS)
   - Raissi et al. (2018): "Forward-Backward Stochastic Neural Networks"

2. **PMP-Informed (NeuralSOC):**
   - Ruthotto et al. (2020): "Machine learning framework for mean-field games" (PNAS)

3. **Course Materials:**
   - Lecture 8 of "Computational Mathematics and AI" course
   - The lecture emphasizes that **sampling strategy determines success!**
