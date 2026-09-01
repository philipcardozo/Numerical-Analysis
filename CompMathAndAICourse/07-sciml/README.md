# Lecture 07: Scientific ML for PDEs

**Course:** Computational Mathematics and AI

This directory contains Jupyter notebooks demonstrating neural network methods for solving partial differential equations, specifically the 2D Darcy Flow problem.

## Notebooks

| Notebook | Topic | Method | Open in Colab |
|----------|-------|--------|---------------|
| `classical-solver.ipynb` | Classical Iterative Solvers | CG with Preconditioners | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lruthotto/CompMathAndAICourse/blob/main/07-sciml/classical-solver.ipynb) |
| `pinn.ipynb` | Physics-Informed Neural Networks | PINN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lruthotto/CompMathAndAICourse/blob/main/07-sciml/pinn.ipynb) |
| `operator-learning.ipynb` | Neural Operators | FNO + DeepONet | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lruthotto/CompMathAndAICourse/blob/main/07-sciml/operator-learning.ipynb) |

## Problem: 2D Darcy Flow

We solve the steady-state elliptic PDE modeling flow through porous media:

$$-\nabla \cdot (\kappa(x,y) \nabla u) = f \quad \text{in } \Omega = [0,1]^2$$
$$u = 0 \quad \text{on } \partial\Omega \quad \text{(Dirichlet boundary conditions)}$$

where:
- $\kappa(x,y) \in \{0.1, 1.0\}$ is the heterogeneous permeability field (input)
- $u(x,y)$ is the pressure/potential field (output)
- $f = 1$ (constant forcing term)

---

## Dataset: PDEBench Darcy Flow

### Attribution

The dataset used in these notebooks comes from **PDEBench**, a benchmark suite for scientific machine learning:

> Takamoto, M., Praditia, T., Leber, R., Ber, M., Morand, L., Hartmann, S., ... & Thuerey, N. (2022).
> **PDEBench: An Extensive Benchmark for Scientific Machine Learning.**
> *Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track.*
> [arXiv:2210.07182](https://arxiv.org/abs/2210.07182)

**Data Repository:** [DaRUS - doi:10.18419/darus-2986](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986)

**License:** The PDEBench dataset is released under CC BY 4.0.

### Dataset Specifications

| Property | Value |
|----------|-------|
| **File** | `2D_DarcyFlow_beta1.0_Train.hdf5` |
| **Samples** | 10,000 (κ, u) pairs |
| **Resolution** | 128 × 128 grid |
| **Domain** | [0, 1]² (unit square) |
| **Boundary Conditions** | Dirichlet (u = 0 on ∂Ω) |
| **Forcing** | f = 1 (constant) |
| **File Size** | ~100 MB |

### HDF5 Data Structure

```
2D_DarcyFlow_beta1.0_Train.hdf5
├── nu          : (10000, 1, 128, 128)  # Permeability field κ
├── tensor      : (10000, 1, 128, 128)  # Solution field u
├── x-coordinate: (128,)                 # Grid x-coordinates
└── y-coordinate: (128,)                 # Grid y-coordinates
```

---

## Data Generation Methodology

### Permeability Field Generation

The permeability field κ(x,y) is generated using a **thresholded Gaussian Random Field (GRF)**:

1. **Generate GRF via spectral method:**
   - Start with white noise on the grid
   - Apply FFT to transform to frequency domain
   - Multiply by Gaussian power spectrum filter: $S(k) = \exp(-\beta \cdot k^2)$
   - Apply inverse FFT to get smooth correlated field

2. **Threshold to binary values:**
   ```
   κ(x,y) = 1.0  if GRF(x,y) > median(GRF)
   κ(x,y) = 0.1  otherwise
   ```

3. **Beta parameter controls smoothness:**
   - `beta = 1.0` (default): Smooth, large-scale features (single blob-like regions)
   - `beta = 0.1`: Intermediate roughness
   - `beta = 0.01`: Fine-grained, rough features

### PDE Solver

PDEBench generates ground truth solutions using a **finite-volume time-stepping approach**:

1. **Parabolic relaxation:** Solve the time-dependent diffusion equation
   $$\frac{\partial u}{\partial t} = \nabla \cdot (\kappa \nabla u) + f$$

2. **Time integration:** Explicit Runge-Kutta (2-stage) with adaptive time stepping
   - CFL condition: $\Delta t \leq 0.25 \cdot \min(\Delta x^2, \Delta y^2) / \max(\kappa)$
   - Integration from t = 0 to t = 2.0 (sufficient for steady-state convergence)

3. **Spatial discretization:** Second-order central finite differences on cell-centered grid

4. **Output:** Final time snapshot approximates the steady-state elliptic solution

**Note:** This is mathematically equivalent to solving the elliptic problem, since as $t \to \infty$, the parabolic solution converges to the steady-state where $\partial u / \partial t = 0$.

### Comparison with Original FNO Dataset

The original Fourier Neural Operator paper (Li et al., 2021) used a different dataset:

| Property | PDEBench (This Course) | FNO Original |
|----------|------------------------|--------------|
| **Resolution** | 128 × 128 | 421 × 421 |
| **Samples** | 10,000 | 1,024 |
| **Solver** | Finite-volume time-stepping | Second-order finite difference (direct solve) |
| **Permeability** | Thresholded GRF | Thresholded GRF |

---

## Classical Solver Implementation

Our classical baseline (`classical-solver.ipynb`) uses:

1. **Discretization:** 5-point finite difference stencil (equivalent to P1 finite elements)
2. **Permeability averaging:** Harmonic mean at cell faces for numerical stability
3. **Linear solver:** Conjugate Gradient (CG) with incomplete Cholesky preconditioner
4. **Boundary conditions:** Dirichlet (u = 0) enforced directly in the system matrix

### Discretization Error Note

The classical solver achieves ~6% relative L² error compared to PDEBench reference solutions. This discrepancy arises from:

1. **Different discretization schemes:** Our 5-point stencil vs. PDEBench's cell-centered finite-volume
2. **Permeability averaging:** Harmonic mean vs. direct cell-center values
3. **Finite-time approximation:** PDEBench uses t = 2.0 (approximately, not exactly, steady-state)

This error is **not** a bug—it demonstrates that different valid discretizations of the same PDE yield slightly different solutions at finite resolution.

---

## Method Comparison

| Method | Rel. L² Error | Time | Training | Use Case |
|--------|---------------|------|----------|----------|
| **Classical CG** | 6.1% vs PDEBench | 0.14s per solve | --- | Single instance, high accuracy |
| **PINN** | 37.9% vs PDEBench | 200s (training) | 200s (20k Adam) | Inverse problems, data-free |
| **FNO** | 1.2% vs PDEBench | 29ms per solve | ~13 min (150 epochs) | Many instances, resolution transfer |
| **DeepONet** | 3.9% vs PDEBench | 3ms per solve | ~12 min (300 epochs) | Irregular domains, arbitrary queries |

**Note:** All accuracy values are relative L² error vs PDEBench reference solutions on held-out test set. PINN struggles with heterogeneous $\kappa$ (37.9% error), while neural operators excel (1.2-3.9% error). Classical CG error (6.1%) reflects discretization differences, not solver error.

---

## Directory Structure

```
07-sciml/
├── README.md                   # This file
├── utils.py                    # Data loading and visualization
├── models.py                   # Neural network implementations (FNO, DeepONet)
├── classical-solver.ipynb      # Classical CG solver demo
├── pinn.ipynb                  # Physics-Informed Neural Networks
├── operator-learning.ipynb     # FNO and DeepONet training
├── generate_operator_figures.py # Standalone script for operator figures
├── figures/                    # Generated figures for slides
│   ├── classical_*.png         # Classical solver figures
│   ├── pinn_*.png              # PINN figures
│   ├── fno_*.png               # FNO figures
│   ├── deeponet_*.png          # DeepONet figures
│   └── operator_*.png          # Shared operator figures
└── saved_models/               # Pre-trained model checkpoints
```

---

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
# Open any notebook and run all cells
```

### Data Download

The notebooks automatically download the PDEBench Darcy Flow dataset on first run:

```python
from utils import download_pdebench_darcy
download_pdebench_darcy(data_dir="data/pdebench", beta="1.0")
```

Data is cached in `data/pdebench/` (~100MB).

---

## References

### Primary Dataset

1. **PDEBench:** Takamoto, M., et al. (2022). "PDEBench: An Extensive Benchmark for Scientific Machine Learning." *NeurIPS Datasets and Benchmarks*. [arXiv:2210.07182](https://arxiv.org/abs/2210.07182)

### Neural Network Methods

2. **PINNs:** Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707. [DOI:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

3. **FNO:** Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). "Fourier Neural Operator for Parametric Partial Differential Equations." *ICLR*. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)

4. **DeepONet:** Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators." *Nature Machine Intelligence*, 3(3), 218-229. [DOI:10.1038/s42256-021-00302-5](https://doi.org/10.1038/s42256-021-00302-5)

### PINN Failure Modes

5. **Spectral Bias:** Krishnapriyan, A., Gholami, A., Zhe, S., Kirby, R., & Mahoney, M. W. (2021). "Characterizing possible failure modes in physics-informed neural networks." *NeurIPS*. [arXiv:2109.01050](https://arxiv.org/abs/2109.01050)

### Code and Data Repositories

- **PDEBench GitHub:** [github.com/pdebench/PDEBench](https://github.com/pdebench/PDEBench)
- **PDEBench Data:** [DaRUS doi:10.18419/darus-2986](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986)
- **NeuralOperator GitHub:** [github.com/neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)

---

## Acknowledgments

This course material uses the PDEBench dataset, which was developed by researchers at Technical University of Munich, Bosch Research, and other institutions. We gratefully acknowledge their contribution to reproducible benchmarking in scientific machine learning.
