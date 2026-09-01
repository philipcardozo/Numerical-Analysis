# Lecture 09: Machine Learning for Inverse Problems

This directory contains Jupyter notebooks demonstrating machine learning approaches for solving Bayesian inverse problems.

## Notebooks

### 1. `dps-gmm.ipynb` - Diffusion Posterior Sampling (DPS)

Demonstrates the DPS algorithm for Bayesian inference using a 1D Gaussian Mixture Model example.

**Problem Setup:**
- **Prior**: 3-component GMM with modes at {-2.5, 0.0, 1.5}
- **Likelihood**: Gaussian observation model with y_obs = 0.5, sigma_y = 0.8
- **Posterior**: Analytically computable GMM (for validation)

**Key Concepts:**
- VP (Variance Preserving) diffusion process
- Posterior score decomposition: `grad log pi(x|y) = grad log pi(x) + grad log p(y|x)`
- Tweedie's formula for denoising
- Reverse SDE sampling with likelihood guidance

**Generated Figures:**
- `prior_distribution.png` - 3-component GMM prior
- `prior_vs_posterior.png` - Bayesian update visualization
- `noise_schedule.png` - VP diffusion coefficients and SNR
- `marginal_evolution.png` - Forward diffusion marginal densities
- `score_functions.png` - Prior and posterior score functions
- `histogram_comparison.png` - DPS vs SDE vs direct sampling comparison
- `spacetime_density.png` - Sample evolution during reverse diffusion
- `true_posterior_evolution.png` - Analytical posterior marginal evolution
- `sample_trajectories.png` - Individual reverse diffusion paths

### 2. `cotflow-demo.ipynb` - COT-Flow for Bayesian Inverse Problems

Demonstrates COT-Flow (Conditional Optimal Transport Flow) for amortized posterior inference.

**Problem:** Given observations from a stochastic Lotka-Volterra predator-prey model, infer the posterior distribution over the 4 model parameters.

**Method:** COT-Flow learns an amortized transport map T(z; y) that pushes samples from z ~ N(0, I) to the posterior theta ~ p(theta|y).

**Generated Figures:**
- `posterior_marginals.png` - 1D marginal distributions
- `posterior_pairwise.png` - Corner plot (2D marginals)
- `training_convergence.png` - Training loss curves

## Requirements

The notebooks require:
- JAX with float64 support
- PDEforGAI library (included as submodule at repository root)

For COT-Flow additionally:
- Equinox >= 0.11.0
- Diffrax >= 0.4.0
- Optax >= 0.1.0

See `requirements.txt` for specific dependencies.

## Usage

### Local Execution

```bash
cd workspace/code/09-inverse-problems
pip install -r requirements.txt
jupyter notebook
```

### Google Colab

The notebooks include automatic setup cells for Colab that:
1. Clone the repository
2. Install dependencies
3. Add PDEforGAI to the Python path

## Learning Objectives

1. Understand how diffusion models can be adapted for inverse problems
2. Learn the DPS algorithm and its approximations
3. Compare DPS sampling with exact posterior sampling
4. Visualize the space-time evolution of diffusion-based sampling
5. Understand amortized inference via neural ODEs

## References

1. **Chung et al. (2023):** *Diffusion Posterior Sampling for General Noisy Inverse Problems*
   [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)

2. **Wang et al. (2023):** *Efficient Neural Network Approaches for Conditional Optimal Transport*
   [arXiv:2310.16975](https://arxiv.org/abs/2310.16975)

## Course Context

This example is part of **Lecture 9: Inverse Problems** in the Computational Mathematics and AI course.

Connections to other lectures:
- **Lecture 8:** HJB equations and optimal control
- **Lecture 6:** Generative modeling (score-based diffusion)
- **Lecture 3:** Optimization (training neural networks)
