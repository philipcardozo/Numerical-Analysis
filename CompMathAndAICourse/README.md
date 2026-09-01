# Computational Mathematics and AI - Code Examples

This repository contains reproducible code examples for the **NSF CBMS 2025: Computational Mathematics and AI** course—a ten-lecture introductory series on research topics at the intersection of computational mathematics and artificial intelligence. The course explores how computational mathematics provides foundations, precise language, and design principles for AI, and how AI enables new capabilities for tackling previously intractable computational problems.

**Course Information:**
- **Dates:** December 8–12, 2025
- **Location:** Houston, Texas
- **Instructor:** Lars Ruthotto (Emory University)
- **Conference:** [Research at the Interface of Applied Mathematics and Machine Learning (CBMS-AMML)](https://www.math.uh.edu/cbms-amml/)

**Resources:**
- [Conference Flyer (PDF)](https://www.math.uh.edu/cbms-amml/resources/NSF_CBMS_2025_flyer.pdf)
- [Lecture Slides](https://www.math.emory.edu/~lruthot/workshops/computational-math-ai-2025/)
- [Lecture Videos (YouTube)](https://www.youtube.com/playlist?list=PLPre92Pl4X2A5xj5-nNWU1r1pAuwykI1V)

## Installation

```bash
pip install -r requirements.txt
```

## Notebooks

| Notebook | Colab | Description |
|----------|-------|-------------|
| `01-polynomial-double-descent.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lruthotto/CompMathAndAICourse/blob/main/01-polynomial-double-descent.ipynb) | Demonstrates the double descent phenomenon in polynomial curve fitting. Includes Picard plot analysis. |
| `03-optimization/peaks_optimization.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lruthotto/CompMathAndAICourse/blob/main/03-optimization/peaks_optimization.ipynb) | Neural network optimization on 2D peaks classification. Compares SGD, Adam, and TR-GN across small, lazy/NTK, and mean-field regimes. |
| `07-sciml/classical-solver.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lruthotto/CompMathAndAICourse/blob/main/07-sciml/classical-solver.ipynb) | Classical CG solver for 2D Darcy flow. Baseline comparison for neural methods. |
| `07-sciml/pinn.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lruthotto/CompMathAndAICourse/blob/main/07-sciml/pinn.ipynb) | Physics-Informed Neural Networks for solving PDEs without training data. |
| `07-sciml/operator-learning.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lruthotto/CompMathAndAICourse/blob/main/07-sciml/operator-learning.ipynb) | Fourier Neural Operator (FNO) and DeepONet for learning solution operators. |
| `08-stochastic-control/stochastic-control.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lruthotto/CompMathAndAICourse/blob/main/08-stochastic-control/stochastic-control.ipynb) | 100D stochastic optimal control. Compares PINN, FBSNN, and NeuralSOC methods. |
| `09-inverse-problems/dps-gmm.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lruthotto/CompMathAndAICourse/blob/main/09-inverse-problems/dps-gmm.ipynb) | Diffusion Posterior Sampling (DPS) for Bayesian inference on 1D GMM. |
| `09-inverse-problems/deblurring-cg.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lruthotto/CompMathAndAICourse/blob/main/09-inverse-problems/deblurring-cg.ipynb) | Image deblurring with conjugate gradient methods. |

## Usage

Open the notebooks in Jupyter:

```bash
jupyter notebook
```

Or in JupyterLab:

```bash
jupyter lab
```

Each notebook is self-contained and includes all helper functions inline.

## Requirements

- Python 3.9+
- NumPy, SciPy, Matplotlib
- JAX (for the JAX version)

See `requirements.txt` for full dependencies.

## Acknowledgments

This conference is supported under **NSF CBMS Award Number 2430460** and by the **Department of Mathematics at the University of Houston**. The course is supported in part by **NSF Award DMS-2038118**. We thank the organizers for the invitation and generous support.

Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

