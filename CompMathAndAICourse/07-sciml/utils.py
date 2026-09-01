"""
Shared Utilities for Lecture 07: Scientific ML for PDEs

This module provides utilities for:
- PDEBench data loading with auto-download
- Matrix assembly for classical FEM solvers
- Visualization functions

Problem: 2D Darcy Flow
    -div(kappa * grad(u)) = f in [0,1]^2
    u = 0 on boundary
"""

import numpy as np
import h5py
import urllib.request
from pathlib import Path
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt

# Try to import JAX (optional - only needed for ML methods)
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# =============================================================================
# PDEBench Data Loading
# =============================================================================

PDEBENCH_URLS = {
    'beta1.0': 'https://darus.uni-stuttgart.de/api/access/datafile/133219',
    'beta0.1': 'https://darus.uni-stuttgart.de/api/access/datafile/133220',
    'beta0.01': 'https://darus.uni-stuttgart.de/api/access/datafile/133221',
}


def download_pdebench(data_path: Path, beta: str = 'beta1.0') -> Path:
    """
    Download PDEBench Darcy Flow data if not present.

    Parameters:
        data_path: Path to save the data file
        beta: Beta value for the dataset ('beta1.0', 'beta0.1', 'beta0.01')

    Returns:
        data_path: Path to the downloaded file
    """
    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if data_path.exists():
        return data_path

    url = PDEBENCH_URLS.get(beta)
    if url is None:
        raise ValueError(f"Unknown beta value: {beta}. Options: {list(PDEBENCH_URLS.keys())}")

    print(f"Downloading PDEBench Darcy Flow (beta={beta})...")
    print(f"  URL: {url}")
    print(f"  Destination: {data_path}")

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    with urllib.request.urlopen(req) as response:
        total_size = response.headers.get('content-length')
        if total_size:
            total_size = int(total_size)
            print(f"  File size: {total_size / 1e6:.1f} MB")

        downloaded = 0
        chunk_size = 8192
        with open(data_path, 'wb') as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = 100 * downloaded / total_size
                    print(f"\r  Progress: {pct:.1f}%", end='', flush=True)
        print()

    print(f"  Download complete!")
    return data_path


def load_pdebench_samples(
    data_path: str,
    n_samples: int = 10,
    start_idx: int = 0,
    beta: str = 'beta1.0',
    auto_download: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load sample permeability and solution fields from PDEBench.

    Parameters:
        data_path: Path to HDF5 file
        n_samples: Number of samples to load
        start_idx: Starting index in dataset
        beta: Beta value for auto-download
        auto_download: If True, download data if not present

    Returns:
        kappas: Permeability fields (n_samples, 128, 128)
        solutions: Reference solutions (n_samples, 128, 128)
    """
    data_path = Path(data_path)

    if not data_path.exists():
        if auto_download:
            download_pdebench(data_path, beta=beta)
        else:
            raise FileNotFoundError(f"PDEBench data not found: {data_path}")

    with h5py.File(data_path, 'r') as f:
        kappas = np.array(f['nu'][start_idx:start_idx + n_samples])
        solutions = np.array(f['tensor'][start_idx:start_idx + n_samples, 0])

    # Ensure correct shape
    if kappas.ndim == 4:
        kappas = kappas[:, 0, :, :]
    if solutions.ndim == 4:
        solutions = solutions[:, 0, :, :]

    return kappas, solutions


def save_data_split(
    split_path: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    config: dict,
) -> None:
    """
    Save train/val/test indices to ensure reproducible data splits.

    This is the ONLY reliable way to ensure evaluation uses the same
    data split as training. Never rely on random seeds alone!
    """
    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        split_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        config=config,
    )
    print(f"Saved data split indices to: {split_path}")


def load_data_split(split_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load saved train/val/test indices.

    Returns:
        train_idx, val_idx, test_idx: Index arrays
        config: Configuration dict used when creating the split
    """
    data = np.load(split_path, allow_pickle=True)
    return (
        data['train_idx'],
        data['val_idx'],
        data['test_idx'],
        data['config'].item() if 'config' in data else {},
    )


def load_pdebench_darcy(
    data_path: str,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 42,
    beta: str = 'beta1.0',
    auto_download: bool = True,
    normalize: bool = False,
    indices_path: Optional[str] = None,
    save_indices: bool = False,
    load_indices: bool = False,
) -> Tuple[Dict, Dict, Dict, int]:
    """
    Load PDEBench Darcy Flow data with train/val/test splits.

    Parameters:
        data_path: Path to HDF5 file
        n_train, n_val, n_test: Number of samples for each split
        seed: Random seed for shuffling (only used if load_indices is None)
        beta: Beta value for auto-download
        auto_download: If True, download data if not present
        normalize: If True, apply z-score normalization
        indices_path: Path to save/load train/val/test indices JSON
        save_indices: If True, save indices to indices_path after split
        load_indices: If True, load indices from indices_path instead of generating

    Returns:
        train_data, val_data, test_data: Dicts with 'kappa' and 'u' arrays
        resolution: Grid resolution (128 for PDEBench)
    """
    data_path = Path(data_path)

    if not data_path.exists():
        if auto_download:
            download_pdebench(data_path, beta=beta)
        else:
            raise FileNotFoundError(f"PDEBench data not found: {data_path}")

    print(f"Loading PDEBench data from: {data_path}")

    with h5py.File(data_path, 'r') as f:
        nu = np.array(f['nu'][:])
        u = np.array(f['tensor'][:])

    # Reshape if needed
    if nu.ndim == 4:
        nu = nu[:, 0, :, :]
    if u.ndim == 4:
        u = u[:, 0, :, :]

    n_total = nu.shape[0]

    # Load or create indices
    if load_indices is not None:
        print(f"  Loading indices from: {load_indices}")
        train_idx, val_idx, test_idx, config = load_data_split(load_indices)
        print(f"  Loaded split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    else:
        n_needed = n_train + n_val + n_test
        if n_needed > n_total:
            print(f"  Warning: Requested {n_needed} samples but only {n_total} available")
            ratio = n_total / n_needed
            n_train = int(n_train * ratio)
            n_val = int(n_val * ratio)
            n_test = n_total - n_train - n_val
            print(f"  Adjusted to: train={n_train}, val={n_val}, test={n_test}")

    # Shuffle and split (or load saved indices for reproducibility)
    if load_indices and indices_path is not None:
        indices_path = Path(indices_path)
        if indices_path.exists():
            import json
            with open(indices_path, 'r') as f:
                saved = json.load(f)
            train_idx = np.array(saved['train_indices'])
            val_idx = np.array(saved['val_indices'])
            test_idx = np.array(saved['test_indices'])
            print(f"  Loaded indices from: {indices_path}")
            # Update counts to match loaded indices
            n_train = len(train_idx)
            n_val = len(val_idx)
            n_test = len(test_idx)
        else:
            raise FileNotFoundError(f"Indices file not found: {indices_path}. Run with save_indices=True first.")
    else:
        # Generate new indices
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_total)

        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:n_train + n_val + n_test]

        # Save indices if requested
        if save_indices and indices_path is not None:
            import json
            indices_path = Path(indices_path)
            indices_path.parent.mkdir(parents=True, exist_ok=True)
            indices_data = {
                'seed': seed,
                'n_train': n_train,
                'n_val': n_val,
                'n_test': n_test,
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist(),
                'test_indices': test_idx.tolist(),
            }
            with open(indices_path, 'w') as f:
                json.dump(indices_data, f, indent=2)
            print(f"  Saved indices to: {indices_path}")

    if normalize:
        nu_mean, nu_std = nu[train_idx].mean(), max(nu[train_idx].std(), 1e-6)
        u_mean, u_std = u[train_idx].mean(), max(u[train_idx].std(), 1e-6)
        nu_data = (nu - nu_mean) / nu_std
        u_data = (u - u_mean) / u_std
        print(f"  Normalization applied")
    else:
        nu_data = nu
        u_data = u
        print(f"  Using UNNORMALIZED data (PDEBench style)")

    print(f"  Data ranges: kappa in [{nu.min():.4f}, {nu.max():.4f}]")
    print(f"               u in [{u.min():.4f}, {u.max():.4f}]")

    if HAS_JAX:
        train_data = {'kappa': jnp.array(nu_data[train_idx]), 'u': jnp.array(u_data[train_idx])}
        val_data = {'kappa': jnp.array(nu_data[val_idx]), 'u': jnp.array(u_data[val_idx])}
        test_data = {'kappa': jnp.array(nu_data[test_idx]), 'u': jnp.array(u_data[test_idx])}
    else:
        train_data = {'kappa': nu_data[train_idx], 'u': u_data[train_idx]}
        val_data = {'kappa': nu_data[val_idx], 'u': u_data[val_idx]}
        test_data = {'kappa': nu_data[test_idx], 'u': u_data[test_idx]}

    resolution = nu.shape[1]
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"  Resolution: {resolution}x{resolution}")

    return train_data, val_data, test_data, resolution


# DEPRECATED: Use load_pdebench_darcy with load_indices parameter instead
def load_pdebench_test_only(
    data_path: str,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 42,
    beta: str = 'beta1.0',
    auto_download: bool = True,
) -> Tuple[Dict, int]:
    """
    DEPRECATED: This function relies on random seed consistency which is fragile.
    Use load_pdebench_darcy with load_indices parameter instead.

    Load ONLY test data from PDEBench Darcy Flow to save memory.
    """
    import warnings
    warnings.warn(
        "load_pdebench_test_only is deprecated. Use load_pdebench_darcy with "
        "load_indices parameter for reliable data splits.",
        DeprecationWarning
    )

    data_path = Path(data_path)

    if not data_path.exists():
        if auto_download:
            download_pdebench(data_path, beta=beta)
        else:
            raise FileNotFoundError(f"PDEBench data not found: {data_path}")

    print(f"Loading PDEBench TEST data only from: {data_path}")

    with h5py.File(data_path, 'r') as f:
        nu = np.array(f['nu'][:])
        u = np.array(f['tensor'][:])

    # Reshape if needed
    if nu.ndim == 4:
        nu = nu[:, 0, :, :]
    if u.ndim == 4:
        u = u[:, 0, :, :]

    n_total = nu.shape[0]

    # Use SAME split logic as load_pdebench_darcy
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)

    # Test indices are at position [n_train + n_val : n_train + n_val + n_test]
    test_idx = perm[n_train + n_val:n_train + n_val + n_test]

    print(f"  Test indices: {test_idx[:5]}... (first 5 of {len(test_idx)})")

    if HAS_JAX:
        test_data = {'kappa': jnp.array(nu[test_idx]), 'u': jnp.array(u[test_idx])}
    else:
        test_data = {'kappa': nu[test_idx], 'u': u[test_idx]}

    resolution = nu.shape[1]
    print(f"  Loaded {n_test} test samples at {resolution}x{resolution}")

    return test_data, resolution


# =============================================================================
# Matrix Assembly for Classical FEM Solver
# =============================================================================

def assemble_darcy_system(kappa: np.ndarray, resolution: int) -> Tuple:
    """
    Assemble finite difference system for Darcy flow using cell-centered grid.

    Discretizes -div(kappa * grad(u)) = f with u=0 on boundary.
    Uses 5-point stencil with harmonic averaging for kappa at interior faces.
    For boundary faces, uses ghost point approach: u_ghost = -u_cell so that
    the average at the boundary is zero (homogeneous Dirichlet BC).

    Cell-centered grid:
        - Grid spacing: h = 1/n (n cells span [0, 1])
        - Cell centers at: x_j = (j + 0.5) * h for j = 0, ..., n-1
        - Boundary flux uses distance h/2 from cell center to boundary

    Parameters:
        kappa: Permeability field (resolution x resolution)
        resolution: Grid resolution (128 for PDEBench)

    Returns:
        A: Sparse stiffness matrix (N x N) in CSR format
        b: Right-hand side vector (N,)
    """
    from scipy.sparse import csr_matrix

    n = resolution
    h = 1.0 / n  # Cell-centered: n cells span [0, 1]
    h2 = h * h
    N = n * n

    # Build COO format arrays
    rows = []
    cols = []
    data = []
    b_vec = np.zeros(N, dtype=np.float64)

    # Ensure float64 for numerical precision in CG solver
    kappa_np = np.asarray(kappa, dtype=np.float64)

    def harmonic_mean(k1, k2):
        return 2.0 / (1.0 / k1 + 1.0 / k2)

    for i in range(n):
        for j in range(n):
            k = i * n + j
            diag = 0.0

            # East face (j+1 direction)
            if j < n - 1:
                # Interior face: harmonic mean of adjacent cells
                kappa_e = harmonic_mean(kappa_np[i, j], kappa_np[i, j + 1])
                rows.append(k)
                cols.append(k + 1)
                data.append(-kappa_e / h2)
                diag += kappa_e / h2
            else:
                # Boundary face: distance to boundary is h/2, factor of 2
                kappa_e = kappa_np[i, j]
                diag += 2 * kappa_e / h2

            # West face (j-1 direction)
            if j > 0:
                kappa_w = harmonic_mean(kappa_np[i, j], kappa_np[i, j - 1])
                rows.append(k)
                cols.append(k - 1)
                data.append(-kappa_w / h2)
                diag += kappa_w / h2
            else:
                kappa_w = kappa_np[i, j]
                diag += 2 * kappa_w / h2

            # North face (i+1 direction)
            if i < n - 1:
                kappa_n = harmonic_mean(kappa_np[i, j], kappa_np[i + 1, j])
                rows.append(k)
                cols.append(k + n)
                data.append(-kappa_n / h2)
                diag += kappa_n / h2
            else:
                kappa_n = kappa_np[i, j]
                diag += 2 * kappa_n / h2

            # South face (i-1 direction)
            if i > 0:
                kappa_s = harmonic_mean(kappa_np[i, j], kappa_np[i - 1, j])
                rows.append(k)
                cols.append(k - n)
                data.append(-kappa_s / h2)
                diag += kappa_s / h2
            else:
                kappa_s = kappa_np[i, j]
                diag += 2 * kappa_s / h2

            # Diagonal entry
            rows.append(k)
            cols.append(k)
            data.append(diag)

            # RHS: constant forcing f = 1
            b_vec[k] = 1.0

    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    return A, b_vec


def solve_darcy_cg(
    A,
    b: np.ndarray,
    resolution: int,
    precond_type: str = 'ic',
    tol: float = 1e-8,
    maxiter: int = 2000,
) -> Tuple[np.ndarray, Dict]:
    """
    Solve Darcy linear system with CG and optional preconditioning.

    Parameters:
        A: Sparse stiffness matrix (N x N)
        b: Right-hand side vector (N,)
        resolution: Grid resolution
        precond_type: 'none', 'jacobi', or 'ic' (incomplete Cholesky)
        tol: Convergence tolerance (relative)
        maxiter: Maximum iterations

    Returns:
        u: Solution field (resolution x resolution)
        info: Dictionary with solver statistics
    """
    import time
    from scipy.sparse.linalg import cg, LinearOperator, spilu

    N = A.shape[0]

    # Setup preconditioner
    t_precond_start = time.time()

    if precond_type == 'none':
        M = None
        precond_name = 'None'
    elif precond_type == 'jacobi':
        diag = A.diagonal().copy()
        diag[diag == 0] = 1.0
        M = LinearOperator(A.shape, lambda x: x / diag)
        precond_name = 'Jacobi'
    elif precond_type == 'ic':
        A_csc = A.tocsc()
        try:
            ilu = spilu(A_csc, drop_tol=1e-6, fill_factor=20, permc_spec='MMD_AT_PLUS_A')
            M = LinearOperator(A.shape, ilu.solve)
            precond_name = 'ILU'
        except Exception:
            # Fallback to diagonal
            diag = A.diagonal().copy()
            diag[diag == 0] = 1.0
            M = LinearOperator(A.shape, lambda x: x / diag)
            precond_name = 'Diag(fallback)'
    else:
        raise ValueError(f"Unknown preconditioner: {precond_type}")

    t_precond = time.time() - t_precond_start

    # Solve with CG
    t_solve_start = time.time()
    iteration_count = [0]
    residuals = []
    b_norm = np.linalg.norm(b)

    def callback(xk):
        iteration_count[0] += 1
        r = b - A @ xk
        residuals.append(np.linalg.norm(r) / b_norm)

    # Handle SciPy version differences: rtol/atol added in 1.12, older uses tol
    import scipy
    if tuple(map(int, scipy.__version__.split('.')[:2])) >= (1, 12):
        u_vec, exit_code = cg(A, b, M=M, rtol=tol, atol=0, maxiter=maxiter, callback=callback)
    else:
        u_vec, exit_code = cg(A, b, M=M, tol=tol, maxiter=maxiter, callback=callback)
    t_solve = time.time() - t_solve_start

    # Reshape to 2D
    u = u_vec.reshape((resolution, resolution))

    # Compute final relative residual
    final_rel_residual = residuals[-1] if residuals else np.linalg.norm(b - A @ u_vec) / b_norm

    info = {
        'precond_type': precond_name,
        'exit_code': exit_code,
        'iterations': iteration_count[0],
        'residuals': residuals,
        'final_rel_residual': final_rel_residual,
        't_precond': t_precond,
        't_solve': t_solve,
        't_total': t_precond + t_solve,
        'converged': exit_code == 0,
    }

    return u, info


def compute_l2_error(u1: np.ndarray, u2: np.ndarray, h: float = None) -> float:
    """
    Compute relative L2 error between two solutions.

    Parameters:
        u1, u2: Solution fields
        h: Grid spacing (optional, only affects absolute error)

    Returns:
        Relative L2 error: ||u1 - u2|| / ||u2||
    """
    diff = u1 - u2
    error_l2 = np.sqrt(np.sum(diff ** 2))
    norm_l2 = np.sqrt(np.sum(u2 ** 2))
    return error_l2 / (norm_l2 + 1e-10)


# =============================================================================
# Visualization
# =============================================================================

def plot_solution_comparison(
    kappa: np.ndarray,
    u_ref: np.ndarray,
    u_pred: np.ndarray,
    title: str = "Solution Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot kappa, reference, prediction, and error side by side.

    Parameters:
        kappa: Permeability field
        u_ref: Reference solution
        u_pred: Predicted solution
        title: Overall title
        save_path: If provided, save figure to this path

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

    # Permeability
    im0 = axes[0].imshow(kappa.T, origin='lower', cmap='viridis',
                         extent=[0, 1, 0, 1], aspect='equal')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Permeability $\\kappa$')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Reference solution
    im1 = axes[1].imshow(u_ref.T, origin='lower', cmap='RdBu_r',
                         extent=[0, 1, 0, 1], aspect='equal')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('Reference $u$')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Predicted solution
    im2 = axes[2].imshow(u_pred.T, origin='lower', cmap='RdBu_r',
                         extent=[0, 1, 0, 1], aspect='equal')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_title('Predicted $u$')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Error
    error = np.abs(u_pred - u_ref)
    rel_err = compute_l2_error(u_pred, u_ref)
    im3 = axes[3].imshow(error.T, origin='lower', cmap='hot',
                         extent=[0, 1, 0, 1], aspect='equal')
    axes[3].set_xlabel('x')
    axes[3].set_ylabel('y')
    axes[3].set_title(f'|Error| (rel L2: {rel_err:.2e})')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_convergence(
    train_history: list,
    val_history: list = None,
    xlabel: str = 'Epoch',
    ylabel: str = 'Loss',
    title: str = 'Training Convergence',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training (and optionally validation) convergence curves.

    Parameters:
        train_history: List of training loss values
        val_history: Optional list of validation loss values
        xlabel, ylabel, title: Axis labels and title
        save_path: If provided, save figure to this path

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(train_history, label='Train', linewidth=2)
    if val_history is not None:
        ax.semilogy(val_history, label='Validation', linewidth=2, linestyle='--')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_permeability_samples(
    kappas: np.ndarray,
    n_show: int = 3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot sample permeability fields from PDEBench.

    Parameters:
        kappas: Array of permeability fields (n, H, W)
        n_show: Number of samples to show
        save_path: If provided, save figure to this path

    Returns:
        Figure object
    """
    n_show = min(n_show, len(kappas))
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 3.5))

    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        im = ax.imshow(kappas[i].T, origin='lower', cmap='viridis',
                       extent=[0, 1, 0, 1], aspect='equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Sample {i+1}: $\\kappa$')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def save_individual_figure(
    data: np.ndarray,
    title: str,
    save_path: str,
    cmap: str = 'viridis',
    vmin: float = None,
    vmax: float = None,
    figsize: Tuple[float, float] = (4, 3.5),
    dpi: int = 150,
) -> None:
    """
    Save a single 2D field as an individual figure file.

    Parameters:
        data: 2D array to plot
        title: Figure title
        save_path: Path to save the figure
        cmap: Colormap name
        vmin, vmax: Color scale limits
        figsize: Figure size in inches
        dpi: Resolution
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data.T, origin='lower', cmap=cmap,
                   extent=[0, 1, 0, 1], aspect='equal',
                   vmin=vmin, vmax=vmax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_cg_convergence(
    results_list: list,
    labels: list = None,
    title: str = 'CG Convergence',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot CG convergence curves for different preconditioners.

    Parameters:
        results_list: List of info dicts from solve_darcy_cg
        labels: Labels for each curve
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

    if labels is None:
        labels = [r['precond_type'] for r in results_list]

    for i, (result, label) in enumerate(zip(results_list, labels)):
        residuals = result['residuals']
        iters = result['iterations']
        ax.semilogy(residuals, label=f'{label} ({iters} iters)',
                    color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Relative Residual', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([1e-10, 10])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
