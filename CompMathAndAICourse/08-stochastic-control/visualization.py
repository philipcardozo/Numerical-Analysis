"""
Visualization utilities for stochastic optimal control experiments.

Provides plotting functions for:
- Training convergence curves
- Value function 2D slices
- Optimal and random walk trajectories
- Control trajectories
- Method comparison figures
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, Optional, List, Dict
import pandas as pd
from pathlib import Path

from problem import (
    ProblemConfig, terminal_cost, analytical_solution_mc,
    sample_controlled_trajectories, sample_random_walk_trajectories,
    terminal_cost_gradient, optimal_control, evaluate_control_objective,
)

# Color scheme
COLORS = {
    'initial': '#2E86AB',       # Blue for initial point
    'target': '#E94F37',        # Red for target
    'trajectory': '#1B998B',    # Teal for trajectories
    'trajectory_mean': '#F46036',  # Orange for mean trajectory
    'learned': '#2E86AB',       # Blue for learned
    'analytical': '#E94F37',    # Red for analytical/true
    'error': '#7B2D8E',         # Purple for error
    'pinn': '#2E86AB',          # Blue for PINN
    'fbsnn': '#E94F37',         # Red for FBSNN
    'neural_soc': '#1B998B',    # Teal for NeuralSOC
}

# Plotting style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def compute_value_function_2d_slice(
    t: float,
    config: ProblemConfig,
    key: jax.Array,
    x_range: Tuple[float, float] = (-2, 2),
    n_points: int = 50,
    n_mc_samples: int = 5000,
    center: Optional[Tuple[float, float]] = None,
    batch_size: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute analytical value function on a 2D slice (x1, x2).

    Args:
        t: Time point
        config: Problem configuration
        key: JAX random key
        x_range: Range for x1, x2 relative to center
        n_points: Grid resolution
        n_mc_samples: MC samples for Hopf-Cole
        center: Center point for the slice (default: target or origin)
        batch_size: Number of grid points to process at once (memory control)

    Returns:
        x1_grid, x2_grid, phi_values (all as numpy arrays)
    """
    if center is None:
        if config.x_target is not None:
            center = (float(config.x_target[0]), float(config.x_target[1]))
        else:
            center = (0.0, 0.0)

    x1 = jnp.linspace(center[0] + x_range[0], center[0] + x_range[1], n_points)
    x2 = jnp.linspace(center[1] + x_range[0], center[1] + x_range[1], n_points)
    x1_grid, x2_grid = jnp.meshgrid(x1, x2)

    if config.x_target is not None:
        base_point = config.x_target
    else:
        base_point = jnp.zeros(config.d)

    def make_point(x1_val, x2_val):
        point = base_point.at[0].set(x1_val)
        point = point.at[1].set(x2_val)
        return point

    x1_flat = x1_grid.ravel()
    x2_flat = x2_grid.ravel()
    n_total = len(x1_flat)
    keys = jr.split(key, n_total)

    def compute_phi(x1_val, x2_val, subkey):
        point = make_point(x1_val, x2_val)
        return analytical_solution_mc(t, point, config, subkey, n_mc_samples)

    # Batched computation to avoid OOM
    # Memory per batch: batch_size * n_mc_samples * d * 4 bytes
    phi_results = []
    for i in range(0, n_total, batch_size):
        end_idx = min(i + batch_size, n_total)
        batch_phi = jax.vmap(compute_phi)(
            x1_flat[i:end_idx], x2_flat[i:end_idx], keys[i:end_idx]
        )
        phi_results.append(np.array(batch_phi))

    phi_flat = np.concatenate(phi_results)
    phi_values = phi_flat.reshape(x1_grid.shape)

    return np.array(x1_grid), np.array(x2_grid), phi_values


def compute_value_function_2d_diagonal_slice(
    t: float,
    config: ProblemConfig,
    key: jax.Array,
    x_range: Tuple[float, float] = (-1, 4),
    n_points: int = 50,
    n_mc_samples: int = 5000,
    learned_fn: Optional[callable] = None,
    batch_size: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Compute value function on 2D slice containing the diagonal from origin to target.

    The 2D plane is parameterized such that:
    - x₁ and x₂ vary independently over x_range
    - x₃ to x_d are set to (x₁ + x₂) / 2

    This ensures:
    - At (0, 0): all coordinates = 0 (origin)
    - At (3, 3): all coordinates = 3 (target for shifted problem)
    - The diagonal line from origin to target lies on this plane

    Args:
        t: Time point
        config: Problem configuration
        key: JAX random key
        x_range: Range for x1, x2 coordinates
        n_points: Grid resolution
        n_mc_samples: MC samples for analytical solution (Hopf-Cole)
        learned_fn: Optional learned value function (t, x) -> float
        batch_size: Number of grid points to process at once (memory control)

    Returns:
        x1_grid, x2_grid, phi_analytical, phi_learned (None if learned_fn not provided)
    """
    x1 = jnp.linspace(x_range[0], x_range[1], n_points)
    x2 = jnp.linspace(x_range[0], x_range[1], n_points)
    x1_grid, x2_grid = jnp.meshgrid(x1, x2)

    def make_diagonal_point(x1_val, x2_val):
        """Create d-dimensional point on the diagonal-containing plane."""
        # First two coordinates are x1 and x2
        # Remaining coordinates are set to (x1 + x2) / 2
        remaining_val = (x1_val + x2_val) / 2.0
        point = jnp.ones(config.d) * remaining_val
        point = point.at[0].set(x1_val)
        point = point.at[1].set(x2_val)
        return point

    x1_flat = x1_grid.ravel()
    x2_flat = x2_grid.ravel()
    n_total = len(x1_flat)
    keys = jr.split(key, n_total)

    # Compute analytical solution with batching
    def compute_phi_analytical(x1_val, x2_val, subkey):
        point = make_diagonal_point(x1_val, x2_val)
        return analytical_solution_mc(t, point, config, subkey, n_mc_samples)

    phi_results = []
    for i in range(0, n_total, batch_size):
        end_idx = min(i + batch_size, n_total)
        batch_phi = jax.vmap(compute_phi_analytical)(
            x1_flat[i:end_idx], x2_flat[i:end_idx], keys[i:end_idx]
        )
        phi_results.append(np.array(batch_phi))

    phi_analytical = np.concatenate(phi_results).reshape(x1_grid.shape)

    # Compute learned solution if provided
    phi_learned = None
    if learned_fn is not None:
        def compute_phi_learned(x1_val, x2_val):
            point = make_diagonal_point(x1_val, x2_val)
            return learned_fn(t, point)

        phi_learned_flat = jax.vmap(compute_phi_learned)(x1_flat, x2_flat)
        phi_learned = np.array(phi_learned_flat.reshape(x1_grid.shape))

    return np.array(x1_grid), np.array(x2_grid), phi_analytical, phi_learned


def plot_value_function_diagonal_slice(
    config: ProblemConfig,
    key: jax.Array,
    t: float = 0.0,
    solver=None,
    x_range: Tuple[float, float] = (-1, 4),
    n_points: int = 50,
    n_mc_samples: int = 5000,
    mode: str = 'analytical',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "Value Function",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
    show_diagonal: bool = True,
) -> Tuple[plt.Axes, Tuple[float, float]]:
    """
    Plot value function on 2D diagonal slice.

    Args:
        config: Problem configuration
        key: JAX random key
        t: Time point
        solver: Trained solver (required if mode='learned')
        x_range: Range for x1, x2 coordinates
        n_points: Grid resolution
        n_mc_samples: MC samples for analytical solution
        mode: 'analytical' or 'learned'
        vmin, vmax: Colorbar limits (for consistent comparison)
        title: Plot title
        ax: Optional axes to plot on
        save_path: Optional path to save figure
        show_diagonal: Whether to show the diagonal line from origin to target

    Returns:
        Matplotlib axes and (vmin, vmax) used for colorbar
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Get learned function if needed
    learned_fn = None
    if mode == 'learned' and solver is not None:
        learned_fn = lambda t, x: solver.evaluate(t, x)

    # Compute value functions
    x1, x2, phi_analytical, phi_learned = compute_value_function_2d_diagonal_slice(
        t, config, key, x_range=x_range, n_points=n_points,
        n_mc_samples=n_mc_samples, learned_fn=learned_fn
    )

    # Select which to plot
    if mode == 'learned' and phi_learned is not None:
        phi = phi_learned
    else:
        phi = phi_analytical

    # Determine colorbar limits
    if vmin is None:
        vmin = float(phi.min())
    if vmax is None:
        vmax = float(phi.max())

    # Plot
    cf = ax.contourf(x1, x2, phi, levels=30, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(cf, ax=ax, label=f'$\\Phi({t}, x)$')

    # Mark initial point (origin)
    ax.scatter([0], [0], c=COLORS['initial'], s=150, marker='s',
               zorder=5, label='Initial $x_0$', edgecolors='white', linewidths=1.5)

    # Mark target
    if config.x_target is not None:
        target_coord = float(config.x_target[0])
        ax.scatter([target_coord], [target_coord], c=COLORS['target'], s=200, marker='*',
                   zorder=5, label='Target', edgecolors='white', linewidths=1.5)

        # Draw diagonal line from origin to target
        if show_diagonal:
            ax.plot([0, target_coord], [0, target_coord], 'w--', linewidth=2,
                    alpha=0.7, label='Diagonal', zorder=4)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.set_aspect('equal')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return ax, (vmin, vmax)


def compute_terminal_cost_2d_slice(
    config: ProblemConfig,
    x_range: Tuple[float, float] = (-2, 2),
    n_points: int = 50,
    center: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute terminal cost on a 2D slice (x1, x2)."""
    if center is None:
        if config.x_target is not None:
            center = (float(config.x_target[0]), float(config.x_target[1]))
        else:
            center = (0.0, 0.0)

    x1 = jnp.linspace(center[0] + x_range[0], center[0] + x_range[1], n_points)
    x2 = jnp.linspace(center[1] + x_range[0], center[1] + x_range[1], n_points)
    x1_grid, x2_grid = jnp.meshgrid(x1, x2)

    if config.x_target is not None:
        base_point = config.x_target
    else:
        base_point = jnp.zeros(config.d)

    def make_point(x1_val, x2_val):
        point = base_point.at[0].set(x1_val)
        point = point.at[1].set(x2_val)
        return point

    x1_flat = x1_grid.ravel()
    x2_flat = x2_grid.ravel()

    def compute_g(x1_val, x2_val):
        point = make_point(x1_val, x2_val)
        return terminal_cost(point, config)

    g_flat = jax.vmap(compute_g)(x1_flat, x2_flat)
    g_values = g_flat.reshape(x1_grid.shape)

    return np.array(x1_grid), np.array(x2_grid), np.array(g_values)


def sample_optimal_trajectories(
    config: ProblemConfig,
    key: jax.Array,
    n_trajectories: int = 20,
    n_steps: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample optimal trajectories using analytical gradient."""
    x0 = jnp.zeros((n_trajectories, config.d))

    def control_fn(t: float, x: jnp.ndarray) -> jnp.ndarray:
        grad_g = terminal_cost_gradient(x, config)
        return optimal_control(grad_g, config)

    trajectory = sample_controlled_trajectories(
        key=key, x0=x0, config=config, n_steps=n_steps,
        control_fn=control_fn, compute_running_costs=False, deterministic=False,
    )

    return np.array(trajectory.t), np.array(trajectory.X)


def plot_training_convergence(
    history,
    optimal_value: Optional[float] = None,
    title: str = "Training Convergence",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plot training loss convergence.

    Args:
        history: TrainingHistory object or dict with 'iterations' and 'losses'
        optimal_value: Optional analytical optimal value to show
        title: Plot title
        ax: Optional axes to plot on
        save_path: Optional path to save figure

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    iterations = history.iterations if hasattr(history, 'iterations') else history['iterations']
    losses = history.losses if hasattr(history, 'losses') else history['losses']

    ax.semilogy(iterations, losses, 'b-', linewidth=1.5, label='Training Loss')

    if optimal_value is not None:
        ax.axhline(y=optimal_value, color='r', linestyle='--', linewidth=1.5,
                   label=f'Optimal J* = {optimal_value:.4f}')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return ax


def plot_value_function_slice(
    config: ProblemConfig,
    key: jax.Array,
    t: float = 0.0,
    learned_fn: Optional[callable] = None,
    x_range: Tuple[float, float] = (-2, 2),
    n_points: int = 40,
    title: str = "Value Function",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plot value function on a 2D slice.

    Args:
        config: Problem configuration
        key: JAX random key
        t: Time point
        learned_fn: Optional learned value function (t, x) -> float
        x_range: Range for x1, x2 relative to center
        n_points: Grid resolution
        title: Plot title
        ax: Optional axes to plot on
        save_path: Optional path to save figure

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    center = None
    if config.x_target is not None:
        center = (float(config.x_target[0]), float(config.x_target[1]))

    x1, x2, phi_true = compute_value_function_2d_slice(
        t, config, key, x_range=x_range, n_points=n_points, center=center
    )

    cf = ax.contourf(x1, x2, phi_true, levels=30, cmap='viridis')
    plt.colorbar(cf, ax=ax, label=f'$\Phi({t}, x)$')

    # Mark initial point (origin) if visible
    if x1.min() <= 0 <= x1.max() and x2.min() <= 0 <= x2.max():
        ax.scatter([0], [0], c=COLORS['initial'], s=100, marker='s',
                   zorder=5, label='Initial $x_0$', edgecolors='white')

    # Mark target
    if center:
        ax.scatter([center[0]], [center[1]], c=COLORS['target'], s=100, marker='*',
                   zorder=5, label='Target', edgecolors='white')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return ax


def plot_trajectories(
    config: ProblemConfig,
    key: jax.Array,
    solver=None,
    method: str = "optimal",
    n_trajectories: int = 20,
    n_steps: int = 100,
    title: str = "Trajectories",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plot trajectory samples (2D projection).

    Args:
        config: Problem configuration
        key: JAX random key
        solver: Optional solver for generating trajectories
        method: 'optimal', 'random_walk', or 'learned'
        n_trajectories: Number of trajectories
        n_steps: Number of time steps
        title: Plot title
        ax: Optional axes to plot on
        save_path: Optional path to save figure

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    x0 = jnp.zeros((n_trajectories, config.d))

    if method == "random_walk":
        trajectory = sample_random_walk_trajectories(
            key=key, x0=x0, config=config, n_steps=n_steps
        )
        X = np.array(trajectory.X)
    elif method == "learned" and solver is not None:
        trajectory = solver.sample_training_trajectories(key, x0)
        X = np.array(trajectory.X)
    else:  # optimal
        _, X = sample_optimal_trajectories(
            config, key, n_trajectories=n_trajectories, n_steps=n_steps
        )

    # Plot individual trajectories (2D projection)
    X_2d = X[:, :, :2]
    for i in range(min(10, X_2d.shape[0])):
        ax.plot(X_2d[i, :, 0], X_2d[i, :, 1], color=COLORS['trajectory'],
                alpha=0.5, linewidth=0.8)

    # Mean trajectory
    mean_traj = X_2d.mean(axis=0)
    ax.plot(mean_traj[:, 0], mean_traj[:, 1], color=COLORS['trajectory_mean'],
            linewidth=2.5, label='Mean trajectory')

    # Mark initial and target
    target_2d = (0.0, 0.0)
    if config.x_target is not None:
        target_2d = (float(config.x_target[0]), float(config.x_target[1]))

    ax.scatter([0], [0], c=COLORS['initial'], s=150, marker='s', zorder=10,
               label='Initial $x_0$', edgecolors='white')
    ax.scatter([target_2d[0]], [target_2d[1]], c=COLORS['target'], s=150, marker='*',
               zorder=10, label='Target', edgecolors='white')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title)
    ax.legend(loc='best')

    # Set axis limits
    x_all = np.concatenate([X_2d[:, :, 0].ravel(), [0, target_2d[0]]])
    y_all = np.concatenate([X_2d[:, :, 1].ravel(), [0, target_2d[1]]])
    margin = 0.5
    ax.set_xlim(x_all.min() - margin, x_all.max() + margin)
    ax.set_ylim(y_all.min() - margin, y_all.max() + margin)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return ax


def plot_control_components(
    config: ProblemConfig,
    solver,
    key: jax.Array,
    n_trajectories: int = 10,
    title: str = "Control Trajectories",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plot control component trajectories (first few dimensions).

    Args:
        config: Problem configuration
        solver: Trained solver
        key: JAX random key
        n_trajectories: Number of trajectories
        title: Plot title
        ax: Optional axes to plot on
        save_path: Optional path to save figure

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    x0 = jnp.zeros((n_trajectories, config.d))
    trajectory = solver.sample_training_trajectories(key, x0)

    ts = np.array(trajectory.t[:-1])  # Controls are at n_steps points
    U = np.array(trajectory.U)

    # Plot first 3 control components
    colors = ['#2E86AB', '#E94F37', '#1B998B']
    for dim in range(min(3, config.d)):
        U_dim = U[:, :, dim]
        mean_u = U_dim.mean(axis=0)
        std_u = U_dim.std(axis=0)

        ax.fill_between(ts, mean_u - std_u, mean_u + std_u,
                        color=colors[dim], alpha=0.2)
        ax.plot(ts, mean_u, color=colors[dim], linewidth=2,
                label=f'$u_{dim+1}$')

    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Control $u$')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return ax


def plot_method_comparison(
    results: Dict[str, Dict],
    metric: str = 'control_objective',
    title: str = "Method Comparison",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plot bar chart comparing methods.

    Args:
        results: Dict mapping method names to result dicts
        metric: Metric to compare
        title: Plot title
        ax: Optional axes to plot on
        save_path: Optional path to save figure

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    methods = list(results.keys())
    values = [results[m].get(metric, 0) for m in methods]
    colors = [COLORS.get(m.lower().replace('-', '_'), '#888888') for m in methods]

    bars = ax.bar(methods, values, color=colors, edgecolor='white', linewidth=1.5)

    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return ax


def save_results_csv(
    results: Dict[str, any],
    filename: str,
    output_dir: str = 'figures',
) -> str:
    """
    Save results to CSV file.

    Args:
        results: Dictionary of results
        filename: Output filename (without extension)
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / f"{filename}.csv"

    if isinstance(results, dict):
        df = pd.DataFrame([results])
    else:
        df = pd.DataFrame(results)

    df.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")

    return str(filepath)


def save_training_convergence_csv(
    history,
    method: str,
    output_dir: str = 'figures',
) -> Tuple[str, str]:
    """
    Save training convergence data to CSV files.

    Creates two CSV files:
    1. Training loss per iteration
    2. Validation metrics per validation iteration

    Args:
        history: TrainingHistory object
        method: Method name for filename
        output_dir: Output directory

    Returns:
        Tuple of (training_csv_path, validation_csv_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Training loss per iteration
    training_df = pd.DataFrame({
        'iteration': history.iterations,
        'loss': history.losses,
    })
    training_path = output_path / f"{method}_training_loss.csv"
    training_df.to_csv(training_path, index=False)
    print(f"Saved: {training_path}")

    # Validation metrics per validation iteration
    val_data = {
        'iteration': history.val_iterations,
        'rel_error': history.val_rel_errors,
        'abs_error': history.val_abs_errors,
    }

    # Add suboptimality and control objective if available
    if hasattr(history, 'val_suboptimalities') and history.val_suboptimalities:
        val_data['suboptimality'] = history.val_suboptimalities
    if hasattr(history, 'val_control_objectives') and history.val_control_objectives:
        val_data['control_objective'] = history.val_control_objectives

    val_df = pd.DataFrame(val_data)
    val_path = output_path / f"{method}_validation_metrics.csv"
    val_df.to_csv(val_path, index=False)
    print(f"Saved: {val_path}")

    return str(training_path), str(val_path)


def save_convergence_comparison_csv(
    histories: Dict[str, any],
    output_dir: str = 'figures',
    filename: str = 'convergence_comparison',
) -> str:
    """
    Save convergence comparison across methods to a single CSV.

    Each method's validation metrics are merged by iteration,
    with columns prefixed by method name.

    Args:
        histories: Dict mapping method names to TrainingHistory objects
        output_dir: Output directory
        filename: Output filename (without extension)

    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all unique validation iterations
    all_iterations = set()
    for method, history in histories.items():
        all_iterations.update(history.val_iterations)
    all_iterations = sorted(all_iterations)

    # Build dataframe
    data = {'iteration': all_iterations}

    for method, history in histories.items():
        method_key = method.lower().replace('-', '_')

        # Create lookup from iteration to metrics
        iter_to_idx = {it: i for i, it in enumerate(history.val_iterations)}

        # Loss (interpolate from training iterations)
        losses = []
        for it in all_iterations:
            if it in history.iterations:
                idx = history.iterations.index(it)
                losses.append(history.losses[idx])
            else:
                # Find closest iteration
                closest = min(history.iterations, key=lambda x: abs(x - it))
                idx = history.iterations.index(closest)
                losses.append(history.losses[idx])
        data[f'{method_key}_loss'] = losses

        # Validation metrics
        rel_errors = []
        suboptimalities = []
        control_objectives = []

        for it in all_iterations:
            if it in iter_to_idx:
                idx = iter_to_idx[it]
                rel_errors.append(history.val_rel_errors[idx])
                if hasattr(history, 'val_suboptimalities') and history.val_suboptimalities:
                    suboptimalities.append(history.val_suboptimalities[idx])
                if hasattr(history, 'val_control_objectives') and history.val_control_objectives:
                    control_objectives.append(history.val_control_objectives[idx])
            else:
                rel_errors.append(None)
                if hasattr(history, 'val_suboptimalities') and history.val_suboptimalities:
                    suboptimalities.append(None)
                if hasattr(history, 'val_control_objectives') and history.val_control_objectives:
                    control_objectives.append(None)

        data[f'{method_key}_rel_error'] = rel_errors
        if suboptimalities:
            data[f'{method_key}_suboptimality'] = suboptimalities
        if control_objectives:
            data[f'{method_key}_control_objective'] = control_objectives

    df = pd.DataFrame(data)
    filepath = output_path / f"{filename}.csv"
    df.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")

    return str(filepath)


def create_full_results_figure(
    config: ProblemConfig,
    solver,
    history,
    method: str,
    key: jax.Array,
    optimal_value: Optional[float] = None,
    save_dir: str = 'figures',
) -> plt.Figure:
    """
    Create a comprehensive 2x3 results figure.

    Panels:
    (1, 1) Value function at t=0
    (1, 2) Value function at t=0.5
    (1, 3) Value function at t=1 (terminal)
    (2, 1) Trajectories
    (2, 2) Control components
    (2, 3) Training convergence

    Args:
        config: Problem configuration
        solver: Trained solver
        history: Training history
        method: Method name
        key: JAX random key
        optimal_value: Analytical optimal value
        save_dir: Directory to save figures

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    keys = jr.split(key, 6)

    # Row 1: Value functions at different times
    for i, t in enumerate([0.0, 0.5, 1.0]):
        plot_value_function_slice(
            config, keys[i], t=t,
            title=f'$\Phi({t}, x)$',
            ax=axes[0, i]
        )

    # Row 2: Trajectories, Controls, Convergence
    plot_trajectories(
        config, keys[3], solver=solver, method='learned',
        title=f'{method.upper()} Trajectories',
        ax=axes[1, 0]
    )

    plot_control_components(
        config, solver, keys[4],
        title=f'{method.upper()} Controls',
        ax=axes[1, 1]
    )

    plot_training_convergence(
        history, optimal_value=optimal_value,
        title='Training Convergence',
        ax=axes[1, 2]
    )

    plt.suptitle(f'{method.upper()} Results', fontsize=14)
    plt.tight_layout()

    # Save
    save_path = Path(save_dir) / f"{method}_results.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")

    return fig
