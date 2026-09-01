"""
Plotting utilities for peaks classification optimization experiments.

Provides functions for visualizing:
- Decision boundaries (matching lecture slide style)
- Training/test loss curves
- Training/test accuracy curves
- Dataset visualization

Style matches lecture03 slides for consistency.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


# =============================================================================
# Style Configuration (matches lecture slides)
# =============================================================================

# Custom 5-class colormap (pastel palette for decision boundaries)
# From math_ml_course/visualization/plots.py
CLASS_COLORS = np.array([
    [0.6, 0.77, 0.89],      # Class 0: light blue
    [0.92, 0.71, 0.63],     # Class 1: light salmon/pink
    [0.969, 0.875, 0.651],  # Class 2: light yellow
    [0.796, 0.671, 0.824],  # Class 3: light purple
    [0.785, 0.867, 0.675],  # Class 4: light green
])

# Line colors for train/test (matching slides)
TRAIN_COLOR = '#1f77b4'  # Blue
TEST_COLOR = '#ff7f0e'   # Orange

DEFAULT_FIGSIZE = (10, 4)  # Wide format for side-by-side
DEFAULT_DPI = 150
SAVE_DPI = 300


def setup_style():
    """Configure matplotlib for clean, publication-quality figures matching slides."""
    plt.rcParams.update({
        'figure.dpi': DEFAULT_DPI,
        'savefig.dpi': SAVE_DPI,
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'figure.figsize': DEFAULT_FIGSIZE,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
    })


# =============================================================================
# Dataset Visualization
# =============================================================================

def plot_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    title: str = "Peaks Classification Dataset",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 7),
) -> plt.Figure:
    """
    Plot training and test data points colored by class.

    Parameters:
        X_train: Training features (N_train, 2)
        y_train: Training labels (N_train,)
        X_test: Optional test features (N_test, 2)
        y_test: Optional test labels (N_test,)
        title: Plot title
        ax: Optional matplotlib axes
        figsize: Figure size if creating new figure

    Returns:
        fig: Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    n_classes = len(np.unique(y_train))
    colors = CLASS_COLORS[:n_classes]

    # Plot training data as circles
    for c in range(n_classes):
        mask = np.array(y_train) == c
        ax.scatter(
            np.array(X_train[mask, 0]), np.array(X_train[mask, 1]),
            c=[colors[c]], marker='o', s=40, alpha=0.7,
            edgecolors='k', linewidths=0.5, label=f'Class {c} (train)'
        )

    # Plot test data as squares
    if X_test is not None and y_test is not None:
        for c in range(n_classes):
            mask = np.array(y_test) == c
            ax.scatter(
                np.array(X_test[mask, 0]), np.array(X_test[mask, 1]),
                c=[colors[c]], marker='s', s=40, alpha=0.7,
                edgecolors='k', linewidths=0.5
            )

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', ncol=2, fontsize=8)

    return fig


# =============================================================================
# Decision Boundary Visualization
# =============================================================================

def plot_decision_boundary(
    model_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    title: str = "Decision Boundary",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (6, 5),
    grid_range: Tuple[float, float] = (-3, 3),
    grid_resolution: int = 150,
    show_misclassified: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot decision boundary with training/test data overlay (slide style).

    Parameters:
        model_fn: Function that takes X (N, 2) and returns logits (N, n_classes)
        X_train: Training features (N_train, 2)
        y_train: Training labels (N_train,)
        X_test: Optional test features (N_test, 2)
        y_test: Optional test labels (N_test,)
        title: Plot title
        ax: Optional matplotlib axes
        figsize: Figure size if creating new figure
        grid_range: Range for grid (min, max)
        grid_resolution: Number of grid points per dimension
        show_misclassified: Highlight misclassified points in red
        save_path: Optional path to save figure

    Returns:
        fig: Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Create grid
    grid = np.linspace(grid_range[0], grid_range[1], grid_resolution)
    X_grid, Y_grid = np.meshgrid(grid, grid)
    X_grid_flat = jnp.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)

    # Compute predictions on grid
    predictions = model_fn(X_grid_flat)
    pred_grid = jnp.argmax(predictions, axis=-1).reshape(X_grid.shape)

    n_classes = int(jnp.max(pred_grid)) + 1
    n_classes = max(n_classes, len(np.unique(y_train)))

    # Use pastel colors matching math_ml_course/visualization/plots.py
    if n_classes <= len(CLASS_COLORS):
        colors = CLASS_COLORS[:n_classes]
        cmap = ListedColormap(colors)
    else:
        cmap = 'viridis'
        colors = plt.cm.viridis(np.linspace(0, 1, n_classes))

    # Plot decision regions
    ax.contourf(
        X_grid, Y_grid, np.array(pred_grid),
        levels=n_classes - 1, cmap=cmap, alpha=0.8
    )
    ax.contour(
        X_grid, Y_grid, np.array(pred_grid),
        levels=n_classes - 1, cmap=cmap, alpha=1.0, linewidths=1.5
    )

    # Get predictions for data points
    pred_train = np.array(jnp.argmax(model_fn(jnp.array(X_train)), axis=-1))
    y_train_arr = np.array(y_train)

    if show_misclassified:
        # Plot correctly classified points by class color
        correct_mask = pred_train == y_train_arr
        for c in range(n_classes):
            mask = correct_mask & (pred_train == c)
            if np.any(mask):
                ax.scatter(
                    X_train[mask, 0], X_train[mask, 1],
                    color=colors[c], marker='o',
                    alpha=0.6, s=30, edgecolors='k', linewidths=0.5
                )
        # Plot misclassified points in red at 2x size
        incorrect_mask = ~correct_mask
        if np.any(incorrect_mask):
            ax.scatter(
                X_train[incorrect_mask, 0], X_train[incorrect_mask, 1],
                color='red', marker='o',
                alpha=0.8, s=60, edgecolors='k', linewidths=0.5
            )
    else:
        # Color by predicted class
        for c in range(n_classes):
            mask = pred_train == c
            if np.any(mask):
                ax.scatter(
                    X_train[mask, 0], X_train[mask, 1],
                    color=colors[c], marker='o',
                    alpha=0.6, s=30, edgecolors='k', linewidths=0.5
                )

    # Plot test data as squares
    if X_test is not None and y_test is not None:
        pred_test = np.array(jnp.argmax(model_fn(jnp.array(X_test)), axis=-1))
        y_test_arr = np.array(y_test)

        if show_misclassified:
            correct_mask = pred_test == y_test_arr
            for c in range(n_classes):
                mask = correct_mask & (pred_test == c)
                if np.any(mask):
                    ax.scatter(
                        X_test[mask, 0], X_test[mask, 1],
                        color=colors[c], marker='s',
                        alpha=0.6, s=30, edgecolors='k', linewidths=0.5
                    )
            incorrect_mask = ~correct_mask
            if np.any(incorrect_mask):
                ax.scatter(
                    X_test[incorrect_mask, 0], X_test[incorrect_mask, 1],
                    color='red', marker='s',
                    alpha=0.8, s=60, edgecolors='k', linewidths=0.5
                )
        else:
            for c in range(n_classes):
                mask = pred_test == c
                if np.any(mask):
                    ax.scatter(
                        X_test[mask, 0], X_test[mask, 1],
                        color=colors[c], marker='s',
                        alpha=0.6, s=30, edgecolors='k', linewidths=0.5
                    )

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Loss and Accuracy Curves (Slide Style)
# =============================================================================

def plot_loss_curves(
    train_losses: List[float],
    test_losses: Optional[List[float]] = None,
    iterations: Optional[List[int]] = None,
    title: str = "Loss Convergence",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (6, 4),
    log_scale: bool = True,
) -> plt.Figure:
    """
    Plot training and test loss curves (slide style).

    Parameters:
        train_losses: List of training loss values
        test_losses: Optional list of test loss values
        iterations: Optional list of iteration numbers (default: 0, 1, 2, ...)
        title: Plot title
        ax: Optional matplotlib axes
        figsize: Figure size if creating new figure
        log_scale: Use logarithmic y-axis

    Returns:
        fig: Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if iterations is None:
        iterations = list(range(len(train_losses)))

    plot_fn = ax.semilogy if log_scale else ax.plot

    plot_fn(iterations, train_losses, 'o-', color=TRAIN_COLOR,
            label='Train', linewidth=2, markersize=5)

    if test_losses is not None:
        plot_fn(iterations, test_losses, 's-', color=TEST_COLOR,
                label='Test', linewidth=2, markersize=5)

    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', frameon=True, fancybox=False,
              edgecolor='gray', framealpha=1.0)

    return fig


def plot_accuracy_curves(
    train_accuracies: List[float],
    test_accuracies: Optional[List[float]] = None,
    iterations: Optional[List[int]] = None,
    title: str = "Accuracy",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (6, 4),
) -> plt.Figure:
    """
    Plot training and test accuracy curves (slide style).

    Parameters:
        train_accuracies: List of training accuracy values
        test_accuracies: Optional list of test accuracy values
        iterations: Optional list of iteration numbers
        title: Plot title
        ax: Optional matplotlib axes
        figsize: Figure size if creating new figure

    Returns:
        fig: Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if iterations is None:
        iterations = list(range(len(train_accuracies)))

    ax.plot(iterations, train_accuracies, 'o-', color=TRAIN_COLOR,
            label='Train', linewidth=2, markersize=5)

    if test_accuracies is not None:
        ax.plot(iterations, test_accuracies, 's-', color=TEST_COLOR,
                label='Test', linewidth=2, markersize=5)

    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='upper right', frameon=True, fancybox=False,
              edgecolor='gray', framealpha=1.0)

    return fig


def plot_convergence(
    train_losses: List[float],
    test_losses: List[float],
    train_accuracies: List[float],
    test_accuracies: List[float],
    iterations: Optional[List[int]] = None,
    optimizer_name: str = "Optimizer",
    width: int = 32,
    figsize: Tuple[float, float] = (12, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create two-panel convergence plot matching lecture slide style.

    Layout: [Loss (log scale)] [Accuracy]

    Parameters:
        train_losses, test_losses: Loss histories
        train_accuracies, test_accuracies: Accuracy histories
        iterations: Optional iteration numbers
        optimizer_name: Name for title (e.g., "SGD", "Adam")
        width: Network width for title
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if iterations is None:
        iterations = list(range(len(train_losses)))

    # Left panel: Loss (log scale)
    ax = axes[0]
    ax.semilogy(iterations, train_losses, 'o-', color=TRAIN_COLOR,
                label='Train', linewidth=2, markersize=5)
    ax.semilogy(iterations, test_losses, 's-', color=TEST_COLOR,
                label='Test', linewidth=2, markersize=5)
    ax.set_xlabel('Iteration', fontsize=16)
    ax.set_ylabel('Loss', fontsize=16)
    ax.set_title(f'{optimizer_name} (width={width}): Loss Convergence', fontsize=16)
    ax.legend(loc='upper right', frameon=True, fancybox=False,
              edgecolor='gray', framealpha=1.0, fontsize=14)
    ax.tick_params(axis='both', labelsize=14)

    # Right panel: Accuracy
    ax = axes[1]
    ax.plot(iterations, train_accuracies, 'o-', color=TRAIN_COLOR,
            label='Train', linewidth=2, markersize=5)
    ax.plot(iterations, test_accuracies, 's-', color=TEST_COLOR,
            label='Test', linewidth=2, markersize=5)
    ax.set_xlabel('Iteration', fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.set_title(f'{optimizer_name} (width={width}): Accuracy', fontsize=16)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower right', frameon=True, fancybox=False,
              edgecolor='gray', framealpha=1.0, fontsize=14)
    ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Combined Visualization
# =============================================================================

def plot_training_summary(
    model_fn: Callable,
    train_losses: List[float],
    test_losses: List[float],
    train_accuracies: List[float],
    test_accuracies: List[float],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    iterations: Optional[List[int]] = None,
    title: str = "Training Summary",
    figsize: Tuple[float, float] = (14, 5),
) -> plt.Figure:
    """
    Create a 3-panel summary: decision boundary, loss curves, accuracy curves.

    Parameters:
        model_fn: Function that takes X and returns logits
        train_losses, test_losses: Loss histories
        train_accuracies, test_accuracies: Accuracy histories
        X_train, y_train, X_test, y_test: Data
        iterations: Optional iteration numbers
        title: Overall figure title
        figsize: Figure size

    Returns:
        fig: Matplotlib figure with 3 panels
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Decision boundary
    plot_decision_boundary(
        model_fn, X_train, y_train, X_test, y_test,
        title="Decision Boundary", ax=axes[0]
    )

    # Panel 2: Loss curves
    plot_loss_curves(
        train_losses, test_losses, iterations,
        title="Loss", ax=axes[1]
    )

    # Panel 3: Accuracy curves
    plot_accuracy_curves(
        train_accuracies, test_accuracies, iterations,
        title="Accuracy", ax=axes[2]
    )

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_regime_comparison(
    results_dict: dict,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """
    Compare multiple regimes/optimizers on same plot.

    Parameters:
        results_dict: Dict mapping name -> TrainingResult
        figsize: Figure size

    Returns:
        fig: Matplotlib figure with loss and accuracy comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss comparison
    for name, result in results_dict.items():
        axes[0].semilogy(
            result.iterations, result.train_losses,
            'o-', label=f'{name}', linewidth=2, markersize=3
        )

    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Loss Comparison')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Accuracy comparison
    for name, result in results_dict.items():
        axes[1].plot(
            result.iterations, result.test_accuracies,
            's-', label=f'{name}', linewidth=2, markersize=3
        )

    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_title('Accuracy Comparison')
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()

    return fig


# =============================================================================
# Save Training Results (Main Export Function)
# =============================================================================

def save_training_results(
    model_fn: Callable,
    result: Any,  # TrainingResult from training.py
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    optimizer_name: str,
    width: int,
    output_dir: str = "figures",
    prefix: str = "",
) -> Dict[str, str]:
    """
    Save all training visualization plots in slide-matching style.

    Creates:
    - {prefix}{optimizer}_loss.png: Loss convergence plot
    - {prefix}{optimizer}_accuracy.png: Accuracy convergence plot
    - {prefix}{optimizer}_decision_boundary.png: Decision boundary visualization

    Parameters:
        model_fn: Function that takes X and returns logits
        result: TrainingResult from training.py
        X_train, y_train, X_test, y_test: Data
        optimizer_name: Name for titles and filenames (e.g., "adam", "sgd_vanilla")
        width: Network width for titles
        output_dir: Directory to save figures
        prefix: Optional prefix for filenames (e.g., "small_" or "lazy_")

    Returns:
        paths: Dictionary of saved file paths
    """
    from pathlib import Path

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Format optimizer name for display
    display_name = get_optimizer_display_name(optimizer_name)

    iterations = result.iterations if result.iterations else list(range(len(result.train_losses)))

    paths = {}

    # 1. Loss convergence plot
    loss_path = output_path / f"{prefix}{optimizer_name}_loss.png"
    fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
    ax_loss.semilogy(iterations, result.train_losses, 'o-', color=TRAIN_COLOR,
                     label='Train', linewidth=2, markersize=5)
    ax_loss.semilogy(iterations, result.test_losses, 's-', color=TEST_COLOR,
                     label='Test', linewidth=2, markersize=5)
    ax_loss.set_xlabel('Iteration', fontsize=16)
    ax_loss.set_ylabel('Loss', fontsize=16)
    ax_loss.set_title(f'{display_name} (width={width}): Loss Convergence', fontsize=16)
    ax_loss.legend(loc='upper right', frameon=True, fancybox=False,
                   edgecolor='gray', framealpha=1.0, fontsize=14)
    ax_loss.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    fig_loss.savefig(loss_path, dpi=SAVE_DPI, bbox_inches='tight', facecolor='white')
    print(f"Saved: {loss_path}")
    plt.close(fig_loss)
    paths['loss'] = str(loss_path)

    # 2. Accuracy convergence plot
    acc_path = output_path / f"{prefix}{optimizer_name}_accuracy.png"
    fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
    ax_acc.plot(iterations, result.train_accuracies, 'o-', color=TRAIN_COLOR,
                label='Train', linewidth=2, markersize=5)
    ax_acc.plot(iterations, result.test_accuracies, 's-', color=TEST_COLOR,
                label='Test', linewidth=2, markersize=5)
    ax_acc.set_xlabel('Iteration', fontsize=16)
    ax_acc.set_ylabel('Accuracy', fontsize=16)
    ax_acc.set_title(f'{display_name} (width={width}): Accuracy', fontsize=16)
    ax_acc.set_ylim([0, 1.05])
    ax_acc.legend(loc='lower right', frameon=True, fancybox=False,
                  edgecolor='gray', framealpha=1.0, fontsize=14)
    ax_acc.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    fig_acc.savefig(acc_path, dpi=SAVE_DPI, bbox_inches='tight', facecolor='white')
    print(f"Saved: {acc_path}")
    plt.close(fig_acc)
    paths['accuracy'] = str(acc_path)

    # 3. Decision boundary
    db_path = output_path / f"{prefix}{optimizer_name}_decision_boundary.png"
    fig_db = plot_decision_boundary(
        model_fn=model_fn,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        title=f"{display_name} (width={width}): Decision Boundary",
        save_path=str(db_path),
    )
    plt.close(fig_db)
    paths['decision_boundary'] = str(db_path)

    print(f"\nSaved {len(paths)} figures to {output_dir}/")
    return paths


def get_optimizer_display_name(optimizer_type: str) -> str:
    """Get display name for optimizer type."""
    names = {
        'sgd_vanilla': 'SGD',
        'sgd_momentum': 'SGD+Momentum',
        'sgd_nesterov': 'SGD+Nesterov',
        'adam': 'Adam',
        'adamw': 'AdamW',
        'lion': 'Lion',
        'tr_gn': 'TR Gauss-Newton',
    }
    return names.get(optimizer_type, optimizer_type.upper())
