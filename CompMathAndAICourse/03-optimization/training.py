"""
Training utilities for peaks classification optimization experiments.

This module provides functional JAX code for training neural networks on the
peaks classification problem. It supports multiple optimizers (SGD, Adam, TR-GN)
and three network regimes (small, lazy/NTK, mean-field).

For use with the peaks_optimization.ipynb notebook.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml


# =============================================================================
# Type Definitions
# =============================================================================

Params = List[Dict[str, jnp.ndarray]]
Activation = Callable[[jnp.ndarray], jnp.ndarray]
Scalar = jnp.ndarray


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(regime: str, optimizer: str, config_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file based on regime and optimizer.

    Parameters:
        regime: One of "small", "lazy", "meanfield"
        optimizer: One of "sgd_vanilla", "adam", "tr_gn", etc.
        config_dir: Optional path to config directory (defaults to ./config)

    Returns:
        config: Configuration dictionary

    Example:
        >>> config = load_config("small", "adam")
        >>> print(config["optimizer"]["learning_rate"])
    """
    if config_dir is None:
        config_dir = Path(__file__).parent / "config"
    else:
        config_dir = Path(config_dir)

    config_path = config_dir / regime / f"{optimizer}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            f"Available regimes: small, lazy, meanfield\n"
            f"Available optimizers: sgd_vanilla, adam, tr_gn"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def print_config(config: Dict[str, Any]) -> None:
    """Print configuration in a readable format."""
    print("=" * 60)
    print(f"Experiment: {config.get('experiment', {}).get('name', 'Unnamed')}")
    print("=" * 60)

    print("\nModel:")
    model = config.get("model", {})
    print(f"  Layer sizes: {model.get('layer_sizes')}")
    if model.get("init_type"):
        print(f"  Init type: {model.get('init_type')}")
    if model.get("width_scale", 0) > 0:
        print(f"  Width scale: {model.get('width_scale')}")
    if model.get("mean_field"):
        print(f"  Mean field: {model.get('mean_field')} (outer layer frozen)")

    print("\nOptimizer:")
    opt = config.get("optimizer", {})
    print(f"  Type: {opt.get('type')}")
    print(f"  Max iterations: {opt.get('max_iter')}")
    if "learning_rate" in opt:
        print(f"  Learning rate: {opt.get('learning_rate')}")
    if "batch_size" in opt:
        print(f"  Batch size: {opt.get('batch_size')}")
    if "lr_decay" in opt and opt.get("lr_decay", 1.0) != 1.0:
        print(f"  LR decay: {opt.get('lr_decay')}")
    if "momentum" in opt:
        print(f"  Momentum: {opt.get('momentum')}")

    print("\nData:")
    data = config.get("data", {})
    print(f"  Samples: {data.get('n_samples')}")
    print(f"  Classes: {data.get('n_classes')}")
    print(f"  Train fraction: {data.get('train_fraction')}")


# =============================================================================
# Dataset Generation
# =============================================================================

def peaks_function(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the MATLAB peaks function.

    z = 3*(1-x)^2*exp(-(x^2) - (y+1)^2)
        - 10*(x/5 - x^3 - y^5)*exp(-x^2-y^2)
        - 1/3*exp(-(x+1)^2 - y^2)
    """
    term1 = 3 * (1 - x)**2 * jnp.exp(-(x**2) - (y + 1)**2)
    term2 = -10 * (x/5 - x**3 - y**5) * jnp.exp(-x**2 - y**2)
    term3 = -(1/3) * jnp.exp(-(x + 1)**2 - y**2)
    return term1 + term2 + term3


def generate_peaks_data(
    config: Dict[str, Any],
    seed: int = 42,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate peaks classification dataset based on config.

    Parameters:
        config: Configuration dictionary with 'data' section
        seed: Random seed for reproducibility

    Returns:
        X_train, y_train, X_test, y_test: Training and test data
    """
    data_config = config.get("data", {})
    n_samples = data_config.get("n_samples", 1000)
    n_classes = data_config.get("n_classes", 5)
    x_range = tuple(data_config.get("x_range", [-3, 3]))
    y_range = tuple(data_config.get("y_range", [-3, 3]))
    train_fraction = data_config.get("train_fraction", 0.8)

    # Create key from seed (matches runner.py behavior)
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    # Split subkey for x, y, noise sampling (matches peaks_classification.py)
    key_x, key_y, _ = jax.random.split(subkey, 3)

    # Sample random points
    x = jax.random.uniform(key_x, (n_samples,), minval=x_range[0], maxval=x_range[1])
    y = jax.random.uniform(key_y, (n_samples,), minval=y_range[0], maxval=y_range[1])

    # Compute peaks values and assign classes via level sets
    z = peaks_function(x, y)
    percentiles = jnp.linspace(0, 100, n_classes + 1)
    thresholds = jnp.percentile(z, percentiles)
    labels = jnp.searchsorted(thresholds[1:-1], z)

    # Stack into feature matrix
    X = jnp.stack([x, y], axis=1)

    # Train/test split
    key, subkey = jax.random.split(key)
    n_train = int(n_samples * train_fraction)
    perm = jax.random.permutation(subkey, n_samples)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_train, y_train = X[train_idx], labels[train_idx]
    X_test, y_test = X[test_idx], labels[test_idx]

    return X_train, y_train, X_test, y_test


# =============================================================================
# Model Initialization
# =============================================================================

def init_params(
    layer_sizes: List[int],
    key: jax.random.PRNGKey,
    init_type: str = "xavier",
    init_scale: float = 1.0,
) -> Params:
    """
    Initialize MLP parameters.

    Parameters:
        layer_sizes: [input_dim, hidden1, ..., output_dim]
        key: JAX random key
        init_type: "xavier", "ntk", or "meanfield"
        init_scale: Scale factor for initialization

    Returns:
        params: List of {'W': weight, 'b': bias} dicts
    """
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    n_layers = len(layer_sizes) - 1

    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        key_w, key_b = jax.random.split(keys[i])
        is_output_layer = (i == n_layers - 1)

        if init_type == "meanfield":
            # Mean-field: LeCun for hidden, partitioned indicator for output
            if is_output_layer:
                neurons_per_class = n_in // n_out
                W_kron = n_out * init_scale * jnp.kron(
                    jnp.eye(n_out), jnp.ones((neurons_per_class, 1))
                )
                if W_kron.shape[0] < n_in:
                    padding = jnp.zeros((n_in - W_kron.shape[0], n_out))
                    W = jnp.concatenate([W_kron, padding], axis=0)
                else:
                    W = W_kron
            else:
                scale = init_scale / jnp.sqrt(n_in)
                W = jax.random.normal(key_w, (n_in, n_out)) * scale
        elif init_type == "ntk":
            # NTK: LeCun for hidden, N(0,1) for output
            if is_output_layer:
                scale = init_scale
            else:
                scale = init_scale / jnp.sqrt(n_in)
            W = jax.random.normal(key_w, (n_in, n_out)) * scale
        else:
            # Xavier/Glorot (default)
            scale = init_scale * jnp.sqrt(2.0 / (n_in + n_out))
            W = jax.random.normal(key_w, (n_in, n_out)) * scale

        b = jnp.zeros(n_out)
        params.append({'W': W, 'b': b})

    return params


def init_model(config: Dict[str, Any], key: jax.random.PRNGKey) -> Params:
    """Initialize model based on config."""
    model_config = config.get("model", {})
    layer_sizes = model_config.get("layer_sizes", [2, 32, 5])
    init_type = model_config.get("init_type", "xavier")
    init_scale = model_config.get("init_scale", 1.0)

    return init_params(layer_sizes, key, init_type, init_scale)


# =============================================================================
# Forward Pass and Loss Functions
# =============================================================================

def F_fn(
    x: jnp.ndarray,
    theta: Params,
    activation: Activation = jax.nn.relu,
    width_scale: float = 0.0,
) -> jnp.ndarray:
    """
    Forward pass for a single data point.

    Parameters:
        x: Single input vector (input_dim,)
        theta: Network parameters
        activation: Activation function
        width_scale: Output scaling (0=standard, 0.5=NTK, 1.0=mean-field)

    Returns:
        logits: Output logits (output_dim,)
    """
    a = x
    for layer in theta[:-1]:
        a = activation(a @ layer['W'] + layer['b'])

    out_layer = theta[-1]
    logits = a @ out_layer['W'] + out_layer['b']

    if width_scale > 0:
        width = out_layer['W'].shape[0]
        logits = logits / (width ** width_scale)

    return logits


def F(
    X: jnp.ndarray,
    theta: Params,
    activation: Activation = jax.nn.relu,
    width_scale: float = 0.0,
) -> jnp.ndarray:
    """Batched forward pass via vmap."""
    return jax.vmap(lambda x: F_fn(x, theta, activation, width_scale))(X)


def softmax_cross_entropy_ell(logits: jnp.ndarray, label: jnp.ndarray) -> Scalar:
    """Single-sample softmax cross-entropy loss (numerically stable)."""
    max_logit = jnp.max(logits)
    shifted = logits - max_logit
    log_sum_exp = max_logit + jnp.log(jnp.sum(jnp.exp(shifted)))
    return log_sum_exp - logits[label]


def L(
    ell_fn: Callable,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    theta: Params,
    activation: Activation = jax.nn.relu,
    width_scale: float = 0.0,
) -> Scalar:
    """Batch mean loss."""
    Y_pred = F(X, theta, activation, width_scale)
    per_sample_losses = jax.vmap(ell_fn)(Y_pred, Y)
    return jnp.mean(per_sample_losses)


def loss_and_grad(
    ell_fn: Callable,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    theta: Params,
    activation: Activation = jax.nn.relu,
    width_scale: float = 0.0,
) -> Tuple[Scalar, Params]:
    """Compute loss and gradient in single forward-backward pass."""
    def loss_fn(params):
        return L(ell_fn, X, Y, params, activation, width_scale)
    return jax.value_and_grad(loss_fn)(theta)


def predict(
    X: jnp.ndarray,
    theta: Params,
    activation: Activation = jax.nn.relu,
    width_scale: float = 0.0,
) -> jnp.ndarray:
    """Predict class labels."""
    logits = F(X, theta, activation, width_scale)
    return jnp.argmax(logits, axis=-1)


def accuracy(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    theta: Params,
    activation: Activation = jax.nn.relu,
    width_scale: float = 0.0,
) -> Scalar:
    """Compute classification accuracy."""
    preds = predict(X, theta, activation, width_scale)
    return jnp.mean(preds == Y)


# =============================================================================
# Parameter Utilities
# =============================================================================

def count_params(theta: Params) -> int:
    """Count total parameters."""
    return sum(layer['W'].size + layer['b'].size for layer in theta)


def flatten_params(theta: Params) -> jnp.ndarray:
    """Flatten parameters to vector."""
    return jnp.concatenate([
        jnp.concatenate([layer['W'].ravel(), layer['b'].ravel()])
        for layer in theta
    ])


def unflatten_params(flat: jnp.ndarray, template: Params) -> Params:
    """Unflatten vector back to parameter structure."""
    params = []
    idx = 0
    for layer in template:
        W_size = layer['W'].size
        b_size = layer['b'].size
        W = flat[idx:idx + W_size].reshape(layer['W'].shape)
        idx += W_size
        b = flat[idx:idx + b_size].reshape(layer['b'].shape)
        idx += b_size
        params.append({'W': W, 'b': b})
    return params


# =============================================================================
# Optimizer State Classes
# =============================================================================

class SGDState(NamedTuple):
    """State for SGD with momentum."""
    step: int
    momentum_buffer: Params


class AdamState(NamedTuple):
    """State for Adam optimizer."""
    step: int
    m: Params
    v: Params


# =============================================================================
# First-Order Optimizers
# =============================================================================

def sgd_vanilla_step(theta: Params, grad: Params, lr: float) -> Params:
    """Vanilla SGD step: θ = θ - lr * grad."""
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, theta, grad)


def sgd_init(theta: Params) -> SGDState:
    """Initialize SGD with momentum state."""
    momentum_buffer = jax.tree_util.tree_map(jnp.zeros_like, theta)
    return SGDState(step=0, momentum_buffer=momentum_buffer)


def sgd_step(
    theta: Params,
    grad: Params,
    state: SGDState,
    lr: float = 0.01,
    momentum: float = 0.9,
    nesterov: bool = False,
) -> Tuple[Params, SGDState]:
    """SGD step with optional momentum and Nesterov."""
    momentum_buffer_new = jax.tree_util.tree_map(
        lambda v, g: momentum * v + g,
        state.momentum_buffer, grad
    )

    if nesterov:
        update = jax.tree_util.tree_map(
            lambda v, g: momentum * v + g,
            momentum_buffer_new, grad
        )
    else:
        update = momentum_buffer_new

    theta_new = jax.tree_util.tree_map(lambda p, u: p - lr * u, theta, update)
    state_new = SGDState(step=state.step + 1, momentum_buffer=momentum_buffer_new)

    return theta_new, state_new


def adam_init(theta: Params) -> AdamState:
    """Initialize Adam optimizer state."""
    zeros = jax.tree_util.tree_map(jnp.zeros_like, theta)
    return AdamState(step=0, m=zeros, v=zeros)


def adam_step(
    theta: Params,
    grad: Params,
    state: AdamState,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8
) -> Tuple[Params, AdamState]:
    """
    Perform one Adam optimization step (functional).

    Update rules:
        m_t = β₁ × m_{t-1} + (1 - β₁) × g_t
        v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²
        m̂ = m_t / (1 - β₁^t)
        v̂ = v_t / (1 - β₂^t)
        θ_t = θ_{t-1} - lr × m̂ / (√v̂ + ε)
    """
    step = state.step + 1

    # Update moments
    m_new = jax.tree_util.tree_map(
        lambda m, g: beta1 * m + (1 - beta1) * g,
        state.m,
        grad
    )
    v_new = jax.tree_util.tree_map(
        lambda v, g: beta2 * v + (1 - beta2) * g**2,
        state.v,
        grad
    )

    # Bias correction
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - beta1**step), m_new)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - beta2**step), v_new)

    # Update parameters
    theta_new = jax.tree_util.tree_map(
        lambda p, m, v: p - lr * m / (jnp.sqrt(v) + eps),
        theta,
        m_hat,
        v_hat
    )

    state_new = AdamState(step=step, m=m_new, v=v_new)

    return theta_new, state_new


# AdamW uses the same state as Adam
AdamWState = AdamState


def adamw_init(theta: Params) -> AdamWState:
    """Initialize AdamW optimizer state."""
    zeros = jax.tree_util.tree_map(jnp.zeros_like, theta)
    return AdamWState(step=0, m=zeros, v=zeros)


def adamw_step(
    theta: Params,
    grad: Params,
    state: AdamWState,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01
) -> Tuple[Params, AdamWState]:
    """
    AdamW optimization step with decoupled weight decay.

    Unlike Adam + L2 regularization, AdamW applies weight decay directly
    to the parameters rather than modifying the gradient.
    """
    step = state.step + 1

    # Update moments (same as Adam)
    m_new = jax.tree_util.tree_map(
        lambda m, g: beta1 * m + (1 - beta1) * g,
        state.m,
        grad
    )
    v_new = jax.tree_util.tree_map(
        lambda v, g: beta2 * v + (1 - beta2) * g**2,
        state.v,
        grad
    )

    # Bias correction
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - beta1**step), m_new)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - beta2**step), v_new)

    # Update parameters with DECOUPLED weight decay
    theta_new = jax.tree_util.tree_map(
        lambda p, m, v: p * (1 - lr * weight_decay) - lr * m / (jnp.sqrt(v) + eps),
        theta,
        m_hat,
        v_hat
    )

    state_new = AdamWState(step=step, m=m_new, v=v_new)
    return theta_new, state_new


class LionState(NamedTuple):
    """State for Lion optimizer."""
    step: int
    momentum_buffer: Params


def lion_init(theta: Params) -> LionState:
    """Initialize Lion optimizer state."""
    momentum_buffer = jax.tree_util.tree_map(jnp.zeros_like, theta)
    return LionState(step=0, momentum_buffer=momentum_buffer)


def lion_step(
    theta: Params,
    grad: Params,
    state: LionState,
    lr: float = 0.0001,
    beta: float = 0.9,
) -> Tuple[Params, LionState]:
    """
    Lion optimization step (sign-based momentum).

    Lion uses the sign of the momentum buffer for updates, making it
    robust and allowing larger learning rates in some regimes.
    """
    # Update momentum buffer: m_t = β × m_{t-1} + g_t
    momentum_buffer_new = jax.tree_util.tree_map(
        lambda m, g: beta * m + g,
        state.momentum_buffer,
        grad
    )

    # Compute sign of momentum
    sign_momentum = jax.tree_util.tree_map(jnp.sign, momentum_buffer_new)

    # Update parameters: θ_t = θ_{t-1} - lr × sign(m_t)
    theta_new = jax.tree_util.tree_map(
        lambda p, s: p - lr * s,
        theta,
        sign_momentum
    )

    state_new = LionState(
        step=state.step + 1,
        momentum_buffer=momentum_buffer_new
    )
    return theta_new, state_new


# =============================================================================
# Gauss-Newton (Second-Order) Optimizer
# =============================================================================

def softmax_cross_entropy_H(logits: jnp.ndarray, label: jnp.ndarray) -> jnp.ndarray:
    """Analytical Hessian of softmax cross-entropy: H = diag(p) - p⊗p."""
    p = jax.nn.softmax(logits)
    return jnp.diag(p) - jnp.outer(p, p)


def J_fn(x: jnp.ndarray, theta: Params, activation: Activation = jax.nn.relu) -> Params:
    """Compute Jacobian of F_fn w.r.t. theta for single sample."""
    return jax.jacrev(lambda params: F_fn(x, params, activation))(theta)


def flatten_jacobian(J_pytree: Params, theta: Params) -> jnp.ndarray:
    """Flatten Jacobian PyTree to dense matrix."""
    first_W_jac = J_pytree[0]['W']
    output_dim = first_W_jac.shape[0]
    flat_parts = []
    for layer_J in J_pytree:
        for key in ['W', 'b']:
            J_param = layer_J[key]
            flat_parts.append(J_param.reshape(output_dim, -1))
    return jnp.concatenate(flat_parts, axis=1)


def H_sqrt(H: jnp.ndarray) -> jnp.ndarray:
    """Matrix square root via SVD."""
    U, S, _ = jnp.linalg.svd(H, full_matrices=False)
    S_sqrt = jnp.sqrt(jnp.maximum(S, 0.0))
    return U @ jnp.diag(S_sqrt) @ U.T


def gauss_newton_hessian(
    H_fn: Callable,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    theta: Params,
    activation: Activation = jax.nn.relu,
) -> jnp.ndarray:
    """Compute Gauss-Newton Hessian: G = Σ_i J_i^T H_i J_i."""
    def weighted_jacobian(x, y):
        y_pred = F_fn(x, theta, activation)
        J_pytree = J_fn(x, theta, activation)
        J_flat = flatten_jacobian(J_pytree, theta)
        H = H_fn(y_pred, y)
        H_half = H_sqrt(H)
        return H_half @ J_flat

    weighted_Js = jax.vmap(weighted_jacobian)(X, Y)
    batch_size = X.shape[0]
    output_dim = weighted_Js.shape[1]
    n_params = weighted_Js.shape[2]
    stacked = weighted_Js.reshape(batch_size * output_dim, n_params)
    return (stacked.T @ stacked) / batch_size


class TRStepInfo(NamedTuple):
    """Information from one Trust Region Gauss-Newton step."""
    loss_before: Scalar
    loss_after: Scalar
    gradient_norm: Scalar
    step_norm: Scalar
    rho: Scalar
    accepted: bool


def tr_gauss_newton_step(
    ell_fn: Callable,
    H_fn: Callable,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    theta: Params,
    delta: Scalar,
    activation: Activation = jax.nn.relu,
    rho0: float = 1e-4,
    rho_low: float = 0.1,
    rho_high: float = 0.75,
    w_down: float = 0.5,
    w_up: float = 1.5,
    delta_max: float = 1e4,
) -> Tuple[Params, Scalar, TRStepInfo]:
    """
    Perform one Trust Region Gauss-Newton step.

    Uses SVD-based solution of TR subproblem with adaptive radius.
    """
    # Compute loss and gradient
    loss_current, grad_pytree = loss_and_grad(ell_fn, X, Y, theta, activation)
    grad_flat = flatten_params(grad_pytree)
    grad_norm = jnp.linalg.norm(grad_flat)

    # Build GN Hessian and compute SVD
    G = gauss_newton_hessian(H_fn, X, Y, theta, activation)
    U, S, Vt = jnp.linalg.svd(G, full_matrices=False)
    V = Vt.T

    # Solve TR subproblem
    rhs = U.T @ grad_flat
    S_max = jnp.max(S)
    S_threshold = jnp.maximum(S_max * 1e-12, 1e-14)
    S_safe = jnp.maximum(S, S_threshold)
    z_unc = -rhs / S_safe
    s_unc = V @ z_unc
    s_unc_norm = jnp.linalg.norm(s_unc)

    if s_unc_norm <= delta:
        s_flat = s_unc
        lambda_reg = 0.0
    else:
        # Bisection to find lambda
        lambda_low = 0.0
        lambda_high = jnp.linalg.norm(rhs) / delta + S_max
        for _ in range(50):
            lambda_mid = 0.5 * (lambda_low + lambda_high)
            coeffs = rhs / (S + lambda_mid)
            s_norm = jnp.sqrt(jnp.sum(coeffs**2))
            if jnp.abs(s_norm - delta) < 1e-10:
                break
            if s_norm > delta:
                lambda_low = lambda_mid
            else:
                lambda_high = lambda_mid
        lambda_reg = lambda_mid
        s_flat = -V @ (rhs / (S + lambda_reg))

    step_norm = jnp.linalg.norm(s_flat)
    s_pytree = unflatten_params(s_flat, theta)

    # Compute actual/predicted reduction
    pred = -jnp.dot(grad_flat, s_flat) - 0.5 * jnp.dot(s_flat, G @ s_flat)
    theta_trial = jax.tree_util.tree_map(lambda p, s: p + s, theta, s_pytree)
    loss_trial = L(ell_fn, X, Y, theta_trial, activation)
    ared = loss_current - loss_trial

    rho = jnp.where(
        jnp.abs(pred) > 1e-14,
        ared / pred,
        jnp.where(ared > 1e-14, 1.0, 0.0)
    )

    # TR decision logic
    if rho < rho0:
        accepted = False
        delta_new = w_down * jnp.minimum(delta, step_norm)
        theta_new = theta
        loss_new = loss_current
    elif rho < rho_low:
        accepted = True
        delta_new = w_down * delta
        theta_new = theta_trial
        loss_new = loss_trial
    elif rho <= rho_high:
        accepted = True
        delta_new = delta
        theta_new = theta_trial
        loss_new = loss_trial
    else:
        accepted = True
        theta_new = theta_trial
        loss_new = loss_trial
        at_boundary = jnp.abs(step_norm - delta) / delta < 1e-3
        if at_boundary:
            delta_new = jnp.minimum(w_up * delta, delta_max)
        else:
            delta_new = delta

    info = TRStepInfo(
        loss_before=loss_current,
        loss_after=loss_new,
        gradient_norm=grad_norm,
        step_norm=step_norm,
        rho=rho,
        accepted=accepted,
    )

    return theta_new, delta_new, info


# =============================================================================
# Training Result Container
# =============================================================================

class TrainingResult(NamedTuple):
    """Container for training results."""
    params: Params
    train_losses: List[float]
    test_losses: List[float]
    train_accuracies: List[float]
    test_accuracies: List[float]
    iterations: List[int]
    n_iterations: int
    final_train_loss: float
    final_test_loss: float
    final_train_acc: float
    final_test_acc: float


# =============================================================================
# Main Training Function
# =============================================================================

def train(
    config: Dict[str, Any],
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_test: jnp.ndarray,
    y_test: jnp.ndarray,
    theta_init: Params,
    verbose: bool = True,
    eval_freq: Optional[int] = None,
) -> TrainingResult:
    """
    Train model using optimizer specified in config.

    Parameters:
        config: Configuration dictionary
        X_train, y_train: Training data
        X_test, y_test: Test data
        theta_init: Initial parameters
        verbose: Print progress
        eval_freq: How often to evaluate (default: ~20 times per run)

    Returns:
        result: TrainingResult with final params and metrics
    """
    optimizer_type = config["optimizer"]["type"]
    opt_params = config["optimizer"]
    max_iter = opt_params.get("max_iter", 1000)
    mean_field = config.get("model", {}).get("mean_field", False)
    width_scale = config.get("model", {}).get("width_scale", 0.0)

    if eval_freq is None:
        eval_freq = max(1, max_iter // 20)

    # Metrics tracking
    iterations = []
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    theta = theta_init

    if verbose:
        print(f"\nTraining with {optimizer_type.upper()}...")
        print(f"Max iterations: {max_iter}")

    if optimizer_type == "tr_gn":
        # Trust Region Gauss-Newton
        delta = jnp.array(1.0)
        atol = float(opt_params.get("atol", 1e-6))

        for i in range(max_iter):
            theta, delta, info = tr_gauss_newton_step(
                softmax_cross_entropy_ell,
                softmax_cross_entropy_H,
                X_train, y_train, theta, delta,
            )

            if i % eval_freq == 0 or i == max_iter - 1:
                test_loss = float(L(softmax_cross_entropy_ell, X_test, y_test, theta))
                train_acc = float(accuracy(X_train, y_train, theta))
                test_acc = float(accuracy(X_test, y_test, theta))

                iterations.append(i)
                train_losses.append(float(info.loss_after))
                test_losses.append(test_loss)
                train_accs.append(train_acc)
                test_accs.append(test_acc)

                if verbose and i % max(1, max_iter // 10) == 0:
                    status = "accept" if info.accepted else "REJECT"
                    print(f"  Iter {i:4d}: loss={info.loss_after:.4f}, "
                          f"acc={train_acc:.4f}/{test_acc:.4f} [{status}]")

            if info.gradient_norm < atol:
                if verbose:
                    print(f"  Converged at iteration {i}")
                break

    else:
        # First-order optimizers
        batch_size = opt_params.get("batch_size", len(X_train))
        batch_size = min(batch_size, len(X_train))
        lr_decay = opt_params.get("lr_decay", 1.0)
        base_lr = opt_params.get("learning_rate", 0.01)

        # Initialize optimizer state
        if optimizer_type in ["sgd", "sgd_momentum", "sgd_nesterov"]:
            state = sgd_init(theta)
            momentum = opt_params.get("momentum", 0.9)
            nesterov = optimizer_type == "sgd_nesterov"
        elif optimizer_type == "adam":
            state = adam_init(theta)
        elif optimizer_type == "adamw":
            state = adamw_init(theta)
            weight_decay = opt_params.get("weight_decay", 0.01)
        elif optimizer_type == "lion":
            state = lion_init(theta)
            beta = opt_params.get("beta", 0.9)
        else:
            state = None  # vanilla SGD

        key = jax.random.PRNGKey(42)

        # JIT-compiled functions
        @jax.jit
        def compute_loss_and_grad(params, X_batch, y_batch):
            return loss_and_grad(
                softmax_cross_entropy_ell, X_batch, y_batch, params,
                width_scale=width_scale
            )

        @jax.jit
        def compute_test_loss(params, X, y):
            return L(softmax_cross_entropy_ell, X, y, params, width_scale=width_scale)

        @jax.jit
        def compute_accuracy(params, X, y):
            return accuracy(X, y, params, width_scale=width_scale)

        def mask_outer_grad(grad):
            """Zero outer layer gradients for mean-field."""
            if mean_field and len(grad) > 1:
                W_grad = grad[-1]['W']
                b_grad = grad[-1]['b']
                return list(grad[:-1]) + [{
                    'W': jnp.zeros_like(W_grad),
                    'b': jnp.zeros_like(b_grad)
                }]
            return grad

        n_train = len(X_train)

        for i in range(max_iter):
            # Mini-batch sampling
            if batch_size < n_train:
                perm = jax.random.permutation(key, n_train)
                key = jax.random.fold_in(key, i)
                batch_idx = perm[:batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
            else:
                X_batch = X_train
                y_batch = y_train

            # Compute gradient
            loss, grad = compute_loss_and_grad(theta, X_batch, y_batch)
            grad = mask_outer_grad(grad)

            # LR schedule
            lr = base_lr * (lr_decay ** (i / max_iter))

            # Optimizer step
            if optimizer_type == "sgd_vanilla":
                theta = sgd_vanilla_step(theta, grad, lr)
            elif optimizer_type in ["sgd", "sgd_momentum", "sgd_nesterov"]:
                theta, state = sgd_step(theta, grad, state, lr, momentum, nesterov)
            elif optimizer_type == "adam":
                theta, state = adam_step(theta, grad, state, lr)
            elif optimizer_type == "adamw":
                theta, state = adamw_step(theta, grad, state, lr, weight_decay=weight_decay)
            elif optimizer_type == "lion":
                theta, state = lion_step(theta, grad, state, lr, beta=beta)

            # Evaluate on FULL datasets (not mini-batch)
            if i % eval_freq == 0 or i == max_iter - 1:
                train_loss = float(compute_test_loss(theta, X_train, y_train))
                test_loss = float(compute_test_loss(theta, X_test, y_test))
                train_acc = float(compute_accuracy(theta, X_train, y_train))
                test_acc = float(compute_accuracy(theta, X_test, y_test))

                iterations.append(i)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                train_accs.append(train_acc)
                test_accs.append(test_acc)

                if verbose and i % max(1, max_iter // 10) == 0:
                    print(f"  Iter {i:4d}: loss={train_loss:.4f}, "
                          f"acc={train_acc:.4f}/{test_acc:.4f}")

    result = TrainingResult(
        params=theta,
        train_losses=train_losses,
        test_losses=test_losses,
        train_accuracies=train_accs,
        test_accuracies=test_accs,
        iterations=iterations,
        n_iterations=len(iterations),
        final_train_loss=train_losses[-1] if train_losses else 0.0,
        final_test_loss=test_losses[-1] if test_losses else 0.0,
        final_train_acc=train_accs[-1] if train_accs else 0.0,
        final_test_acc=test_accs[-1] if test_accs else 0.0,
    )

    if verbose:
        print(f"\nTraining complete!")
        print(f"  Final train loss: {result.final_train_loss:.4f}")
        print(f"  Final test loss:  {result.final_test_loss:.4f}")
        print(f"  Final train acc:  {result.final_train_acc:.4f}")
        print(f"  Final test acc:   {result.final_test_acc:.4f}")

    return result
