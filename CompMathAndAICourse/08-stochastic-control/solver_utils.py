"""
Shared utility functions for stochastic optimal control solvers.

These functions encapsulate common operations used across PINNSolver,
FBSNNSolver, and NeuralSOCSolver to eliminate code duplication.

All functions accept the neural network (phi_net) and configuration (config)
as explicit arguments rather than accessing them from self, making them
pure functions that work with any Equinox module.
"""

import jax
import jax.numpy as jnp
from typing import Tuple
import equinox as eqx

from problem import (
    ProblemConfig,
    terminal_cost,
    terminal_cost_gradient,
    sample_controlled_trajectories,
)


# =============================================================================
# Value Function Evaluation Utilities
# =============================================================================

def compute_optimal_control(
    phi_net: eqx.Module,
    t: float,
    x: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute optimal control from PMP: u* = -grad_x Phi.

    Args:
        phi_net: Value function network (callable with signature phi_net(t, x))
        t: Current time
        x: Current state, shape (d,)

    Returns:
        Optimal control vector, shape (d,)
    """
    grad_phi = jax.grad(lambda xi: phi_net(t, xi))(x)
    return -grad_phi


def evaluate_value_function(
    phi_net: eqx.Module,
    t: float,
    x: jnp.ndarray
) -> float:
    """
    Evaluate learned value function at (t, x).

    Args:
        phi_net: Value function network
        t: Current time
        x: Current state, shape (d,)

    Returns:
        Scalar value function Phi(t, x)
    """
    return phi_net(t, x)


def evaluate_value_function_batch(
    phi_net: eqx.Module,
    t: float,
    x: jnp.ndarray
) -> jnp.ndarray:
    """
    Evaluate value function for batch of states.

    Args:
        phi_net: Value function network
        t: Current time
        x: Batch of states, shape (batch, d)

    Returns:
        Batch of values, shape (batch,)
    """
    return jax.vmap(lambda xi: phi_net(t, xi))(x)


# =============================================================================
# Terminal Condition Losses
# =============================================================================

def terminal_match_loss(
    phi_net: eqx.Module,
    X_T: jnp.ndarray,
    config: ProblemConfig
) -> float:
    """
    Compute terminal matching loss: (Phi(T, X_T) - g(X_T))^2.

    This loss enforces the terminal boundary condition of the HJB equation.

    Args:
        phi_net: Value function network
        X_T: Terminal state, shape (d,)
        config: Problem configuration

    Returns:
        Scalar squared error
    """
    T = config.T
    phi_T = phi_net(T, X_T)
    g_T = terminal_cost(X_T, config)
    return (phi_T - g_T) ** 2


def terminal_gradient_loss(
    phi_net: eqx.Module,
    X_T: jnp.ndarray,
    config: ProblemConfig
) -> float:
    """
    Compute terminal gradient matching loss.

    Enforces that grad_x Phi(T, x) = grad_x g(x) at terminal time,
    scaled by diffusion coefficient sigma.

    Args:
        phi_net: Value function network
        X_T: Terminal state, shape (d,)
        config: Problem configuration

    Returns:
        Mean squared error of gradient mismatch
    """
    sigma = config.sigma
    T = config.T

    grad_phi_T = jax.grad(lambda x: phi_net(T, x))(X_T)
    Z_T = grad_phi_T * sigma

    grad_g_T = terminal_cost_gradient(X_T, config)
    Z_target = grad_g_T * sigma

    return jnp.mean((Z_T - Z_target) ** 2)


# =============================================================================
# Trajectory Generation
# =============================================================================

def generate_optimal_trajectory(
    phi_net: eqx.Module,
    key: jax.Array,
    x0: jnp.ndarray,
    config: ProblemConfig,
    n_steps: int,
    deterministic: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate optimal trajectory using learned policy.

    Simulates the controlled SDE using the optimal control derived from
    the value function gradient: u* = -grad_x Phi.

    Args:
        phi_net: Value function network
        key: JAX random key
        x0: Initial state, shape (d,)
        config: Problem configuration
        n_steps: Number of time steps
        deterministic: If True, no diffusion noise

    Returns:
        Tuple of (t, X, U):
            t: Time grid, shape (n_steps + 1,)
            X: State trajectory, shape (n_steps + 1, d)
            U: Control trajectory, shape (n_steps, d)
    """
    def control_fn(t, x):
        return compute_optimal_control(phi_net, t, x)

    trajectory = sample_controlled_trajectories(
        key=key, x0=x0, config=config, n_steps=n_steps,
        control_fn=control_fn, compute_running_costs=False,
        deterministic=deterministic
    )
    return trajectory.t, trajectory.X, trajectory.U
