"""
100-Dimensional Stochastic Optimal Control Benchmark Problem.

This implements the standard benchmark from the Deep BSDE literature:
- Terminal cost: g(x) = log((1 + ||x - x_target||^2) / 2)  or quadratic
- Running cost: L(x, u) = ||u||^2
- Dynamics: dX = 2u dt + sqrt(2) dW
- Optimal control: u* = -grad_x Phi (from PMP)

The HJB equation is:
    -dPhi/dt + Delta Phi - ||grad Phi||^2 = 0
    Phi(T, x) = g(x)

References:
- E, Han, Jentzen (2017): Deep learning-based numerical methods for high-D PDEs
- Han, Jentzen, E (2018): Solving high-dimensional PDEs using deep learning (PNAS)
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax
from typing import NamedTuple, Optional, Callable, Tuple


class ProblemConfig(NamedTuple):
    """Configuration for the 100D benchmark problem."""
    d: int = 100                    # State dimension
    T: float = 1.0                  # Terminal time
    sigma: float = jnp.sqrt(2.0)    # Diffusion coefficient
    x_target: Optional[jnp.ndarray] = None  # Target for terminal cost (default: origin)
    term_weight: float = 1.0        # Weight for terminal cost
    terminal_cost_type: str = "log"  # "log" or "quadratic"


def terminal_cost(x: jnp.ndarray, config: ProblemConfig) -> float:
    """
    Terminal cost g(x).

    Two types supported:
    - "log": g(x) = term_weight * log((1 + ||x - x_target||^2) / 2)
    - "quadratic": g(x) = term_weight * ||x - x_target||^2

    Args:
        x: State vector, shape (d,)
        config: Problem configuration

    Returns:
        Scalar terminal cost
    """
    x_target = config.x_target if config.x_target is not None else jnp.zeros(config.d)
    diff = x - x_target
    dist_sq = jnp.sum(diff**2)

    if config.terminal_cost_type == "quadratic":
        return config.term_weight * dist_sq
    else:  # "log" (default)
        return config.term_weight * jnp.log((1.0 + dist_sq) / 2.0)


def terminal_cost_gradient(x: jnp.ndarray, config: ProblemConfig) -> jnp.ndarray:
    """
    Gradient of terminal cost.

    Args:
        x: State vector, shape (d,)
        config: Problem configuration

    Returns:
        Gradient vector, shape (d,)
    """
    x_target = config.x_target if config.x_target is not None else jnp.zeros(config.d)
    diff = x - x_target

    if config.terminal_cost_type == "quadratic":
        return config.term_weight * 2.0 * diff
    else:  # "log" (default)
        return config.term_weight * 2.0 * diff / (1.0 + jnp.sum(diff**2))


def running_cost(x: jnp.ndarray, u: jnp.ndarray, config: ProblemConfig) -> float:
    """Running cost L(x, u) = ||u||^2."""
    return jnp.sum(u**2)


def drift(x: jnp.ndarray, u: jnp.ndarray, config: ProblemConfig) -> jnp.ndarray:
    """Drift coefficient f(x, u) = 2u."""
    return 2.0 * u


def diffusion(x: jnp.ndarray, config: ProblemConfig) -> float:
    """Diffusion coefficient sigma(x) = sqrt(2)."""
    return config.sigma


def optimal_control(grad_phi: jnp.ndarray, config: ProblemConfig) -> jnp.ndarray:
    """Optimal control from PMP: u* = -grad_x Phi."""
    return -grad_phi


def hjb_residual(
    phi: float,
    dphi_dt: float,
    grad_phi: jnp.ndarray,
    laplacian_phi: float,
    config: ProblemConfig
) -> float:
    """
    HJB residual: -dPhi/dt + (sigma^2/2) * Delta Phi - ||grad Phi||^2.
    """
    sigma_sq_half = config.sigma**2 / 2.0  # = 1.0
    grad_norm_sq = jnp.sum(grad_phi**2)
    return -dphi_dt + sigma_sq_half * laplacian_phi - grad_norm_sq


def analytical_solution_quadratic(
    t: float,
    x: jnp.ndarray,
    config: ProblemConfig
) -> float:
    """
    Closed-form analytical solution for quadratic terminal cost.

    Args:
        t: Current time
        x: Current state, shape (d,)
        config: Problem configuration

    Returns:
        Exact value function Phi(t, x)
    """
    tau = config.T - t  # Time remaining
    x_target = config.x_target if config.x_target is not None else jnp.zeros(config.d)
    diff = x - x_target
    dist_sq = jnp.sum(diff**2)

    # P(t) from Riccati equation
    P_t = config.term_weight / (1.0 + 4.0 * config.term_weight * tau)

    # r(t) from the trace term
    r_t = (config.sigma**2 / 4.0) * jnp.log(1.0 + 4.0 * config.term_weight * tau) * config.d

    return P_t * dist_sq + r_t


def analytical_solution_mc(
    t: float,
    x: jnp.ndarray,
    config: ProblemConfig,
    key: jax.Array,
    n_samples: int = 10000
) -> float:
    """
    Analytical solution via Monte Carlo (Hopf-Cole for log cost).

    For quadratic terminal cost: closed-form solution.
    For log terminal cost: Monte Carlo estimate.

    Args:
        t: Current time
        x: Current state, shape (d,)
        config: Problem configuration
        key: JAX random key
        n_samples: Number of MC samples

    Returns:
        Value function Phi(t, x)
    """
    if config.terminal_cost_type == "quadratic":
        return analytical_solution_quadratic(t, x, config)

    tau = config.T - t
    if tau <= 0:
        return terminal_cost(x, config)

    # Sample W_{T-t} ~ N(0, tau * I)
    W = jr.normal(key, (n_samples, config.d)) * jnp.sqrt(tau)
    X_T = x + config.sigma * W

    g_samples = jax.vmap(lambda x_i: terminal_cost(x_i, config))(X_T)

    # Hopf-Cole: Phi = -log E[exp(-g)]
    log_exp_neg_g = -g_samples
    max_val = jnp.max(log_exp_neg_g)
    phi = -max_val - jnp.log(jnp.mean(jnp.exp(log_exp_neg_g - max_val)))

    return phi


def sample_initial_states(
    key: jax.Array,
    batch_size: int,
    config: ProblemConfig,
    scale: float = 0.1
) -> jnp.ndarray:
    """Sample initial states for training."""
    return scale * jr.normal(key, (batch_size, config.d))


# Unified Trajectory Data Structure
class Trajectory(NamedTuple):
    """Unified trajectory container for all solvers."""
    t: jnp.ndarray                          # Time grid (n_steps + 1,)
    X: jnp.ndarray                          # States (batch, n_steps + 1, d)
    dW: jnp.ndarray                         # Brownian increments (batch, n_steps, d)
    U: Optional[jnp.ndarray] = None         # Controls (batch, n_steps, d)
    running_costs: Optional[jnp.ndarray] = None  # Running costs (batch, n_steps)


def sample_sde_trajectories(
    key: jax.Array,
    x0: jnp.ndarray,
    config: ProblemConfig,
    n_steps: int,
    drift_fn: Optional[Callable[[float, jnp.ndarray], jnp.ndarray]] = None,
    control_fn: Optional[Callable[[float, jnp.ndarray], jnp.ndarray]] = None,
    compute_running_costs: bool = False,
    deterministic: bool = False
) -> Trajectory:
    """
    Unified SDE trajectory sampling using diffrax.

    Supports both random walk (drift_fn=None) and controlled SDEs.

    Args:
        key: JAX random key
        x0: Initial states, shape (batch, d) or (d,)
        config: Problem configuration
        n_steps: Number of time discretization steps
        drift_fn: If None, pure random walk. If provided, controlled SDE.
        control_fn: Function to compute control at (t, x).
        compute_running_costs: If True, compute running costs.
        deterministic: If True, no diffusion noise.

    Returns:
        Trajectory with t, X, dW, and optionally U and running_costs
    """
    if x0.ndim == 1:
        x0 = x0[None, :]
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size, d = x0.shape
    T = config.T
    sigma = config.sigma
    dt = T / n_steps
    ts = jnp.linspace(0, T, n_steps + 1)

    def solve_single(x0_single: jnp.ndarray, key_single: jax.Array):
        if drift_fn is not None:
            def drift_coeff(t, x, args):
                return drift_fn(t, x)
        else:
            def drift_coeff(t, x, args):
                return jnp.zeros(d)

        if deterministic:
            def diffusion_coeff(t, x, args):
                return jnp.zeros(d)
        else:
            def diffusion_coeff(t, x, args):
                return sigma * jnp.ones(d)

        brownian = diffrax.VirtualBrownianTree(
            t0=0.0, t1=T, tol=1e-3, shape=(d,), key=key_single
        )

        terms = diffrax.MultiTerm(
            diffrax.ODETerm(drift_coeff),
            diffrax.ControlTerm(
                lambda t, x, args: jnp.diag(diffusion_coeff(t, x, args)),
                brownian
            )
        )

        sol = diffrax.diffeqsolve(
            terms, diffrax.Euler(), t0=0.0, t1=T, dt0=dt,
            y0=x0_single, saveat=diffrax.SaveAt(ts=ts),
        )

        if deterministic:
            dW_single = jnp.zeros((n_steps, d))
        else:
            def get_increment(k):
                return brownian.evaluate(ts[k], ts[k + 1])
            dW_single = jax.vmap(get_increment)(jnp.arange(n_steps))

        return sol.ys, dW_single

    keys = jr.split(key, batch_size)
    X, dW = jax.vmap(solve_single)(x0, keys)

    U = None
    L = None

    if control_fn is not None:
        def compute_controls_single(X_single):
            U_list = []
            for k in range(n_steps):
                t_k = ts[k]
                x_k = X_single[k]
                u_k = control_fn(t_k, x_k)
                U_list.append(u_k)
            return jnp.stack(U_list, axis=0)

        U = jax.vmap(compute_controls_single)(X)

        if compute_running_costs:
            def compute_costs_single(X_single, U_single):
                L_list = []
                for k in range(n_steps):
                    x_k = X_single[k]
                    u_k = U_single[k]
                    L_k = running_cost(x_k, u_k, config)
                    L_list.append(L_k)
                return jnp.array(L_list)

            L = jax.vmap(compute_costs_single)(X, U)

    if squeeze_output:
        X = X[0]
        dW = dW[0]
        if U is not None:
            U = U[0]
        if L is not None:
            L = L[0]

    return Trajectory(t=ts, X=X, dW=dW, U=U, running_costs=L)


def sample_random_walk_trajectories(
    key: jax.Array,
    x0: jnp.ndarray,
    config: ProblemConfig,
    n_steps: int
) -> Trajectory:
    """Sample random walk trajectories (no drift)."""
    return sample_sde_trajectories(
        key=key, x0=x0, config=config, n_steps=n_steps,
        drift_fn=None, control_fn=None
    )


def sample_controlled_trajectories(
    key: jax.Array,
    x0: jnp.ndarray,
    config: ProblemConfig,
    n_steps: int,
    control_fn: Callable[[float, jnp.ndarray], jnp.ndarray],
    compute_running_costs: bool = True,
    deterministic: bool = False
) -> Trajectory:
    """Sample controlled SDE trajectories."""
    def drift_fn(t, x):
        u_star = control_fn(t, x)
        return drift(x, u_star, config)

    return sample_sde_trajectories(
        key=key, x0=x0, config=config, n_steps=n_steps,
        drift_fn=drift_fn, control_fn=control_fn,
        compute_running_costs=compute_running_costs,
        deterministic=deterministic
    )


# Evaluation Utilities
class EvaluationResult(NamedTuple):
    """Container for control objective evaluation results."""
    mean_objective: float
    std_objective: float
    mean_running_cost: float
    mean_terminal_cost: float
    optimal_value: float
    relative_suboptimality: float


def evaluate_control_objective(
    policy_fn: Callable[[float, jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    config: ProblemConfig,
    key: jax.Array,
    n_steps: int = 100,
    n_mc_samples: int = 10000
) -> EvaluationResult:
    """
    Evaluate the stochastic optimal control objective for a given policy.

    Computes J = E[integral_0^T ||u(t,X_t)||^2 dt + g(X_T)] by Monte Carlo.

    Args:
        policy_fn: Control policy u(t, x)
        x0: Initial states, shape (batch, d)
        config: Problem configuration
        key: JAX random key
        n_steps: Number of time discretization steps
        n_mc_samples: Number of MC samples for analytical solution

    Returns:
        EvaluationResult with control objective statistics
    """
    if x0.ndim == 1:
        x0 = x0[None, :]

    batch_size = x0.shape[0]
    dt = config.T / n_steps

    key, traj_key = jr.split(key)
    trajectory = sample_controlled_trajectories(
        key=traj_key, x0=x0, config=config, n_steps=n_steps,
        control_fn=policy_fn, compute_running_costs=True
    )

    total_running_costs = jnp.sum(trajectory.running_costs, axis=-1) * dt
    X_T = trajectory.X[:, -1, :]
    terminal_costs = jax.vmap(lambda x: terminal_cost(x, config))(X_T)
    objectives = total_running_costs + terminal_costs

    key, mc_key = jr.split(key)
    mc_keys = jr.split(mc_key, batch_size)
    optimal_values = jax.vmap(
        lambda x, k: analytical_solution_mc(0.0, x, config, k, n_mc_samples)
    )(x0, mc_keys)

    mean_objective = float(jnp.mean(objectives))
    std_objective = float(jnp.std(objectives))
    mean_running = float(jnp.mean(total_running_costs))
    mean_terminal = float(jnp.mean(terminal_costs))
    mean_optimal = float(jnp.mean(optimal_values))

    relative_subopt = (mean_objective - mean_optimal) / (jnp.abs(mean_optimal) + 1e-8)

    return EvaluationResult(
        mean_objective=mean_objective,
        std_objective=std_objective,
        mean_running_cost=mean_running,
        mean_terminal_cost=mean_terminal,
        optimal_value=mean_optimal,
        relative_suboptimality=float(relative_subopt)
    )


# Default configurations
DEFAULT_CONFIG = ProblemConfig(d=100, T=1.0, sigma=jnp.sqrt(2.0))

SHIFTED_CONFIG = ProblemConfig(
    d=100, T=1.0, sigma=jnp.sqrt(2.0),
    x_target=3.0 * jnp.ones(100),
    terminal_cost_type='quadratic'
)


def create_problem_config(
    d: int = 100,
    T: float = 1.0,
    shifted_target: bool = False,
    terminal_cost_type: str = "log",
    seed: int = 42
) -> ProblemConfig:
    """
    Factory function to create problem configuration.

    Args:
        d: State dimension
        T: Terminal time
        shifted_target: If True, target is at (3, 3, ..., 3)
        terminal_cost_type: "log" or "quadratic"
        seed: Random seed (unused but included for config compatibility)

    Returns:
        ProblemConfig instance
    """
    x_target = 3.0 * jnp.ones(d) if shifted_target else None
    return ProblemConfig(
        d=d, T=T, sigma=jnp.sqrt(2.0),
        x_target=x_target,
        terminal_cost_type=terminal_cost_type
    )
