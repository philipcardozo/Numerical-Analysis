"""
Neural Solvers for High-Dimensional Stochastic Optimal Control.

Three solver implementations:

1. PINNSolver: Physics-Informed Neural Network
   - Enforces HJB equation at collocation points
   - Random walk or uniform sampling
   - Works with trajectory-based mode

2. FBSNNSolver: Forward-Backward Stochastic Neural Network
   - BSDE formulation with random walk sampling
   - Baseline method from Raissi et al. (2018)
   - FAILS on shifted targets (demonstrates sampling importance)

3. NeuralSOCSolver: Neural Stochastic Optimal Control
   - PMP-guided sampling with full backprop through SDE
   - Integrates loss as augmented state
   - Works in high dimensions with shifted targets

Key insight: The sampling strategy (not just the neural network) determines success!
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Tuple, Dict, Optional, Any
from dataclasses import dataclass

from problem import (
    ProblemConfig, Trajectory, drift, running_cost, terminal_cost,
    terminal_cost_gradient, hjb_residual as compute_hjb_residual_value,
    sample_random_walk_trajectories, sample_controlled_trajectories,
)
from solver_utils import (
    compute_optimal_control,
    evaluate_value_function,
    evaluate_value_function_batch,
    terminal_match_loss,
    terminal_gradient_loss,
    generate_optimal_trajectory,
)


# =============================================================================
# PINN Solver
# =============================================================================

@dataclass
class PINNLossWeights:
    """Configurable weights for PINN loss components."""
    hjb_residual: float = 1.0
    terminal_match: float = 1.0


class PINNSolver(eqx.Module):
    """
    Physics-Informed Neural Network solver for stochastic optimal control.

    Re-implementation for teaching purposes based on:
        Hu, Zhao, Karniadakis. "Hutchinson trace estimation for
        high-dimensional and high-order physics-informed neural networks."
        Comput. Methods Appl. Mech. Engrg., 2024.
    Results may differ from the original implementation.

    Enforces the HJB equation via loss function:
        Loss = w1 * ||HJB_residual||^2 + w2 * ||terminal_condition_error||^2

    Two sampling modes:
    1. "trajectory": Random walk sampling along trajectories (default)
    2. "uniform": Uniform sampling in [0,T] x domain
    """

    phi_net: eqx.Module
    config: ProblemConfig
    n_steps: int
    dt: float
    sampling_mode: str
    use_trace_estimator: bool
    trace_samples: int
    x_scale: float
    loss_weights: PINNLossWeights

    def __init__(
        self,
        phi_net: eqx.Module,
        config: ProblemConfig,
        n_steps: int = 20,
        sampling_mode: str = "trajectory",
        use_trace_estimator: bool = False,
        trace_samples: int = 1,
        x_scale: float = 1.0,
        loss_weights: Optional[PINNLossWeights] = None
    ):
        self.phi_net = phi_net
        self.config = config
        self.n_steps = n_steps
        self.dt = config.T / n_steps
        self.sampling_mode = sampling_mode
        self.use_trace_estimator = use_trace_estimator
        self.trace_samples = trace_samples
        self.x_scale = x_scale
        self.loss_weights = loss_weights if loss_weights is not None else PINNLossWeights()

    def optimal_control(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        """Compute optimal control: u* = -grad_x Phi."""
        return compute_optimal_control(self.phi_net, t, x)

    def evaluate(self, t: float, x: jnp.ndarray) -> float:
        """Evaluate learned value function at (t, x)."""
        return evaluate_value_function(self.phi_net, t, x)

    def evaluate_batch(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate value function for batch of states."""
        return evaluate_value_function_batch(self.phi_net, t, x)

    def generate_trajectory(
        self, key: jax.Array, x0: jnp.ndarray, deterministic: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Generate optimal trajectory using learned policy."""
        return generate_optimal_trajectory(
            self.phi_net, key, x0, self.config, self.n_steps, deterministic
        )

    def sample_training_trajectories(
        self, key: jax.Array, x0: jnp.ndarray
    ) -> Trajectory:
        """Sample collocation points for training."""
        if self.sampling_mode == "trajectory":
            return sample_random_walk_trajectories(
                key=key, x0=x0, config=self.config, n_steps=self.n_steps
            )
        else:  # uniform mode
            batch_size, d = x0.shape
            ts = jnp.linspace(0, self.config.T, self.n_steps + 1)
            key, x_key = jr.split(key)
            X = self.x_scale * jr.normal(x_key, (batch_size, self.n_steps + 1, d))
            dW = jnp.zeros((batch_size, self.n_steps, d))
            return Trajectory(t=ts, X=X, dW=dW, U=None, running_costs=None)

    def _hjb_residual_at_point(
        self, t: float, x: jnp.ndarray, key: Optional[jax.Array] = None, eps_t: float = 1e-4
    ) -> float:
        """Compute squared HJB residual at a single point."""
        phi, grad_phi = jax.value_and_grad(lambda xi: self.phi_net(t, xi))(x)

        if self.use_trace_estimator and key is not None:
            grad_fn = lambda xi: jax.grad(lambda xj: self.phi_net(t, xj))(xi)
            d = x.shape[0]
            z = jr.choice(key, jnp.array([-1.0, 1.0]), shape=(d,))
            _, hvp = jax.jvp(grad_fn, (x,), (z,))
            laplacian = jnp.dot(z, hvp)
        else:
            if hasattr(self.phi_net, 'value_gradient_and_laplacian'):
                _, _, laplacian = self.phi_net.value_gradient_and_laplacian(t, x)
            else:
                hess = jax.hessian(lambda xi: self.phi_net(t, xi))(x)
                laplacian = jnp.trace(hess)

        phi_plus = self.phi_net(t + eps_t, x)
        dphi_dt = (phi_plus - phi) / eps_t

        residual = compute_hjb_residual_value(phi, dphi_dt, grad_phi, laplacian, self.config)
        return residual ** 2

    def compute_loss_single(
        self, X_traj: jnp.ndarray, ts: jnp.ndarray,
        key: Optional[jax.Array] = None, weights: Optional[PINNLossWeights] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Compute total loss for a single trajectory."""
        w = weights if weights is not None else self.loss_weights
        n_steps = self.n_steps

        if w.hjb_residual > 0:
            if self.use_trace_estimator and key is not None:
                keys = jr.split(key, n_steps)
                hjb_residuals = jax.vmap(
                    lambda k, t_k, x_k: self._hjb_residual_at_point(t_k, x_k, k)
                )(keys, ts[:n_steps], X_traj[:n_steps])
            else:
                hjb_residuals = jax.vmap(
                    lambda t_k, x_k: self._hjb_residual_at_point(t_k, x_k, None)
                )(ts[:n_steps], X_traj[:n_steps])
            hjb_loss = jnp.mean(hjb_residuals)
        else:
            hjb_loss = 0.0

        X_T = X_traj[-1]
        term_match = terminal_match_loss(self.phi_net, X_T, self.config) if w.terminal_match > 0 else 0.0

        total_loss = w.hjb_residual * hjb_loss + w.terminal_match * term_match

        return total_loss, {
            "hjb_residual": hjb_loss,
            "terminal_match": term_match,
            "total": total_loss,
        }

    def compute_loss(
        self, trajectory: Trajectory, weights: Tuple[float, float] = (1.0, 1.0),
        key: Optional[jax.Array] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Compute PINN loss over batch of trajectories."""
        w_hjb, w_term = weights
        loss_weights = PINNLossWeights(hjb_residual=w_hjb, terminal_match=w_term)

        ts = trajectory.t
        X = trajectory.X

        if X.ndim == 2:
            return self.compute_loss_single(X, ts, key, loss_weights)

        batch_size = X.shape[0]

        if self.use_trace_estimator and key is not None:
            keys = jr.split(key, batch_size)
            losses, components_batch = jax.vmap(
                lambda X_i, key_i: self.compute_loss_single(X_i, ts, key_i, loss_weights)
            )(X, keys)
        else:
            losses, components_batch = jax.vmap(
                lambda X_i: self.compute_loss_single(X_i, ts, None, loss_weights)
            )(X)

        total_loss = jnp.mean(losses)
        components = {k: jnp.mean(v) for k, v in components_batch.items()}
        components["total"] = total_loss

        return total_loss, components


# =============================================================================
# FBSNN Solver
# =============================================================================

@dataclass
class FBSNNLossWeights:
    """Configurable weights for FBSNN loss components."""
    bsde_residual: float = 1.0
    terminal_match: float = 1.0
    terminal_gradient: float = 1.0


class FBSNNSolver(eqx.Module):
    """
    Forward-Backward Stochastic Neural Network solver.

    Re-implementation for teaching purposes based on:
        Raissi. "Forward-backward stochastic neural networks: Deep learning
        of high-dimensional partial differential equations." arXiv:1804.07010, 2018.
    Results may differ from the original implementation.

    Uses RANDOM WALK sampling (dX = sigma dW, no drift) and trains the
    value function by minimizing BSDE residuals.

    IMPORTANT: Random walk sampling fails when target is shifted because
    trajectories don't explore the relevant regions.
    """

    phi_net: eqx.Module
    config: ProblemConfig
    n_steps: int
    dt: float
    loss_weights: FBSNNLossWeights

    def __init__(
        self,
        phi_net: eqx.Module,
        config: ProblemConfig,
        n_steps: int = 20,
        loss_weights: Optional[FBSNNLossWeights] = None
    ):
        self.phi_net = phi_net
        self.config = config
        self.n_steps = n_steps
        self.dt = config.T / n_steps
        self.loss_weights = loss_weights if loss_weights is not None else FBSNNLossWeights()

    def optimal_control(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        """Compute optimal control: u* = -grad_x Phi."""
        return compute_optimal_control(self.phi_net, t, x)

    def evaluate(self, t: float, x: jnp.ndarray) -> float:
        return evaluate_value_function(self.phi_net, t, x)

    def evaluate_batch(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return evaluate_value_function_batch(self.phi_net, t, x)

    def generate_trajectory(
        self, key: jax.Array, x0: jnp.ndarray, deterministic: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return generate_optimal_trajectory(
            self.phi_net, key, x0, self.config, self.n_steps, deterministic
        )

    def sample_training_trajectories(
        self, key: jax.Array, x0: jnp.ndarray
    ) -> Trajectory:
        """Sample random walk trajectories for training."""
        return sample_random_walk_trajectories(
            key=key, x0=x0, config=self.config, n_steps=self.n_steps
        )

    def _bsde_residual_at_step(
        self, t_k: float, t_kp1: float,
        X_k: jnp.ndarray, X_kp1: jnp.ndarray, dW_k: jnp.ndarray
    ) -> float:
        """Compute squared BSDE residual at a single time step."""
        dt = t_kp1 - t_k
        sigma = self.config.sigma
        sigma_sq = sigma ** 2

        Y_k, grad_phi_k = jax.value_and_grad(lambda x: self.phi_net(t_k, x))(X_k)
        Z_k = grad_phi_k * sigma
        Y_kp1 = self.phi_net(t_kp1, X_kp1)

        # Generator: f = ||Z||^2 / sigma^2
        f_k = jnp.sum(Z_k ** 2) / sigma_sq

        # Random walk: cost-to-go INCREASES -> +f*dt
        Y_kp1_expected = Y_k + f_k * dt + jnp.dot(Z_k, dW_k)
        residual = Y_kp1 - Y_kp1_expected

        return residual ** 2

    def compute_loss_single(
        self, X_traj: jnp.ndarray, dW_traj: jnp.ndarray, ts: jnp.ndarray,
        weights: Optional[FBSNNLossWeights] = None
    ) -> Tuple[float, Dict[str, float]]:
        w = weights if weights is not None else self.loss_weights
        n_steps = self.n_steps

        if w.bsde_residual > 0:
            bsde_residuals = jax.vmap(
                lambda k: self._bsde_residual_at_step(
                    ts[k], ts[k + 1], X_traj[k], X_traj[k + 1], dW_traj[k]
                )
            )(jnp.arange(n_steps))
            bsde_loss = jnp.mean(bsde_residuals)
        else:
            bsde_loss = 0.0

        X_T = X_traj[-1]
        term_match = terminal_match_loss(self.phi_net, X_T, self.config) if w.terminal_match > 0 else 0.0
        term_grad = terminal_gradient_loss(self.phi_net, X_T, self.config) if w.terminal_gradient > 0 else 0.0

        total_loss = (
            w.bsde_residual * bsde_loss +
            w.terminal_match * term_match +
            w.terminal_gradient * term_grad
        )

        return total_loss, {
            "bsde_residual": bsde_loss,
            "terminal_match": term_match,
            "terminal_gradient": term_grad,
            "total": total_loss,
        }

    def compute_loss(
        self, trajectory: Trajectory, weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        key: Optional[jax.Array] = None
    ) -> Tuple[float, Dict[str, float]]:
        w_bsde, w_term, w_grad = weights
        loss_weights = FBSNNLossWeights(
            bsde_residual=w_bsde, terminal_match=w_term, terminal_gradient=w_grad
        )

        ts = trajectory.t
        X = trajectory.X
        dW = trajectory.dW

        if X.ndim == 2:
            return self.compute_loss_single(X, dW, ts, loss_weights)

        losses, components_batch = jax.vmap(
            lambda X_i, dW_i: self.compute_loss_single(X_i, dW_i, ts, loss_weights)
        )(X, dW)

        total_loss = jnp.mean(losses)
        components = {k: jnp.mean(v) for k, v in components_batch.items()}
        components["total"] = total_loss

        return total_loss, components

# =============================================================================
# NeuralSOC Solver
# =============================================================================

@dataclass
class NeuralSOCLossWeights:
    """Configurable weights for NeuralSOC loss components."""
    control_objective: float = 1.0
    terminal_match: float = 1.0
    terminal_gradient: float = 1.0
    hjb_residual: float = 0.0
    bsde_martingale: float = 0.0


class NeuralSOCSolver(eqx.Module):
    """
    Neural Stochastic Optimal Control solver with full backprop through SDE.

    Re-implementation for teaching purposes based on:
        Böttcher, Asikis, Ruthotto. "Solving stochastic optimal control
        problems using neural networks and a policy gradient method."
        SIAM J. Sci. Comput., 2024. https://doi.org/10.1137/23M155832X
    Results may differ from the original implementation.

    IMPORTANT: This solver differs fundamentally from FBSNN and PINN:
    - FBSNN/PINN: Sample trajectories (detached) -> compute loss on samples
    - NeuralSOC: Solve augmented SDE with loss as state -> full gradient flow

    Augmented state: y = [X, J] where
        X: Physical state (d dimensions, with noise)
        J: Integrated running cost (1 dimension, NO noise)
    """

    phi_net: eqx.Module
    config: ProblemConfig
    n_steps: int
    dt: float
    loss_weights: NeuralSOCLossWeights

    def __init__(
        self,
        phi_net: eqx.Module,
        config: ProblemConfig,
        n_steps: int = 20,
        terminal_match_weight: float = 1.0,
        loss_weights: Optional[NeuralSOCLossWeights] = None
    ):
        self.phi_net = phi_net
        self.config = config
        self.n_steps = n_steps
        self.dt = config.T / n_steps

        if loss_weights is not None:
            self.loss_weights = loss_weights
        else:
            self.loss_weights = NeuralSOCLossWeights(
                control_objective=1.0,
                terminal_match=terminal_match_weight,
                terminal_gradient=0.0,
                hjb_residual=0.0,
                bsde_martingale=0.0
            )

    def optimal_control(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        """Compute optimal control from PMP: u* = -grad_x Phi."""
        return compute_optimal_control(self.phi_net, t, x)

    def solve_augmented_sde(
        self, key: jax.Array, x0: jnp.ndarray, return_trajectory: bool = False
    ) -> Tuple[jnp.ndarray, float, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        """
        Solve the augmented SDE with loss integrated as state.

        This is the core of NeuralSOC: gradients flow through this entire solve.
        """
        d = self.config.d
        T = self.config.T
        sigma = self.config.sigma
        dt = self.dt
        ts = jnp.linspace(0.0, T, self.n_steps + 1)

        y0 = jnp.concatenate([x0, jnp.array([0.0])])
        sqrt_dt = jnp.sqrt(dt)

        dW_all = sqrt_dt * jr.normal(key, (self.n_steps, d))

        def euler_step(carry, inputs):
            y = carry
            t_k, dW_k = inputs
            X = y[:d]
            J = y[d]

            u_star = self.optimal_control(t_k, X)
            X_new = X + drift(X, u_star, self.config) * dt + sigma * dW_k
            J_new = J + running_cost(X, u_star, self.config) * dt

            y_new = jnp.concatenate([X_new, jnp.array([J_new])])
            return y_new, y_new if return_trajectory else None

        y_final, y_history = jax.lax.scan(euler_step, y0, (ts[:-1], dW_all))

        if return_trajectory:
            Y_traj = jnp.concatenate([y0[None, :], y_history], axis=0)
            X_traj = Y_traj[:, :d]
            X_T = y_final[:d]
            J_T = y_final[d]
            return X_T, J_T, X_traj, dW_all
        else:
            X_T = y_final[:d]
            J_T = y_final[d]
            return X_T, J_T, None, None

    def compute_loss_single(
        self, key: jax.Array, x0: jnp.ndarray,
        weights: Optional[NeuralSOCLossWeights] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Compute total loss for a single trajectory."""
        w = weights if weights is not None else self.loss_weights

        need_trajectory = (w.hjb_residual > 0 or w.bsde_martingale > 0)

        X_T, J_T, X_traj, dW_traj = self.solve_augmented_sde(
            key, x0, return_trajectory=need_trajectory
        )

        g_T = terminal_cost(X_T, self.config)
        control_objective = J_T + g_T

        phi_T = self.phi_net(self.config.T, X_T)
        terminal_mismatch = (phi_T - g_T) ** 2

        term_grad = terminal_gradient_loss(self.phi_net, X_T, self.config) if w.terminal_gradient > 0 else 0.0

        hjb_loss = 0.0
        bsde_loss = 0.0

        total_loss = (
            w.control_objective * control_objective +
            w.terminal_match * terminal_mismatch +
            w.terminal_gradient * term_grad +
            w.hjb_residual * hjb_loss +
            w.bsde_martingale * bsde_loss
        )

        return total_loss, {
            "control_objective": control_objective,
            "running_cost": J_T,
            "terminal_cost": g_T,
            "terminal_mismatch": terminal_mismatch,
            "terminal_gradient": term_grad,
            "hjb_residual": hjb_loss,
            "bsde_martingale": bsde_loss,
            "phi_T": phi_T,
            "total": total_loss,
        }

    def compute_loss(
        self, key: jax.Array, x0: jnp.ndarray,
        weights: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute loss over a batch of initial states.

        IMPORTANT: Unlike FBSNN/PINN, this method does NOT take a pre-sampled
        trajectory. It generates trajectories internally with full gradient flow.
        """
        if weights is not None:
            w_ctrl, w_term_match = weights
            loss_weights = NeuralSOCLossWeights(
                control_objective=w_ctrl,
                terminal_match=w_term_match,
                terminal_gradient=self.loss_weights.terminal_gradient,
                hjb_residual=self.loss_weights.hjb_residual,
                bsde_martingale=self.loss_weights.bsde_martingale
            )
        else:
            loss_weights = None

        if x0.ndim == 1:
            return self.compute_loss_single(key, x0, loss_weights)

        batch_size = x0.shape[0]
        keys = jr.split(key, batch_size)

        losses, components_batch = jax.vmap(
            lambda k, x: self.compute_loss_single(k, x, loss_weights)
        )(keys, x0)

        total_loss = jnp.mean(losses)
        components = {k: jnp.mean(v) for k, v in components_batch.items()}
        components["total"] = total_loss

        return total_loss, components

    def sample_training_trajectories(
        self, key: jax.Array, x0: jnp.ndarray
    ) -> Trajectory:
        """Sample trajectories for visualization/analysis."""
        d = self.config.d
        T = self.config.T
        ts = jnp.linspace(0.0, T, self.n_steps + 1)

        if x0.ndim == 1:
            x0 = x0[None, :]
            squeeze = True
        else:
            squeeze = False

        batch_size = x0.shape[0]
        keys = jr.split(key, batch_size)

        def solve_single(key_i, x0_i):
            X_T, J_T, X_traj, dW_traj = self.solve_augmented_sde(key_i, x0_i, return_trajectory=True)
            def compute_at_step(k):
                t_k = ts[k]
                x_k = X_traj[k]
                u_k = self.optimal_control(t_k, x_k)
                L_k = running_cost(x_k, u_k, self.config)
                return u_k, L_k
            U, L = jax.vmap(compute_at_step)(jnp.arange(self.n_steps))
            return X_traj, dW_traj, U, L

        X_batch, dW_batch, U_batch, L_batch = jax.vmap(solve_single)(keys, x0)

        if squeeze:
            X_batch = X_batch[0]
            dW_batch = dW_batch[0]
            U_batch = U_batch[0]
            L_batch = L_batch[0]

        return Trajectory(t=ts, X=X_batch, dW=dW_batch, U=U_batch, running_costs=L_batch)

    def generate_trajectory(
        self, key: jax.Array, x0: jnp.ndarray, deterministic: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Generate optimal trajectory from initial state."""
        X_T, J_T, X_traj, _ = self.solve_augmented_sde(key, x0, return_trajectory=True)
        ts = jnp.linspace(0.0, self.config.T, self.n_steps + 1)

        def compute_control(k):
            return self.optimal_control(ts[k], X_traj[k])
        U = jax.vmap(compute_control)(jnp.arange(self.n_steps))

        return ts, X_traj, U

    def evaluate(self, t: float, x: jnp.ndarray) -> float:
        return evaluate_value_function(self.phi_net, t, x)

    def evaluate_batch(self, t: float, x: jnp.ndarray) -> jnp.ndarray:
        return evaluate_value_function_batch(self.phi_net, t, x)


# =============================================================================
# Factory function
# =============================================================================

def create_solver(
    method: str,
    phi_net: eqx.Module,
    config: ProblemConfig,
    n_steps: int = 20,
    **kwargs
) -> eqx.Module:
    """
    Factory function to create solvers.

    Args:
        method: 'pinn', 'fbsnn', or 'neural_soc'
        phi_net: Value function network
        config: Problem configuration
        n_steps: Number of time steps
        **kwargs: Method-specific parameters

    Returns:
        Initialized solver
    """
    if method == 'pinn':
        return PINNSolver(phi_net, config, n_steps, **kwargs)
    elif method == 'fbsnn':
        return FBSNNSolver(phi_net, config, n_steps, **kwargs)
    elif method == 'neural_soc':
        return NeuralSOCSolver(phi_net, config, n_steps, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pinn', 'fbsnn', or 'neural_soc'.")
