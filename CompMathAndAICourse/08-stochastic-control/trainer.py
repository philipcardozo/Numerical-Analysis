"""
Training utilities for high-dimensional PDE solvers.

Provides a unified training interface for all solvers:
- PINNSolver: Physics-Informed Neural Network
- FBSNNSolver: Forward-Backward Stochastic Neural Network
- NeuralSOCSolver: Neural Stochastic Optimal Control with guided sampling
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from typing import Tuple, NamedTuple, Optional, Dict, Any, List
from dataclasses import dataclass
import time

from problem import (
    ProblemConfig, sample_initial_states, analytical_solution_mc,
    evaluate_control_objective, sample_controlled_trajectories,
    terminal_cost, running_cost,
)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimization
    learning_rate: float = 1e-3
    lr_decay: float = 0.1
    lr_decay_steps: int = 1500
    max_iterations: int = 5000
    batch_size: int = 64

    # Loss weights (method-specific)
    loss_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 0.1)

    # Logging
    print_freq: int = 100
    val_freq: int = 200

    # Early stopping
    patience: int = 500
    min_delta: float = 1e-6

    # Validation
    n_val_samples: int = 128
    n_mc_samples: int = 10000


class TrainingState(NamedTuple):
    """Container for training state."""
    iteration: int
    opt_state: Any
    best_loss: float
    best_model: Any
    patience_counter: int
    loss_history: list
    val_history: list


class TrainingHistory(NamedTuple):
    """Container for training history."""
    iterations: List[int]
    losses: List[float]
    val_iterations: List[int]
    val_rel_errors: List[float]
    val_abs_errors: List[float]
    val_suboptimalities: List[float]  # Policy suboptimality (J - J*) / J*
    val_control_objectives: List[float]  # Control objective J
    final_loss: float
    final_rel_error: float
    final_suboptimality: float
    total_time: float


class Trainer:
    """
    Unified trainer for all high-dimensional PDE solvers.

    Works with any solver: PINNSolver, FBSNNSolver, NeuralSOCSolver.
    """

    METHOD_ALIASES = {
        "deepbsde": "fbsnn",
        "pmp": "neural_soc",
    }

    def __init__(
        self,
        solver,
        config: TrainingConfig,
        method: str = "neural_soc"
    ):
        """
        Initialize trainer.

        Args:
            solver: Any solver (PINNSolver, FBSNNSolver, NeuralSOCSolver)
            config: Training configuration
            method: Training method ("pinn", "fbsnn", or "neural_soc")
        """
        self.solver = solver
        self.config = config
        self.method = self.METHOD_ALIASES.get(method, method)

        # Create optimizer with learning rate schedule
        schedule = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=config.lr_decay_steps,
            decay_rate=config.lr_decay,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(schedule)
        )

    def loss_fn(
        self,
        phi_net: eqx.Module,
        key: jax.Array,
        x0: jnp.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Compute loss for a batch."""
        solver = eqx.tree_at(lambda s: s.phi_net, self.solver, phi_net)

        if self.method == "neural_soc":
            weights = self.config.loss_weights[:2]
            loss, components = solver.compute_loss(key, x0, weights=weights)
        else:
            key, traj_key = jr.split(key)
            trajectory = solver.sample_training_trajectories(traj_key, x0)

            if self.method == "pinn":
                weights = self.config.loss_weights[:2]
            else:  # fbsnn
                weights = self.config.loss_weights[:3]

            loss, components = solver.compute_loss(trajectory, weights=weights, key=key)

        return loss, components

    @eqx.filter_jit
    def train_step(
        self,
        phi_net: eqx.Module,
        opt_state: Any,
        key: jax.Array,
        x0: jnp.ndarray
    ) -> Tuple[eqx.Module, Any, float, Dict[str, float]]:
        """Single training step."""
        (loss, components), grads = eqx.filter_value_and_grad(
            lambda net: self.loss_fn(net, key, x0), has_aux=True
        )(phi_net)

        updates, opt_state = self.optimizer.update(grads, opt_state, phi_net)
        phi_net = eqx.apply_updates(phi_net, updates)

        return phi_net, opt_state, loss, components

    def validate(
        self,
        phi_net: eqx.Module,
        key: jax.Array,
        problem_config: ProblemConfig
    ) -> Dict[str, float]:
        """Validate against analytical solution and compute policy suboptimality."""
        solver = eqx.tree_at(lambda s: s.phi_net, self.solver, phi_net)

        key, x_key, mc_key, ctrl_key = jr.split(key, 4)
        x_val = sample_initial_states(
            x_key, self.config.n_val_samples, problem_config, scale=0.1
        )

        phi_learned = solver.evaluate_batch(0.0, x_val)

        mc_keys = jr.split(mc_key, self.config.n_val_samples)
        phi_true = jax.vmap(
            lambda x, k: analytical_solution_mc(
                0.0, x, problem_config, k, self.config.n_mc_samples
            )
        )(x_val, mc_keys)

        abs_error = jnp.abs(phi_learned - phi_true)
        rel_error = abs_error / (jnp.abs(phi_true) + 1e-8)

        # Compute control objective for learned policy
        # Use a small batch starting from origin
        n_eval_trajs = min(64, self.config.n_val_samples)
        x0_eval_batch = jnp.zeros((n_eval_trajs, problem_config.d))

        def learned_policy(t, x):
            return solver.optimal_control(t, x)

        eval_result = evaluate_control_objective(
            learned_policy, x0_eval_batch, problem_config, ctrl_key,
            n_steps=50, n_mc_samples=self.config.n_mc_samples
        )

        j_learned = eval_result.mean_objective
        j_optimal = eval_result.optimal_value
        suboptimality = eval_result.relative_suboptimality

        return {
            "mean_abs_error": float(jnp.mean(abs_error)),
            "max_abs_error": float(jnp.max(abs_error)),
            "mean_rel_error": float(jnp.mean(rel_error)),
            "max_rel_error": float(jnp.max(rel_error)),
            "phi_learned_mean": float(jnp.mean(phi_learned)),
            "phi_true_mean": float(jnp.mean(phi_true)),
            "control_objective": float(j_learned),
            "optimal_value": float(j_optimal),
            "suboptimality": float(suboptimality),
        }

    def train(
        self,
        key: jax.Array,
        problem_config: ProblemConfig,
        verbose: bool = True
    ) -> Tuple[eqx.Module, TrainingHistory]:
        """
        Full training loop.

        Args:
            key: Random key
            problem_config: Problem configuration
            verbose: Print progress

        Returns:
            Tuple of (trained_network, training_history)
        """
        phi_net = self.solver.phi_net
        opt_state = self.optimizer.init(eqx.filter(phi_net, eqx.is_array))

        best_loss = float('inf')
        best_model = phi_net
        patience_counter = 0

        iterations_list = []
        losses_list = []
        val_iterations = []
        val_rel_errors = []
        val_abs_errors = []
        val_suboptimalities = []
        val_control_objectives = []

        start_time = time.time()

        for iteration in range(self.config.max_iterations):
            key, x_key, step_key = jr.split(key, 3)
            x0 = sample_initial_states(
                x_key, self.config.batch_size, problem_config, scale=0.1
            )

            phi_net, opt_state, loss, components = self.train_step(
                phi_net, opt_state, step_key, x0
            )

            if not jnp.isfinite(loss):
                raise RuntimeError(f"Training diverged at iteration {iteration}: loss={loss}")

            iterations_list.append(iteration)
            losses_list.append(float(loss))

            if loss < best_loss - self.config.min_delta:
                best_loss = loss
                best_model = phi_net
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and iteration % self.config.print_freq == 0:
                elapsed = time.time() - start_time
                print(f"Iter {iteration:5d} | Loss: {loss:.6f} | "
                      f"Best: {best_loss:.6f} | Time: {elapsed:.1f}s")

            if iteration % self.config.val_freq == 0:
                key, val_key = jr.split(key)
                val_metrics = self.validate(phi_net, val_key, problem_config)
                val_iterations.append(iteration)
                val_rel_errors.append(val_metrics['mean_rel_error'])
                val_abs_errors.append(val_metrics['mean_abs_error'])
                val_suboptimalities.append(val_metrics['suboptimality'])
                val_control_objectives.append(val_metrics['control_objective'])

                if verbose:
                    print(f"  Val | Rel Error: {val_metrics['mean_rel_error']:.4f} | "
                          f"Subopt: {val_metrics['suboptimality']:.4f}")

            if self.config.patience is not None and patience_counter >= self.config.patience:
                if verbose:
                    print(f"Early stopping at iteration {iteration}")
                break

        total_time = time.time() - start_time

        key, val_key = jr.split(key)
        final_val = self.validate(best_model, val_key, problem_config)

        history = TrainingHistory(
            iterations=iterations_list,
            losses=losses_list,
            val_iterations=val_iterations,
            val_rel_errors=val_rel_errors,
            val_abs_errors=val_abs_errors,
            val_suboptimalities=val_suboptimalities,
            val_control_objectives=val_control_objectives,
            final_loss=float(best_loss),
            final_rel_error=final_val['mean_rel_error'],
            final_suboptimality=final_val['suboptimality'],
            total_time=total_time,
        )

        if verbose:
            print(f"\nTraining complete!")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Final rel error: {final_val['mean_rel_error']:.4f}")
            print(f"  Final suboptimality: {final_val['suboptimality']:.4f}")

        return best_model, history


def train_solver(
    solver,
    problem_config: ProblemConfig,
    method: str,
    key: jax.Array,
    max_iterations: int = 5000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    lr_decay: float = 0.1,
    lr_decay_steps: int = 1500,
    loss_weights: Tuple[float, ...] = (1.0, 1.0, 1.0),
    print_freq: int = 100,
    val_freq: int = 200,
    patience: int = 500,
    verbose: bool = True
) -> Tuple[eqx.Module, TrainingHistory]:
    """
    Convenience function to train a solver.

    Args:
        solver: Initialized solver (PINNSolver, FBSNNSolver, NeuralSOCSolver)
        problem_config: Problem configuration
        method: 'pinn', 'fbsnn', or 'neural_soc'
        key: JAX random key
        max_iterations: Maximum training iterations
        batch_size: Batch size
        learning_rate: Initial learning rate
        lr_decay: Learning rate decay factor
        lr_decay_steps: Steps between decay applications
        loss_weights: Loss component weights
        print_freq: Print frequency
        val_freq: Validation frequency
        patience: Early stopping patience
        verbose: Print progress

    Returns:
        Tuple of (trained_network, training_history)
    """
    config = TrainingConfig(
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        lr_decay_steps=lr_decay_steps,
        max_iterations=max_iterations,
        batch_size=batch_size,
        loss_weights=loss_weights,
        print_freq=print_freq,
        val_freq=val_freq,
        patience=patience,
    )

    trainer = Trainer(solver, config, method=method)
    return trainer.train(key, problem_config, verbose=verbose)
