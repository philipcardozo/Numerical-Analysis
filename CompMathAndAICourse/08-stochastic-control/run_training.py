#!/usr/bin/env python3
"""
Standalone training script for stochastic control methods.
Generates data and figures for lecture slides.

Outputs:
- convergence_comparison.csv with validation metrics
- trajectory_comparison.png
- value_function_t0.png
- Training times for all methods
"""
import gc
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Local modules
from problem import (
    ProblemConfig, create_problem_config, DEFAULT_CONFIG, SHIFTED_CONFIG,
    terminal_cost, analytical_solution_mc, sample_initial_states,
    evaluate_control_objective,
)
from networks import create_value_network, PhiResNet
from solvers import create_solver, PINNSolver, FBSNNSolver, NeuralSOCSolver
from trainer import train_solver, TrainingConfig, Trainer
from visualization import (
    plot_training_convergence, plot_value_function_slice,
    plot_value_function_diagonal_slice, compute_value_function_2d_diagonal_slice,
    plot_trajectories, plot_control_components, plot_method_comparison,
    save_results_csv, save_training_convergence_csv, save_convergence_comparison_csv,
    create_full_results_figure, COLORS,
)
from config_loader import (
    load_method_config, load_all_configs, print_config, MethodConfig
)

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

# ============================================
# Configuration
# ============================================
D = 100
T = 1.0
SHIFTED_TARGET = True  # Set to True for shifted target
MAX_ITERATIONS = 1000
VAL_FREQ = 200  # Validation frequency for CSV output

# Output directories - save directly to lecture slides
LECTURE_DIR = Path("/workspace/lectures/lecture08-high-dim-pdes/slides")
suffix = "shifted" if SHIFTED_TARGET else "default"
DATA_DIR = LECTURE_DIR / "data" / suffix
FIGURES_DIR = LECTURE_DIR / "figures" / suffix
MODEL_DIR = Path.cwd() / "saved_models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

print(f"\nOutput directories:")
print(f"  Data: {DATA_DIR}")
print(f"  Figures: {FIGURES_DIR}")
print(f"  Models: {MODEL_DIR}")

# Load configs
configs = load_all_configs(shifted=SHIFTED_TARGET, config_dir='config')
pinn_cfg = MethodConfig(**{**configs['pinn'].__dict__, 'max_iterations': MAX_ITERATIONS, 'val_freq': VAL_FREQ})
fbsnn_cfg = MethodConfig(**{**configs['fbsnn'].__dict__, 'max_iterations': MAX_ITERATIONS, 'val_freq': VAL_FREQ})
neural_soc_cfg = MethodConfig(**{**configs['neural_soc'].__dict__, 'max_iterations': MAX_ITERATIONS, 'val_freq': VAL_FREQ})

# Problem config
TERMINAL_COST_TYPE = "quadratic" if SHIFTED_TARGET else "log"
config = create_problem_config(d=D, T=T, shifted_target=SHIFTED_TARGET, terminal_cost_type=TERMINAL_COST_TYPE)

print(f"\nProblem: d={D}, T={T}, shifted={SHIFTED_TARGET}")
print(f"Max iterations: {MAX_ITERATIONS}")
print(f"Validation frequency: {VAL_FREQ}")

# Master key
MASTER_KEY = jr.PRNGKey(42)
key = MASTER_KEY

# Compute analytical value
key, opt_key = jr.split(key)
x0 = jnp.zeros(config.d)
optimal_value = float(analytical_solution_mc(0.0, x0, config, opt_key, n_samples=50000))
print(f"\nAnalytical optimal value Phi(0, 0) = {optimal_value:.6f}")


def save_model(params, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(params, f)
    print(f"Saved model to: {filepath}")


def cleanup():
    """Clean up memory."""
    gc.collect()
    jax.clear_caches()
    print("Memory cleaned")


def evaluate_at_checkpoint(solver, cfg, config, key, method_name):
    """Evaluate control objective at a checkpoint."""
    key, eval_key = jr.split(key)
    x0_eval = sample_initial_states(eval_key, 64, config, scale=0.1)

    solver_trained = create_solver(method_name, solver, config, n_steps=cfg.n_steps)
    key, eval_key = jr.split(key)
    result = evaluate_control_objective(
        lambda t, x: solver_trained.optimal_control(t, x),
        x0_eval, config, eval_key, n_steps=50
    )
    return result, key


# Storage for validation metrics
all_metrics = {
    'pinn': {'iterations': [], 'losses': [], 'rel_errors': [], 'suboptimalities': [], 'control_objectives': []},
    'fbsnn': {'iterations': [], 'losses': [], 'rel_errors': [], 'suboptimalities': [], 'control_objectives': []},
    'neuralsoc': {'iterations': [], 'losses': [], 'rel_errors': [], 'suboptimalities': [], 'control_objectives': []},
}
training_times = {}

# ============================================
# PINN Training
# ============================================
print("\n" + "="*60)
print("Training PINN")
print("="*60)

key, net_key = jr.split(key)
phi_net_pinn = create_value_network(
    net_key, D, architecture=pinn_cfg.architecture,
    hidden_width=pinn_cfg.hidden_width, depth=pinn_cfg.depth
)
pinn_solver = create_solver(
    'pinn', phi_net_pinn, config, n_steps=pinn_cfg.n_steps,
    sampling_mode='trajectory', use_trace_estimator=True,
)

start_time = time.time()
key, train_key = jr.split(key)
trained_pinn, history_pinn = train_solver(
    pinn_solver, config, method='pinn', key=train_key,
    max_iterations=pinn_cfg.max_iterations,
    batch_size=pinn_cfg.batch_size,
    learning_rate=pinn_cfg.learning_rate,
    lr_decay=pinn_cfg.lr_decay,
    lr_decay_steps=pinn_cfg.lr_decay_steps,
    loss_weights=pinn_cfg.loss_weights,
    print_freq=pinn_cfg.print_freq,
    val_freq=pinn_cfg.val_freq,
    patience=pinn_cfg.patience,
    verbose=True
)
pinn_time = time.time() - start_time
training_times['PINN'] = pinn_time
print(f"PINN training time: {pinn_time:.1f}s ({pinn_time/60:.1f} min)")

# Extract validation metrics from history
all_metrics['pinn']['iterations'] = history_pinn.iterations
all_metrics['pinn']['losses'] = history_pinn.losses
if hasattr(history_pinn, 'val_iterations'):
    all_metrics['pinn']['val_iterations'] = history_pinn.val_iterations
    all_metrics['pinn']['rel_errors'] = history_pinn.val_rel_errors
    all_metrics['pinn']['suboptimalities'] = history_pinn.val_suboptimalities
    all_metrics['pinn']['control_objectives'] = history_pinn.val_control_objectives

# Save model
model_suffix = "_shifted" if SHIFTED_TARGET else ""
PINN_MODEL_PATH = MODEL_DIR / f'pinn_params{model_suffix}.pkl'
save_model(trained_pinn, PINN_MODEL_PATH)

# Cleanup
del pinn_solver, phi_net_pinn
cleanup()


# ============================================
# FBSNN Training
# ============================================
print("\n" + "="*60)
print("Training FBSNN")
print("="*60)

key, net_key = jr.split(key)
phi_net_fbsnn = create_value_network(
    net_key, D, architecture=fbsnn_cfg.architecture,
    hidden_width=fbsnn_cfg.hidden_width, depth=fbsnn_cfg.depth
)
fbsnn_solver = create_solver('fbsnn', phi_net_fbsnn, config, n_steps=fbsnn_cfg.n_steps)

start_time = time.time()
key, train_key = jr.split(key)
trained_fbsnn, history_fbsnn = train_solver(
    fbsnn_solver, config, method='fbsnn', key=train_key,
    max_iterations=fbsnn_cfg.max_iterations,
    batch_size=fbsnn_cfg.batch_size,
    learning_rate=fbsnn_cfg.learning_rate,
    lr_decay=fbsnn_cfg.lr_decay,
    lr_decay_steps=fbsnn_cfg.lr_decay_steps,
    loss_weights=fbsnn_cfg.loss_weights,
    print_freq=fbsnn_cfg.print_freq,
    val_freq=fbsnn_cfg.val_freq,
    patience=fbsnn_cfg.patience,
    verbose=True
)
fbsnn_time = time.time() - start_time
training_times['FBSNN'] = fbsnn_time
print(f"FBSNN training time: {fbsnn_time:.1f}s ({fbsnn_time/60:.1f} min)")

# Extract validation metrics
all_metrics['fbsnn']['iterations'] = history_fbsnn.iterations
all_metrics['fbsnn']['losses'] = history_fbsnn.losses
if hasattr(history_fbsnn, 'val_iterations'):
    all_metrics['fbsnn']['val_iterations'] = history_fbsnn.val_iterations
    all_metrics['fbsnn']['rel_errors'] = history_fbsnn.val_rel_errors
    all_metrics['fbsnn']['suboptimalities'] = history_fbsnn.val_suboptimalities
    all_metrics['fbsnn']['control_objectives'] = history_fbsnn.val_control_objectives

# Save model
FBSNN_MODEL_PATH = MODEL_DIR / f'fbsnn_params{model_suffix}.pkl'
save_model(trained_fbsnn, FBSNN_MODEL_PATH)

# Cleanup
del fbsnn_solver, phi_net_fbsnn
cleanup()


# ============================================
# NeuralSOC Training
# ============================================
print("\n" + "="*60)
print("Training NeuralSOC")
print("="*60)

key, net_key = jr.split(key)
phi_net_neural_soc = create_value_network(
    net_key, D, architecture=neural_soc_cfg.architecture,
    hidden_width=neural_soc_cfg.hidden_width, depth=neural_soc_cfg.depth
)
neural_soc_solver = create_solver('neural_soc', phi_net_neural_soc, config, n_steps=neural_soc_cfg.n_steps)

start_time = time.time()
key, train_key = jr.split(key)
trained_neural_soc, history_neural_soc = train_solver(
    neural_soc_solver, config, method='neural_soc', key=train_key,
    max_iterations=neural_soc_cfg.max_iterations,
    batch_size=neural_soc_cfg.batch_size,
    learning_rate=neural_soc_cfg.learning_rate,
    lr_decay=neural_soc_cfg.lr_decay,
    lr_decay_steps=neural_soc_cfg.lr_decay_steps,
    loss_weights=neural_soc_cfg.loss_weights,
    print_freq=neural_soc_cfg.print_freq,
    val_freq=neural_soc_cfg.val_freq,
    patience=neural_soc_cfg.patience,
    verbose=True
)
neuralsoc_time = time.time() - start_time
training_times['NeuralSOC'] = neuralsoc_time
print(f"NeuralSOC training time: {neuralsoc_time:.1f}s ({neuralsoc_time/60:.1f} min)")

# Extract validation metrics
all_metrics['neuralsoc']['iterations'] = history_neural_soc.iterations
all_metrics['neuralsoc']['losses'] = history_neural_soc.losses
if hasattr(history_neural_soc, 'val_iterations'):
    all_metrics['neuralsoc']['val_iterations'] = history_neural_soc.val_iterations
    all_metrics['neuralsoc']['rel_errors'] = history_neural_soc.val_rel_errors
    all_metrics['neuralsoc']['suboptimalities'] = history_neural_soc.val_suboptimalities
    all_metrics['neuralsoc']['control_objectives'] = history_neural_soc.val_control_objectives

# Save model
NEURAL_SOC_MODEL_PATH = MODEL_DIR / f'neural_soc_params{model_suffix}.pkl'
save_model(trained_neural_soc, NEURAL_SOC_MODEL_PATH)

cleanup()


# ============================================
# Evaluation
# ============================================
print("\n" + "="*60)
print("EVALUATION")
print("="*60)

key, eval_key = jr.split(key)
x0_eval = sample_initial_states(eval_key, 128, config, scale=0.1)
results = {}

# Evaluate PINN
pinn_solver_trained = create_solver('pinn', trained_pinn, config, n_steps=pinn_cfg.n_steps)
key, eval_key = jr.split(key)
pinn_result = evaluate_control_objective(
    lambda t, x: pinn_solver_trained.optimal_control(t, x),
    x0_eval, config, eval_key, n_steps=100
)
results['PINN'] = {
    'control_objective': pinn_result.mean_objective,
    'std_objective': pinn_result.std_objective,
    'optimal_value': pinn_result.optimal_value,
    'relative_suboptimality': pinn_result.relative_suboptimality,
    'training_time': training_times['PINN'],
}
print(f"\nPINN: J = {pinn_result.mean_objective:.4f} ± {pinn_result.std_objective:.4f}, subopt = {pinn_result.relative_suboptimality:.2%}")

# Evaluate FBSNN
fbsnn_solver_trained = create_solver('fbsnn', trained_fbsnn, config, n_steps=fbsnn_cfg.n_steps)
key, eval_key = jr.split(key)
fbsnn_result = evaluate_control_objective(
    lambda t, x: fbsnn_solver_trained.optimal_control(t, x),
    x0_eval, config, eval_key, n_steps=100
)
results['FBSNN'] = {
    'control_objective': fbsnn_result.mean_objective,
    'std_objective': fbsnn_result.std_objective,
    'optimal_value': fbsnn_result.optimal_value,
    'relative_suboptimality': fbsnn_result.relative_suboptimality,
    'training_time': training_times['FBSNN'],
}
print(f"FBSNN: J = {fbsnn_result.mean_objective:.4f} ± {fbsnn_result.std_objective:.4f}, subopt = {fbsnn_result.relative_suboptimality:.2%}")

# Evaluate NeuralSOC
neural_soc_solver_trained = create_solver('neural_soc', trained_neural_soc, config, n_steps=neural_soc_cfg.n_steps)
key, eval_key = jr.split(key)
neural_soc_result = evaluate_control_objective(
    lambda t, x: neural_soc_solver_trained.optimal_control(t, x),
    x0_eval, config, eval_key, n_steps=100
)
results['NeuralSOC'] = {
    'control_objective': neural_soc_result.mean_objective,
    'std_objective': neural_soc_result.std_objective,
    'optimal_value': neural_soc_result.optimal_value,
    'relative_suboptimality': neural_soc_result.relative_suboptimality,
    'training_time': training_times['NeuralSOC'],
}
print(f"NeuralSOC: J = {neural_soc_result.mean_objective:.4f} ± {neural_soc_result.std_objective:.4f}, subopt = {neural_soc_result.relative_suboptimality:.2%}")

print(f"\nOptimal (analytical): {optimal_value:.4f}")


# ============================================
# Generate Convergence Comparison CSV
# ============================================
print("\n" + "="*60)
print("GENERATING CONVERGENCE CSV")
print("="*60)

# Combine all training histories into lecture-format CSV
# Get all unique iterations from all methods
all_iters = set()
for method in ['pinn', 'fbsnn', 'neuralsoc']:
    all_iters.update(all_metrics[method]['iterations'])
    if 'val_iterations' in all_metrics[method]:
        all_iters.update(all_metrics[method]['val_iterations'])
all_iters = sorted(all_iters)

# Build the combined dataframe
csv_data = {'iteration': all_iters}

for method, prefix in [('pinn', 'pinn'), ('fbsnn', 'fbsnn'), ('neuralsoc', 'neuralsoc')]:
    metrics = all_metrics[method]

    # Create lookup dicts for loss and validation metrics
    loss_lookup = dict(zip(metrics['iterations'], metrics['losses']))

    val_lookup = {}
    if 'val_iterations' in metrics and metrics['val_iterations']:
        for i, it in enumerate(metrics['val_iterations']):
            val_lookup[it] = {
                'rel_error': metrics['rel_errors'][i] if i < len(metrics['rel_errors']) else None,
                'suboptimality': metrics['suboptimalities'][i] if i < len(metrics['suboptimalities']) else None,
                'control_objective': metrics['control_objectives'][i] if i < len(metrics['control_objectives']) else None,
            }

    # Fill in the columns
    csv_data[f'{prefix}_loss'] = [loss_lookup.get(it, '') for it in all_iters]
    csv_data[f'{prefix}_rel_error'] = [val_lookup.get(it, {}).get('rel_error', '') for it in all_iters]
    csv_data[f'{prefix}_suboptimality'] = [val_lookup.get(it, {}).get('suboptimality', '') for it in all_iters]
    csv_data[f'{prefix}_control_objective'] = [val_lookup.get(it, {}).get('control_objective', '') for it in all_iters]

convergence_df = pd.DataFrame(csv_data)
convergence_csv_path = DATA_DIR / 'convergence_comparison.csv'
convergence_df.to_csv(convergence_csv_path, index=False)
print(f"Saved: {convergence_csv_path}")


# ============================================
# Generate Results Summary CSV
# ============================================
results_df = pd.DataFrame({
    'Method': ['PINN', 'FBSNN', 'NeuralSOC'],
    'Control_Objective': [results[m]['control_objective'] for m in ['PINN', 'FBSNN', 'NeuralSOC']],
    'Std_Objective': [results[m]['std_objective'] for m in ['PINN', 'FBSNN', 'NeuralSOC']],
    'Optimal_Value': [results[m]['optimal_value'] for m in ['PINN', 'FBSNN', 'NeuralSOC']],
    'Relative_Suboptimality': [results[m]['relative_suboptimality'] for m in ['PINN', 'FBSNN', 'NeuralSOC']],
    'Training_Time_s': [results[m]['training_time'] for m in ['PINN', 'FBSNN', 'NeuralSOC']],
})
results_csv_path = DATA_DIR / 'results_summary.csv'
results_df.to_csv(results_csv_path, index=False)
print(f"Saved: {results_csv_path}")


# ============================================
# Plots
# ============================================
print("\n" + "="*60)
print("GENERATING PLOTS")
print("="*60)

# Trajectory comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
key, k1, k2, k3 = jr.split(key, 4)
plot_trajectories(config, k1, solver=pinn_solver_trained, method='learned', n_trajectories=20, title='PINN Trajectories', ax=axes[0])
plot_trajectories(config, k2, solver=fbsnn_solver_trained, method='learned', n_trajectories=20, title='FBSNN Trajectories', ax=axes[1])
plot_trajectories(config, k3, solver=neural_soc_solver_trained, method='learned', n_trajectories=20, title='NeuralSOC Trajectories', ax=axes[2])
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'trajectory_comparison.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {FIGURES_DIR / 'trajectory_comparison.png'}")

# Value function at t=0
fig, ax = plt.subplots(figsize=(8, 6))
key, plot_key = jr.split(key)
plot_value_function_slice(config, plot_key, t=0.0, title='Analytical Value Function $\\Phi(0, x)$', ax=ax)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'value_function_t0.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {FIGURES_DIR / 'value_function_t0.png'}")


# ============================================
# Print Summary for LaTeX tables
# ============================================
print("\n" + "="*60)
print("SUMMARY FOR LATEX TABLES")
print("="*60)
print(f"\nProblem: {'Shifted' if SHIFTED_TARGET else 'Default'} Target")
print(f"Optimal value: {optimal_value:.4f}")
print()
print("Method       | Suboptimality | Training Time")
print("-" * 50)
for method in ['PINN', 'FBSNN', 'NeuralSOC']:
    subopt = results[method]['relative_suboptimality'] * 100
    time_min = results[method]['training_time'] / 60
    print(f"{method:12} | {subopt:10.2f}%  | {time_min:.1f} min")

print("\n" + "="*60)
print("DONE!")
print("="*60)
