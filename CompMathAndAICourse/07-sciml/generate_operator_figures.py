"""
Standalone script to generate operator learning figures from saved models.

DEPRECATED: This script is deprecated. Use operator-learning.ipynb instead,
which properly saves and loads data split indices for reproducibility.

Used when notebook execution has memory issues.
"""
import warnings
warnings.warn(
    "generate_operator_figures.py is deprecated. Use operator-learning.ipynb "
    "with TRAIN_MODE=False for reliable figure generation with saved indices.",
    DeprecationWarning
)

import matplotlib
matplotlib.use('Agg')

import sys
from pathlib import Path

# Setup paths
NOTEBOOK_DIR = Path(__file__).parent
sys.path.insert(0, str(NOTEBOOK_DIR))

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import json

from utils import load_pdebench_darcy, compute_l2_error, save_individual_figure
from models import create_fno, create_deeponet, count_params, rel_l2_error

print(f"JAX devices: {jax.devices()}")

# Paths
OUTPUT_DIR = NOTEBOOK_DIR / "figures"
MODEL_DIR = NOTEBOOK_DIR / "saved_models"
DATA_PATH = NOTEBOOK_DIR / "data" / "pdebench" / "2D_DarcyFlow_beta1.0_Train.hdf5"
INDICES_PATH = MODEL_DIR / "data_split_indices.npz"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load test data using SAVED INDICES (not relying on random seed consistency!)
print("Loading test data...")
if INDICES_PATH.exists():
    _, _, test_data, resolution = load_pdebench_darcy(
        DATA_PATH,
        n_train=1000, n_val=200, n_test=200,  # Ignored when load_indices is set
        seed=42,
        auto_download=True,
        normalize=False,
        load_indices=str(INDICES_PATH),  # Use saved indices!
    )
else:
    raise FileNotFoundError(
        f"No saved indices found at {INDICES_PATH}. "
        f"Run operator-learning.ipynb with TRAIN_MODE=True first."
    )
print(f"Loaded {len(test_data['kappa'])} test samples")

# Configuration (must match saved models)
fno_config = {'modes': 12, 'width': 20, 'n_layers': 4, 'use_grid': True, 'n_epochs': 150}
deeponet_config = {'n_sensors': 4096, 'latent_dim': 256,
                   'branch_hidden': [512, 512, 512], 'trunk_hidden': [512, 512, 512],
                   'n_epochs': 300}

# Initialize models
print("Initializing models...")
fno_model = create_fno(
    modes=fno_config['modes'],
    width=fno_config['width'],
    n_layers=fno_config['n_layers'],
    use_grid=fno_config['use_grid'],
)
fno_params = fno_model.init(random.PRNGKey(0), resolution)

deeponet_model = create_deeponet(
    n_sensors=deeponet_config['n_sensors'],
    latent_dim=deeponet_config['latent_dim'],
    branch_hidden=deeponet_config['branch_hidden'],
    trunk_hidden=deeponet_config['trunk_hidden'],
)
deeponet_params = deeponet_model.init(random.PRNGKey(100), resolution)

n_params_fno = count_params(fno_params)
n_params_deeponet = count_params(deeponet_params)
print(f"FNO params: {n_params_fno:,}")
print(f"DeepONet params: {n_params_deeponet:,}")

# Load saved models
print("Loading saved models...")
with open(MODEL_DIR / 'best_fno_params.pkl', 'rb') as f:
    best_fno_params = pickle.load(f)
print("  FNO loaded")

with open(MODEL_DIR / 'best_deeponet_params.pkl', 'rb') as f:
    best_deeponet_params = pickle.load(f)
print("  DeepONet loaded")

# Evaluate on test set
print("\nEvaluating models...")

def compute_test_metrics(model, params, test_data):
    """Compute test MSE and relative L2 error."""
    u_pred = model.apply(params, test_data['kappa'])
    mse = float(jnp.mean((u_pred - test_data['u'])**2))
    rel_l2 = float(rel_l2_error(u_pred, test_data['u']))
    return mse, rel_l2

test_loss_fno, test_rel_l2_fno = compute_test_metrics(fno_model, best_fno_params, test_data)
test_loss_don, test_rel_l2_don = compute_test_metrics(deeponet_model, best_deeponet_params, test_data)

print(f"FNO:      MSE={test_loss_fno:.6e}, Rel L2={test_rel_l2_fno*100:.1f}%")
print(f"DeepONet: MSE={test_loss_don:.6e}, Rel L2={test_rel_l2_don*100:.1f}%")

# Inference time
from time import time
_ = fno_model.apply(best_fno_params, test_data['kappa'][:10])  # warmup
start = time()
for _ in range(10):
    _ = fno_model.apply(best_fno_params, test_data['kappa'])
fno_inference_time = (time() - start) / 10 / len(test_data['kappa']) * 1000

_ = deeponet_model.apply(best_deeponet_params, test_data['kappa'][:10])  # warmup
start = time()
for _ in range(10):
    _ = deeponet_model.apply(best_deeponet_params, test_data['kappa'])
deeponet_inference_time = (time() - start) / 10 / len(test_data['kappa']) * 1000

print(f"FNO inference: {fno_inference_time:.2f} ms/sample")
print(f"DeepONet inference: {deeponet_inference_time:.2f} ms/sample")

# Generate individual figures
print("\nGenerating figures...")
test_indices = [0, 1, 2]

for idx in test_indices:
    kappa = np.array(test_data['kappa'][idx])
    u_ref = np.array(test_data['u'][idx])
    u_fno = np.array(fno_model.apply(best_fno_params, test_data['kappa'][idx:idx+1])[0])
    u_don = np.array(deeponet_model.apply(best_deeponet_params, test_data['kappa'][idx:idx+1])[0])

    err_fno = compute_l2_error(u_fno, u_ref)
    err_don = compute_l2_error(u_don, u_ref)

    # Color limits
    u_vmin = min(u_ref.min(), u_fno.min(), u_don.min())
    u_vmax = max(u_ref.max(), u_fno.max(), u_don.max())
    err_max = max(np.abs(u_fno - u_ref).max(), np.abs(u_don - u_ref).max())

    # Shared figures
    save_individual_figure(kappa, f'$\\kappa$ (test {idx})',
                          OUTPUT_DIR / f'operator_test{idx}_kappa.png', cmap='viridis')
    save_individual_figure(u_ref, f'Reference $u$ (test {idx})',
                          OUTPUT_DIR / f'operator_test{idx}_reference.png', cmap='RdBu_r',
                          vmin=u_vmin, vmax=u_vmax)

    # FNO
    save_individual_figure(u_fno, f'FNO ({err_fno*100:.1f}%)',
                          OUTPUT_DIR / f'fno_test{idx}_prediction.png', cmap='RdBu_r',
                          vmin=u_vmin, vmax=u_vmax)
    save_individual_figure(np.abs(u_fno - u_ref), '|FNO Error|',
                          OUTPUT_DIR / f'fno_test{idx}_error.png', cmap='hot',
                          vmin=0, vmax=err_max)

    # DeepONet
    save_individual_figure(u_don, f'DeepONet ({err_don*100:.1f}%)',
                          OUTPUT_DIR / f'deeponet_test{idx}_prediction.png', cmap='RdBu_r',
                          vmin=u_vmin, vmax=u_vmax)
    save_individual_figure(np.abs(u_don - u_ref), '|DeepONet Error|',
                          OUTPUT_DIR / f'deeponet_test{idx}_error.png', cmap='hot',
                          vmin=0, vmax=err_max)

    print(f"Test {idx}: FNO={err_fno*100:.1f}%, DeepONet={err_don*100:.1f}%")

# Save summary
summary = {
    'fno': {
        'n_params': n_params_fno,
        'test_rel_l2_error': float(test_rel_l2_fno),
        'inference_time_ms': float(fno_inference_time),
        'epochs': fno_config['n_epochs'],
    },
    'deeponet': {
        'n_params': n_params_deeponet,
        'test_rel_l2_error': float(test_rel_l2_don),
        'inference_time_ms': float(deeponet_inference_time),
        'epochs': deeponet_config['n_epochs'],
    }
}
with open(OUTPUT_DIR / 'operator_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {OUTPUT_DIR / 'operator_summary.json'}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"FNO:      {test_rel_l2_fno*100:.1f}% error, {fno_inference_time:.2f}ms inference")
print(f"DeepONet: {test_rel_l2_don*100:.1f}% error, {deeponet_inference_time:.2f}ms inference")
print("=" * 60)
