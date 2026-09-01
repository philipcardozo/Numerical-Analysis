"""
Configuration loader for stochastic control experiments.

Loads hyperparameters from YAML files produced by HPO tuning.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any


@dataclass
class MethodConfig:
    """Configuration for a single method."""
    # Network
    architecture: str
    hidden_width: int
    depth: int

    # Solver
    n_steps: int

    # Training
    learning_rate: float
    lr_decay: float
    lr_decay_steps: int
    max_iterations: int
    batch_size: int
    loss_weights: Tuple[float, ...]

    # Validation
    print_freq: int = 100
    val_freq: int = 200
    patience: int = 500

    # HPO metadata (optional)
    hpo_best_value: Optional[float] = None
    hpo_tuned_date: Optional[str] = None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load raw YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_method_config(raw_config: Dict[str, Any]) -> MethodConfig:
    """Parse raw YAML config into MethodConfig dataclass."""
    network = raw_config.get('network', {})
    solver = raw_config.get('solver', {})
    training = raw_config.get('training', {})
    output = raw_config.get('output', {})
    hpo_metadata = output.get('_hpo_metadata', {})

    return MethodConfig(
        # Network
        architecture=network.get('architecture', 'resnet'),
        hidden_width=network.get('hidden_width', 128),
        depth=network.get('depth', 4),

        # Solver
        n_steps=solver.get('n_steps', 30),

        # Training
        learning_rate=training.get('learning_rate', 1e-3),
        lr_decay=training.get('lr_decay', 0.1),
        lr_decay_steps=training.get('lr_decay_steps', 1500),
        max_iterations=training.get('max_iterations', 5000),
        batch_size=training.get('batch_size', 64),
        loss_weights=tuple(training.get('loss_weights', [1.0, 1.0])),

        # Validation
        print_freq=training.get('print_freq', 100),
        val_freq=training.get('val_freq', 200),
        patience=training.get('patience', 500),

        # HPO metadata
        hpo_best_value=hpo_metadata.get('best_value'),
        hpo_tuned_date=hpo_metadata.get('tuned_date'),
    )


def get_config_path(method: str, shifted: bool, config_dir: str = 'config') -> str:
    """Get path to config file for given method and target type."""
    target_type = 'shifted' if shifted else 'default'

    # Special case for neural_soc shifted - use no_bsde version
    if method == 'neural_soc' and shifted:
        filename = f'neural_soc_shifted_no_bsde_tuned.yaml'
    else:
        filename = f'{method}_{target_type}_tuned.yaml'

    return str(Path(config_dir) / filename)


def load_method_config(
    method: str,
    shifted: bool = False,
    config_dir: str = 'config'
) -> MethodConfig:
    """
    Load tuned configuration for a method.

    Args:
        method: 'pinn', 'fbsnn', or 'neural_soc'
        shifted: Whether to use shifted target config
        config_dir: Directory containing config files

    Returns:
        MethodConfig with tuned hyperparameters
    """
    config_path = get_config_path(method, shifted, config_dir)
    raw_config = load_config(config_path)
    return parse_method_config(raw_config)


def load_all_configs(
    shifted: bool = False,
    config_dir: str = 'config'
) -> Dict[str, MethodConfig]:
    """
    Load tuned configurations for all methods.

    Args:
        shifted: Whether to use shifted target configs
        config_dir: Directory containing config files

    Returns:
        Dict mapping method names to MethodConfig
    """
    methods = ['pinn', 'fbsnn', 'neural_soc']
    return {
        method: load_method_config(method, shifted, config_dir)
        for method in methods
    }


def print_config(config: MethodConfig, method: str = "Method") -> None:
    """Print configuration summary."""
    print(f"\n{method.upper()} Configuration:")
    print(f"  Network: {config.architecture}, width={config.hidden_width}, depth={config.depth}")
    print(f"  Solver: n_steps={config.n_steps}")
    print(f"  Training: lr={config.learning_rate:.6f}, batch={config.batch_size}")
    print(f"  LR schedule: decay={config.lr_decay:.4f}, steps={config.lr_decay_steps}")
    print(f"  Loss weights: {config.loss_weights}")
    if config.hpo_best_value is not None:
        print(f"  HPO best value: {config.hpo_best_value:.4f} (tuned {config.hpo_tuned_date})")


# Default configurations (fallback if YAML not found)
DEFAULT_CONFIGS = {
    'pinn': MethodConfig(
        architecture='resnet', hidden_width=128, depth=4, n_steps=30,
        learning_rate=1e-3, lr_decay=0.1, lr_decay_steps=1500,
        max_iterations=3000, batch_size=64, loss_weights=(1.0, 1.0),
    ),
    'fbsnn': MethodConfig(
        architecture='resnet', hidden_width=128, depth=4, n_steps=30,
        learning_rate=1e-3, lr_decay=0.1, lr_decay_steps=1500,
        max_iterations=3000, batch_size=64, loss_weights=(1.0, 1.0, 0.5),
    ),
    'neural_soc': MethodConfig(
        architecture='resnet', hidden_width=128, depth=4, n_steps=30,
        learning_rate=1e-3, lr_decay=0.1, lr_decay_steps=1500,
        max_iterations=3000, batch_size=64, loss_weights=(1.0, 1.0),
    ),
}
