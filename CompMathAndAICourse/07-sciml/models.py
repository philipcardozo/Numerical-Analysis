"""
Neural Network Models for Lecture 07: Scientific ML for PDEs

Self-contained implementations of:
- MLP (for PINN)
- FNO (Fourier Neural Operator)
- DeepONet (Deep Operator Network)

All implementations use pure JAX without external dependencies like Equinox.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional


# =============================================================================
# MLP (Multi-Layer Perceptron)
# =============================================================================

def initialize_mlp(
    layer_sizes: List[int],
    key: jax.random.PRNGKey,
    scale: float = 1.0,
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Initialize MLP parameters with Xavier initialization.

    Parameters:
        layer_sizes: List [input_dim, hidden1, ..., output_dim]
        key: JAX random key
        scale: Scale factor for initialization

    Returns:
        params: List of (W, b) tuples for each layer
    """
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)

    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        key_w, key_b = jax.random.split(keys[i])
        init_scale = scale * jnp.sqrt(2.0 / (n_in + n_out))
        W = jax.random.normal(key_w, (n_in, n_out)) * init_scale
        b = jnp.zeros(n_out)
        params.append((W, b))

    return params


def mlp_forward(
    params: List[Tuple[jnp.ndarray, jnp.ndarray]],
    x: jnp.ndarray,
    activation: Callable = jax.nn.tanh,
) -> jnp.ndarray:
    """
    Forward pass through MLP.

    Parameters:
        params: List of (W, b) tuples
        x: Input (batch_size, input_dim) or (input_dim,)
        activation: Activation function

    Returns:
        Output (batch_size, output_dim) or (output_dim,)
    """
    for W, b in params[:-1]:
        x = activation(x @ W + b)

    W, b = params[-1]
    return x @ W + b


def count_params(params) -> int:
    """Count total number of parameters (arrays only)."""
    return sum(p.size for p in jax.tree_util.tree_leaves(params) if hasattr(p, 'size'))


# =============================================================================
# FNO (Fourier Neural Operator)
# =============================================================================

class SpectralConv2d:
    """
    2D Spectral convolution layer operating in Fourier space.

    Uses dual weight matrices for positive and negative frequency modes.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

    def init(self, key: jax.random.PRNGKey) -> Dict:
        """Initialize Fourier weights."""
        key1_r, key1_i, key2_r, key2_i = jax.random.split(key, 4)

        scale = 1.0 / (self.in_channels * self.out_channels)
        shape = (self.in_channels, self.out_channels, self.modes1, self.modes2)

        return {
            'weights1_real': jax.random.uniform(key1_r, shape) * scale,
            'weights1_imag': jax.random.uniform(key1_i, shape) * scale,
            'weights2_real': jax.random.uniform(key2_r, shape) * scale,
            'weights2_imag': jax.random.uniform(key2_i, shape) * scale,
        }

    def apply(self, params: Dict, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through spectral convolution.

        Parameters:
            params: Dict with weight arrays
            x: Input (batch, height, width, in_channels)

        Returns:
            Output (batch, height, width, out_channels)
        """
        batch, height, width, _ = x.shape

        # Form complex weights
        weights1 = params['weights1_real'] + 1j * params['weights1_imag']
        weights2 = params['weights2_real'] + 1j * params['weights2_imag']

        # Permute to channels-first for FFT
        x_perm = jnp.transpose(x, (0, 3, 1, 2))

        # 2D FFT
        x_ft = jnp.fft.rfft2(x_perm)

        # Output in frequency domain
        out_ft = jnp.zeros((batch, self.out_channels, height, width // 2 + 1), dtype=x_ft.dtype)

        # Positive modes
        x_ft_pos = x_ft[:, :, :self.modes1, :self.modes2]
        out_pos = jnp.einsum("bixy,ioxy->boxy", x_ft_pos, weights1)
        out_ft = out_ft.at[:, :, :self.modes1, :self.modes2].set(out_pos)

        # Negative modes
        x_ft_neg = x_ft[:, :, -self.modes1:, :self.modes2]
        out_neg = jnp.einsum("bixy,ioxy->boxy", x_ft_neg, weights2)
        out_ft = out_ft.at[:, :, -self.modes1:, :self.modes2].set(out_neg)

        # Inverse FFT
        out = jnp.fft.irfft2(out_ft, s=(height, width))

        # Permute back to channels-last
        return jnp.transpose(out, (0, 2, 3, 1))


class FNOBlock:
    """FNO block: spectral conv + 1x1 conv skip connection."""

    def __init__(self, channels: int, modes1: int, modes2: int, activation: Callable = jax.nn.gelu):
        self.channels = channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.activation = activation
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)

    def init(self, key: jax.random.PRNGKey) -> Dict:
        key_spectral, key_skip = jax.random.split(key)

        spectral_params = self.spectral_conv.init(key_spectral)

        scale = jnp.sqrt(2.0 / (self.channels + self.channels))
        W_skip = jax.random.normal(key_skip, (self.channels, self.channels)) * scale

        return {'spectral': spectral_params, 'W_skip': W_skip}

    def apply(self, params: Dict, x: jnp.ndarray) -> jnp.ndarray:
        # Spectral path
        v1 = self.spectral_conv.apply(params['spectral'], x)

        # Skip path (1x1 conv)
        v2 = jnp.einsum("bhwi,io->bhwo", x, params['W_skip'])

        return self.activation(v1 + v2)


class FNO2d:
    """
    2D Fourier Neural Operator.

    Architecture:
    1. Lifting: in_channels -> width
    2. N spectral conv blocks
    3. Projection: width -> 128 -> out_channels
    """

    def __init__(
        self,
        modes: int = 12,
        width: int = 32,
        n_layers: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        use_grid: bool = True,
        activation: Callable = jax.nn.gelu,
    ):
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_grid = use_grid
        self.activation = activation

        # Input channels with grid coordinates
        self.lift_channels = in_channels + 2 if use_grid else in_channels

        self.blocks = [FNOBlock(width, modes, modes, activation) for _ in range(n_layers)]

    def init(self, key: jax.random.PRNGKey, resolution: int) -> Dict:
        """
        Initialize FNO parameters.

        Parameters:
            key: JAX random key
            resolution: Grid resolution

        Returns:
            params: Dict containing all parameters
        """
        keys = jax.random.split(key, self.n_layers + 3)

        # Lifting
        scale_lift = jnp.sqrt(2.0 / (self.lift_channels + self.width))
        W_lift = jax.random.normal(keys[0], (self.lift_channels, self.width)) * scale_lift

        # Blocks
        blocks_params = [block.init(keys[i + 1]) for i, block in enumerate(self.blocks)]

        # Two-layer projection: width -> 128 -> out_channels
        hidden_proj = 128
        scale_proj1 = jnp.sqrt(2.0 / (self.width + hidden_proj))
        W_proj1 = jax.random.normal(keys[-2], (self.width, hidden_proj)) * scale_proj1

        scale_proj2 = jnp.sqrt(2.0 / (hidden_proj + self.out_channels))
        W_proj2 = jax.random.normal(keys[-1], (hidden_proj, self.out_channels)) * scale_proj2

        return {
            'W_lift': W_lift,
            'blocks': blocks_params,
            'W_proj1': W_proj1,
            'W_proj2': W_proj2,
        }

    def _get_grid(self, batch: int, height: int, width: int) -> jnp.ndarray:
        """Create normalized grid coordinates [0, 1] x [0, 1]."""
        x = jnp.linspace(0, 1, height)
        y = jnp.linspace(0, 1, width)
        xx, yy = jnp.meshgrid(x, y, indexing='ij')
        grid = jnp.stack([xx, yy], axis=-1)
        return jnp.broadcast_to(grid, (batch, height, width, 2))

    def apply(self, params: Dict, kappa: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass: kappa [B, H, W] -> u [B, H, W]

        Parameters:
            params: Model parameters
            kappa: Input permeability field (batch, height, width)

        Returns:
            u: Predicted solution (batch, height, width)
        """
        batch, height, width = kappa.shape

        # Add channel dimension
        x = kappa[..., None]  # (B, H, W, 1)

        # Add grid coordinates
        if self.use_grid:
            grid = self._get_grid(batch, height, width)
            x = jnp.concatenate([x, grid], axis=-1)  # (B, H, W, 3)

        # Lifting
        h = jnp.einsum("bhwi,io->bhwo", x, params['W_lift'])

        # FNO blocks
        for i, block in enumerate(self.blocks):
            h = block.apply(params['blocks'][i], h)

        # Two-layer projection
        h = jnp.einsum("bhwi,io->bhwo", h, params['W_proj1'])
        h = self.activation(h)
        out = jnp.einsum("bhwi,io->bhwo", h, params['W_proj2'])

        return out[..., 0]  # Remove channel dimension


def create_fno(
    modes: int = 12,
    width: int = 20,
    n_layers: int = 4,
    use_grid: bool = True,
) -> FNO2d:
    """
    Create FNO model with specified architecture.

    Parameters:
        modes: Number of Fourier modes to keep
        width: Hidden channel dimension
        n_layers: Number of FNO blocks
        use_grid: Whether to add grid coordinates as input

    Returns:
        FNO2d model instance
    """
    return FNO2d(
        modes=modes,
        width=width,
        n_layers=n_layers,
        use_grid=use_grid,
    )


# =============================================================================
# DeepONet (Deep Operator Network)
# =============================================================================

class DeepONet:
    """
    Deep Operator Network for learning operators.

    Architecture:
    - Branch network: encodes input function at sensor locations
    - Trunk network: encodes query coordinates
    - Output: inner product of branch and trunk embeddings

    Parameters:
        n_sensors: Number of sensor points for input function
        latent_dim: Dimension of latent embedding
        branch_hidden: Hidden layer sizes for branch network
        trunk_hidden: Hidden layer sizes for trunk network
        activation: Activation function
    """

    def __init__(
        self,
        n_sensors: int = 64,
        latent_dim: int = 50,
        branch_hidden: List[int] = None,
        trunk_hidden: List[int] = None,
        activation: Callable = jax.nn.gelu,
    ):
        self.n_sensors = n_sensors
        self.latent_dim = latent_dim
        self.branch_hidden = branch_hidden or [100, 100]
        self.trunk_hidden = trunk_hidden or [100, 100]
        self.activation = activation

        # Layer sizes
        self.branch_layers = [n_sensors] + self.branch_hidden + [latent_dim]
        self.trunk_layers = [2] + self.trunk_hidden + [latent_dim]  # 2D coordinates

    def init(self, key: jax.random.PRNGKey, resolution: int = 128) -> Dict:
        """
        Initialize DeepONet parameters.

        Parameters:
            key: JAX random key
            resolution: Grid resolution for query points

        Returns:
            params: Dict containing all parameters
        """
        key_branch, key_trunk, key_sensors = jax.random.split(key, 3)

        # Generate sensor locations (uniform grid)
        sensor_grid_size = int(np.ceil(np.sqrt(self.n_sensors)))
        x = jnp.linspace(0, 1, sensor_grid_size)
        y = jnp.linspace(0, 1, sensor_grid_size)
        xx, yy = jnp.meshgrid(x, y)
        sensor_coords = jnp.stack([xx.ravel(), yy.ravel()], axis=1)[:self.n_sensors]

        # Generate query coordinates (full grid)
        x_query = jnp.linspace(0, 1, resolution)
        y_query = jnp.linspace(0, 1, resolution)
        xx_q, yy_q = jnp.meshgrid(x_query, y_query)
        query_coords = jnp.stack([xx_q.ravel(), yy_q.ravel()], axis=1)

        # Initialize branch and trunk networks
        branch_params = initialize_mlp(self.branch_layers, key_branch)
        trunk_params = initialize_mlp(self.trunk_layers, key_trunk)

        # Store resolution as instance attribute (not in params for grad compatibility)
        self.resolution = resolution

        return {
            'branch': branch_params,
            'trunk': trunk_params,
            'sensor_coords': sensor_coords,
            'query_coords': query_coords,
        }

    def _sample_at_sensors(self, field: jnp.ndarray, sensor_coords: jnp.ndarray) -> jnp.ndarray:
        """
        Sample field at sensor locations using bilinear interpolation.

        Parameters:
            field: Field values (batch, H, W)
            sensor_coords: Sensor locations (n_sensors, 2) in [0, 1]^2

        Returns:
            values: Field values at sensors (batch, n_sensors)
        """
        n_grid = field.shape[1]
        sensor_x = sensor_coords[:, 0] * (n_grid - 1)
        sensor_y = sensor_coords[:, 1] * (n_grid - 1)

        x0 = jnp.floor(sensor_x).astype(jnp.int32)
        y0 = jnp.floor(sensor_y).astype(jnp.int32)
        x1 = jnp.minimum(x0 + 1, n_grid - 1)
        y1 = jnp.minimum(y0 + 1, n_grid - 1)

        dx = sensor_x - x0
        dy = sensor_y - y0

        def sample_one(f):
            v00, v01 = f[y0, x0], f[y0, x1]
            v10, v11 = f[y1, x0], f[y1, x1]
            return v00 * (1 - dx) * (1 - dy) + v01 * dx * (1 - dy) + v10 * (1 - dx) * dy + v11 * dx * dy

        return jax.vmap(sample_one)(field)

    def apply(self, params: Dict, kappa: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass: kappa [B, H, W] -> u [B, H, W]

        Parameters:
            params: Model parameters
            kappa: Input permeability field (batch, height, width)

        Returns:
            u: Predicted solution (batch, height, width)
        """
        batch = kappa.shape[0]
        resolution = self.resolution

        # Sample kappa at sensor locations
        kappa_sensors = self._sample_at_sensors(kappa, params['sensor_coords'])

        # Branch network: (batch, n_sensors) -> (batch, latent_dim)
        branch_out = mlp_forward(params['branch'], kappa_sensors, self.activation)

        # Trunk network: (n_queries, 2) -> (n_queries, latent_dim)
        trunk_out = mlp_forward(params['trunk'], params['query_coords'], self.activation)

        # Inner product: (batch, latent_dim) x (n_queries, latent_dim) -> (batch, n_queries)
        u_flat = jnp.einsum('bp,qp->bq', branch_out, trunk_out)

        # Reshape to grid
        return u_flat.reshape(batch, resolution, resolution)


def create_deeponet(
    n_sensors: int = 4096,
    latent_dim: int = 256,
    branch_hidden: List[int] = None,
    trunk_hidden: List[int] = None,
    activation: Callable = jax.nn.gelu,
) -> DeepONet:
    """
    Create DeepONet model with specified architecture.

    Parameters:
        n_sensors: Number of sensor points
        latent_dim: Latent embedding dimension
        branch_hidden: Hidden layer sizes for branch net
        trunk_hidden: Hidden layer sizes for trunk net
        activation: Activation function

    Returns:
        DeepONet model instance
    """
    return DeepONet(
        n_sensors=n_sensors,
        latent_dim=latent_dim,
        branch_hidden=branch_hidden or [512, 512, 512],
        trunk_hidden=trunk_hidden or [512, 512, 512],
        activation=activation,
    )


# =============================================================================
# Training Utilities
# =============================================================================

def rel_l2_error(u_pred: jnp.ndarray, u_true: jnp.ndarray) -> float:
    """Compute relative L2 error."""
    return float(jnp.sqrt(jnp.mean((u_pred - u_true) ** 2)) /
                 jnp.sqrt(jnp.mean(u_true ** 2)))


def mse_loss(u_pred: jnp.ndarray, u_true: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error loss."""
    return jnp.mean((u_pred - u_true) ** 2)


# =============================================================================
# PINN Utilities
# =============================================================================

def compute_darcy_residual(
    network_fn: Callable,
    params,
    xy: jnp.ndarray,
    kappa: jnp.ndarray,
    f: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute PDE residual for Darcy flow: -div(kappa * grad(u)) - f

    Parameters:
        network_fn: Network function (params, xy) -> u
        params: Network parameters
        xy: Collocation points (n_points, 2)
        kappa: Permeability at collocation points (n_points,)
        f: Forcing at collocation points (n_points,)

    Returns:
        residual: PDE residual at each point (n_points,)
    """
    def u_fn(xy_single):
        return network_fn(params, xy_single[None, :])[0, 0]

    # Compute gradients
    def grad_u(xy_single):
        return jax.grad(u_fn)(xy_single)

    # Compute Hessian (second derivatives)
    def hess_u(xy_single):
        return jax.hessian(u_fn)(xy_single)

    # Vectorize over collocation points
    grads = jax.vmap(grad_u)(xy)  # (n_points, 2)
    hessians = jax.vmap(hess_u)(xy)  # (n_points, 2, 2)

    # Extract derivatives
    u_x = grads[:, 0]
    u_y = grads[:, 1]
    u_xx = hessians[:, 0, 0]
    u_yy = hessians[:, 1, 1]

    # PDE: -div(kappa * grad(u)) = f
    # Expand: -kappa * laplacian(u) - grad(kappa) . grad(u) = f
    # For cell-centered grid, we assume kappa is constant within each cell
    # So: -kappa * (u_xx + u_yy) = f
    residual = -kappa * (u_xx + u_yy) - f

    return residual
