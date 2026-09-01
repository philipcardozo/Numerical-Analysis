"""
Neural Network Architectures for Value Function Approximation.

Two architectures provided:

1. ValueNetwork (MLP):
   - Standard multilayer perceptron
   - Flexible activation functions (sin, tanh, silu, relu)
   - Use for: Baseline comparisons, simple problems

2. PhiResNet (Residual Network):
   - Residual connections for gradient stability
   - Efficient O(d) Laplacian computation via Hutchinson/HVP
   - Use for: PINN solver, high-dimensional problems (d>50)

Solver Recommendations:
- FBSNNSolver: ValueNetwork (simpler, no Laplacian needed)
- PINNSolver: PhiResNet (requires Laplacian)
- NeuralSOCSolver: PhiResNet (ResNet stability helps)
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import List, Tuple, Optional, Callable


class ValueNetwork(eqx.Module):
    """
    MLP for approximating the value function Phi(t, x).

    Architecture follows FBSNNs:
    - Input: [t, x] of dimension 1 + d
    - Hidden layers with configurable activation
    - Output: scalar Phi(t, x)

    Attributes:
        layers: List of linear layers
        activation: Activation function
        d: Spatial dimension
    """

    layers: List[eqx.nn.Linear]
    activation: Callable
    d: int

    def __init__(
        self,
        key: jax.Array,
        d: int,
        hidden_dims: List[int] = None,
        activation: str = "tanh"
    ):
        """
        Initialize ValueNetwork.

        Args:
            key: JAX random key
            d: Spatial dimension (input is d+1 including time)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('sin', 'tanh', 'silu', 'relu')
        """
        self.d = d

        if hidden_dims is None:
            hidden_dims = [256, 256, 256, 256]

        if activation == "sin":
            self.activation = jnp.sin
        elif activation == "tanh":
            self.activation = jnp.tanh
        elif activation == "silu":
            self.activation = jax.nn.silu
        elif activation == "relu":
            self.activation = jax.nn.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layer_dims = [d + 1] + hidden_dims + [1]
        n_layers = len(layer_dims) - 1

        keys = jr.split(key, n_layers)
        self.layers = []
        for i in range(n_layers):
            self.layers.append(
                eqx.nn.Linear(layer_dims[i], layer_dims[i + 1], key=keys[i])
            )

    def __call__(self, t: float, x: jnp.ndarray) -> float:
        """
        Forward pass: compute Phi(t, x).

        Args:
            t: Time (scalar)
            x: State vector, shape (d,)

        Returns:
            Scalar value function Phi(t, x)
        """
        h = jnp.concatenate([jnp.atleast_1d(t), x])

        for layer in self.layers[:-1]:
            h = self.activation(layer(h))

        output = self.layers[-1](h)
        return output.squeeze()

    def value_and_gradient(self, t: float, x: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        """Compute both Phi(t, x) and grad_x Phi(t, x)."""
        phi, grad_phi = jax.value_and_grad(lambda x: self(t, x))(x)
        return phi, grad_phi


class PhiResNet(eqx.Module):
    """
    Residual network for value function.

    Architecture:
        input = [t, x]
        -> Input layer with activation
        -> Residual blocks: h = h + activation(layer(h))
        -> Output layer (scalar)

    Uses SiLU activation and residual connections for stable gradients.

    Attributes:
        layers: List of linear layers
        d: Spatial dimension
        m: Hidden layer width
        depth: Number of residual blocks
    """

    layers: List[eqx.nn.Linear]
    d: int
    m: int
    depth: int

    def __init__(
        self,
        key: jax.Array,
        d: int,
        m: int = 64,
        depth: int = 4
    ):
        """
        Initialize PhiResNet.

        Args:
            key: JAX random key
            d: Spatial dimension (input is d+1 including time)
            m: Hidden layer width
            depth: Number of residual blocks
        """
        self.d = d
        self.m = m
        self.depth = depth

        keys = jr.split(key, depth + 2)

        # Input layer: (d+1) -> m
        self.layers = [eqx.nn.Linear(d + 1, m, key=keys[0])]

        # Residual layers: m -> m
        for i in range(depth):
            self.layers.append(eqx.nn.Linear(m, m, key=keys[i + 1]))

        # Output layer: m -> 1
        self.layers.append(eqx.nn.Linear(m, 1, key=keys[-1]))

    def __call__(self, t: float, x: jnp.ndarray) -> float:
        """
        Forward pass: compute Phi(t, x).

        Args:
            t: Time (scalar)
            x: State vector, shape (d,)

        Returns:
            Scalar value function Phi(t, x)
        """
        h = jnp.concatenate([jnp.atleast_1d(t), x])

        # Input layer with activation
        h = jax.nn.silu(self.layers[0](h))

        # Residual blocks
        for i in range(1, self.depth + 1):
            h_residual = jax.nn.silu(self.layers[i](h))
            h = h + h_residual

        # Output layer (no activation)
        output = self.layers[-1](h)
        return output.squeeze()

    def value_and_gradient(self, t: float, x: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        """Compute both Phi(t, x) and grad_x Phi(t, x)."""
        phi, grad_phi = jax.value_and_grad(lambda x: self(t, x))(x)
        return phi, grad_phi

    def value_gradient_and_laplacian(
        self, t: float, x: jnp.ndarray
    ) -> Tuple[float, jnp.ndarray, float]:
        """
        Compute Phi, grad_x Phi, and Laplacian (trace of Hessian).

        Uses efficient O(d) computation via Hessian-vector products.

        Args:
            t: Time (scalar)
            x: State vector, shape (d,)

        Returns:
            Tuple of (phi, grad_phi, laplacian_phi)
        """
        phi, grad_phi = jax.value_and_grad(lambda x: self(t, x))(x)

        d = x.shape[0]

        def hvp_diagonal(i):
            """Compute H[i,i] via forward-over-reverse AD."""
            e_i = jnp.zeros(d).at[i].set(1.0)
            _, hvp = jax.jvp(
                jax.grad(lambda x: self(t, x)),
                (x,),
                (e_i,)
            )
            return hvp[i]

        laplacian_phi = jnp.sum(jax.vmap(hvp_diagonal)(jnp.arange(d)))

        return phi, grad_phi, laplacian_phi

    def laplacian(self, t: float, x: jnp.ndarray) -> float:
        """Compute Laplacian (trace of Hessian) efficiently."""
        d = x.shape[0]

        def hvp_diagonal(i):
            e_i = jnp.zeros(d).at[i].set(1.0)
            _, hvp = jax.jvp(
                jax.grad(lambda x: self(t, x)),
                (x,),
                (e_i,)
            )
            return hvp[i]

        return jnp.sum(jax.vmap(hvp_diagonal)(jnp.arange(d)))


def create_value_network(
    key: jax.Array,
    d: int,
    architecture: str = "resnet",
    **kwargs
) -> eqx.Module:
    """
    Factory function to create value networks.

    Args:
        key: JAX random key
        d: Spatial dimension
        architecture: 'mlp' or 'resnet'
        **kwargs: Architecture-specific parameters
            - mlp: hidden_dims (list), activation (str)
            - resnet: m (int, hidden_width), depth (int)

    Returns:
        Initialized network

    Examples:
        >>> net = create_value_network(key, d=100, architecture="resnet", m=64, depth=4)
        >>> net = create_value_network(key, d=100, architecture="mlp", hidden_dims=[256, 256])
    """
    if architecture == "mlp":
        hidden_dims = kwargs.get("hidden_dims", [256, 256, 256, 256])
        activation = kwargs.get("activation", "tanh")
        return ValueNetwork(key, d, hidden_dims, activation)

    elif architecture == "resnet":
        # Map 'hidden_width' to 'm' for compatibility with config files
        m = kwargs.get("m", kwargs.get("hidden_width", 64))
        depth = kwargs.get("depth", 4)
        return PhiResNet(key, d, m, depth)

    else:
        raise ValueError(f"Unknown architecture: {architecture}. Use 'mlp' or 'resnet'.")
