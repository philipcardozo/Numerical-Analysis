"""
Utilities for Diffusion Posterior Sampling (DPS) demonstration.

This module provides:
- GaussianMixtureDistribution: GMM with analytical score function
- NoiseSchedule: VP diffusion noise schedule
"""

import jax
import jax.numpy as jnp
from typing import Union, Tuple, Optional, List


class GaussianMixtureDistribution:
    """
    Gaussian mixture distribution with analytical score function.

    For p(x) = sum_i w_i N(x; mu_i, sigma_i^2), the score is:
        s(x) = grad log p(x) = (1/p(x)) sum_i w_i N(x; mu_i, sigma_i^2) * (-(x - mu_i)/sigma_i^2)

    Attributes:
        means: Array of component means
        stds: Array of component standard deviations
        weights: Array of mixture weights (normalized)
        n_components: Number of mixture components
    """

    def __init__(
        self,
        means: Union[List[float], jnp.ndarray],
        stds: Union[List[float], jnp.ndarray],
        weights: Optional[Union[List[float], jnp.ndarray]] = None
    ):
        """
        Initialize Gaussian mixture distribution.

        Args:
            means: Array/list of component means
            stds: Array/list of component standard deviations
            weights: Array/list of mixture weights (default: uniform)
        """
        self.means = jnp.asarray(means)
        self.stds = jnp.asarray(stds)
        self.n_components = len(self.means)

        if weights is None:
            self.weights = jnp.ones(self.n_components) / self.n_components
        else:
            weights = jnp.asarray(weights)
            self.weights = weights / jnp.sum(weights)

    def _component_pdf(self, x: jnp.ndarray, i: int) -> jnp.ndarray:
        """Evaluate PDF of component i."""
        return jnp.exp(-0.5 * ((x - self.means[i]) / self.stds[i]) ** 2) / (
            self.stds[i] * jnp.sqrt(2 * jnp.pi)
        )

    def pdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate probability density function.

        Args:
            x: (n,) array of evaluation points

        Returns:
            (n,) array of density values
        """
        result = jnp.zeros_like(x)
        for i in range(self.n_components):
            result = result + self.weights[i] * self._component_pdf(x, i)
        return result

    def score(self, x: jnp.ndarray, eps: float = 1e-10) -> jnp.ndarray:
        """
        Evaluate analytical score function: s(x) = grad log p(x).

        The score is computed as:
            s(x) = sum_i w_i N(x; mu_i, sigma_i^2) * (-(x - mu_i)/sigma_i^2) / p(x)

        Args:
            x: (n,) array of evaluation points
            eps: Small constant to avoid division by zero

        Returns:
            (n,) array of score values
        """
        p_x = self.pdf(x) + eps

        # Weighted sum of component contributions
        numerator = jnp.zeros_like(x)
        for i in range(self.n_components):
            component_pdf = self._component_pdf(x, i)
            component_score = -(x - self.means[i]) / (self.stds[i] ** 2)
            numerator = numerator + self.weights[i] * component_pdf * component_score

        return numerator / p_x

    def sample(self, key: jax.Array, n: int) -> jnp.ndarray:
        """
        Generate samples from the mixture distribution.

        Args:
            key: JAX random key
            n: Number of samples

        Returns:
            (n,) array of samples
        """
        key1, key2 = jax.random.split(key)

        # Sample component indices
        indices = jax.random.choice(key1, self.n_components, shape=(n,), p=self.weights)

        # Sample from each component
        z = jax.random.normal(key2, (n,))

        # Select mean and std based on component
        selected_means = self.means[indices]
        selected_stds = self.stds[indices]

        return z * selected_stds + selected_means


class NoiseSchedule:
    """
    Variance-preserving (VP) noise schedule for diffusion models.

    The VP-SDE has the form:
        dx = -0.5 beta(t) x dt + sqrt(beta(t)) dW

    The noise schedule beta(t) controls the diffusion rate.

    For VP diffusion starting from data x_0:
        x_t = alpha(t) x_0 + sigma(t) eps,  where eps ~ N(0, I)
        alpha(t)^2 = exp(-integral_0^t beta(s) ds)
        sigma(t)^2 = 1 - alpha(t)^2

    Attributes:
        beta_min: Minimum beta value
        beta_max: Maximum beta value
        schedule_type: 'linear' or 'cosine'
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        schedule_type: str = 'linear'
    ):
        """
        Initialize noise schedule.

        Args:
            beta_min: Minimum beta value (at t=0)
            beta_max: Maximum beta value (at t=1)
            schedule_type: 'linear' or 'cosine'
        """
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.schedule_type = schedule_type

        if schedule_type not in ['linear', 'cosine']:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    def beta(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate beta(t) at given times.

        Args:
            t: (n,) array of time values in [0, 1]

        Returns:
            (n,) array of beta values
        """
        if self.schedule_type == 'linear':
            return self.beta_min + t * (self.beta_max - self.beta_min)
        elif self.schedule_type == 'cosine':
            # Cosine schedule
            s = 0.008  # offset to prevent singularity
            return jnp.clip(
                jnp.pi * jnp.tan((t + s) / (1 + s) * jnp.pi / 2) / (1 + s),
                self.beta_min,
                self.beta_max
            )
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def integral_beta(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate integral_0^t beta(s) ds.

        Args:
            t: (n,) array of time values in [0, 1]

        Returns:
            (n,) array of integral values
        """
        if self.schedule_type == 'linear':
            # integral_0^t (beta_min + s(beta_max - beta_min)) ds
            return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        elif self.schedule_type == 'cosine':
            s = 0.008
            f_t = jnp.cos((t + s) / (1 + s) * jnp.pi / 2) ** 2
            f_0 = jnp.cos(s / (1 + s) * jnp.pi / 2) ** 2
            return -jnp.log(f_t / f_0 + 1e-10)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def get_coefficients(self, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get signal and noise coefficients alpha(t) and sigma(t).

        For VP diffusion: x_t = alpha(t) x_0 + sigma(t) eps

        Args:
            t: (n,) array of time values in [0, 1]

        Returns:
            alpha: (n,) signal coefficient alpha(t) = exp(-0.5 integral beta(s) ds)
            sigma: (n,) noise coefficient sigma(t) = sqrt(1 - alpha(t)^2)
        """
        int_beta = self.integral_beta(t)
        alpha = jnp.exp(-0.5 * int_beta)
        sigma = jnp.sqrt(1 - alpha ** 2 + 1e-10)
        return alpha, sigma
