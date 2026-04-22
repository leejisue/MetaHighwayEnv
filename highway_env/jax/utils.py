"""Helper functions for JAX highway environment."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def not_zero(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Return x if |x| > eps, else eps with the sign of x.

    Note: ``jnp.sign(0) == 0``, so a naive ``sign(x) * eps`` would still
    return 0 when ``x == 0``.  We fall back to a positive ``eps`` in
    that case so callers (e.g. ``lmap`` divisor) never see a zero.
    """
    safe_sign = jnp.where(x >= 0, 1.0, -1.0)
    return jnp.where(jnp.abs(x) > eps, x, safe_sign * eps)


def lmap(v: jnp.ndarray, x_range: tuple, y_range: tuple) -> jnp.ndarray:
    """Linear map v from x_range to y_range."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    return y_min + (v - x_min) * (y_max - y_min) / not_zero(x_max - x_min)


def wrap_to_pi(angle: jnp.ndarray) -> jnp.ndarray:
    """Wrap angle to [-pi, pi]."""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
