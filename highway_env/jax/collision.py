"""AABB collision detection for JAX highway environment.

Ported from ``highway_env.vehicle.objects.RoadObject._is_colliding()``.
Uses Axis-Aligned Bounding Box (AABB) instead of SAT polygon
intersection for JIT compatibility.
"""

from __future__ import annotations

import jax.numpy as jnp

from .state import VehicleState, N_MAX


def check_ego_collision(vehicles: VehicleState) -> jnp.ndarray:
    """Check if the ego vehicle (index 0) collides with any other active vehicle.

    Uses AABB overlap: two rectangles overlap when their center
    distances in both axes are less than the sum of half-sizes.
    Since vehicles are roughly axis-aligned on a highway, AABB is
    a good enough approximation for this setting.

    Returns:
        Scalar bool — True if ego collides with any active vehicle.
    """
    ego_x = vehicles.x[0]
    ego_y = vehicles.y[0]
    ego_half_l = vehicles.length[0] / 2.0
    ego_half_w = vehicles.width[0] / 2.0

    other_half_l = vehicles.length / 2.0
    other_half_w = vehicles.width / 2.0

    dx = jnp.abs(vehicles.x - ego_x)
    dy = jnp.abs(vehicles.y - ego_y)

    overlap_x = dx < (ego_half_l + other_half_l)
    overlap_y = dy < (ego_half_w + other_half_w)
    overlap = overlap_x & overlap_y & vehicles.active

    # Exclude self (index 0)
    overlap = overlap.at[0].set(False)

    return jnp.any(overlap)


def check_all_collisions(vehicles: VehicleState) -> jnp.ndarray:
    """Check pairwise collisions between all active vehicles.

    Returns:
        (N_MAX,) bool array — True for vehicles involved in any collision.
    """
    # Pairwise distances
    dx = jnp.abs(vehicles.x[:, None] - vehicles.x[None, :])  # (N, N)
    dy = jnp.abs(vehicles.y[:, None] - vehicles.y[None, :])  # (N, N)

    half_l = vehicles.length / 2.0
    half_w = vehicles.width / 2.0

    overlap_x = dx < (half_l[:, None] + half_l[None, :])
    overlap_y = dy < (half_w[:, None] + half_w[None, :])

    both_active = vehicles.active[:, None] & vehicles.active[None, :]
    not_self = ~jnp.eye(N_MAX, dtype=bool)

    colliding = overlap_x & overlap_y & both_active & not_self

    # A vehicle is crashed if it collides with any other
    return jnp.any(colliding, axis=1)
