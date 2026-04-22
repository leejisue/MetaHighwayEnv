"""Pandas-free kinematic observation for JAX highway environment.

Ported from ``highway_env.envs.common.observation.KinematicObservation``.
Returns a flat observation vector: (N_OBS * 5,) with features
[presence, x, y, vx, vy] for each observed vehicle (ego-relative, normalized).
"""

from __future__ import annotations

import jax.numpy as jnp

from .state import EnvState, EnvParams, N_MAX, N_OBS


# Normalisation scales for [presence, x, y, vx, vy]
# presence is already 0/1; x, y are relative positions; vx, vy are velocities
_SCALES = jnp.array([1.0, 100.0, 100.0, 40.0, 40.0])


def observe(state: EnvState, params: EnvParams) -> jnp.ndarray:
    """Compute kinematic observation (pandas-free).

    Matches ``KinematicObservation.observe()`` with features
    ``["presence", "x", "y", "vx", "vy"]``, relative to ego,
    sorted by distance, normalized.

    Args:
        state: current environment state.
        params: environment parameters.

    Returns:
        Flat observation vector of shape ``(N_OBS * 5,)``.
    """
    vehicles = state.vehicles

    # Ego state
    ego_x = vehicles.x[0]
    ego_y = vehicles.y[0]
    ego_vx = vehicles.vx[0]
    ego_vy = vehicles.vy[0]

    # Features: [presence, x_rel, y_rel, vx_rel, vy_rel]
    presence = vehicles.active.astype(jnp.float32)
    x_rel = vehicles.x - ego_x
    y_rel = vehicles.y - ego_y
    vx_rel = vehicles.vx - ego_vx
    vy_rel = vehicles.vy - ego_vy

    features = jnp.stack([presence, x_rel, y_rel, vx_rel, vy_rel], axis=-1)  # (N_MAX, 5)

    # Sort by distance (ego first, then closest)
    dist_sq = x_rel ** 2 + y_rel ** 2
    # Set ego distance to -inf so it's always first
    dist_sq = dist_sq.at[0].set(-1.0)
    # Set inactive vehicles to very large distance
    dist_sq = jnp.where(vehicles.active, dist_sq, 1e10)

    # FIX P1-4: use argpartition (O(N)) instead of full argsort (O(N log N)).
    # We only need N_OBS=5 nearest out of N_MAX=50; partition then sort only
    # the 5 selected elements to restore distance order.
    part_indices = jnp.argpartition(dist_sq, N_OBS)[:N_OBS]  # unordered top-5
    selected_dist = dist_sq[part_indices]
    order = jnp.argsort(selected_dist)                        # sort only 5 elements
    indices = part_indices[order]

    # Gather observed features
    obs_features = features[indices]  # (N_OBS, 5)

    # For ego (first row), set presence=1, relative features are 0
    ego_row = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
    obs_features = obs_features.at[0].set(ego_row)

    # Normalize
    obs_normalized = obs_features / _SCALES[None, :]

    # Ego absolute speed instead of 0 for vx
    ego_speed_norm = ego_vx / _SCALES[3]
    obs_normalized = obs_normalized.at[0, 3].set(ego_speed_norm)

    return obs_normalized.flatten()  # (N_OBS * 5,)


def obs_shape(params: EnvParams) -> tuple[int]:
    """Return the observation shape."""
    return (N_OBS * 5,)
