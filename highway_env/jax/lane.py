"""Straight-lane utilities for JAX highway environment.

Ported from ``highway_env.road.lane.StraightLane``.
For highway-env the road is always a straight horizontal road
with ``num_lanes`` parallel lanes.  This module provides purely
functional helpers for lane geometry queries.
"""

from __future__ import annotations

import jax.numpy as jnp

from .state import EnvParams


def lane_center_y(lane_id: jnp.ndarray, params: EnvParams) -> jnp.ndarray:
    """Y coordinate of lane center.

    Lane 0 is the leftmost lane; center_y = lane_id * lane_width + lane_width/2.
    This matches highway-env where lanes go from y=0 outward.
    """
    return lane_id * params.lane_width + params.lane_width / 2.0


def get_lane_id(y: jnp.ndarray, params: EnvParams) -> jnp.ndarray:
    """Compute the current lane index from y position.

    Clamps to [0, num_lanes - 1].
    """
    lane = jnp.floor(y / params.lane_width).astype(jnp.int32)
    return jnp.clip(lane, 0, params.num_lanes - 1)


def on_road(y: jnp.ndarray, params: EnvParams) -> jnp.ndarray:
    """Check if y position is within road boundaries."""
    road_top = params.num_lanes * params.lane_width
    return (y >= -params.vehicle_width / 2.0) & (y <= road_top + params.vehicle_width / 2.0)


def steering_to_lane(
    y: jnp.ndarray,
    heading: jnp.ndarray,
    target_lane_id: jnp.ndarray,
    speed: jnp.ndarray,
    params: EnvParams,
    kp_lateral: float = 0.3,
    kp_heading: float = 1.0,
) -> jnp.ndarray:
    """Simple proportional steering controller toward target lane center.

    Used by IDM traffic vehicles to follow their lane / execute lane changes.

    Args:
        y: current y position.
        heading: current heading angle.
        target_lane_id: target lane index.
        speed: current speed (for scaling).
        params: environment parameters.
        kp_lateral: lateral error gain.
        kp_heading: heading error gain.

    Returns:
        Steering command (clamped to [-pi/4, pi/4]).
    """
    target_y = lane_center_y(target_lane_id, params)
    lateral_error = target_y - y
    # Desired heading to correct lateral error
    desired_heading = jnp.arctan2(lateral_error * kp_lateral, jnp.maximum(speed, 1.0))
    heading_error = desired_heading - heading
    steering = kp_heading * heading_error
    return jnp.clip(steering, -jnp.pi / 4.0, jnp.pi / 4.0)
