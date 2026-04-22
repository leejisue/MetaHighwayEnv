"""Kinematic bicycle model for JAX highway environment.

Ported from ``highway_env.vehicle.kinematics.Vehicle.step()``.
All functions are pure and JIT-compatible.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .state import VehicleState, EnvParams, N_MAX


def vehicle_step_single(
    x: float,
    y: float,
    speed: float,
    heading: float,
    accel: float,
    steering: float,
    dt: float,
    length: float = 5.0,
    max_speed: float = 40.0,
) -> tuple[float, float, float, float]:
    """Kinematic bicycle model for a single vehicle.

    Matches ``highway_env.vehicle.kinematics.Vehicle.step()``:
        beta = arctan(0.5 * tan(steering))
        v_vec = speed * [cos(heading + beta), sin(heading + beta)]
        position += v_vec * dt
        heading  += speed * sin(beta) / (length/2) * dt
        speed    += accel * dt

    Returns:
        (new_x, new_y, new_speed, new_heading)
    """
    beta = jnp.arctan(0.5 * jnp.tan(steering))
    vx = speed * jnp.cos(heading + beta)
    vy = speed * jnp.sin(heading + beta)
    new_x = x + vx * dt
    new_y = y + vy * dt
    new_heading = heading + speed * jnp.sin(beta) / (length / 2.0) * dt
    new_speed = jnp.clip(speed + accel * dt, 0.0, max_speed)
    return new_x, new_y, new_speed, new_heading


def update_ego(
    vehicles: VehicleState,
    accel: float,
    steering: float,
    params: EnvParams,
) -> VehicleState:
    """Apply ego action (index 0) via bicycle model."""
    dt = params.dt
    x, y, speed, heading = vehicle_step_single(
        vehicles.x[0],
        vehicles.y[0],
        vehicles.vx[0],
        vehicles.heading[0],
        accel,
        steering,
        dt,
        length=vehicles.length[0],
        max_speed=params.max_speed,
    )
    return vehicles.replace(
        x=vehicles.x.at[0].set(x),
        y=vehicles.y.at[0].set(y),
        vx=vehicles.vx.at[0].set(speed),
        heading=vehicles.heading.at[0].set(heading),
    )


# P2-2: Hoist vmap to module level so it is not recreated on each Python call.
# JAX caches the result, but re-creating the closure every call is a bad pattern.
_vmapped_vehicle_step = jax.vmap(  # FIX P2-2
    vehicle_step_single, in_axes=(0, 0, 0, 0, 0, 0, None, 0, None)
)


def update_vehicles_kinematics(
    vehicles: VehicleState,
    accels: jnp.ndarray,
    steerings: jnp.ndarray,
    dt: float,
    max_speed: float = 40.0,
) -> VehicleState:
    """Update all vehicles (vectorized bicycle model).

    Args:
        vehicles: current vehicle states.
        accels: (N_MAX,) acceleration per vehicle.
        steerings: (N_MAX,) steering per vehicle.
        dt: simulation timestep.
        max_speed: speed clamp.

    Returns:
        Updated VehicleState.
    """
    new_x, new_y, new_speed, new_heading = _vmapped_vehicle_step(
        vehicles.x, vehicles.y, vehicles.vx, vehicles.heading,
        accels, steerings, dt, vehicles.length, max_speed,
    )
    # Only update active vehicles; inactive keep old values
    active = vehicles.active
    return vehicles.replace(
        x=jnp.where(active, new_x, vehicles.x),
        y=jnp.where(active, new_y, vehicles.y),
        vx=jnp.where(active, new_speed, vehicles.vx),
        heading=jnp.where(active, new_heading, vehicles.heading),
    )
