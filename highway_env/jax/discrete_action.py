"""Discrete meta-action support for JAX highway environments.

Implements Highway-Env's ``DiscreteMetaAction`` for the JAX envs:

  **Full (lateral + longitudinal) — highway, merge, roundabout:**

  ===  ============  ==========================================
  Idx  Label         Effect
  ===  ============  ==========================================
  0    LANE_LEFT     ``target_lane -= 1`` (clipped to 0)
  1    IDLE          no change
  2    LANE_RIGHT    ``target_lane += 1`` (clipped to max lane)
  3    FASTER        ``speed_index += 1``
  4    SLOWER        ``speed_index -= 1``
  ===  ============  ==========================================

  **Longitudinal only — intersection:**

  ===  ============  ==========================================
  Idx  Label         Effect
  ===  ============  ==========================================
  0    SLOWER        ``speed_index -= 1``
  1    IDLE          no change
  2    FASTER        ``speed_index += 1``
  ===  ============  ==========================================

After the meta-action updates the ego's ``target_lane_id`` and
``target_speed``, a PD controller (matching Highway-Env's
``ControlledVehicle``) computes continuous acceleration and steering
commands for the kinematic bicycle model.

Controller gains from ``highway_env.vehicle.controller.ControlledVehicle``:

  =============  ======  ======================================
  Parameter      Value   Source
  =============  ======  ======================================
  KP_A           1.667   1 / TAU_ACC  (TAU_ACC = 0.6 s)
  KP_HEADING     5.0     1 / TAU_HEADING  (TAU_HEADING = 0.2 s)
  KP_LATERAL     1.667   1 / TAU_LATERAL  (TAU_LATERAL = 0.6 s)
  MAX_STEERING   π/3     60 degrees
  =============  ======  ======================================

Per-environment target speed arrays (from Highway-Env default configs):

  ==============  ==================  ========================
  Environment     target_speeds       Source
  ==============  ==================  ========================
  highway         [20, 25, 30]        MDPVehicle default
  merge           [20, 25, 30]        MDPVehicle default
  roundabout      [0, 8, 16]          roundabout_env.py L66
  intersection    [0, 4.5, 9]         intersection_env.py L82
  ==============  ==================  ========================
"""

from __future__ import annotations

import jax.numpy as jnp

from .state import VehicleState
from .lane import lane_center_y

# ======================================================================
#  Action constants
# ======================================================================

# Full action set (lateral + longitudinal) — highway, merge, roundabout
LANE_LEFT = 0
IDLE = 1
LANE_RIGHT = 2
FASTER = 3
SLOWER = 4
N_DISCRETE_ACTIONS = 5

# Longitudinal-only action set — intersection
LONGI_SLOWER = 0
LONGI_IDLE = 1
LONGI_FASTER = 2
N_DISCRETE_ACTIONS_LONGI = 3

# ======================================================================
#  Per-environment target speed arrays
# ======================================================================

HIGHWAY_TARGET_SPEEDS = jnp.array([20.0, 25.0, 30.0])
MERGE_TARGET_SPEEDS = jnp.array([20.0, 25.0, 30.0])
ROUNDABOUT_TARGET_SPEEDS = jnp.array([0.0, 8.0, 16.0])
INTERSECTION_TARGET_SPEEDS = jnp.array([0.0, 4.5, 9.0])

# ======================================================================
#  Controller gains  (Highway-Env ControlledVehicle)
# ======================================================================

TAU_ACC = 0.6
TAU_HEADING = 0.2
TAU_LATERAL = 0.6

KP_A = 1.0 / TAU_ACC            # ≈ 1.667
KP_HEADING = 1.0 / TAU_HEADING  # = 5.0
KP_LATERAL = 1.0 / TAU_LATERAL  # ≈ 1.667
MAX_STEERING_ANGLE = jnp.pi / 3.0  # 60 degrees


# ======================================================================
#  Meta-action application
# ======================================================================

def apply_meta_action(
    action: jnp.ndarray,
    vehicles: VehicleState,
    num_lanes: int,
    target_speeds: jnp.ndarray,
) -> VehicleState:
    """Apply a discrete meta-action to the ego vehicle (index 0).

    Updates ``target_lane_id[0]`` and ``target_speed[0]`` based on
    the integer action.

    Args:
        action: scalar integer in ``{0, 1, 2, 3, 4}``.
        vehicles: current vehicle states.
        num_lanes: number of lanes (for lane clipping).
        target_speeds: array of allowed target speeds, shape ``(K,)``.

    Returns:
        Updated VehicleState.
    """
    ego_target_lane = vehicles.target_lane_id[0]
    ego_target_speed = vehicles.target_speed[0]

    # --- Lane changes ---
    new_target_lane = jnp.where(
        action == LANE_LEFT,
        jnp.maximum(ego_target_lane - 1, 0),
        jnp.where(
            action == LANE_RIGHT,
            jnp.minimum(ego_target_lane + 1, num_lanes - 1),
            ego_target_lane,
        ),
    )

    # --- Speed changes ---
    n_speeds = target_speeds.shape[0]
    speed_diffs = jnp.abs(target_speeds - ego_target_speed)
    current_idx = jnp.argmin(speed_diffs)

    new_idx = jnp.where(
        action == FASTER,
        jnp.minimum(current_idx + 1, n_speeds - 1),
        jnp.where(
            action == SLOWER,
            jnp.maximum(current_idx - 1, 0),
            current_idx,
        ),
    )
    new_target_speed = target_speeds[new_idx]

    return vehicles.replace(
        target_lane_id=vehicles.target_lane_id.at[0].set(new_target_lane),
        target_speed=vehicles.target_speed.at[0].set(new_target_speed),
    )


def apply_meta_action_longi(
    action: jnp.ndarray,
    vehicles: VehicleState,
    target_speeds: jnp.ndarray,
) -> VehicleState:
    """Apply a longitudinal-only discrete action (intersection).

    Action mapping: ``{0: SLOWER, 1: IDLE, 2: FASTER}``.
    Only ``target_speed[0]`` is modified; lane is unchanged.

    Args:
        action: scalar integer in ``{0, 1, 2}``.
        vehicles: current vehicle states.
        target_speeds: array of allowed target speeds.

    Returns:
        Updated VehicleState.
    """
    ego_target_speed = vehicles.target_speed[0]

    n_speeds = target_speeds.shape[0]
    speed_diffs = jnp.abs(target_speeds - ego_target_speed)
    current_idx = jnp.argmin(speed_diffs)

    new_idx = jnp.where(
        action == LONGI_FASTER,
        jnp.minimum(current_idx + 1, n_speeds - 1),
        jnp.where(
            action == LONGI_SLOWER,
            jnp.maximum(current_idx - 1, 0),
            current_idx,
        ),
    )
    new_target_speed = target_speeds[new_idx]

    return vehicles.replace(
        target_speed=vehicles.target_speed.at[0].set(new_target_speed),
    )


# ======================================================================
#  PD controller
# ======================================================================

def compute_ego_control(
    vehicles: VehicleState,
    params,
) -> tuple:
    """Compute ego acceleration and steering via PD controller.

    Matches Highway-Env's ``ControlledVehicle.speed_control()`` and
    ``steering_control()`` for straight-road environments (highway,
    merge, roundabout).

    The controller is a three-stage cascade:
      1. **Lateral** — proportional drive toward the target lane center.
      2. **Heading** — proportional heading correction toward the
         desired lateral velocity direction.
      3. **Steering** — convert heading rate to bicycle-model steering
         angle.

    Args:
        vehicles: current vehicle states.
        params: environment parameters (must have ``lane_width``).

    Returns:
        ``(acceleration, steering)`` — scalar commands for the ego.
    """
    ego_speed = vehicles.vx[0]
    ego_y = vehicles.y[0]
    ego_heading = vehicles.heading[0]
    ego_target_speed = vehicles.target_speed[0]
    vehicle_length = vehicles.length[0]

    # Avoid division by zero at very low speeds
    safe_speed = jnp.maximum(jnp.abs(ego_speed), 1.0)

    # ------ Speed control ------
    acceleration = KP_A * (ego_target_speed - ego_speed)
    acceleration = jnp.clip(
        acceleration,
        -params.idm_comfort_decel,
        params.idm_comfort_accel,
    )

    # ------ Steering control (cascaded PD) ------
    target_y = lane_center_y(vehicles.target_lane_id[0], params)

    # Stage 1: lateral error → desired lateral speed
    lateral_error = ego_y - target_y  # positive when above target
    lateral_speed_cmd = -KP_LATERAL * lateral_error

    # Stage 2: desired lateral speed → heading reference
    heading_cmd = jnp.arcsin(
        jnp.clip(lateral_speed_cmd / safe_speed, -1.0, 1.0)
    )
    heading_cmd = jnp.clip(heading_cmd, -jnp.pi / 4.0, jnp.pi / 4.0)

    # For straight road the lane heading is 0
    heading_ref = heading_cmd

    # Heading error (wrap to [-π, π])
    heading_error = heading_ref - ego_heading
    heading_error = jnp.arctan2(
        jnp.sin(heading_error), jnp.cos(heading_error)
    )
    heading_rate_cmd = KP_HEADING * heading_error

    # Stage 3: heading rate → steering angle
    #   From kinematic bicycle: heading_rate ≈ speed·sin(β)/(L/2)
    #   where β = arctan(0.5·tan(δ)).  For small δ: β ≈ δ/2
    #   → heading_rate ≈ speed·δ/(2·L/2) = speed·δ/L
    #   → δ ≈ heading_rate·L/speed
    #   Highway-Env uses slip_angle = arcsin(2L/speed · heading_rate).
    steering = jnp.arcsin(
        jnp.clip(
            2.0 * vehicle_length / safe_speed * heading_rate_cmd,
            -1.0, 1.0,
        )
    )
    steering = jnp.clip(steering, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

    return acceleration, steering


def compute_ego_control_intersection(
    vehicles: VehicleState,
    params,
) -> tuple:
    """Compute ego acceleration for intersection (speed control only).

    Intersection vehicles drive along a fixed heading; only speed
    changes.  No steering control — the vehicle maintains its approach
    heading set at reset.

    Args:
        vehicles: current vehicle states.
        params: intersection environment parameters.

    Returns:
        ``(acceleration, steering)`` where steering is always 0.
    """
    ego_speed = jnp.sqrt(vehicles.vx[0] ** 2 + vehicles.vy[0] ** 2)
    ego_target_speed = vehicles.target_speed[0]

    acceleration = KP_A * (ego_target_speed - ego_speed)
    acceleration = jnp.clip(
        acceleration,
        -params.idm_comfort_decel,
        params.idm_comfort_accel,
    )

    # Apply acceleration along current heading direction
    # (the calling env will decompose into vx/vy components)
    steering = jnp.array(0.0)

    return acceleration, steering
