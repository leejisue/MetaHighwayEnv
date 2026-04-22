"""IDM acceleration and simplified MOBIL lane-change for JAX.

Ported from ``highway_env.vehicle.behavior.IDMVehicle``.
All functions are pure and JIT-compatible.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .state import VehicleState, EnvParams, N_MAX, N_LANES_MAX
from .lane import get_lane_id, steering_to_lane

# P2-1: Hoist jnp.eye out of find_leader_batched — XLA would fold it anyway
# but hoisting makes the tracing graph cleaner.
_SELF_MASK = jnp.eye(N_MAX, dtype=bool)  # FIX P2-1


def _not_zero(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    return jnp.where(jnp.abs(x) > eps, x, eps)


# ------------------------------------------------------------------
#  IDM (Intelligent Driver Model)
# ------------------------------------------------------------------

def desired_gap(
    ego_speed: jnp.ndarray,
    front_speed: jnp.ndarray,
    params: EnvParams,
) -> jnp.ndarray:
    """Compute desired following gap (s*).

    s* = s0 + v*T + v*(v - v_front) / (2*sqrt(a*b))

    Matches ``IDMVehicle.desired_gap()``.
    """
    d0 = params.idm_min_gap + params.vehicle_length  # DISTANCE_WANTED = min_gap + LENGTH
    tau = params.idm_time_gap
    ab = params.idm_comfort_accel * params.idm_comfort_decel
    dv = ego_speed - front_speed
    s_star = d0 + ego_speed * tau + ego_speed * dv / (2.0 * jnp.sqrt(jnp.maximum(ab, 1e-6)))
    return jnp.maximum(s_star, d0)


def idm_acceleration(
    ego_speed: jnp.ndarray,
    front_speed: jnp.ndarray,
    gap: jnp.ndarray,
    has_front: jnp.ndarray,
    target_speed: jnp.ndarray,
    params: EnvParams,
) -> jnp.ndarray:
    """Compute IDM longitudinal acceleration.

    Matches ``IDMVehicle.acceleration()``:
        a = a_max * [1 - (v/v0)^delta - (s*/s)^2]

    Args:
        ego_speed: ego vehicle speed.
        front_speed: front vehicle speed (ignored if has_front=False).
        gap: distance to front vehicle.
        has_front: bool, whether there is a front vehicle.
        target_speed: desired speed of this vehicle.
        params: environment parameters.

    Returns:
        Acceleration command.
    """
    # Free-flow term
    v_ratio = jnp.maximum(ego_speed, 0.0) / _not_zero(target_speed)
    accel = params.idm_comfort_accel * (1.0 - jnp.power(v_ratio, params.idm_delta))

    # Front vehicle interaction term
    s_star = desired_gap(ego_speed, front_speed, params)
    gap_clamped = jnp.maximum(gap, 0.1)
    interaction = params.idm_comfort_accel * jnp.power(s_star / gap_clamped, 2)
    accel = accel - jnp.where(has_front, interaction, 0.0)

    return jnp.clip(accel, -params.idm_max_accel, params.idm_max_accel)


# ------------------------------------------------------------------
#  Leader finding
# ------------------------------------------------------------------

def find_leader(
    vehicles: VehicleState,
    vehicle_idx: int,
    lane_id: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Find the nearest vehicle ahead in the same lane.

    Returns:
        (front_speed, gap, has_front)
    """
    ego_x = vehicles.x[vehicle_idx]
    # Distances to all vehicles in the same lane, ahead
    same_lane = (vehicles.lane_id == lane_id) & vehicles.active
    ahead = vehicles.x > ego_x
    valid = same_lane & ahead
    # Set invalid distances to very large
    dx = jnp.where(valid, vehicles.x - ego_x, 1e10)
    # Subtract vehicle lengths for bumper-to-bumper distance
    dx = dx - vehicles.length
    leader_idx = jnp.argmin(dx)
    gap = dx[leader_idx]
    has_front = jnp.any(valid)
    front_speed = vehicles.vx[leader_idx]
    return front_speed, jnp.maximum(gap, 0.1), has_front


def find_leader_batched(
    vehicles: VehicleState,
    query_x: jnp.ndarray,
    query_lane: jnp.ndarray,
    query_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Find nearest leader for each vehicle (vectorized).

    Args:
        vehicles: all vehicle states.
        query_x: (N_MAX,) x positions of querying vehicles.
        query_lane: (N_MAX,) lane IDs to check.
        query_mask: (N_MAX,) mask of which queries are valid.

    Returns:
        (front_speeds, gaps, has_fronts) each (N_MAX,).
    """
    # For each vehicle i, find leader in query_lane[i]
    # dx[i, j] = vehicles.x[j] - query_x[i]  if same lane and ahead
    dx = vehicles.x[None, :] - query_x[:, None]  # (N, N)
    same_lane = vehicles.lane_id[None, :] == query_lane[:, None]  # (N, N)
    ahead = dx > 0
    other_active = vehicles.active[None, :]  # (1, N)
    valid = same_lane & ahead & other_active  # (N, N)
    # Exclude self: set diagonal to invalid (FIX P2-1: use module-level constant)
    valid = valid & ~_SELF_MASK

    # Bumper-to-bumper gap
    bumper_gap = dx - vehicles.length[None, :]  # (N, N)
    bumper_gap = jnp.where(valid, bumper_gap, 1e10)

    leader_indices = jnp.argmin(bumper_gap, axis=1)  # (N,)
    gaps = jnp.take_along_axis(bumper_gap, leader_indices[:, None], axis=1).squeeze(-1)
    front_speeds = vehicles.vx[leader_indices]
    has_fronts = jnp.any(valid, axis=1)

    return front_speeds, jnp.maximum(gaps, 0.1), has_fronts


# ------------------------------------------------------------------
#  Simplified MOBIL lane change
# ------------------------------------------------------------------

# ------------------------------------------------------------------
#  Full traffic step
# ------------------------------------------------------------------

def _simplified_mobil_with_idm(
    vehicles: VehicleState,
    params: EnvParams,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """MOBIL lane-change + IDM acceleration in one fused pass.

    FIX P0-2: returns both the new target lanes AND the current-lane IDM
    accelerations, so traffic_step can reuse the leader results without
    a 4th call to find_leader_batched.
    """
    current_lanes = vehicles.lane_id
    current_speeds = vehicles.vx
    target_speeds = vehicles.target_speed

    # FIX P0-2: fuse 3 find_leader_batched calls (current/left/right) into one
    # vectorized pass over lane offsets [-1, 0, +1] using jax.vmap.
    # This replaces 3 separate O(N²) matrix allocations with a single shared pass.
    # FIX P1-1 FINAL: use N_LANES_MAX (Python int) instead of params.num_lanes-1
    # so lane_max is a compile-time constant — prevents JIT recompile when
    # num_lanes varies across meta-RL tasks.
    lane_max = jnp.array(N_LANES_MAX - 1, dtype=jnp.int32)

    def _find_for_offset(offset):
        lane = jnp.clip(current_lanes + offset, 0, lane_max)
        return find_leader_batched(vehicles, vehicles.x, lane, vehicles.active)

    # offsets: left=-1, current=0, right=+1
    # vmap returns a tuple (all_front_speeds, all_gaps, all_has_fronts)
    # each element has shape (3, N_MAX): axis-0 = offset index
    all_front_speeds, all_gaps, all_has_fronts = jax.vmap(_find_for_offset)(
        jnp.array([-1, 0, 1])
    )
    # left=offset[0], current=offset[1], right=offset[2]
    front_speeds_left, gaps_left, has_fronts_left = (
        all_front_speeds[0], all_gaps[0], all_has_fronts[0],
    )
    front_speeds_cur, gaps_cur, has_fronts_cur = (
        all_front_speeds[1], all_gaps[1], all_has_fronts[1],
    )
    front_speeds_right, gaps_right, has_fronts_right = (
        all_front_speeds[2], all_gaps[2], all_has_fronts[2],
    )

    left_lane = jnp.clip(current_lanes - 1, 0, lane_max)
    right_lane = jnp.clip(current_lanes + 1, 0, lane_max)

    accel_current = idm_acceleration(
        current_speeds, front_speeds_cur, gaps_cur,
        has_fronts_cur, target_speeds, params,
    )
    accel_left = idm_acceleration(
        current_speeds, front_speeds_left, gaps_left,
        has_fronts_left, target_speeds, params,
    )
    accel_right = idm_acceleration(
        current_speeds, front_speeds_right, gaps_right,
        has_fronts_right, target_speeds, params,
    )

    # Lane change decision
    threshold = params.mobil_lane_change_min_acc_gain
    gain_left = accel_left - accel_current
    gain_right = accel_right - accel_current

    can_go_left = current_lanes > 0
    # FIX P1-1 FINAL: use N_LANES_MAX (Python int) for traced comparison — avoids
    # JIT recompile when params.num_lanes differs across meta-RL task batches.
    can_go_right = current_lanes < (N_LANES_MAX - 1)

    want_left = can_go_left & (gain_left > threshold) & (gain_left >= gain_right)
    want_right = can_go_right & (gain_right > threshold) & (gain_right > gain_left)

    timer_fire = jax.random.uniform(key, (N_MAX,)) < (params.dt / params.mobil_lane_change_delay)

    already_changing = vehicles.target_lane_id != current_lanes
    can_change = timer_fire & ~already_changing & (jnp.abs(current_speeds) >= 1.0)

    new_target = jnp.where(want_left & can_change, left_lane, vehicles.target_lane_id)
    new_target = jnp.where(want_right & can_change, right_lane, new_target)

    # Ego (index 0) keeps its current target (controlled by policy)
    new_target = new_target.at[0].set(vehicles.target_lane_id[0])

    # Return new targets AND the current-lane IDM accels (reused in traffic_step)
    return new_target, accel_current


def simplified_mobil(
    vehicles: VehicleState,
    params: EnvParams,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """Simplified MOBIL lane-change decision for all traffic vehicles.

    For each non-ego active vehicle, compares IDM acceleration in
    current lane vs. adjacent lanes, and changes lane if the gain
    exceeds the threshold.

    Args:
        vehicles: current vehicle states.
        params: environment parameters.
        key: PRNG key (for tie-breaking / stochastic timing).

    Returns:
        new_target_lane_id: (N_MAX,) updated target lane indices.
    """
    new_target, _ = _simplified_mobil_with_idm(vehicles, params, key)
    return new_target


def traffic_step(
    vehicles: VehicleState,
    params: EnvParams,
    key: jnp.ndarray,
) -> tuple[VehicleState, jnp.ndarray, jnp.ndarray]:
    """One simulation step for all traffic vehicles (non-ego).

    1. MOBIL lane-change decisions (fused with IDM for current lane)
    2. IDM longitudinal acceleration (reused from MOBIL pass — FIX P0-2)
    3. Proportional steering toward target lane

    Args:
        vehicles: current vehicle states.
        params: environment parameters.
        key: PRNG key.

    Returns:
        (accels, steerings) for all vehicles (N_MAX,) each.
        Traffic vehicles get IDM+MOBIL; ego (index 0) gets zeros
        (ego action is applied separately).
    """
    key1, key2 = jax.random.split(key)

    # FIX P0-2: MOBIL + IDM in one fused pass (4 find_leader_batched → 1 vmap)
    # _simplified_mobil_with_idm returns current-lane IDM accels for free,
    # eliminating the separate find_leader_batched call that was here before.
    new_targets, accels = _simplified_mobil_with_idm(vehicles, params, key1)

    # 3. Steering toward target lane
    steerings = steering_to_lane(
        vehicles.y, vehicles.heading, new_targets, vehicles.vx, params,
    )

    # Zero out ego (index 0) — ego action applied separately
    accels = accels.at[0].set(0.0)
    steerings = steerings.at[0].set(0.0)

    # Zero out inactive vehicles
    accels = jnp.where(vehicles.active, accels, 0.0)
    steerings = jnp.where(vehicles.active, steerings, 0.0)

    # Update target lane IDs
    updated_vehicles = vehicles.replace(target_lane_id=new_targets)

    return updated_vehicles, accels, steerings
