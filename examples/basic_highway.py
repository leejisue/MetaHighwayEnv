"""Minimal MetaHighwayEnv usage example.

Demonstrates:
  1. Basic meta-RL rollout (task sampling, reset, step)
  2. Causal intervention (do-calculus via intervene())
  3. Normalized score utility
"""

import numpy as np
from highway_env.meta_rl.meta_env import MetaHighwayEnv


# ---------------------------------------------------------------------------
# 1. Basic meta-RL rollout
# ---------------------------------------------------------------------------

env = MetaHighwayEnv(env_type="highway", n_tasks=10, seed=42)

obs, info = env.reset()
print(f"Observation shape : {obs.shape}")
print(f"Action space      : {env.action_space}")

total_reward = 0.0
for _ in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        obs, info = env.reset()

print(f"Total reward (50 steps): {total_reward:.3f}")

env.close()


# ---------------------------------------------------------------------------
# 2. Causal intervention
# ---------------------------------------------------------------------------

causal_env = MetaHighwayEnv(
    env_type="highway",
    n_tasks=10,
    enable_causal_logging=True,
    seed=0,
)

causal_env.reset()

# Apply a do-calculus intervention: set driver aggressiveness to 0.9
result = causal_env.intervene("driver_aggressiveness", 0.9)
print(f"\nIntervention result : {result}")

log = causal_env.get_intervention_log()
print(f"Intervention log length: {len(log)}")

causal_env.close()


# ---------------------------------------------------------------------------
# 3. Normalized score
# ---------------------------------------------------------------------------

env2 = MetaHighwayEnv(env_type="highway", n_tasks=10, seed=1)
raw_return = 22.5
norm = env2.get_normalized_score(raw_return, task_name="easy")
print(f"\nNormalized score for raw_return={raw_return}: {norm:.4f}")
env2.close()
