# NYC TLC reinforcement learning

A research prototype for dynamic NYC yellow-taxi fleet rebalancing with a shared graph policy and centralized critic.

## Notebooks

- **[NYC_RL_revised.ipynb](NYC_RL_revised.ipynb)** — current experiment. It uses chronological train/validation/test splits, train-only feature fitting, integer origin-destination demand, vehicle-conserving delayed trip dynamics, a stay action, corrected PPO/GAE updates, transparent baselines, paired uncertainty estimates, and CPU/GPU timing.
- [NYC_RL_1.ipynb](NYC_RL_1.ipynb) — original prototype retained for comparison.

The revised notebook defaults to `quick_mode=True` for a pipeline check. Set it to `False` to run the predeclared five-seed experiment, including the rolling mixed-integer baseline.

## Interpretation boundary

NYC TLC records represent completed historical trips, not all latent passenger requests. The notebook therefore reports service of observed trips and does not claim to estimate real unmet demand or passenger wait time.
