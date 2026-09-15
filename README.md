# NYC TLC reinforcement learning

A research prototype for dynamic NYC yellow-taxi fleet rebalancing with a shared graph policy and centralized critic.

## Notebooks

- **[NYC_RL_revised.ipynb](NYC_RL_revised.ipynb)** — current experiment. It uses all 12 months of 2023 with chronological January–August train, September–October validation, and November–December locked test splits; train-only feature fitting; integer origin-destination demand; vehicle-conserving delayed trip dynamics; a stay action; corrected PPO/GAE updates; transparent baselines; paired uncertainty estimates; and CPU/GPU timing. The full-year OD cube remains in host memory and only the active time slice is copied to GPU.
- [NYC_RL_1.ipynb](NYC_RL_1.ipynb) — original prototype retained for comparison.

The revised notebook defaults to `quick_mode=False` and runs the predeclared five-seed full-year experiment, including the rolling mixed-integer baseline. Set it to `True` only for a short pipeline smoke test. It also runs no-retraining cost sweeps, seasonal/lag/shuffled/perfect-foresight forecast ablations, and a parameter-shared MLP control, exporting each diagnostic table.

## Interpretation boundary

NYC TLC records represent completed historical trips, not all latent passenger requests. The notebook therefore reports service of observed trips and does not claim to estimate real unmet demand or passenger wait time.
