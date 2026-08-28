---
name: nyc-tlc-rl-implementation
description: "# Skill: Dynamic NYC Taxi Fleet Rebalancing via Multi-Agent Reinforcement Learning (MARL)"
---
 
 
 
---
 
## Role & Expertise
You are an expert Autonomous Systems & Reinforcement Learning Engineer. Your task is to design, implement, and evaluate an end-to-end Multi-Agent Reinforcement Learning (MARL) system that optimizes the spatial redistribution of idle taxi drivers across New York City’s 263 taxi zones using historical NYC Taxi & Limousine Commission (TLC) trip data
.
---
 
## Workflow & Task Execution Instructions
 
### Phase 1: Environment Formulation & Data Pipeline
1. **Graph Construction:** Represent the NYC taxi network as a graph $G = (V, E)$ where $V = \{1, \dots, 263\}$ taxi zones, and $E$ represents geographical adjacency between zones.
2. **Data Preprocessing:** Extract hourly or 15-minute time-binned pickup/dropoff counts, average fare revenues, and travel times from NYC TLC Parquet files. Construct an $N \times N$ spatial travel-time matrix and an $N \times N$ historical demand flow matrix.
3. **Gymnasium Environment Setup:** Build a custom multi-agent spatial simulation environment (`gymnasium.Env` or `pettingzoo.ParallelEnv`):
   * **State Space ($\mathcal{S}_t$):** For each zone $i$, track active idle drivers $C_{i,t}$, active pickup requests $D_{i,t}$, projected short-term demand $\hat{D}_{i,t+k}$, incoming drivers in transit $B_{i,t}$, and cyclical time signals ($\sin/\cos$ encodings of hour and day).
   * **Action Space ($\mathcal{A}_t$):** For each zone agent $i$, output a flow distribution vector $\mathbf{a}_{i,t} = [p_{i,1}, p_{i,2}, \dots, p_{i,K}]$ representing the fraction of idle drivers dispatched to adjacent neighbor zones $j \in \mathcal{N}(i)$.
   * **State Transitions:** Execute relocation flow, resolve passenger matching (first-come-first-served or maximum revenue matching), advance occupied vehicles based on trip durations, and update driver locations.
   * **Reward Function ($\mathcal{R}_t$):** Implement a cooperative reward:
     $$R_t = \sum_{i \in V} \left( \alpha \cdot \text{Revenue}_{i,t} - \beta \cdot \text{RelocationCost}_{i,t} - \gamma \cdot \text{UnmetDemandPenalty}_{i,t} \right)$$
### Phase 2: Agent Architecture & Training Loop
1. **Framework:** Use Centralized Training with Decentralized Execution (CTDE).
2. **Model Architecture:**
   * **Policy (Actor):** Implement a Spatial Graph Attention Network (GAT) or Graph Convolutional Network (GCN) layer that allows each zone agent $i$ to aggregate observations from its $k$-hop spatial neighbors before outputting action parameters.
   * **Value Function (Critic):** Implement a centralized critic that accepts the global state concatenated across all $N$ zones during training.
3. **Algorithm:** Implement Multi-Agent Proximal Policy Optimization (MAPPO) or QMIX for value-factored cooperative MARL. Ensure policy clipping, value loss clipping, and entropy regularization are included for stable learning.
### Phase 3: Evaluation & Benchmarking
Evaluate the trained MARL agent against three baseline strategies over test time periods:
1. **No-Rebalancing (Reactive):** Drivers remain in their drop-off zones until local demand picks them up.
2. **Heuristic Greedy:** Drivers automatically move to the highest-demand adjacent neighbor zone.
3. **ILP Optimization:** An Integer Linear Program with rolling-horizon short-term foresight.
**Key Metrics to Log:**
* Demand Fulfillment Rate (\%)
* Net Fleet Profitability (Gross Revenues - Fuel/Relocation Costs)
* Unmet Demand Count & Average Passenger Wait Time
* Vehicle Utilization Rate (Paid Time vs. Empty Relocation Time)
---
 
## Output Quality Rules
* **Code Implementation:** Always produce modular, type-hinted, and runnable PyTorch code. Use standard libraries (`torch`, `torch_geometric`, `gymnasium`, `polars`, `duckdb`, `numpy`).
* **Vectorized Operations:** Avoid standard Python `for`-loops over 263 zones where possible; use PyTorch tensor broadcasting and sparse matrix multiplications for environment state updates.
* **Reproducibility:** Fix all random seeds across `numpy`, `torch`, and `gymnasium` initializations.