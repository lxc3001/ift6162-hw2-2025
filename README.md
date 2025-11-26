# Flash Clay Calciner - Neural Surrogate MPC

Economic Model Predictive Control for flash clay calciner using a learned neural surrogate.

## Overview

This project implements:
1. **Physics-based PDE model** of a flash clay calciner (140-dimensional state)
2. **Neural surrogate** that learns discrete-time dynamics (61× faster than physics)
3. **MPPI controller** for energy-optimal control with conversion constraints

## Key Results

| Metric | Value |
|--------|-------|
| Surrogate speedup | **61×** (25ms vs 1.5s per rollout) |
| Energy savings | **72%** vs constant-temperature baseline |
| Conversion target | ✓ Achieved (96.8% vs 95% target) |
| MPC solve time | 1.5s/step (CPU) |

![MPC Control Results](figures/mpc_control_results.png)

## Project Structure

```
├── src/                          # Source code
│   ├── flash_calciner.py         # Physics-based PDE/ODE model
│   ├── surrogate_flash_calciner.py  # Neural surrogate + MPPI/MPC
│   └── run_surrogate_mpc.py      # Closed-loop control evaluation
│
├── models/                       # Trained model weights
│   └── surrogate_model.pt
│
├── figures/                      # Output figures
│   ├── mpc_control_results.png
│   ├── mpc_state_profiles.png
│   └── surrogate_training.png
│
├── baselines/                    # RL baseline experiments
│   ├── td3_flash_calciner.py
│   ├── ppo_flash_calciner.py
│   ├── rl_flash_calciner.py
│   └── figures/
│
├── archive/                      # Old experiments
│
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train surrogate (generates ~900 transitions, trains neural net)
python src/surrogate_flash_calciner.py

# Run closed-loop MPC control
python src/run_surrogate_mpc.py
```

## Method

### 1. Neural Surrogate

Learns discrete-time dynamics of the 140D PDE system:
```
x_{t+1} = f_θ(x_t, u_t)
```

- **Architecture**: Spatially-aware 1D convolutions (261K params)
- **Training**: Residual learning on 900 physics simulation transitions
- **Accuracy**: ~10% relative error (sufficient for MPC)

### 2. MPPI Controller

Model Predictive Path Integral control:
- Samples 96 random control sequences
- Rolls out trajectories using surrogate (in parallel)
- Weights by cost, returns weighted average

**Cost function**: Energy + soft conversion constraint + terminal temperature

### 3. Closed-Loop Evaluation

- MPC plans with surrogate, executes on physics simulator
- 80 steps (8 seconds simulated time)
- Compares against constant-temperature baseline

## Physics Model

Based on Cantisani et al. "Dynamic modeling and simulation of a flash clay calciner":

- **Reaction**: Kaolinite → Metakaolin + 2H₂O (3rd order Arrhenius)
- **State**: 5 species × 20 cells + 2 temperatures × 20 cells = 140D
- **Discretization**: Finite volume with upwind convection

## Reference

Cantisani, N., Svensen, J. L., Hansen, O. F., & Jørgensen, J. B. (2024). 
Dynamic modeling and simulation of a flash clay calciner.
