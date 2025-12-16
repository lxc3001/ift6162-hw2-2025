# Flash Calciner Control (HW2)


See `ASSIGNMENT.md` for the full problem statement and grading rubric.

This repo is set up so you run scripts directly:
- Part 1: `scripts/train_part1.py`
- Part 2: `scripts/train_part2.py`

Both scripts have a small configuration block at the bottom under:

```python
if __name__ == "__main__":
        ...
```

Edit those values, then run the script.

## Setup

```bash
# (Recommended) create a virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install deps
pip install -r requirements.txt
```

The training scripts use Matplotlib in non-interactive mode (`Agg`) and will save figures to disk.

## Run Part 1 (REINFORCE / PPO / TD3)

1) Open `scripts/train_part1.py` and edit the config under `__main__` (algorithm, steps/episodes, device, etc.).

2) Run:

```bash
python scripts/train_part1.py
```

### Expected outputs (Part 1)

- Checkpoints in `models/`:
    - `models/part1_reinforce.pt`
    - `models/part1_ppo.pt`
    - `models/part1_td3.pt`
- Learning curves:
    - `figures/part1_learning_curves.png`
- Optional trajectory plot(s) if enabled in the config.

Example console output (values will vary):

```text
Baseline:
{'mean_energy': ..., 'mean_violations': ..., 'mean_final_conversion': ...}
Training REINFORCE for 300 episodes...
REINFORCE eval:
{'mean_energy': ..., 'mean_violations': ..., 'mean_final_conversion': ...}
Training PPO for ... steps...
PPO eval:
{...}
Training TD3 for ... steps...
TD3 eval:
{...}
Saved learning curves to ...\figures\part1_learning_curves.png
```

## Run Part 2 (PPO on surrogate, evaluated on physics)

Part 2 trains PPO on the surrogate environment and evaluates the resulting policy on the true physics simulator.

1) Ensure the provided surrogate checkpoint exists:
- `models/surrogate_model.pt`

2) Open `scripts/train_part2.py` and edit the config under `__main__`.

3) Run:

```bash
python scripts/train_part2.py
```

### Expected outputs (Part 2)

- Checkpoint in `models/`:
    - `models/part2_ppo_1d.pt` (if controlling only `T_g_in`)
    - `models/part2_ppo_2d.pt` (if controlling both `T_g_in` and `T_s_in`)
- Learning curves:
    - `figures/part2_learning_curves.png`
- Spatial profiles (physics):
    - `figures/part2_spatial_profiles_physics.png`
- Optional architecture comparison plot (if enabled):
    - `figures/part2_arch_compare.png`

Example console output (values will vary):

```text
Training Part2 PPO: steps=20000, episode_length=50, control_T_s=True, arch=mlp
Saved checkpoint to ...\models\part2_ppo_2d.pt
Physics eval:
{'mean_energy': 21.83, 'mean_violations': 50.0, 'mean_final_conversion': 0.895}
Physics baseline (constant high heat) eval:
{'mean_energy': 50.0, 'mean_violations': 46.0, 'mean_final_conversion': 0.983}
Saved learning curves to ...\figures\part2_learning_curves.png
Saved spatial profile plot to figures/part2_spatial_profiles_physics.png
```

## Configurations

In `scripts/train_part1.py`:
- `algo`: `"reinforce" | "ppo" | "td3" | "all"`
- `episodes` / `steps`
- `device`: `"cpu"` or `"cuda"`

In `scripts/train_part2.py`:
- `steps`: total environment steps
- `control_ts`: `False` = 1D action (`T_g_in`), `True` = 2D action (`T_g_in`, `T_s_in`)
- `arch`: `"mlp" | "conv" | "hybrid"`
- `profile_snaps`: how many snapshot times to show in the spatial profile plot
- `compare_archs`: trains `mlp/conv/hybrid` and saves a combined comparison figure

