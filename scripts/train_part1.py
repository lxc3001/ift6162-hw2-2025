#!/usr/bin/env python3
"""Train Part 1 algorithms (REINFORCE, PPO, TD3) on CalcinerEnv.

Creates plots and saves checkpoints under ./models.

Example:
  python scripts/train_part1.py --algo ppo --steps 12000
  python scripts/train_part1.py --algo all --episodes 300
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calciner import CalcinerEnv, ConstantTemperatureController, evaluate_baseline
from calciner.part1.algorithms import (
    ReinforceConfig,
    PPOConfig,
    TD3Config,
    train_reinforce,
    train_ppo,
    train_td3,
    evaluate_policy,
)


def _moving_avg(x: List[float], w: int = 10) -> np.ndarray:
    if len(x) == 0:
        return np.array([])
    x = np.array(x, dtype=np.float32)
    if len(x) < w:
        return x
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(x, kernel, mode="valid")


def _moving_avg_xy(y: List[float], x: List[float], w: int = 10) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=np.float32)
    x_arr = np.asarray(x, dtype=np.float32)
    if len(y_arr) == 0:
        return x_arr, y_arr
    if len(y_arr) != len(x_arr):
        # Fallback to implicit index if metadata is inconsistent
        x_arr = np.arange(len(y_arr), dtype=np.float32)
    if len(y_arr) < w:
        return x_arr, y_arr
    kernel = np.ones(w, dtype=np.float32) / w
    y_ma = np.convolve(y_arr, kernel, mode="valid")
    x_ma = x_arr[w - 1 :]
    return x_ma, y_ma


def _plot_history(out_path: Path, histories: Dict[str, Dict[str, List[float]]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=False)

    for name, hist in histories.items():
        x_steps = hist.get("x_steps", [])

        if "ep_return" in hist:
            y = hist["ep_return"]
            if x_steps:
                x, y_ma = _moving_avg_xy(y, x_steps, 10)
                axes[0].plot(x, y_ma, label=f"{name} (MA10)")
            else:
                axes[0].plot(_moving_avg(y, 10), label=f"{name} (MA10)")
        elif "episode_return" in hist:
            y = hist["episode_return"]
            if x_steps:
                x, y_ma = _moving_avg_xy(y, x_steps, 10)
                axes[0].plot(x, y_ma, label=f"{name} (MA10)")
            else:
                axes[0].plot(_moving_avg(y, 10), label=f"{name} (MA10)")
        else:
            y = hist.get("update_return", [])
            if x_steps:
                axes[0].plot(x_steps, y, label=name)
            else:
                axes[0].plot(y, label=name)

        if "ep_energy" in hist:
            y = hist["ep_energy"]
            if x_steps:
                x, y_ma = _moving_avg_xy(y, x_steps, 10)
                axes[1].plot(x, y_ma, label=f"{name} (MA10)")
            else:
                axes[1].plot(_moving_avg(y, 10), label=f"{name} (MA10)")
        elif "episode_energy" in hist:
            y = hist["episode_energy"]
            if x_steps:
                x, y_ma = _moving_avg_xy(y, x_steps, 10)
                axes[1].plot(x, y_ma, label=f"{name} (MA10)")
            else:
                axes[1].plot(_moving_avg(y, 10), label=f"{name} (MA10)")
        else:
            y = hist.get("update_energy", [])
            if x_steps:
                axes[1].plot(x_steps, y, label=name)
            else:
                axes[1].plot(y, label=name)

        if "ep_violations" in hist:
            y = hist["ep_violations"]
            if x_steps:
                x, y_ma = _moving_avg_xy(y, x_steps, 10)
                axes[2].plot(x, y_ma, label=f"{name} (MA10)")
            else:
                axes[2].plot(_moving_avg(y, 10), label=f"{name} (MA10)")
        elif "episode_violations" in hist:
            y = hist["episode_violations"]
            if x_steps:
                x, y_ma = _moving_avg_xy(y, x_steps, 10)
                axes[2].plot(x, y_ma, label=f"{name} (MA10)")
            else:
                axes[2].plot(_moving_avg(y, 10), label=f"{name} (MA10)")
        else:
            y = hist.get("update_violations", [])
            if x_steps:
                axes[2].plot(x_steps, y, label=name)
            else:
                axes[2].plot(y, label=name)

    axes[0].set_title("Return")
    axes[0].set_ylabel("Episode return")
    axes[0].legend()

    axes[1].set_title("Energy proxy")
    axes[1].set_ylabel("Sum(power)")
    axes[1].legend()

    axes[2].set_title("Constraint violations")
    axes[2].set_ylabel("#steps with alpha < alpha_min")
    axes[2].set_xlabel("Environment steps")
    axes[2].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _rollout_trajectory(env: CalcinerEnv, policy, *, seed: int, device: str) -> Dict[str, np.ndarray]:
    dev = torch.device(device)

    obs = env.reset(seed=seed)
    done = False

    ts: List[int] = []
    alpha: List[float] = []
    alpha_min: List[float] = []
    u: List[float] = []
    power: List[float] = []
    reward: List[float] = []

    while not done:
        if isinstance(policy, torch.nn.Module):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
            with torch.no_grad():
                if hasattr(policy, "deterministic"):
                    act = float(policy.deterministic(obs_t).cpu().numpy().item())
                else:
                    act = float(policy(obs_t).cpu().numpy().item())
        else:
            act = float(policy(obs))

        obs, rew, done, info = env.step(act)

        ts.append(env.t - 1)
        alpha.append(float(info.get("alpha", np.nan)))
        alpha_min.append(float(env.alpha_min[env.t - 1]))
        u.append(float(info.get("u", act)))
        power.append(float(info.get("power", 0.0)))
        reward.append(float(rew))

    return {
        "t": np.asarray(ts, dtype=np.int32),
        "alpha": np.asarray(alpha, dtype=np.float32),
        "alpha_min": np.asarray(alpha_min, dtype=np.float32),
        "u": np.asarray(u, dtype=np.float32),
        "power": np.asarray(power, dtype=np.float32),
        "reward": np.asarray(reward, dtype=np.float32),
    }


def _plot_trajectory(out_path: Path, traj: Dict[str, np.ndarray], title: str) -> None:
    t = traj["t"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, traj["u"], label="T_g,in (u)")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title(title)

    axes[1].plot(t, traj["alpha"], label="alpha")
    axes[1].plot(t, traj["alpha_min"], label="alpha_min")
    axes[1].set_ylabel("Conversion")
    axes[1].legend()

    axes[2].plot(t, traj["power"], label="power")
    axes[2].set_ylabel("Power")
    axes[2].set_xlabel("t (step)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["reinforce", "ppo", "td3", "all"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=300, help="Used by REINFORCE")
    parser.add_argument("--steps", type=int, default=40 * 300, help="Used by PPO/TD3")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--outdir", type=str, default="models")
    parser.add_argument(
        "--plot-trajectory",
        action="store_true",
        help="Also save a 1-episode trajectory plot (u, alpha vs alpha_min, power).",
    )
    args = parser.parse_args()

    env = CalcinerEnv(episode_length=40, dt=0.5)

    outdir = (ROOT / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    baseline = ConstantTemperatureController(T_g_in=1261.15)
    baseline_stats = evaluate_baseline(env, baseline, n_episodes=5)
    print("Baseline:")
    print(baseline_stats)

    histories: Dict[str, Dict[str, List[float]]] = {}

    if args.algo in ("reinforce", "all"):
        print(f"Training REINFORCE for {args.episodes} episodes...")
        cfg = ReinforceConfig(seed=args.seed, episodes=args.episodes, device=args.device)
        policy, hist = train_reinforce(env, cfg)
        hist["x_steps"] = [env.episode_length * (i + 1) for i in range(len(hist.get("ep_return", [])))]
        histories["reinforce"] = hist
        torch.save(
            {
                "algo": "reinforce",
                "config": cfg.__dict__,
                "state_dict": policy.state_dict(),
            },
            outdir / "part1_reinforce.pt",
        )
        stats = evaluate_policy(env, policy, n_episodes=10, device=args.device)
        print("REINFORCE eval:")
        print(stats)
        if args.plot_trajectory:
            traj = _rollout_trajectory(env, policy, seed=0, device=args.device)
            _plot_trajectory(
                (ROOT / "figures" / "part1_trajectory_reinforce.png").resolve(),
                traj,
                title="REINFORCE trajectory (deterministic policy)",
            )

    if args.algo in ("ppo", "all"):
        print(f"Training PPO for {args.steps} environment steps...")
        cfg = PPOConfig(seed=args.seed, total_steps=args.steps, device=args.device)
        policy, hist = train_ppo(env, cfg)
        hist["x_steps"] = [cfg.rollout_steps * (i + 1) for i in range(len(hist.get("update_return", [])))]
        histories["ppo"] = hist
        torch.save(
            {
                "algo": "ppo",
                "config": cfg.__dict__,
                "state_dict": policy.state_dict(),
            },
            outdir / "part1_ppo.pt",
        )
        stats = evaluate_policy(env, policy, n_episodes=10, device=args.device)
        print("PPO eval:")
        print(stats)
        if args.plot_trajectory:
            traj = _rollout_trajectory(env, policy, seed=0, device=args.device)
            _plot_trajectory(
                (ROOT / "figures" / "part1_trajectory_ppo.png").resolve(),
                traj,
                title="PPO trajectory (deterministic policy)",
            )

    if args.algo in ("td3", "all"):
        print(f"Training TD3 for {args.steps} environment steps...")
        cfg = TD3Config(seed=args.seed, total_steps=args.steps, device=args.device)
        actor, hist = train_td3(env, cfg)
        hist["x_steps"] = [env.episode_length * (i + 1) for i in range(len(hist.get("episode_return", [])))]
        histories["td3"] = hist
        torch.save(
            {
                "algo": "td3",
                "config": cfg.__dict__,
                "state_dict": actor.state_dict(),
            },
            outdir / "part1_td3.pt",
        )
        stats = evaluate_policy(env, actor, n_episodes=10, device=args.device)
        print("TD3 eval:")
        print(stats)
        if args.plot_trajectory:
            traj = _rollout_trajectory(env, actor, seed=0, device=args.device)
            _plot_trajectory(
                (ROOT / "figures" / "part1_trajectory_td3.png").resolve(),
                traj,
                title="TD3 trajectory (deterministic policy)",
            )

    fig_path = (ROOT / "figures" / "part1_learning_curves.png").resolve()
    _plot_history(fig_path, histories)
    print(f"Saved learning curves to {fig_path}")

    if args.plot_trajectory:
        baseline_traj = _rollout_trajectory(env, baseline.get_action, seed=0, device=args.device)
        _plot_trajectory(
            (ROOT / "figures" / "part1_trajectory_baseline.png").resolve(),
            baseline_traj,
            title="Baseline constant-temperature trajectory",
        )


if __name__ == "__main__":
    main()
