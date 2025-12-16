#!/usr/bin/env python3
"""Train Part 2 PPO on SurrogateCalcinerEnv (140D) and validate on true physics.

This script is added without modifying existing repo files.

This script is intended to be edited/run directly (no CLI args).

Edit the configuration at the bottom under `if __name__ == "__main__":`.
"""

from __future__ import annotations

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

from calciner import CalcinerSimulator, SpatiallyAwareDynamics, SurrogateModel, SurrogateCalcinerEnv
from calciner.physics import N_SPECIES, L
from calciner.part2.ppo import PPOConfig, train_ppo, evaluate_policy_on_physics


def _moving_avg(x: List[float], w: int = 10) -> np.ndarray:
    if len(x) == 0:
        return np.array([])
    x = np.asarray(x, dtype=np.float32)
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
        x_arr = np.arange(len(y_arr), dtype=np.float32)
    if len(y_arr) < w:
        return x_arr, y_arr
    kernel = np.ones(w, dtype=np.float32) / w
    y_ma = np.convolve(y_arr, kernel, mode="valid")
    x_ma = x_arr[w - 1 :]
    return x_ma, y_ma


def _plot_history(out_path: Path, hist: Dict[str, List[float]], *, rollout_steps: int) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=False)

    x_steps = [rollout_steps * (i + 1) for i in range(len(hist.get("update_return", [])))]

    x0, y0 = _moving_avg_xy(hist.get("update_return", []), x_steps, 5)
    axes[0].plot(x0, y0)
    axes[0].set_title("Return")

    x1, y1 = _moving_avg_xy(hist.get("update_energy", []), x_steps, 5)
    axes[1].plot(x1, y1)
    axes[1].set_title("Energy (normalized sum)")

    x2, y2 = _moving_avg_xy(hist.get("update_violations", []), x_steps, 5)
    axes[2].plot(x2, y2)
    axes[2].set_title("Constraint violations (#steps with violation>0)")

    x3, y3 = _moving_avg_xy(hist.get("update_final_alpha", []), x_steps, 5)
    axes[3].plot(x3, y3)
    axes[3].set_title("Final conversion (alpha)")
    axes[3].set_xlabel("Environment steps")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_arch_comparison(out_path: Path, hists: Dict[str, Dict[str, List[float]]], *, rollout_steps: int) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=False)

    keys = [
        ("update_return", "Return"),
        ("update_energy", "Energy (normalized sum)"),
        ("update_violations", "Constraint violations (#steps with violation>0)"),
        ("update_final_alpha", "Final conversion (alpha)"),
    ]

    # Use env steps on x-axis, aligned to update index.
    max_updates = max((len(h.get("update_return", [])) for h in hists.values()), default=0)
    x_steps = [rollout_steps * (i + 1) for i in range(max_updates)]

    for ax, (k, title) in zip(axes, keys):
        for arch, hist in hists.items():
            # Align per-arch curve length
            y_raw = hist.get(k, [])
            x_local = x_steps[: len(y_raw)]
            x_ma, y_ma = _moving_avg_xy(y_raw, x_local, 5)
            ax.plot(x_ma, y_ma, label=arch)
        ax.set_title(title)
        ax.legend(fontsize=9)

    axes[-1].set_xlabel("Environment steps")
    fig.suptitle("Part 2 architecture comparison (PPO)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _slice_temperatures(x: np.ndarray, N_z: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (T_s, T_g) profiles from packed state vector."""
    x = np.asarray(x)
    off = N_SPECIES * N_z
    T_s = x[off : off + N_z]
    T_g = x[off + N_z : off + 2 * N_z]
    return np.asarray(T_s, dtype=np.float32), np.asarray(T_g, dtype=np.float32)


def _plot_spatial_profiles(
    out_path: Path,
    z: np.ndarray,
    profiles: Dict[str, List[np.ndarray]],
    title: str,
) -> None:
    """Plot T_s and T_g spatial profiles for multiple snapshot times."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Parse snapshot times and assign colors by t (same color for policy/baseline).
    ts: List[int] = []
    parsed: List[tuple[str, int, List[np.ndarray]]] = []
    for label, ts_list in profiles.items():
        which, t_lab = label.split(":", 1)
        if not t_lab.startswith("t="):
            raise ValueError(f"Unexpected profile label format: {label}")
        t = int(t_lab.split("=", 1)[1])
        ts.append(t)
        parsed.append((which, t, ts_list))

    unique_ts = sorted(set(ts))
    cmap = plt.get_cmap("tab10")
    t_to_color = {t: cmap(i % 10) for i, t in enumerate(unique_ts)}

    for which, t, ts_list in parsed:
        style = "-" if which == "policy" else "--"
        color = t_to_color[t]
        T_s, T_g = ts_list
        axes[0].plot(z, T_s, style, color=color, alpha=0.95)
        axes[1].plot(z, T_g, style, color=color, alpha=0.95)

    axes[0].set_title(title)
    axes[0].set_ylabel("Solid temperature T_s [K]")
    axes[1].set_ylabel("Gas temperature T_g [K]")
    axes[1].set_xlabel("Position along reactor [m]")

    # Two compact legends: (1) time->color, (2) policy/baseline->linestyle
    time_handles = [
        plt.Line2D([0], [0], color=t_to_color[t], linestyle="-", linewidth=2, label=f"t={t}")
        for t in unique_ts
    ]
    style_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="policy"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="baseline"),
    ]

    leg1 = axes[0].legend(handles=time_handles, title="Time", fontsize=9, title_fontsize=9, loc="upper left", ncol=min(3, len(time_handles)))
    axes[0].add_artist(leg1)
    axes[0].legend(handles=style_handles, title="Style", fontsize=9, title_fontsize=9, loc="lower left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _rollout_states_surrogate(env: SurrogateCalcinerEnv, policy, obs_norm, obs_adapter, *, device: str) -> List[np.ndarray]:
    dev = torch.device(device)
    obs = env.reset(seed=0)
    states: List[np.ndarray] = [obs.copy()]
    done = False
    while not done:
        model_obs = obs_adapter.transform(obs) if obs_adapter is not None else obs
        obs_in = obs_norm.normalize(model_obs) if obs_norm is not None else model_obs
        obs_t = torch.as_tensor(obs_in, dtype=torch.float32, device=dev).unsqueeze(0)
        with torch.no_grad():
            act = policy.deterministic(obs_t).squeeze(0).cpu().numpy()
        obs, _, done, _ = env.step(act)
        states.append(obs.copy())
    return states


def _rollout_states_physics(simulator: CalcinerSimulator, policy, obs_norm, obs_adapter, *, episode_length: int, control_T_s: bool, device: str) -> List[np.ndarray]:
    dev = torch.device(device)
    # Match SurrogateCalcinerEnv cold start
    N_z = simulator.N_z
    c = np.zeros((N_SPECIES, N_z), dtype=np.float32)
    c[0, :] = 0.15
    c[1, :] = 0.1
    c[2, :] = 0.05
    c[3, :] = 18.0
    c[4, :] = 0.1
    T_s = np.ones(N_z, dtype=np.float32) * 600.0
    T_g = np.ones(N_z, dtype=np.float32) * 600.0
    x = np.concatenate([c.flatten(), T_s, T_g]).astype(np.float32)

    states: List[np.ndarray] = [x.copy()]
    for _ in range(episode_length):
        model_obs = obs_adapter.transform(x) if obs_adapter is not None else x
        obs_in = obs_norm.normalize(model_obs) if obs_norm is not None else model_obs
        obs_t = torch.as_tensor(obs_in, dtype=torch.float32, device=dev).unsqueeze(0)
        with torch.no_grad():
            act = policy.deterministic(obs_t).squeeze(0).cpu().numpy()

        if not control_T_s:
            T_g_in = float(act[0])
            T_s_in = 657.15
        else:
            T_g_in = float(act[0])
            T_s_in = float(act[1])
        u = np.array([T_g_in, T_s_in], dtype=np.float32)
        x = simulator.step(x, u)
        states.append(np.asarray(x, dtype=np.float32).copy())
    return states


def _baseline_action(env: SurrogateCalcinerEnv, *, control_T_s: bool) -> np.ndarray:
    # High-heat baseline: tends to satisfy conversion but uses more energy.
    if not control_T_s:
        return np.array([env.T_g_max], dtype=np.float32)
    return np.array([env.T_g_max, env.T_s_max], dtype=np.float32)


def _eval_baseline_physics(
    simulator: CalcinerSimulator,
    *,
    episode_length: int,
    alpha_min: float,
    control_T_s: bool,
    n_episodes: int,
) -> Dict[str, float]:
    energies: List[float] = []
    violations: List[int] = []
    final_alpha: List[float] = []

    # Build constant action equivalent to surrogate baseline
    if not control_T_s:
        T_g_in = 1350.0
        T_s_in = 657.15
    else:
        T_g_in = 1350.0
        T_s_in = 800.0

    for _ in range(n_episodes):
        # cold start
        N_z = simulator.N_z
        c = np.zeros((N_SPECIES, N_z), dtype=np.float32)
        c[0, :] = 0.15
        c[1, :] = 0.1
        c[2, :] = 0.05
        c[3, :] = 18.0
        c[4, :] = 0.1
        T_s = np.ones(N_z, dtype=np.float32) * 600.0
        T_g = np.ones(N_z, dtype=np.float32) * 600.0
        x = np.concatenate([c.flatten(), T_s, T_g]).astype(np.float32)

        ep_energy = 0.0
        ep_vio = 0
        alpha = 0.0

        for _t in range(episode_length):
            u = np.array([T_g_in, T_s_in], dtype=np.float32)
            x = simulator.step(x, u)
            # same conversion definition as SurrogateCalcinerEnv
            c_kao_out = float(x[N_z - 1])
            alpha = float(np.clip(1.0 - c_kao_out / 0.15, 0.0, 1.0))

            energy = (T_g_in - 900.0) / (1350.0 - 900.0)
            violation = max(0.0, alpha_min - alpha)
            ep_energy += float(energy)
            if violation > 0.0:
                ep_vio += 1

        energies.append(ep_energy)
        violations.append(ep_vio)
        final_alpha.append(alpha)

    return {
        "mean_energy": float(np.mean(energies)),
        "mean_violations": float(np.mean(violations)),
        "mean_final_conversion": float(np.mean(final_alpha)),
    }


def _load_surrogate(model_path: Path) -> tuple[SurrogateModel, float]:
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    N_z = int(checkpoint["N_z"])
    dt = float(checkpoint["dt"])

    model = SpatiallyAwareDynamics(N_z=N_z)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    norm_params = {k: np.array(v) for k, v in checkpoint["norm_params"].items()}
    surrogate = SurrogateModel(model, norm_params)
    return surrogate, dt


def run_part2(
    *,
    seed: int = 0,
    steps: int = 50_000,
    device: str = "cpu",
    outdir: str = "models",
    episode_length: int = 50,
    alpha_min: float = 0.95,
    control_ts: bool = False,
    arch: str = "mlp",
    compare_archs: bool = False,
    profile_snaps: int = 3,
    plot_surrogate_profiles: bool = False,
    eval_episodes: int = 5,
) -> None:
    if arch not in {"mlp", "conv", "hybrid"}:
        raise ValueError(f"Invalid arch={arch}")

    model_path = (ROOT / "models" / "surrogate_model.pt").resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Missing surrogate checkpoint: {model_path}")

    surrogate, dt = _load_surrogate(model_path)

    env = SurrogateCalcinerEnv(
        surrogate,
        episode_length=episode_length,
        alpha_min=alpha_min,
        control_T_s=bool(control_ts),
    )

    if compare_archs:
        arches = ["mlp", "conv", "hybrid"]
        hists: Dict[str, Dict[str, List[float]]] = {}

        print(
            f"Comparing architectures: {arches} (steps={steps}, episode_length={episode_length}, control_T_s={control_ts})"
        )

        for arch_i in arches:
            cfg = PPOConfig(seed=seed, total_steps=steps, device=device, arch=arch_i)
            outdir_i = (ROOT / outdir / f"compare_{arch_i}").resolve()
            outdir_i.mkdir(parents=True, exist_ok=True)

            outdir.mkdir(parents=True, exist_ok=True)
            env_i = SurrogateCalcinerEnv(
                surrogate,
                episode_length=episode_length,
                alpha_min=alpha_min,
                control_T_s=bool(control_ts),
            )

            print(f"\n=== Training arch={arch_i} ===")
            policy, obs_norm, obs_adapter, hist = train_ppo(env_i, cfg)
            hists[arch_i] = hist

            ckpt = {
                "algo": "ppo",
                "part": 2,
                "config": cfg.__dict__,
                "env": {
                    "episode_length": episode_length,
                    "alpha_min": alpha_min,
                    "control_T_s": bool(control_ts),
                    "T_g_min": env_i.T_g_min,
                    "T_g_max": env_i.T_g_max,
                    "T_s_min": env_i.T_s_min,
                    "T_s_max": env_i.T_s_max,
                    "T_s_default": env_i.T_s_default,
                },
                "obs_normalizer": obs_norm.state_dict(),
                "state_dict": policy.state_dict(),
                "arch": cfg.arch,
            }

            out_path = outdir_i / ("part2_ppo_2d.pt" if control_ts else "part2_ppo_1d.pt")
            torch.save(ckpt, out_path)
            print(f"Saved checkpoint to {out_path}")

            simulator = CalcinerSimulator(N_z=20, dt=dt)
            phys_stats = evaluate_policy_on_physics(
                simulator,
                policy,
                obs_norm,
                episode_length=episode_length,
                alpha_min=alpha_min,
                control_T_s=bool(control_ts),
                device=device,
                n_episodes=eval_episodes,
                obs_adapter=obs_adapter,
            )
            print("Physics eval:")
            print(phys_stats)

        fig_path = (ROOT / "figures" / "part2_arch_compare.png").resolve()
        _plot_arch_comparison(fig_path, hists, rollout_steps=int(cfg.rollout_steps))
        print(f"\nSaved architecture comparison plot to {fig_path}")
        return

    cfg = PPOConfig(seed=seed, total_steps=steps, device=device, arch=arch)

    outdir = (ROOT / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"Training Part2 PPO: steps={cfg.total_steps}, episode_length={episode_length}, control_T_s={control_ts}, arch={cfg.arch}"
    )

    policy, obs_norm, obs_adapter, hist = train_ppo(env, cfg)

    ckpt = {
        "algo": "ppo",
        "part": 2,
        "config": cfg.__dict__,
        "env": {
            "episode_length": episode_length,
            "alpha_min": alpha_min,
            "control_T_s": bool(control_ts),
            "T_g_min": env.T_g_min,
            "T_g_max": env.T_g_max,
            "T_s_min": env.T_s_min,
            "T_s_max": env.T_s_max,
            "T_s_default": env.T_s_default,
        },
        "obs_normalizer": obs_norm.state_dict(),
        "state_dict": policy.state_dict(),
        "arch": cfg.arch,
    }

    out_path = outdir / ("part2_ppo_2d.pt" if control_ts else "part2_ppo_1d.pt")
    torch.save(ckpt, out_path)
    print(f"Saved checkpoint to {out_path}")

    # Validate on true physics simulator
    simulator = CalcinerSimulator(N_z=20, dt=dt)
    phys_stats = evaluate_policy_on_physics(
        simulator,
        policy,
        obs_norm,
        episode_length=episode_length,
        alpha_min=alpha_min,
        control_T_s=bool(control_ts),
        device=device,
        n_episodes=eval_episodes,
        obs_adapter=obs_adapter,
    )
    print("Physics eval:")
    print(phys_stats)

    baseline_phys = _eval_baseline_physics(
        simulator,
        episode_length=episode_length,
        alpha_min=alpha_min,
        control_T_s=bool(control_ts),
        n_episodes=eval_episodes,
    )
    print("Physics baseline (constant high heat) eval:")
    print(baseline_phys)

    fig_path = (ROOT / "figures" / "part2_learning_curves.png").resolve()
    _plot_history(fig_path, hist, rollout_steps=int(cfg.rollout_steps))
    print(f"Saved learning curves to {fig_path}")

    # Spatial profile visualization (physics required by assignment; surrogate optional)
    N_z = 20
    z = np.linspace(0.0, float(L), N_z)
    # Snapshot indices for profiles (keep it readable; default 3 snapshots)
    n_snaps = int(max(1, profile_snaps))
    if n_snaps == 1:
        snap_idx = [episode_length]
    else:
        snap_idx = np.linspace(0, episode_length, n_snaps, dtype=int).tolist()
    snap_idx = sorted(set(int(i) for i in snap_idx if 0 <= int(i) <= episode_length))

    if plot_surrogate_profiles:
        # Surrogate closed-loop states for policy
        states_surr_policy = _rollout_states_surrogate(env, policy, obs_norm, obs_adapter, device=device)
        # Surrogate closed-loop states for baseline
        env_b = SurrogateCalcinerEnv(
            surrogate,
            episode_length=episode_length,
            alpha_min=alpha_min,
            control_T_s=bool(control_ts),
        )
        act_b = _baseline_action(env_b, control_T_s=bool(control_ts))
        states_surr_base: List[np.ndarray] = [env_b.reset(seed=0).copy()]
        done = False
        while not done:
            obs_b, _, done, _ = env_b.step(act_b)
            states_surr_base.append(obs_b.copy())

        profiles_surr: Dict[str, List[np.ndarray]] = {}
        for i in snap_idx:
            T_s, T_g = _slice_temperatures(states_surr_policy[i], N_z)
            profiles_surr[f"policy:t={i}"] = [T_s, T_g]
            T_s, T_g = _slice_temperatures(states_surr_base[i], N_z)
            profiles_surr[f"baseline:t={i}"] = [T_s, T_g]

        _plot_spatial_profiles(
            (ROOT / "figures" / "part2_spatial_profiles_surrogate.png").resolve(),
            z,
            profiles_surr,
            title="Part 2 spatial temperature profiles (surrogate closed-loop)",
        )

    # Physics closed-loop states for policy and baseline
    states_phys_policy = _rollout_states_physics(
        simulator,
        policy,
        obs_norm,
        obs_adapter,
        episode_length=episode_length,
        control_T_s=bool(control_ts),
        device=device,
    )

    # baseline physics rollout (same cold start + constant action)
    if not control_ts:
        T_g_in = 1350.0
        T_s_in = 657.15
    else:
        T_g_in = 1350.0
        T_s_in = 800.0
    x = states_phys_policy[0].copy()
    states_phys_base: List[np.ndarray] = [x.copy()]
    for _ in range(episode_length):
        x = simulator.step(x, np.array([T_g_in, T_s_in], dtype=np.float32))
        states_phys_base.append(np.asarray(x, dtype=np.float32).copy())

    profiles_phys: Dict[str, List[np.ndarray]] = {}
    for i in snap_idx:
        T_s, T_g = _slice_temperatures(states_phys_policy[i], N_z)
        profiles_phys[f"policy:t={i}"] = [T_s, T_g]
        T_s, T_g = _slice_temperatures(states_phys_base[i], N_z)
        profiles_phys[f"baseline:t={i}"] = [T_s, T_g]

    _plot_spatial_profiles(
        (ROOT / "figures" / "part2_spatial_profiles_physics.png").resolve(),
        z,
        profiles_phys,
        title="Part 2 spatial temperature profiles (physics closed-loop)",
    )

    if plot_surrogate_profiles:
        print("Saved spatial profile plots to figures/part2_spatial_profiles_surrogate.png and figures/part2_spatial_profiles_physics.png")
    else:
        print("Saved spatial profile plot to figures/part2_spatial_profiles_physics.png")


if __name__ == "__main__":
    # Edit these settings directly.
    run_part2(
        seed=0,
        steps=20_000,
        device="cpu",
        outdir="models",
        episode_length=50,
        alpha_min=0.95,
        control_ts=False,  # 1D if False, 2D if True
        arch="mlp",  # "mlp" | "conv" | "hybrid"
        compare_archs=False, # set True to compare "mlp" | "conv" | "hybrid"
        profile_snaps=3,
        plot_surrogate_profiles=False, # set True to plot surrogate closed-loop profiles
        eval_episodes=3,
    )
