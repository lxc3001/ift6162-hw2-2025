from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Arch = Literal["mlp", "conv", "hybrid"]


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class ObsNormalizer:
    """Running mean/std normalizer (Welford)."""

    def __init__(self, size: int, eps: float = 1e-6):
        self.size = int(size)
        self.eps = float(eps)
        self.count = 0
        self.mean = np.zeros(self.size, dtype=np.float64)
        self.m2 = np.zeros(self.size, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        for row in x:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.m2 += delta * delta2

    @property
    def var(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.size, dtype=np.float64)
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + self.eps)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean.astype(np.float32)) / self.std.astype(np.float32)

    def state_dict(self) -> Dict[str, np.ndarray | int | float]:
        return {
            "size": self.size,
            "eps": self.eps,
            "count": self.count,
            "mean": self.mean.astype(np.float64),
            "m2": self.m2.astype(np.float64),
        }

    @classmethod
    def from_state_dict(cls, d: Dict) -> "ObsNormalizer":
        obj = cls(int(d["size"]), float(d["eps"]))
        obj.count = int(d["count"])
        obj.mean = np.asarray(d["mean"], dtype=np.float64)
        obj.m2 = np.asarray(d["m2"], dtype=np.float64)
        return obj


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: nn.Module = nn.Tanh(),
    ):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(activation)
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Conv1DEncoder(nn.Module):
    """1D conv encoder over spatial dimension.

    Expects observation packed as:
      [species(5)*N_z, T_s(N_z), T_g(N_z)]  => channels = 5 + 2
    """

    def __init__(
        self,
        *,
        N_z: int = 20,
        n_species: int = 5,
        conv_channels: Tuple[int, ...] = (32, 64),
        kernel_size: int = 3,
        activation: nn.Module = nn.ReLU(),
        pool: str = "avg",
    ):
        super().__init__()
        self.N_z = int(N_z)
        self.n_species = int(n_species)
        self.in_channels = self.n_species + 2

        layers: List[nn.Module] = []
        c_in = self.in_channels
        pad = kernel_size // 2
        for c_out in conv_channels:
            layers.append(nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=pad))
            layers.append(activation)
            c_in = c_out
        self.conv = nn.Sequential(*layers)

        if pool not in {"avg", "max"}:
            raise ValueError(f"Unsupported pool={pool}")
        self.pool = pool
        self.out_dim = int(conv_channels[-1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, 140]
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        B = obs.shape[0]
        N_z = self.N_z
        ns = self.n_species
        expected = ns * N_z + 2 * N_z
        if obs.shape[1] != expected:
            raise ValueError(f"Conv1DEncoder expected obs_dim={expected}, got {obs.shape[1]}")

        c = obs[:, : ns * N_z].reshape(B, ns, N_z)
        T_s = obs[:, ns * N_z : ns * N_z + N_z].reshape(B, 1, N_z)
        T_g = obs[:, ns * N_z + N_z : ns * N_z + 2 * N_z].reshape(B, 1, N_z)
        x = torch.cat([c, T_s, T_g], dim=1)

        y = self.conv(x)
        if self.pool == "avg":
            y = y.mean(dim=-1)
        else:
            y = y.max(dim=-1).values
        return y


class ObsAdapter:
    """Maps raw 140D state to model input space.

    - mlp/conv: identity (140D)
    - hybrid: hand-crafted features (default 5D)
    """

    def __init__(self, *, arch: Arch, N_z: int = 20, c_in_nominal: float = 0.15):
        self.arch: Arch = arch
        self.N_z = int(N_z)
        self.c_in_nominal = float(c_in_nominal)

    @property
    def obs_dim(self) -> int:
        if self.arch in ("mlp", "conv"):
            return 7 * self.N_z
        # alpha_out, T_s_out, T_g_out, dT_s, dT_g
        return 5

    def transform(self, raw_obs: np.ndarray) -> np.ndarray:
        raw_obs = np.asarray(raw_obs, dtype=np.float32)
        if self.arch in ("mlp", "conv"):
            return raw_obs

        N_z = self.N_z
        c_kao_out = float(raw_obs[N_z - 1])
        alpha = float(np.clip(1.0 - c_kao_out / self.c_in_nominal, 0.0, 1.0))

        off = 5 * N_z
        T_s = raw_obs[off : off + N_z]
        T_g = raw_obs[off + N_z : off + 2 * N_z]

        T_s_out = float(T_s[-1])
        T_g_out = float(T_g[-1])
        dT_s = float(T_s[-1] - T_s[0])
        dT_g = float(T_g[-1] - T_g[0])
        return np.asarray([alpha, T_s_out, T_g_out, dT_s, dT_g], dtype=np.float32)


class SquashedGaussianPolicy(nn.Module):
    """Diagonal Gaussian with tanh squashing mapped to per-dim [low, high]."""

    def __init__(
        self,
        obs_dim: int,
        act_low: np.ndarray,
        act_high: np.ndarray,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        encoder: Optional[nn.Module] = None,
        encoder_out_dim: Optional[int] = None,
        log_std_init: float = -0.5,
    ):
        super().__init__()
        act_low = np.asarray(act_low, dtype=np.float32)
        act_high = np.asarray(act_high, dtype=np.float32)
        assert act_low.shape == act_high.shape

        self.act_dim = int(act_low.size)

        self.register_buffer("act_low", torch.as_tensor(act_low))
        self.register_buffer("act_high", torch.as_tensor(act_high))
        self.register_buffer("act_mid", 0.5 * (self.act_low + self.act_high))
        self.register_buffer("act_scale", 0.5 * (self.act_high - self.act_low))

        self.encoder = encoder
        in_dim = int(encoder_out_dim) if encoder is not None else int(obs_dim)
        self.mu_net = MLP(in_dim, self.act_dim, hidden_sizes=hidden_sizes, activation=nn.Tanh())
        self.log_std = nn.Parameter(torch.ones(self.act_dim, dtype=torch.float32) * log_std_init)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs) if self.encoder is not None else obs

    def _distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        feat = self._features(obs)
        mu = self.mu_net(feat)
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mu, std)

    def _squash(self, raw: torch.Tensor) -> torch.Tensor:
        y = torch.tanh(raw)
        return self.act_mid + self.act_scale * y

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Log-prob of an already-squashed action in [low, high]."""
        if action.ndim == 1:
            action = action.unsqueeze(0)

        y = (action - self.act_mid) / (self.act_scale + 1e-8)
        y = torch.clamp(y, -1.0 + 1e-6, 1.0 - 1e-6)
        raw = 0.5 * (torch.log1p(y) - torch.log1p(-y))

        dist = self._distribution(obs)
        log_prob_raw = dist.log_prob(raw)
        log_det = torch.log(self.act_scale * (1.0 - y.pow(2)) + 1e-6)
        return (log_prob_raw - log_det).sum(dim=-1)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self._distribution(obs)
        raw = dist.rsample()
        action = self._squash(raw)

        log_prob_raw = dist.log_prob(raw)
        log_det = torch.log(self.act_scale * (1.0 - torch.tanh(raw).pow(2)) + 1e-6)
        log_prob = (log_prob_raw - log_det).sum(dim=-1)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self._distribution(obs)
        raw = dist.mean
        action = self._squash(raw)
        return action


class ValueNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        encoder: Optional[nn.Module] = None,
        encoder_out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = encoder
        in_dim = int(encoder_out_dim) if encoder is not None else int(obs_dim)
        self.v = MLP(in_dim, 1, hidden_sizes=hidden_sizes, activation=nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(obs) if self.encoder is not None else obs
        return self.v(feat).squeeze(-1)


@torch.no_grad()
def _compute_gae(
    rews: np.ndarray,
    vals: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        delta = rews[t] + gamma * vals[t + 1] * next_nonterminal - vals[t]
        last = delta + gamma * lam * next_nonterminal * last
        adv[t] = last
    ret = adv + vals[:-1]
    return adv, ret


@dataclass
class PPOConfig:
    seed: int = 0
    total_steps: int = 200_000
    rollout_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    lr_policy: float = 3e-4
    lr_value: float = 1e-3
    train_epochs: int = 10
    minibatch_size: int = 256
    max_grad_norm: float = 1.0
    hidden_sizes: Tuple[int, ...] = (256, 256)
    device: str = "cpu"
    arch: Arch = "mlp"


def evaluate_policy(
    env,
    policy: nn.Module,
    obs_norm: Optional[ObsNormalizer],
    obs_adapter: Optional[ObsAdapter],
    n_episodes: int,
    device: str,
) -> Dict[str, float]:
    dev = torch.device(device)
    energies: List[float] = []
    violations: List[int] = []
    final_alpha: List[float] = []

    for ep in range(n_episodes):
        obs = env.reset(seed=ep)
        done = False
        ep_energy = 0.0
        ep_vio = 0
        info = {}

        while not done:
            model_obs = obs_adapter.transform(obs) if obs_adapter is not None else obs
            obs_in = obs_norm.normalize(model_obs) if obs_norm is not None else model_obs
            obs_t = _to_tensor(obs_in, dev).unsqueeze(0)
            with torch.no_grad():
                act = policy.deterministic(obs_t).squeeze(0).cpu().numpy()

            obs, rew, done, info = env.step(act)
            ep_energy += float(info.get("energy", 0.0))
            if float(info.get("violation", 0.0)) > 0.0:
                ep_vio += 1

        energies.append(ep_energy)
        violations.append(ep_vio)
        final_alpha.append(float(info.get("alpha", 0.0)))

    return {
        "mean_energy": float(np.mean(energies)),
        "mean_violations": float(np.mean(violations)),
        "mean_final_conversion": float(np.mean(final_alpha)),
    }


def _cold_start_state_vector(N_z: int = 20) -> np.ndarray:
    # Matches SurrogateCalcinerEnv.reset()
    n_species = 5
    c = np.zeros((n_species, N_z), dtype=np.float32)
    c[0, :] = 0.15
    c[1, :] = 0.1
    c[2, :] = 0.05
    c[3, :] = 18.0
    c[4, :] = 0.1
    T_s = np.ones(N_z, dtype=np.float32) * 600.0
    T_g = np.ones(N_z, dtype=np.float32) * 600.0
    return np.concatenate([c.flatten(), T_s, T_g]).astype(np.float32)


def _compute_conversion_from_state(x: np.ndarray, N_z: int = 20, c_in_nominal: float = 0.15) -> float:
    c_kao_out = float(x[N_z - 1])
    alpha = 1.0 - c_kao_out / c_in_nominal
    return float(np.clip(alpha, 0.0, 1.0))


def evaluate_policy_on_physics(
    simulator,
    policy: nn.Module,
    obs_norm: Optional[ObsNormalizer],
    episode_length: int,
    alpha_min: float,
    control_T_s: bool,
    device: str,
    n_episodes: int = 5,
    obs_adapter: Optional[ObsAdapter] = None,
) -> Dict[str, float]:
    dev = torch.device(device)

    energies: List[float] = []
    violations: List[int] = []
    final_alpha: List[float] = []

    for ep in range(n_episodes):
        x = _cold_start_state_vector(simulator.N_z)
        ep_energy = 0.0
        ep_vio = 0
        alpha = 0.0

        for _ in range(episode_length):
            model_obs = obs_adapter.transform(x) if obs_adapter is not None else x
            obs_in = obs_norm.normalize(model_obs) if obs_norm is not None else model_obs
            obs_t = _to_tensor(obs_in, dev).unsqueeze(0)
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

            alpha = _compute_conversion_from_state(x, simulator.N_z)
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


def train_ppo(env, cfg: PPOConfig) -> Tuple[SquashedGaussianPolicy, ObsNormalizer, ObsAdapter, Dict[str, List[float]]]:
    dev = torch.device(cfg.device)
    _set_seed(cfg.seed)

    raw_obs_dim = int(getattr(env, "state_dim"))
    act_dim = int(getattr(env, "action_dim"))

    N_z = int(getattr(env, "N_z", 20))
    # For hybrid we use c_in_nominal=0.15 (matches SurrogateCalcinerEnv)
    obs_adapter = ObsAdapter(arch=cfg.arch, N_z=N_z, c_in_nominal=0.15)
    obs_dim = int(obs_adapter.obs_dim)

    if cfg.arch in ("mlp", "conv") and raw_obs_dim != obs_dim:
        # Identity adapter should match raw state dim
        raise ValueError(f"Unexpected raw_obs_dim={raw_obs_dim} for arch={cfg.arch} (expected {obs_dim})")

    # Bounds from SurrogateCalcinerEnv
    if act_dim == 1:
        act_low = np.array([float(getattr(env, "T_g_min"))], dtype=np.float32)
        act_high = np.array([float(getattr(env, "T_g_max"))], dtype=np.float32)
    else:
        act_low = np.array([float(getattr(env, "T_g_min")), float(getattr(env, "T_s_min"))], dtype=np.float32)
        act_high = np.array([float(getattr(env, "T_g_max")), float(getattr(env, "T_s_max"))], dtype=np.float32)

    obs_norm = ObsNormalizer(obs_dim)

    if cfg.arch == "conv":
        encoder_pi = Conv1DEncoder(N_z=N_z, n_species=5, conv_channels=(32, 64)).to(dev)
        encoder_v = Conv1DEncoder(N_z=N_z, n_species=5, conv_channels=(32, 64)).to(dev)
        policy = SquashedGaussianPolicy(
            obs_dim,
            act_low,
            act_high,
            hidden_sizes=cfg.hidden_sizes,
            encoder=encoder_pi,
            encoder_out_dim=encoder_pi.out_dim,
        ).to(dev)
        value = ValueNetwork(
            obs_dim,
            hidden_sizes=cfg.hidden_sizes,
            encoder=encoder_v,
            encoder_out_dim=encoder_v.out_dim,
        ).to(dev)
    else:
        policy = SquashedGaussianPolicy(obs_dim, act_low, act_high, hidden_sizes=cfg.hidden_sizes).to(dev)
        value = ValueNetwork(obs_dim, hidden_sizes=cfg.hidden_sizes).to(dev)

    opt_pi = torch.optim.Adam(policy.parameters(), lr=cfg.lr_policy)
    opt_v = torch.optim.Adam(value.parameters(), lr=cfg.lr_value)

    history: Dict[str, List[float]] = {
        "update_return": [],
        "update_energy": [],
        "update_violations": [],
        "update_final_alpha": [],
    }

    steps_done = 0
    obs = env.reset(seed=cfg.seed)

    while steps_done < cfg.total_steps:
        obs_buf: List[np.ndarray] = []
        act_buf: List[np.ndarray] = []
        logp_buf: List[float] = []
        rew_buf: List[float] = []
        done_buf: List[float] = []
        val_buf: List[float] = []

        ep_returns: List[float] = []
        ep_energy: List[float] = []
        ep_vio: List[int] = []
        ep_final_alpha: List[float] = []

        ep_ret = 0.0
        energy = 0.0
        vio = 0
        last_alpha = 0.0

        for _ in range(cfg.rollout_steps):
            model_obs = obs_adapter.transform(obs)
            obs_norm.update(model_obs)
            obs_in = obs_norm.normalize(model_obs).astype(np.float32)

            obs_t = _to_tensor(obs_in, dev).unsqueeze(0)
            with torch.no_grad():
                act_t, logp_t = policy.sample(obs_t)
                v_t = value(obs_t)

            act = act_t.squeeze(0).cpu().numpy().astype(np.float32)
            next_obs, rew, done, info = env.step(act)

            obs_buf.append(obs_in)
            act_buf.append(act)
            logp_buf.append(float(logp_t.cpu().numpy().item()))
            rew_buf.append(float(rew))
            done_buf.append(float(done))
            val_buf.append(float(v_t.cpu().numpy().item()))

            ep_ret += float(rew)
            energy += float(info.get("energy", 0.0))
            if float(info.get("violation", 0.0)) > 0.0:
                vio += 1
            last_alpha = float(info.get("alpha", 0.0))

            obs = next_obs
            steps_done += 1

            if done:
                ep_returns.append(ep_ret)
                ep_energy.append(energy)
                ep_vio.append(vio)
                ep_final_alpha.append(last_alpha)

                obs = env.reset(seed=cfg.seed + steps_done)
                ep_ret = 0.0
                energy = 0.0
                vio = 0
                last_alpha = 0.0

            if steps_done >= cfg.total_steps:
                break

        # bootstrap
        model_obs = obs_adapter.transform(obs)
        obs_norm.update(model_obs)
        obs_in = obs_norm.normalize(model_obs).astype(np.float32)
        obs_t = _to_tensor(obs_in, dev).unsqueeze(0)
        with torch.no_grad():
            last_v = float(value(obs_t).cpu().numpy().item())

        rews = np.asarray(rew_buf, dtype=np.float32)
        dones = np.asarray(done_buf, dtype=np.float32)
        vals = np.asarray(val_buf + [last_v], dtype=np.float32)

        adv, ret = _compute_gae(rews, vals, dones, cfg.gamma, cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = _to_tensor(np.stack(obs_buf), dev)
        act_t = _to_tensor(np.stack(act_buf), dev)
        logp_old_t = _to_tensor(np.asarray(logp_buf, dtype=np.float32), dev)
        adv_t = _to_tensor(adv, dev)
        ret_t = _to_tensor(ret, dev)

        n = obs_t.shape[0]
        idx = np.arange(n)
        for _ in range(cfg.train_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, cfg.minibatch_size):
                mb = idx[start : start + cfg.minibatch_size]
                mb_obs = obs_t[mb]
                mb_act = act_t[mb]
                mb_adv = adv_t[mb]
                mb_ret = ret_t[mb]
                mb_logp_old = logp_old_t[mb]

                mb_logp = policy.log_prob(mb_obs, mb_act)
                ratio = torch.exp(mb_logp - mb_logp_old)
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
                pi_loss = -(torch.min(ratio * mb_adv, clipped * mb_adv)).mean()

                v_pred = value(mb_obs)
                v_loss = F.mse_loss(v_pred, mb_ret)

                if cfg.ent_coef != 0.0:
                    dist = policy._distribution(mb_obs)
                    entropy = dist.entropy().sum(dim=-1).mean()
                else:
                    entropy = torch.tensor(0.0, device=mb_obs.device)

                loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                opt_pi.zero_grad()
                opt_v.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value.parameters()), cfg.max_grad_norm)
                opt_pi.step()
                opt_v.step()

        history["update_return"].append(float(np.mean(ep_returns) if ep_returns else 0.0))
        history["update_energy"].append(float(np.mean(ep_energy) if ep_energy else 0.0))
        history["update_violations"].append(float(np.mean(ep_vio) if ep_vio else 0.0))
        history["update_final_alpha"].append(float(np.mean(ep_final_alpha) if ep_final_alpha else 0.0))

    return policy, obs_norm, obs_adapter, history

