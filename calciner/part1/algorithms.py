from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)

# Compute discounted cumulative rewards G_t
def _discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(x))):
        running = x[t] + gamma * running
        out[t] = running
    return out


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
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


class TanhGaussianPolicy(nn.Module):
    """Gaussian policy with tanh squashing mapped to [act_low, act_high]."""

    def __init__(
        self,
        obs_dim: int,
        act_low: float,
        act_high: float,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.act_low = float(act_low)
        self.act_high = float(act_high)
        self.act_mid = 0.5 * (self.act_low + self.act_high)
        self.act_scale = 0.5 * (self.act_high - self.act_low)

        self.mu_net = MLP(obs_dim, 1, hidden_sizes=hidden_sizes, activation=nn.Tanh())
        self.log_std = nn.Parameter(torch.tensor([log_std_init], dtype=torch.float32))

    def _distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mu = self.mu_net(obs)
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mu, std)

    def _squash(self, raw: torch.Tensor) -> torch.Tensor:
        # [-1, 1]
        squashed = torch.tanh(raw)
        # map to [low, high]
        return self.act_mid + self.act_scale * squashed

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Log-prob of an already-squashed action in [act_low, act_high]."""
        if action.ndim == 1:
            action = action.unsqueeze(-1)

        # Map to (-1, 1) then invert tanh via atanh
        y = (action - self.act_mid) / (self.act_scale + 1e-8)
        y = torch.clamp(y, -1.0 + 1e-6, 1.0 - 1e-6)
        raw = 0.5 * (torch.log1p(y) - torch.log1p(-y))  # atanh(y)

        dist = self._distribution(obs)
        log_prob_raw = dist.log_prob(raw)
        log_det = torch.log(self.act_scale * (1.0 - y.pow(2)) + 1e-6)
        return (log_prob_raw - log_det).sum(dim=-1)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob) with tanh correction."""
        dist = self._distribution(obs)
        raw = dist.rsample()
        action = self._squash(raw)

        # log prob with tanh change-of-variables
        # a = mid + scale * tanh(raw)
        # log|da/draw| = log(scale * (1 - tanh(raw)^2))
        log_prob_raw = dist.log_prob(raw)
        # add small epsilon for numerical stability
        log_det = torch.log(self.act_scale * (1.0 - torch.tanh(raw).pow(2)) + 1e-6)
        log_prob = (log_prob_raw - log_det).sum(dim=-1)
        return action.squeeze(-1), log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self._distribution(obs)
        raw = dist.mean
        action = self._squash(raw)
        return action.squeeze(-1)


class ValueNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
    ):
        super().__init__()
        self.v = MLP(obs_dim, 1, hidden_sizes=hidden_sizes, activation=nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.q = MLP(obs_dim + 1, 1, hidden_sizes=hidden_sizes, activation=nn.ReLU())

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        if act.ndim == 1:
            act = act.unsqueeze(-1)
        x = torch.cat([obs, act], dim=-1)
        return self.q(x).squeeze(-1)


@dataclass
class ReinforceConfig:
    seed: int = 0
    episodes: int = 300
    gamma: float = 0.99
    lr_policy: float = 3e-4
    lr_value: float = 1e-3
    hidden_sizes: Tuple[int, ...] = (64, 64)
    device: str = "cpu"


@dataclass
class PPOConfig:
    seed: int = 0
    total_steps: int = 40 * 300
    rollout_steps: int = 40 * 10
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
    hidden_sizes: Tuple[int, ...] = (64, 64)
    device: str = "cpu"


@dataclass
class TD3Config:
    seed: int = 0
    total_steps: int = 40 * 300
    start_steps: int = 1000
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 20.0
    noise_clip: float = 50.0
    expl_noise: float = 40.0
    policy_delay: int = 2
    batch_size: int = 256
    replay_size: int = 200_000
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    hidden_sizes_actor: Tuple[int, ...] = (256, 256)
    hidden_sizes_critic: Tuple[int, ...] = (256, 256)
    device: str = "cpu"


class ReplayBuffer:
    def __init__(self, obs_dim: int, size: int, device: torch.device):
        self.device = device
        self.size = int(size)
        self.obs_buf = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.size,), dtype=np.float32)
        self.rew_buf = np.zeros((self.size,), dtype=np.float32)
        self.done_buf = np.zeros((self.size,), dtype=np.float32)
        self.ptr = 0
        self.full = False

    def add(self, obs, act, rew, next_obs, done) -> None:
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr += 1
        if self.ptr >= self.size:
            self.ptr = 0
            self.full = True

    def __len__(self) -> int:
        return self.size if self.full else self.ptr

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        n = len(self)
        idx = np.random.randint(0, n, size=batch_size)
        batch = {
            "obs": _to_tensor(self.obs_buf[idx], self.device),
            "act": _to_tensor(self.act_buf[idx], self.device),
            "rew": _to_tensor(self.rew_buf[idx], self.device),
            "next_obs": _to_tensor(self.next_obs_buf[idx], self.device),
            "done": _to_tensor(self.done_buf[idx], self.device),
        }
        return batch


def evaluate_policy(env, policy: nn.Module, n_episodes: int = 5, device: str = "cpu") -> Dict[str, float]:
    dev = torch.device(device)

    total_energy: List[float] = []
    violations: List[int] = []
    final_alpha: List[float] = []

    for ep in range(n_episodes):
        obs = env.reset(seed=ep)
        done = False
        energy = 0.0
        vio = 0
        info = {}
        while not done:
            obs_t = _to_tensor(obs, dev).unsqueeze(0)
            with torch.no_grad():
                if hasattr(policy, "deterministic"):
                    act = policy.deterministic(obs_t).cpu().numpy().item()
                else:
                    act = policy(obs_t).cpu().numpy().item()
            obs, reward, done, info = env.step(act)
            energy += float(info.get("power", 0.0))
            if info.get("alpha", 0.0) < env.alpha_min[env.t - 1]:
                vio += 1

        total_energy.append(energy)
        violations.append(vio)
        final_alpha.append(float(info.get("alpha", 0.0)))

    return {
        "mean_energy": float(np.mean(total_energy)),
        "mean_violations": float(np.mean(violations)),
        "mean_final_conversion": float(np.mean(final_alpha)),
    }



# ———————————————————————— Algo REINFORCE —————————————————————— #

def train_reinforce(env, cfg: ReinforceConfig) -> Tuple[TanhGaussianPolicy, Dict[str, List[float]]]:
    dev = torch.device(cfg.device)
    _set_seed(cfg.seed)

    obs_dim = 3
    policy = TanhGaussianPolicy(obs_dim, env.u_min, env.u_max, hidden_sizes=cfg.hidden_sizes).to(dev)
    value = ValueNetwork(obs_dim, hidden_sizes=cfg.hidden_sizes).to(dev)

    opt_pi = torch.optim.Adam(policy.parameters(), lr=cfg.lr_policy)
    opt_v = torch.optim.Adam(value.parameters(), lr=cfg.lr_value)

    history: Dict[str, List[float]] = {"ep_return": [], "ep_energy": [], "ep_violations": []}

    for ep in range(cfg.episodes):
        obs = env.reset(seed=cfg.seed + ep)
        done = False

        obs_list: List[np.ndarray] = []
        act_list: List[float] = []
        logp_list: List[float] = []
        rew_list: List[float] = []
        energy = 0.0
        vio = 0

        while not done:
            obs_t = _to_tensor(obs, dev).unsqueeze(0)
            with torch.no_grad():
                act_t, logp_t = policy.sample(obs_t)
            act = float(act_t.cpu().numpy().item())
            obs2, rew, done, info = env.step(act)

            obs_list.append(obs)
            act_list.append(act)
            logp_list.append(float(logp_t.cpu().numpy().item()))
            rew_list.append(float(rew))

            energy += float(info.get("power", 0.0))
            if info.get("alpha", 0.0) < env.alpha_min[env.t - 1]:
                vio += 1

            obs = obs2

        rews = np.array(rew_list, dtype=np.float32)
        returns = _discount_cumsum(rews, cfg.gamma)

        obs_t = _to_tensor(np.stack(obs_list), dev)
        returns_t = _to_tensor(returns, dev)

        # baseline value loss
        v_pred = value(obs_t)
        v_loss = F.mse_loss(v_pred, returns_t)
        opt_v.zero_grad()
        v_loss.backward()
        opt_v.step()

        # policy loss
        with torch.no_grad():
            adv = returns_t - value(obs_t)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        act_t = _to_tensor(np.array(act_list, dtype=np.float32), dev)
        logp_t = policy.log_prob(obs_t, act_t)

        pi_loss = -(logp_t * adv).mean()
        opt_pi.zero_grad()
        pi_loss.backward()
        opt_pi.step()

        history["ep_return"].append(float(np.sum(rews)))
        history["ep_energy"].append(float(energy))
        history["ep_violations"].append(float(vio))

    return policy, history


# ———————————————————————— Algo PPO —————————————————————— #

@torch.no_grad()
def _compute_gae(rews: np.ndarray, vals: np.ndarray, dones: np.ndarray, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        next_value = vals[t + 1]
        delta = rews[t] + gamma * next_value * next_nonterminal - vals[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + vals[:-1]
    return adv, returns


def train_ppo(env, cfg: PPOConfig) -> Tuple[TanhGaussianPolicy, Dict[str, List[float]]]:
    dev = torch.device(cfg.device)
    _set_seed(cfg.seed)

    obs_dim = 3
    policy = TanhGaussianPolicy(obs_dim, env.u_min, env.u_max, hidden_sizes=cfg.hidden_sizes).to(dev)
    value = ValueNetwork(obs_dim, hidden_sizes=cfg.hidden_sizes).to(dev)

    opt_pi = torch.optim.Adam(policy.parameters(), lr=cfg.lr_policy)
    opt_v = torch.optim.Adam(value.parameters(), lr=cfg.lr_value)

    history: Dict[str, List[float]] = {"update_return": [], "update_energy": [], "update_violations": []}

    steps_done = 0
    while steps_done < cfg.total_steps:
        # rollout
        obs_buf: List[np.ndarray] = []
        act_buf: List[float] = []
        logp_buf: List[float] = []
        rew_buf: List[float] = []
        done_buf: List[float] = []
        val_buf: List[float] = []

        ep_returns: List[float] = []
        ep_energy: List[float] = []
        ep_vio: List[int] = []

        obs = env.reset(seed=cfg.seed + steps_done)
        ep_ret = 0.0
        energy = 0.0
        vio = 0

        for _ in range(cfg.rollout_steps):
            obs_t = _to_tensor(obs, dev).unsqueeze(0)
            with torch.no_grad():
                act_t, logp_t = policy.sample(obs_t)
                v_t = value(obs_t)

            act = float(act_t.cpu().numpy().item())
            obs2, rew, done, info = env.step(act)

            obs_buf.append(obs)
            act_buf.append(act)
            logp_buf.append(float(logp_t.cpu().numpy().item()))
            rew_buf.append(float(rew))
            done_buf.append(float(done))
            val_buf.append(float(v_t.cpu().numpy().item()))

            ep_ret += float(rew)
            energy += float(info.get("power", 0.0))
            if info.get("alpha", 0.0) < env.alpha_min[env.t - 1]:
                vio += 1

            obs = obs2
            steps_done += 1

            if done:
                ep_returns.append(ep_ret)
                ep_energy.append(energy)
                ep_vio.append(vio)
                obs = env.reset(seed=cfg.seed + steps_done)
                ep_ret = 0.0
                energy = 0.0
                vio = 0

            if steps_done >= cfg.total_steps:
                break

        # bootstrap value
        obs_t = _to_tensor(obs, dev).unsqueeze(0)
        with torch.no_grad():
            last_v = float(value(obs_t).cpu().numpy().item())

        rews = np.array(rew_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.float32)
        vals = np.array(val_buf + [last_v], dtype=np.float32)
        adv, ret = _compute_gae(rews, vals, dones, cfg.gamma, cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = _to_tensor(np.stack(obs_buf), dev)
        act_t = _to_tensor(np.array(act_buf, dtype=np.float32), dev)
        logp_old_t = _to_tensor(np.array(logp_buf, dtype=np.float32), dev)
        adv_t = _to_tensor(adv, dev)
        ret_t = _to_tensor(ret, dev)

        n = obs_t.shape[0]
        idx = np.arange(n)

        for _ in range(cfg.train_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, cfg.minibatch_size):
                mb = idx[start : start + cfg.minibatch_size]
                mb_obs = obs_t[mb]
                mb_adv = adv_t[mb]
                mb_ret = ret_t[mb]
                mb_logp_old = logp_old_t[mb]

                mb_act = act_t[mb]

                mb_logp = policy.log_prob(mb_obs, mb_act)
                ratio = torch.exp(mb_logp - mb_logp_old)

                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
                pi_loss = -(torch.min(ratio * mb_adv, clipped * mb_adv)).mean()

                v_pred = value(mb_obs)
                v_loss = F.mse_loss(v_pred, mb_ret)

                # Entropy bonus (use underlying Normal entropy; squash correction ignored)
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

    return policy, history


# ———————————————————————— Algo TD3 —————————————————————— #

def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau)
            tp.data.add_(tau * sp.data)


def train_td3(env, cfg: TD3Config) -> Tuple[nn.Module, Dict[str, List[float]]]:
    dev = torch.device(cfg.device)
    _set_seed(cfg.seed)

    obs_dim = 3

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = MLP(obs_dim, 1, hidden_sizes=cfg.hidden_sizes_actor, activation=nn.ReLU())

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            raw = self.net(obs)
            # map tanh -> [u_min, u_max]
            act_mid = 0.5 * (env.u_min + env.u_max)
            act_scale = 0.5 * (env.u_max - env.u_min)
            return (act_mid + act_scale * torch.tanh(raw)).squeeze(-1)

        def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
            return self.forward(obs)

    actor = Actor().to(dev)
    actor_target = Actor().to(dev)
    actor_target.load_state_dict(actor.state_dict())

    act_mid = 0.5 * (env.u_min + env.u_max)
    act_scale = 0.5 * (env.u_max - env.u_min)

    class TD3QNetwork(QNetwork):
        def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
            # Normalize action from Kelvin to roughly [-1, 1] to avoid scale mismatch.
            act_norm = (act - act_mid) / (act_scale + 1e-8)
            return super().forward(obs, act_norm)

    q1 = TD3QNetwork(obs_dim, hidden_sizes=cfg.hidden_sizes_critic).to(dev)
    q2 = TD3QNetwork(obs_dim, hidden_sizes=cfg.hidden_sizes_critic).to(dev)
    q1_t = TD3QNetwork(obs_dim, hidden_sizes=cfg.hidden_sizes_critic).to(dev)
    q2_t = TD3QNetwork(obs_dim, hidden_sizes=cfg.hidden_sizes_critic).to(dev)
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())

    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_c = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg.lr_critic)

    rb = ReplayBuffer(obs_dim, cfg.replay_size, dev)

    history: Dict[str, List[float]] = {"episode_return": [], "episode_energy": [], "episode_violations": []}

    obs = env.reset(seed=cfg.seed)
    ep_ret = 0.0
    ep_energy = 0.0
    ep_vio = 0

    for t in range(cfg.total_steps):
        if t < cfg.start_steps:
            act = float(np.random.uniform(env.u_min, env.u_max))
        else:
            obs_t = _to_tensor(obs, dev).unsqueeze(0)
            with torch.no_grad():
                act = float(actor(obs_t).cpu().numpy().item())
            act += float(np.random.normal(0.0, cfg.expl_noise))
            act = float(np.clip(act, env.u_min, env.u_max))

        next_obs, rew, done, info = env.step(act)

        rb.add(obs, act, rew, next_obs, float(done))

        ep_ret += float(rew)
        ep_energy += float(info.get("power", 0.0))
        if info.get("alpha", 0.0) < env.alpha_min[env.t - 1]:
            ep_vio += 1

        obs = next_obs

        if done:
            history["episode_return"].append(float(ep_ret))
            history["episode_energy"].append(float(ep_energy))
            history["episode_violations"].append(float(ep_vio))
            obs = env.reset(seed=cfg.seed + t + 1)
            ep_ret = 0.0
            ep_energy = 0.0
            ep_vio = 0

        if len(rb) < cfg.batch_size:
            continue

        batch = rb.sample(cfg.batch_size)
        b_obs = batch["obs"]
        b_act = batch["act"]
        b_rew = batch["rew"]
        b_next = batch["next_obs"]
        b_done = batch["done"]

        with torch.no_grad():
            next_act = actor_target(b_next)
            noise = torch.clamp(
                torch.randn_like(next_act) * cfg.policy_noise,
                -cfg.noise_clip,
                cfg.noise_clip,
            )
            next_act = torch.clamp(next_act + noise, env.u_min, env.u_max)

            q1_next = q1_t(b_next, next_act)
            q2_next = q2_t(b_next, next_act)
            q_next = torch.min(q1_next, q2_next)
            target_q = b_rew + cfg.gamma * (1.0 - b_done) * q_next

        q1_pred = q1(b_obs, b_act)
        q2_pred = q2(b_obs, b_act)
        critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        opt_c.zero_grad()
        critic_loss.backward()
        opt_c.step()

        if t % cfg.policy_delay == 0:
            actor_loss = -q1(b_obs, actor(b_obs)).mean()
            opt_a.zero_grad()
            actor_loss.backward()
            opt_a.step()

            _soft_update(actor_target, actor, cfg.tau)
            _soft_update(q1_t, q1, cfg.tau)
            _soft_update(q2_t, q2, cfg.tau)

    return actor, history


