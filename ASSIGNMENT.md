# Assignment: Deep Reinforcement Learning for Flash Calciner Control

## Overview

In this assignment, you will implement deep reinforcement learning algorithms to control a flash clay calciner—an industrial reactor that converts kaolinite clay into metakaolin through thermal dehydroxylation. The control objective is to achieve high conversion while minimizing energy consumption.

The assignment has two parts:

| Part | Problem | State Dim | Algorithms |
|------|---------|-----------|------------|
| **Part 1** | Simplified 1D dynamics | 3 | REINFORCE, PPO, TD3 |
| **Part 2** | Full PDE-based dynamics | 140 | Scale up your best algorithm |

## Background

A flash calciner is an industrial reactor that converts clay into a valuable material used in high-performance concrete. The chemistry is straightforward: heating kaolinite (Al₂Si₂O₅(OH)₄) removes hydroxyl groups, producing metakaolin (Al₂Si₂O₇) and water vapor. This dehydroxylation reaction is endothermic—it absorbs heat—so we must continuously supply energy.

The reactor itself is a 10-meter vertical tube. Clay particles enter at the top and fall downward by gravity, while hot gas (nitrogen heated to 900-1300 K) flows upward. This counter-current arrangement maximizes heat transfer: the hottest gas meets the most-reacted particles at the bottom, while cooler gas preheats incoming clay at the top.

Your control input is the gas inlet temperature $T_{g,in}$. Higher temperatures accelerate the reaction, but consume more energy to heat the gas. Lower temperatures save energy, but risk incomplete conversion—particles may exit before fully reacting. The objective is to achieve a target conversion $\alpha \geq \alpha_{min}$ while minimizing heater power.

Conversion $\alpha$ is simply the fraction of kaolinite that has transformed: $\alpha = 1 - c_{\text{out}} / c_{\text{in}}$, where concentrations are measured at the outlet. For example, $\alpha = 0.95$ means 95% of the clay has converted to metakaolin.

The full physics involves tracking five chemical species (kaolinite, metakaolin, quartz, nitrogen, water) at 20 spatial positions, plus temperature profiles for both solid and gas phases—a 140-dimensional state. Part 1 simplifies this to a scalar model that captures the essential control challenge. Part 2 tackles the full complexity using a neural surrogate.

---

---

## Part 1: Simplified Control Problem (50 points)

### Environment

#### Simplified Dynamics Model

The full flash calciner involves coupled PDEs for mass and energy transport across 20 spatial cells (see `docs/model.md`). For Part 1, we abstract this to a scalar system that captures the essential control challenge.

The **outlet conversion** $\alpha \in [0,1]$ (fraction of kaolinite converted to metakaolin) evolves as a first-order lag approaching a temperature-dependent equilibrium:

$$\alpha_{k+1} = e^{-\Delta t/\tau} \alpha_k + (1 - e^{-\Delta t/\tau}) \, \alpha_{ss}(u_k)$$

where:
- $\tau = 2$ s is the thermal time constant (how fast the reactor responds to temperature changes)
- $\Delta t = 0.5$ s is the control period
- $\alpha_{ss}(u)$ is the **steady-state conversion**: the equilibrium $\alpha$ would reach if gas inlet temperature were held constant at $u$ indefinitely

In other words, $\alpha_{ss}(u) = \lim_{t \to \infty} \alpha(t)$ when applying constant control $u$. The discrete dynamics exponentially relax toward this equilibrium with rate $1/\tau$.

The steady-state conversion follows a **sigmoidal Arrhenius-like relationship**:

$$\alpha_{ss}(T) = \frac{0.999}{1 + \exp(-0.025 \cdot (T - 1000))}$$

This captures two key physics phenomena:

1. **Arrhenius kinetics**: Reaction rate increases exponentially with temperature according to
   $$k(T) = A \exp\left(-\frac{E_a}{RT}\right)$$
   where $E_a$ is the activation energy barrier. At low temperatures, most molecular collisions lack sufficient energy to break the hydroxyl bonds. As $T$ increases, more collisions succeed → faster reaction → higher equilibrium conversion.

2. **Saturation**: Conversion cannot exceed 100% (can't convert more kaolinite than you have), so $\alpha_{ss}(T)$ approaches an asymptote near 1.

Numerically: $\alpha_{ss}(900\text{K}) \approx 50\%$, $\alpha_{ss}(1000\text{K}) \approx 73\%$, $\alpha_{ss}(1261\text{K}) \approx 99.8\%$. 

The inflection point at $T = 1000$ K is where the sigmoid transitions from "slow reaction" to "fast reaction" regimes. Below this temperature, the reactor residence time isn't long enough for particles to fully react; above it, kinetics are fast enough that most particles convert before exiting.

#### Why This Simplification?

The first-order model is a **linear approximation around the dominant eigenvalue** of the linearized PDE system. It preserves:
1. The fundamental tradeoff: higher $T$ → faster conversion but more energy
2. Temporal dynamics: the system doesn't respond instantaneously
3. The constraint satisfaction problem: track a time-varying $\alpha_{min}(t)$

What it ignores: spatial profiles, species concentrations, solid/gas temperature differences. You'll tackle those in Part 2.

#### Environment API

```python
from calciner import CalcinerEnv

env = CalcinerEnv(episode_length=40, dt=0.5)
obs = env.reset()  # Returns [alpha, alpha_min, t/T]
obs, reward, done, info = env.step(action)  # action = T_g_in in [900, 1300]
```

**State space** $\mathcal{S} \subset \mathbb{R}^3$:
| Component | Description | Range |
|-----------|-------------|-------|
| $\alpha$ | Current outlet conversion | $[0, 1]$ |
| $\alpha_{min}$ | Target minimum (constraint) | $[0.90, 0.99]$ |
| $t/T$ | Normalized episode time | $[0, 1]$ |

**Action space** $\mathcal{A} \subset \mathbb{R}$:
- $u = T_{g,in}$: Gas inlet temperature in $[900, 1300]$ K

**Reward function**:
$$r(s, a) = -\underbrace{c \cdot (T_{g,in} - 300)}_{\text{energy cost}} - \underbrace{10 \cdot \max(0, \alpha_{min} - \alpha)^2}_{\text{constraint violation}}$$

The energy cost is linear in temperature (heating gas from ambient). The constraint penalty is a soft barrier—you can violate $\alpha_{min}$ but it's expensive. This is a **constrained MDP** relaxed via penalty methods.

### What to Implement

Start with REINFORCE (15 points). The algorithm should maintain a policy parametrization (linear or small neural network) and use the Monte Carlo return as an estimate of the state-action value. Include a baseline to reduce variance—either a learned value function or the mean return works. Train for at least 200 episodes and track the return over time.

Next implement PPO (20 points). The clipped surrogate objective prevents excessively large policy updates, which is essential for stable learning. You will need both a policy network and a value network. Consider using generalized advantage estimation to further reduce variance, though this is optional. Compare the sample efficiency to REINFORCE—how many episodes does each algorithm need to reach similar performance?

Finally, implement TD3 (15 points). This algorithm uses twin Q-networks to reduce overestimation bias, delayed policy updates to decouple actor from critic training, and target policy smoothing for regularization. You will need a replay buffer to store transitions. TD3 should be more sample-efficient than the policy gradient methods, but requires careful tuning of the exploration noise schedule.

For all three algorithms, compare against the constant-temperature baseline provided in `calciner.ConstantTemperatureController`. This baseline simply sets $T_{g,in} = 1261$ K throughout the episode, achieving high conversion at the cost of maximum energy consumption. Your learned policies should achieve similar conversion while reducing energy use.

Report learning curves showing return versus episode number for each algorithm. Analyze the final policy: what is its average energy consumption, how often does it violate the conversion constraint, and how does the chosen temperature vary with the state? The last question is particularly revealing—does the policy learn to modulate temperature based on current conversion and the target threshold?

---

## Part 2: Full 140-Dimensional Problem (50 points)

The simplified model ignores spatial dynamics. The real calciner is a 10-meter reactor where temperature and concentration profiles evolve along the length. The full state has 140 dimensions: 5 species × 20 spatial cells for concentrations, plus 2 temperatures × 20 cells. Simulating this PDE-based model requires integrating a stiff ODE system at approximately 25 ms per step, which makes RL training impractical without approximation.

The neural surrogate provided in `models/surrogate_model.pt` approximates these dynamics in under 0.02 ms—a 60× speedup. It was trained on 10,000 state transitions sampled from diverse initial conditions (cold starts, partial reaction fronts, near-steady-state), achieving 19% mean relative error on challenging 80-step rollouts with time-varying controls. This accuracy suffices for learning policies, especially since we can validate on the true physics simulator afterward.

We provide a Gym-compatible environment that wraps the surrogate and handles state normalization, conversion computation, and reward shaping. You are free to modify the reward function or experiment with different state representations if you wish, but this is optional. The main task is adapting your RL algorithms to the higher-dimensional state space.

### What to Implement

Adapt your best-performing algorithm from Part 1 to work with the 140-dimensional state (25 points). The state vector concatenates concentrations and temperatures across all spatial cells, so you face a representation learning challenge. A linear policy is unlikely to work—you will need a neural network with sufficient capacity. Consider how to encode spatial structure: should your policy use fully connected layers, or would 1D convolutions (like the surrogate itself uses) better capture the fact that neighboring cells interact?

You may need different hyperparameters than Part 1. The higher dimensionality can slow learning, so you might increase network capacity, adjust learning rates, or change the exploration noise schedule. If you implemented TD3, you might need a larger replay buffer or different batch sizes. If you used PPO, the entropy coefficient and value function loss weighting may require tuning.

Start simple. A reasonable first policy is one that looks only at the outlet conversion (the first element of the state after unpacking) and outputs a temperature based on how far that conversion is from the target. This ignores spatial structure but establishes a baseline. From there, incorporate more of the state: perhaps the temperature profile, or the concentration gradient along the reactor.

### Evaluation

Validate your learned policy on the true physics simulator (15 points), not just the surrogate. This tests whether the policy generalizes beyond the approximation errors inherent in the neural dynamics model. Run multiple episodes with different initial conditions and compare performance to the constant-temperature baseline.

Beyond scalar metrics (energy consumption, constraint violations), visualize the closed-loop behavior. Plot conversion over time, the control trajectory, and—most revealingly—the spatial profiles. Does the learned policy create smooth temperature gradients? Does it avoid creating reaction fronts that could damage equipment? These qualitative assessments matter in process control applications.

The surrogate is differentiable, so policy gradient computation can backpropagate through the dynamics model. Whether you exploit this is up to you—some algorithms naturally benefit from differentiable simulators, while others do not.

---

## Deliverables

1. **Code**: Your implementations in a clean, documented format
2. **Report** (max 6 pages): 
   - Algorithm descriptions
   - Experimental setup
   - Results with plots
   - Discussion of what worked/didn't work

3. **Trained models**: Save your best policy checkpoints

---

## Getting Started

```bash
# Install dependencies
pip install numpy torch matplotlib scipy

# Test the environments
python -c "from calciner import CalcinerEnv; env = CalcinerEnv(); print(env.reset())"

# Check the surrogate
python -c "import torch; print(torch.load('models/surrogate_model.pt', weights_only=False).keys())"
```

## Grading

| Component | Points |
|-----------|--------|
| Part 1: REINFORCE | 15 |
| Part 1: PPO | 20 |
| Part 1: TD3 | 15 |
| Part 2: Environment | 10 |
| Part 2: Algorithm | 25 |
| Part 2: Evaluation | 15 |
| **Total** | **100** |

Partial credit is given. A working but suboptimal implementation is better than no implementation.

---

## References

- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods" (TD3, 2018)
- Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (REINFORCE, 1992)
- `docs/model.md` for flash calciner physics

