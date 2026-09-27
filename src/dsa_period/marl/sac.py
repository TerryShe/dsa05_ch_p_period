"""SAC-Discrete as used by XR-SBPC (EXP-DSA05-005), under DSA05's information structure.

One policy network and twin Q networks with targets, shared by all UEs; every
UE's transition enters one proportional prioritized replay buffer. Inputs are the
flattened features of EXP-DSA05-001's actor (``flat_features``), so a UE sees
exactly what A's actor sees in the same slot; the networks are feedforward.

Every UE's transition is stored every slot, idle UEs included (their only legal
action is OFF). The Q loss uses all transitions, so bootstrapping from an idle
next state uses a trained value; the actor and temperature losses use only
transitions whose state had more than one legal action. Expectations over the
policy (targets, actor loss, entropy) run over legal actions only.
"""

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from dsa_period.marl.features import DualSettings, LocalMemory, decode_actions, legal_mask, stack
from dsa_period.marl.policy import sample_from_logits
from dsa_period.physics import PhysicalConfig, PowerMenu

SAC_CHECKPOINT_SCHEMA = 1


@dataclass(frozen=True)
class SacSettings:
    """Hyperparameters of XR-SBPC §V-A; the PER parameters follow Schaul et al."""

    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 512
    buffer_size: int = 10_000
    tau: float = 0.005
    target_entropy: float = 0.05
    alpha_init: float = 1.0
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_eps: float = 1e-6
    hidden: tuple[int, ...] = (400, 300, 200, 100)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden", tuple(int(h) for h in self.hidden))
        if not 0 <= self.gamma < 1 or self.batch_size > self.buffer_size or self.alpha_init <= 0:
            raise ValueError("need 0 <= gamma < 1, batch_size <= buffer_size, alpha_init > 0")


def mlp(inputs: int, outputs: int, hidden: tuple[int, ...]) -> nn.Sequential:
    """LeakyReLU on every hidden layer but the last, Tanh on the last (XR-SBPC §V-A)."""
    layers: list[nn.Module] = []
    width = inputs
    for i, size in enumerate(hidden):
        layers += [nn.Linear(width, size), nn.Tanh() if i == len(hidden) - 1 else nn.LeakyReLU()]
        width = size
    layers.append(nn.Linear(width, outputs))
    return nn.Sequential(*layers)


def flat_features(memory: LocalMemory, obs) -> NDArray[np.float32]:
    """(N, K * token_dim + context_dim): A's actor features of this slot, flattened."""
    tokens, context = memory.features(obs)
    return np.concatenate((tokens.reshape(tokens.shape[0], -1), context), axis=1)


def masked_policy(logits: torch.Tensor, legal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Probabilities and log-probabilities over legal actions; both are 0 on illegal ones."""
    log_probs = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=-1)
    log_probs = torch.where(legal, log_probs, torch.zeros_like(log_probs))
    return log_probs.exp() * legal, log_probs


def soft_targets(
    reward: torch.Tensor,
    next_logits: torch.Tensor,
    next_legal: torch.Tensor,
    next_q1: torch.Tensor,
    next_q2: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    """y = r + gamma * sum_a' pi(a'|s') [min_j Q'_j(s', a') - alpha log pi(a'|s')]."""
    probs, log_probs = masked_policy(next_logits, next_legal)
    q = torch.minimum(next_q1, next_q2).masked_fill(~next_legal, 0.0)
    return reward + gamma * (probs * (q - alpha * log_probs)).sum(-1)


def actor_objective(
    logits: torch.Tensor, legal: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row sum_a pi(a|s) [alpha log pi(a|s) - min_j Q_j(s, a)] and the policy entropy."""
    probs, log_probs = masked_policy(logits, legal)
    q = torch.minimum(q1, q2).masked_fill(~legal, 0.0)
    return (probs * (alpha * log_probs - q)).sum(-1), -(probs * log_probs).sum(-1)


class PrioritizedReplay:
    """Proportional prioritized replay (Schaul et al.) over a circular buffer."""

    def __init__(
        self, capacity: int, obs_dim: int, actions: int, alpha: float, eps: float, seed
    ) -> None:
        self.capacity, self.alpha, self.eps = capacity, alpha, eps
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.action = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.legal = np.zeros((capacity, actions), np.bool_)
        self.next_legal = np.zeros((capacity, actions), np.bool_)
        self.priority = np.zeros(capacity, np.float64)
        self.size, self.cursor, self.max_priority = 0, 0, 1.0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, legal, next_legal) -> None:
        """Append rows in order, overwriting the oldest; new rows get the max priority."""
        rows = len(action)
        if rows > self.capacity:
            raise ValueError("cannot add more rows than the buffer holds")
        index = (self.cursor + np.arange(rows)) % self.capacity
        self.obs[index], self.next_obs[index] = obs, next_obs
        self.action[index], self.reward[index] = action, reward
        self.legal[index], self.next_legal[index] = legal, next_legal
        self.priority[index] = self.max_priority
        self.cursor = int((self.cursor + rows) % self.capacity)
        self.size = min(self.size + rows, self.capacity)

    def sample(self, batch: int, beta: float) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Indices drawn with probability proportional to priority**alpha, and IS weights."""
        scaled = self.priority[: self.size] ** self.alpha
        probs = scaled / scaled.sum()
        index = self.rng.choice(self.size, size=batch, p=probs)
        weights = (self.size * probs[index]) ** (-beta)
        return index, (weights / weights.max()).astype(np.float32)

    def update(self, index: NDArray[np.int64], td: NDArray) -> None:
        priority = np.abs(np.asarray(td, np.float64)) + self.eps
        self.priority[index] = priority
        self.max_priority = max(self.max_priority, float(priority.max()))


class SacAgent:
    def __init__(self, obs_dim: int, actions: int, settings: SacSettings, seed: int) -> None:
        torch.manual_seed(seed)
        self.settings, self.obs_dim, self.actions = settings, obs_dim, actions
        h = settings.hidden
        self.policy = mlp(obs_dim, actions, h)
        self.q1, self.q2 = mlp(obs_dim, actions, h), mlp(obs_dim, actions, h)
        self.q1_target, self.q2_target = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        for net in (self.q1_target, self.q2_target):
            net.requires_grad_(False)
        self.log_alpha = torch.tensor([np.log(settings.alpha_init)], requires_grad=True)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=settings.lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=settings.lr
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=settings.lr)

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.detach().exp())

    def update(self, buffer: PrioritizedReplay, beta: float) -> dict:
        s = self.settings
        index, weights = buffer.sample(s.batch_size, beta)
        obs = torch.from_numpy(buffer.obs[index])
        next_obs = torch.from_numpy(buffer.next_obs[index])
        action = torch.from_numpy(buffer.action[index])
        reward = torch.from_numpy(buffer.reward[index])
        legal = torch.from_numpy(buffer.legal[index])
        next_legal = torch.from_numpy(buffer.next_legal[index])
        weight = torch.from_numpy(weights)
        alpha = self.alpha
        with torch.no_grad():
            target = soft_targets(
                reward,
                self.policy(next_obs),
                next_legal,
                self.q1_target(next_obs),
                self.q2_target(next_obs),
                alpha,
                s.gamma,
            )
        q1 = self.q1(obs).gather(1, action[:, None]).squeeze(1)
        q2 = self.q2(obs).gather(1, action[:, None]).squeeze(1)
        q_loss = (weight * 0.5 * ((q1 - target) ** 2 + (q2 - target) ** 2)).mean()
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()
        td = (0.5 * ((q1 - target).abs() + (q2 - target).abs())).detach().numpy()
        buffer.update(index, td)

        # Actor and temperature: only states with a real choice (a packet to send).
        active = legal.sum(-1) > 1
        metrics = {
            "q_loss": q_loss.item(),
            "alpha": alpha,
            "active_share": float(active.float().mean()),
        }
        if active.any():
            with torch.no_grad():
                q1_all, q2_all = self.q1(obs[active]), self.q2(obs[active])
            objective, entropy = actor_objective(
                self.policy(obs[active]), legal[active], q1_all, q2_all, alpha
            )
            actor_loss = objective.mean()
            self.policy_opt.zero_grad()
            actor_loss.backward()
            self.policy_opt.step()
            alpha_loss = (self.log_alpha.exp() * (entropy.detach() - s.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            metrics |= {"actor_loss": actor_loss.item(), "entropy": entropy.mean().item()}
        with torch.no_grad():
            for net, target_net in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for p, tp in zip(net.parameters(), target_net.parameters(), strict=True):
                    tp.mul_(1.0 - s.tau).add_(s.tau * p)
        return metrics


class SacPolicy:
    """A trained SAC policy network behind the protocol ``Policy`` interface.

    Features come from the UE's own ``LocalMemory`` (A's actor inputs, fixed
    variant, no dual); sampling uses the protocol RNGs exactly like ``MarlPolicy``.
    """

    def __init__(
        self,
        network: nn.Module,
        config: PhysicalConfig,
        menu: PowerMenu,
        epsilon: float,
        name: str = "sac",
    ) -> None:
        self.network, self.config, self.menu, self.name = network, config, menu, name
        self.actions = 1 + config.num_subchannels * (menu.num_levels - 1)
        self._memory = LocalMemory(
            config, menu, config.num_ues, DualSettings("fixed", epsilon=epsilon)
        )

    def reset(self, rngs) -> None:
        if len(rngs) != self.config.num_ues:
            raise ValueError("need one generator per UE")
        self._rngs = list(rngs)
        self._memory.reset(np.zeros(self.config.num_ues))

    def act(self, observations) -> NDArray[np.int64]:
        obs = stack(observations)
        self._memory.update_duals(self._memory.observe(obs))
        features = flat_features(self._memory, obs)
        with torch.no_grad():
            logits = self.network(torch.from_numpy(features)).numpy()
        legal = legal_mask(obs, self.actions)
        active = obs.remaining > 0
        index = np.zeros(self.config.num_ues, dtype=np.int64)
        if active.any():
            uniforms = np.array([self._rngs[ue].random() for ue in np.flatnonzero(active)])
            index[active] = sample_from_logits(logits[active], legal[active], uniforms)
        return decode_actions(index, self.menu)


def policy_checkpoint(agent: SacAgent, **metadata) -> dict:
    """Evaluation checkpoint: the policy network only (Q networks are not needed)."""
    return {
        "schema": SAC_CHECKPOINT_SCHEMA,
        "kind": "sac",
        "model": {
            "obs_dim": agent.obs_dim,
            "num_actions": agent.actions,
            "hidden": list(agent.settings.hidden),
        },
        "sac": asdict(agent.settings),
        "policy": agent.policy.state_dict(),
        **metadata,
    }


def sac_policy_factory(checkpoint: dict, scenario):
    """A factory of protocol policies from a ``policy_checkpoint`` dictionary."""
    if checkpoint.get("kind") != "sac" or checkpoint["schema"] != SAC_CHECKPOINT_SCHEMA:
        raise ValueError("not a supported SAC checkpoint")
    model = checkpoint["model"]
    network = mlp(model["obs_dim"], model["num_actions"], tuple(model["hidden"]))
    network.load_state_dict(checkpoint["policy"])
    network.eval()
    epsilon = scenario.objective["max_violation_rate"]

    def make() -> SacPolicy:
        return SacPolicy(network, scenario.physical, scenario.menu, epsilon)

    return make
