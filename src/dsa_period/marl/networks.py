"""Channel-shared GRU actor (CSG, adapted from dsa04) and the centralized critic.

The actor runs one GRU stream per subchannel with weights shared across
subchannels and UEs, pools the streams of the same UE (DeepSets mean), and
scores OFF plus every (subchannel, power level). Logits are channel-major:
index 1 + k * L + (l - 1) for subchannel k and power ID l.
"""

import torch
from torch import Tensor, nn


def _init(module: nn.Module, output_layers: tuple[nn.Linear, ...]) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=0.01 if layer in output_layers else 1.0)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.GRUCell):
            for name, parameter in layer.named_parameters():
                if "weight" in name:
                    for gate in parameter.chunk(3, dim=0):
                        nn.init.orthogonal_(gate)
                else:
                    nn.init.zeros_(parameter)


class ChannelSetActor(nn.Module):
    def __init__(
        self, token_dim: int, context_dim: int, num_channels: int, num_levels: int, hidden: int
    ) -> None:
        super().__init__()
        self.num_channels, self.num_levels, self.hidden = num_channels, num_levels, hidden
        h, u = hidden, context_dim
        self.encoder = nn.Sequential(nn.Linear(token_dim + u, h), nn.Tanh())
        self.memory = nn.GRUCell(h, h)
        self.phi = nn.Sequential(nn.Linear(h, h), nn.Tanh())
        self.eta = nn.Sequential(nn.Linear(h + u, h), nn.Tanh())
        self.psi = nn.Sequential(nn.Linear(2 * h + u, h), nn.Tanh(), nn.Linear(h, num_levels))
        self.omega = nn.Sequential(nn.Linear(h + u, h), nn.Tanh(), nn.Linear(h, 1))
        _init(self, (self.psi[-1], self.omega[-1]))

    @property
    def num_actions(self) -> int:
        return 1 + self.num_channels * self.num_levels

    def initial_hidden(self, batch: int) -> Tensor:
        return torch.zeros(batch, self.num_channels, self.hidden)

    def step(self, tokens: Tensor, context: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """tokens (B, K, Dt), context (B, Du), hidden (B, K, h) -> logits (B, A), hidden."""
        logits, _, hidden = self._forward(tokens, context, hidden)
        return logits, hidden

    def _forward(self, tokens: Tensor, context: Tensor, hidden: Tensor):
        """Action logits, the pooled per-UE summary and the new hidden state."""
        batch, k = tokens.shape[0], self.num_channels
        expanded = context.unsqueeze(1).expand(batch, k, context.shape[-1])
        encoded = self.encoder(torch.cat((tokens, expanded), dim=-1))
        hidden = self.memory(
            encoded.reshape(-1, self.hidden), hidden.reshape(-1, self.hidden)
        ).reshape(batch, k, self.hidden)
        pooled = self.eta(torch.cat((self.phi(hidden).mean(dim=1), context), dim=-1))
        summary = torch.cat((pooled, context), dim=-1)
        per_channel = self.psi(
            torch.cat((hidden, summary.unsqueeze(1).expand(batch, k, summary.shape[-1])), dim=-1)
        )
        logits = torch.cat((self.omega(summary), per_channel.reshape(batch, -1)), dim=-1)
        return logits, summary, hidden

    def sequence(
        self,
        tokens: Tensor,
        context: Tensor,
        hidden: Tensor,
        starts: Tensor,
        restart: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Ordered replay: tokens (T, B, K, Dt), starts (T, B) clear memory first.

        With ``restart`` (T, B, K, h) an episode start takes that hidden state
        instead of zeros (EXP-DSA05-012 branch episodes resume a snapshot's memory).
        """
        outputs = []
        for t in range(tokens.shape[0]):
            if restart is None:
                hidden = hidden * (~starts[t]).to(hidden.dtype)[:, None, None]
            else:
                hidden = torch.where(starts[t][:, None, None], restart[t], hidden)
            logits, hidden = self.step(tokens[t], context[t], hidden)
            outputs.append(logits)
        return torch.stack(outputs), hidden


class MessageActor(ChannelSetActor):
    """ChannelSetActor with a second head: the uplink message bit (EXP-DSA05-011, content L).

    ``step`` returns the action logits followed by the two message logits,
    (B, num_actions + 2); ``split_logits`` separates them. The message head reads
    the same pooled summary as the OFF score. A UE's message matters only when it
    transmits; the joint log-probability adds the message term only then.
    """

    MESSAGES = 2

    def __init__(
        self, token_dim: int, context_dim: int, num_channels: int, num_levels: int, hidden: int
    ) -> None:
        super().__init__(token_dim, context_dim, num_channels, num_levels, hidden)
        self.message = nn.Sequential(
            nn.Linear(hidden + context_dim, hidden), nn.Tanh(), nn.Linear(hidden, self.MESSAGES)
        )
        _init(self.message, (self.message[-1],))

    def step(self, tokens: Tensor, context: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        logits, summary, hidden = self._forward(tokens, context, hidden)
        return torch.cat((logits, self.message(summary)), dim=-1), hidden


def split_logits(logits: Tensor, num_actions: int) -> tuple[Tensor, Tensor | None]:
    """Action logits and, for a ``MessageActor``, the message logits (else None)."""
    if logits.shape[-1] == num_actions:
        return logits, None
    if logits.shape[-1] != num_actions + MessageActor.MESSAGES:
        raise ValueError("unexpected number of logits")
    return logits[..., :num_actions], logits[..., num_actions:]


class CentralCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        _init(self, ())

    def forward(self, state: Tensor) -> Tensor:
        return self.network(state).squeeze(-1)


class SetCritic(nn.Module):
    """N-independent centralized critic (scenario v2.0, EXP-DSA05-006).

    Every UE's input is encoded by a shared network phi; the encodings of each
    environment (segment) are pooled by mean and max and joined with N / 32
    before the value head. Permuting the UEs of an environment, or changing their
    number, needs no change of weights.
    """

    COUNT_SCALE = 32.0

    def __init__(self, ue_dim: int, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.phi = nn.Sequential(
            nn.Linear(ue_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.rho = nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        _init(self, ())

    def forward(self, ue_features: Tensor, segment: Tensor, num_segments: int) -> Tensor:
        """ue_features (U, D), segment (U,) in [0, S) -> values (S,)."""
        encoded = self.phi(ue_features)
        count = torch.zeros(num_segments, dtype=encoded.dtype).index_add_(
            0, segment, torch.ones_like(segment, dtype=encoded.dtype)
        )
        mean = torch.zeros(num_segments, self.hidden, dtype=encoded.dtype).index_add(
            0, segment, encoded
        ) / count.clamp(min=1.0).unsqueeze(-1)
        top = torch.full((num_segments, self.hidden), -1.0, dtype=encoded.dtype).scatter_reduce(
            0, segment.unsqueeze(-1).expand_as(encoded), encoded, reduce="amax", include_self=False
        )
        pooled = torch.cat((mean, top, (count / self.COUNT_SCALE).unsqueeze(-1)), dim=-1)
        return self.rho(pooled).squeeze(-1)


def masked_log_probs(logits: Tensor, legal: Tensor) -> Tensor:
    return torch.log_softmax(logits.masked_fill(~legal, -torch.inf), dim=-1)
