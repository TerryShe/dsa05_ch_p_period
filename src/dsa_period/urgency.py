"""Urgency state summary: the L1 information level of EXP-DSA05-011 (design §2).

Timeline: in slot t every transmitting UE attaches one uplink bit u_i(t). The AP
reads it iff that transmission's TIN SINR >= theta_f, recovered from the
AP-reported mutual information as in the broadcast probe B3s (``theta_db`` None:
every bit is read, probe B3). At the start of slot t + 1 the AP broadcasts
b_k(t), the OR of the read bits on subchannel k, together with the public RSSI;
the UEs use it to decide slot t + 1. Contents of the uplink bit (design §2.2):

* urgency: the UE is urgent, slack < ``urgent_slack`` slots, with slack = slots
  to deadline - ceil(remaining bits / r_max) and r_max the interference-free
  max-power rate of its best own subchannel (local only, as in step 0 and the
  probe);
* occupancy: every transmission (no uplink bit; the AP detects the
  transmission under the same reading condition);
* message: chosen by the policy's message head.

A UE's actor sees only the K broadcast bits (after an optional evaluation-time
hiding mask, design §3.3) and its own previous uplink bit; nothing else about
other UEs. The same ``SummaryLink`` serves training and evaluation.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dsa_period.marl.features import ObsArrays, stack
from dsa_period.physics import PhysicalConfig, PowerMenu, shannon_bits

CONTENTS = ("urgency", "occupancy", "message")
HIDING = ("none", "all", "nonurgent", "urgent")
URGENT_SLACK = 3
PRIMARY_THETA_DB = -10.0
NOT_PENDING = 10**6  # slack of a UE without a packet


@dataclass(frozen=True)
class SummarySettings:
    """Content of the uplink bit, the AP reading threshold and the urgency threshold."""

    content: str = "urgency"
    theta_db: float | None = PRIMARY_THETA_DB
    urgent_slack: int = URGENT_SLACK

    def __post_init__(self) -> None:
        if self.content not in CONTENTS:
            raise ValueError(f"content must be one of {CONTENTS}")
        if self.urgent_slack < 1:
            raise ValueError("urgent_slack must be a positive number of slots")

    @property
    def message(self) -> bool:
        return self.content == "message"


def slack(config: PhysicalConfig, menu: PowerMenu, obs: ObsArrays) -> NDArray[np.float64]:
    """Per UE: slots to deadline minus the slots still needed at its best rate alone."""
    if obs.own_gain is None:
        raise ValueError("the urgency summary needs scenario v2.0 observations (own gains)")
    rate = shannon_bits(config, menu.power_w[-1] * obs.own_gain.max(axis=1) / config.noise_power_w)
    need = np.ceil(np.maximum(obs.remaining, 0) / np.maximum(rate, 1e-9) - 1e-9)
    return np.where(obs.remaining > 0, obs.to_deadline - need, NOT_PENDING)


def urgent(
    config: PhysicalConfig, menu: PowerMenu, obs: ObsArrays, threshold: int = URGENT_SLACK
) -> NDArray[np.bool_]:
    return (obs.remaining > 0) & (slack(config, menu, obs) < threshold)


def visible(
    broadcast: NDArray[np.bool_], hide: str, urgent_now: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """(N, K) broadcast each UE's actor sees; ``hide`` zeroes it for some UEs (design §3.3)."""
    if hide not in HIDING:
        raise ValueError(f"hide must be one of {HIDING}")
    seen = np.broadcast_to(broadcast, (urgent_now.size, broadcast.size)).copy()
    if hide == "all":
        seen[:] = False
    elif hide == "nonurgent":
        seen[~urgent_now] = False
    elif hide == "urgent":
        seen[urgent_now] = False
    return seen


class SummaryLink:
    """The AP side of the summary for one environment of ``num_ues`` UEs.

    Per slot: ``broadcast(obs)`` turns the bits attached to the previous slot's
    transmissions into b (it must run once per observation, before the slot is
    decided); ``commit(bits, transmitting)`` stores the bits of the slot just
    decided. ``own_previous`` is each UE's own bit of the previous slot.
    """

    def __init__(
        self, settings: SummarySettings, config: PhysicalConfig, menu: PowerMenu, num_ues: int
    ) -> None:
        self.settings, self.config, self.menu, self.num_ues = settings, config, menu, num_ues
        theta = settings.theta_db
        self.threshold_bits = (
            None if theta is None else float(shannon_bits(config, 10.0 ** (theta / 10.0)))
        )
        self.reset()

    def reset(self) -> None:
        self.sent = np.zeros(self.num_ues, dtype=np.bool_)  # bits attached in the last slot
        self.read = np.zeros(self.num_ues, dtype=np.bool_)
        self.bits_sent = self.bits_read = 0

    @property
    def own_previous(self) -> NDArray[np.bool_]:
        """The UE's own uplink bit of the previous slot; always 0 for occupancy (design §2.5).

        Occupancy carries no uplink bit: the AP detects the transmission itself, so
        the actor's own-bit input stays 0 rather than repeating "I transmitted".
        """
        if self.settings.content == "occupancy":
            return np.zeros(self.num_ues, dtype=np.bool_)
        return self.sent.copy()

    def broadcast(self, obs: ObsArrays) -> NDArray[np.bool_]:
        """b of the previous slot: OR of the read bits per subchannel."""
        transmitted = obs.prev_power > 0
        if np.any(self.sent & ~transmitted):
            raise AssertionError("an uplink bit was committed for a UE that did not transmit")
        flagged = self.sent & transmitted
        heard = flagged.copy()
        if self.threshold_bits is not None:
            heard &= obs.prev_service >= self.threshold_bits
        self.read = heard
        self.bits_sent += int(flagged.sum())
        self.bits_read += int(heard.sum())
        out = np.zeros(self.config.num_subchannels, dtype=np.bool_)
        out[obs.prev_channel[heard]] = True
        return out

    def uplink(
        self,
        obs: ObsArrays,
        transmitting: NDArray[np.bool_],
        messages: NDArray[np.int64] | None = None,
    ) -> NDArray[np.bool_]:
        """The bit each transmitting UE attaches to the slot being decided."""
        content = self.settings.content
        if content == "urgency":
            bits = urgent(self.config, self.menu, obs, self.settings.urgent_slack)
        elif content == "occupancy":
            bits = np.ones(self.num_ues, dtype=np.bool_)
        else:
            if messages is None:
                raise ValueError("the message content needs the policy's messages")
            bits = np.asarray(messages) == 1
        return bits & transmitting

    def commit(self, bits: NDArray[np.bool_], transmitting: NDArray[np.bool_]) -> None:
        self.sent = np.asarray(bits, dtype=np.bool_) & np.asarray(transmitting, dtype=np.bool_)


def with_summary(
    tokens: NDArray[np.float32],
    context: NDArray[np.float32],
    seen: NDArray[np.bool_],
    own_previous: NDArray[np.bool_],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Append b_k to every subchannel token and the own previous bit to the context."""
    tokens = np.concatenate((tokens, seen[:, :, None].astype(np.float32)), axis=2)
    context = np.concatenate((context, own_previous[:, None].astype(np.float32)), axis=1)
    return tokens, context


class YieldPolicy:
    """Any policy plus the NR-style forced yield (M+Y, best_local+Y; design §3.1).

    The probe's B3 rule with the reading threshold of ``settings``: a UE with a
    packet that is not urgent and whose chosen subchannel is marked in the
    broadcast stays OFF this slot. A UE ignores the mark of the subchannel it
    used itself in the previous slot when it attached a bit there (the 1-bit OR
    cannot tell whether the mark is its own). Urgent UEs act as the inner policy
    says. The rule draws no random numbers.
    """

    def __init__(self, inner, config: PhysicalConfig, menu: PowerMenu, settings: SummarySettings):
        if settings.content != "urgency":
            raise ValueError("the yield rule reads the urgency content")
        self.inner, self.config, self.menu, self.settings = inner, config, menu, settings
        self.link = SummaryLink(settings, config, menu, config.num_ues)
        theta = "all" if settings.theta_db is None else f"{settings.theta_db:g}"
        self.name = f"yield[theta={theta}]@{getattr(inner, 'name', 'policy')}"

    def reset(self, rngs) -> None:
        self.inner.reset(rngs)
        self.link.reset()
        self.deferred = np.zeros(self.config.num_ues, dtype=np.int64)

    def act(self, observations) -> NDArray[np.int64]:
        obs = stack(observations)
        urgent_now = urgent(self.config, self.menu, obs, self.settings.urgent_slack)
        marked = self.link.broadcast(obs)
        own = self.link.own_previous & (obs.prev_power > 0)
        actions = np.array(self.inner.act(observations), dtype=np.int64, copy=True)
        for ue in np.flatnonzero((obs.remaining > 0) & ~urgent_now & (actions[:, 1] > 0)):
            k = actions[ue, 0]
            if marked[k] and not (own[ue] and obs.prev_channel[ue] == k):
                actions[ue] = 0
                self.deferred[ue] += 1
        transmitting = actions[:, 1] > 0
        self.link.commit(self.link.uplink(obs, transmitting), transmitting)
        return actions


WRAPPERS = ("yield", "summary")


def parse_wrapper(spec: str) -> tuple[str, dict, str] | None:
    """``name[key=value,...]@inner`` -> (name, options, inner spec); None if not wrapped.

    ``yield[theta=T]@SPEC`` puts the yield rule on any policy; ``summary[theta=T,
    hide=H]@marl:PATH`` evaluates a summary checkpoint under another reading
    threshold or hiding mask. T is a number of dB or ``all`` (every bit read).
    """
    name, bracket, rest = spec.partition("[")
    if not bracket or name not in WRAPPERS:
        return None
    body, close, inner = rest.partition("]@")
    if not close or not inner:
        raise ValueError(f"malformed wrapper spec {spec!r}")
    options: dict = {}
    for item in filter(None, body.split(",")):
        key, eq, value = item.partition("=")
        if not eq:
            raise ValueError(f"malformed option {item!r} in {spec!r}")
        if key == "theta":
            options["theta_db"] = None if value == "all" else float(value)
        elif key == "hide" and name == "summary":
            if value not in HIDING:
                raise ValueError(f"hide must be one of {HIDING}")
            options["hide"] = value
        else:
            raise ValueError(f"unknown option {key!r} for {name}")
    if name == "yield" and "theta_db" not in options:
        raise ValueError("the yield rule needs an explicit theta")
    if name == "summary" and not inner.startswith("marl:"):
        raise ValueError("summary[...] applies to a marl: checkpoint")
    return name, options, inner
