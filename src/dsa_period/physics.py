"""Shannon-rate uplink kernel: each UE picks one subchannel and a discrete power.

There is no MCS action. A transmission in one slot carries the mutual information
tau * B * log2(1 + SINR) under treating interference as noise. This is the
HARQ-IR / rateless-coding abstraction: the transmitter need not know its SINR in
advance, and a packet is decoded once the mutual information accumulated over
its transmissions reaches the packet size (queue handling lives in ``env``).

Link gains enter as an (N, K) array, so a frequency-selective or fading channel
model can replace the static path-loss channel without changing this kernel.
All quantities use SI units. No traffic, reward, history or randomness here.
"""

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import NDArray

LN2 = math.log(2.0)


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _dbm_to_w(dbm: float) -> float:
    return 10.0 ** ((dbm - 30.0) / 10.0)


def _frozen(values: object) -> np.ndarray:
    """Detached array whose immutable byte owner also blocks setflags(write=True)."""
    array = np.asarray(values)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class PhysicalConfig:
    """Public physical-layer parameters; link gains belong to the channel model."""

    num_ues: int
    num_subchannels: int
    slot_duration_s: float
    subchannel_bandwidth_hz: float
    noise_psd_dbm_per_hz: float = -174.0
    noise_figure_db: float = 7.0
    max_tx_power_dbm: float = 23.0

    def __post_init__(self) -> None:
        for name in ("num_ues", "num_subchannels"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in (
            "slot_duration_s",
            "subchannel_bandwidth_hz",
            "noise_psd_dbm_per_hz",
            "noise_figure_db",
            "max_tx_power_dbm",
        ):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name))
        if self.slot_duration_s <= 0 or self.subchannel_bandwidth_hz <= 0:
            raise ValueError("slot_duration_s and subchannel_bandwidth_hz must be positive")

    @property
    def noise_power_w(self) -> float:
        noise_dbm = (
            self.noise_psd_dbm_per_hz
            + 10.0 * math.log10(self.subchannel_bandwidth_hz)
            + self.noise_figure_db
        )
        return _dbm_to_w(noise_dbm)

    @property
    def channel_uses_per_slot(self) -> float:
        """B * tau: bits carried per slot per bit/s/Hz of spectral efficiency."""
        return self.slot_duration_s * self.subchannel_bandwidth_hz


@dataclass(frozen=True)
class PowerMenu:
    """Discrete transmit powers. ID 0 is the literal OFF (exactly 0 W).

    Later entries are active powers in strictly increasing dBm. Numeric 0 dBm is
    an active 1 mW transmitter, not OFF. Power IDs index ``power_dbm``/``power_w``.
    """

    power_dbm: tuple[Literal["OFF"] | float, ...]

    def __post_init__(self) -> None:
        raw = self.power_dbm
        if isinstance(raw, (str, bytes)):
            raise ValueError("power_dbm must be a sequence")
        entries = tuple(raw)
        if len(entries) < 2 or entries[0] != "OFF":
            raise ValueError("power_dbm must begin with 'OFF' and hold at least one active power")
        active = tuple(_finite_real(value, "power_dbm") for value in entries[1:])
        if any(right <= left for left, right in zip(active, active[1:])):
            raise ValueError("active powers must be strictly increasing")
        object.__setattr__(self, "power_dbm", ("OFF", *active))

    @property
    def num_levels(self) -> int:
        return len(self.power_dbm)

    @property
    def power_w(self) -> tuple[float, ...]:
        return (0.0, *(_dbm_to_w(value) for value in self.power_dbm[1:]))

    def validate_for(self, config: PhysicalConfig) -> None:
        if not isinstance(config, PhysicalConfig):
            raise ValueError("config must be a PhysicalConfig")
        if self.power_dbm[-1] > config.max_tx_power_dbm:
            raise ValueError("power_dbm exceeds max_tx_power_dbm")


@dataclass(frozen=True)
class StepResult:
    """Simulator-internal slot outcome, not a decentralized UE observation.

    ``actions`` are canonical (OFF rows become (0, 0)). ``sinr_linear`` is NaN
    for OFF UEs. ``service_bits`` is the unclipped mutual information of each
    transmission; ``rssi_w`` is noise plus all received power per subchannel.
    """

    actions: NDArray[np.int64]
    active: NDArray[np.bool_]
    sinr_linear: NDArray[np.float64]
    service_bits: NDArray[np.float64]
    tx_energy_j: NDArray[np.float64]
    rssi_w: NDArray[np.float64]


def validate_actions(
    config: PhysicalConfig, menu: PowerMenu, joint_actions: object
) -> NDArray[np.int64]:
    """Validate raw (subchannel, power_id) rows for every UE, then canonicalize OFF."""
    raw = np.asarray(joint_actions, dtype=object)
    if raw.shape != (config.num_ues, 2):
        raise ValueError(f"joint_actions must have shape ({config.num_ues}, 2)")
    for value in raw.flat:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError("action IDs must be integers, excluding booleans")
    for column, limit in enumerate((config.num_subchannels, menu.num_levels)):
        if any(not 0 <= value < limit for value in raw[:, column]):
            raise ValueError(f"action column {column} IDs must lie in [0, {limit})")
    actions = np.array(raw, dtype=np.int64)
    actions[actions[:, 1] == 0] = 0
    return actions


def validate_gains(config: PhysicalConfig, gains: object) -> NDArray[np.float64]:
    array = np.asarray(gains, dtype=np.float64)
    if array.shape != (config.num_ues, config.num_subchannels):
        raise ValueError(f"gains must have shape ({config.num_ues}, {config.num_subchannels})")
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("gains must be positive and finite")
    return array


def shannon_bits(config: PhysicalConfig, sinr_linear: object) -> NDArray[np.float64]:
    """Mutual information of one slot, tau * B * log2(1 + SINR), accurate at low SINR."""
    return config.channel_uses_per_slot * np.log1p(np.asarray(sinr_linear, np.float64)) / LN2


def compute_step(
    config: PhysicalConfig,
    menu: PowerMenu,
    gains: object,
    joint_actions: object,
) -> StepResult:
    """Evaluate one synchronous slot on all subchannels.

    ``gains[i, k]`` is UE i's linear power gain to the AP on subchannel k. Each
    active UE interferes with every other transmitter on the same subchannel
    for the full slot; different subchannels are orthogonal.
    """
    menu.validate_for(config)
    gain = validate_gains(config, gains)
    actions = validate_actions(config, menu, joint_actions)
    power = np.asarray(menu.power_w)[actions[:, 1]]
    active = actions[:, 1] > 0
    ues = np.arange(config.num_ues)
    received = np.where(active, power * gain[ues, actions[:, 0]], 0.0)
    noise = config.noise_power_w
    rssi = np.full(config.num_subchannels, noise)
    sinr = np.full(config.num_ues, np.nan)
    for channel in range(config.num_subchannels):
        members = np.flatnonzero(active & (actions[:, 0] == channel))
        rssi[channel] = math.fsum((noise, *received[members]))
        for ue in members:
            # Sum the other transmitters directly: RSSI minus self loses small terms.
            interference = math.fsum((noise, *received[members[members != ue]]))
            sinr[ue] = received[ue] / interference
    service = np.where(active, shannon_bits(config, np.nan_to_num(sinr)), 0.0)
    energy = config.slot_duration_s * power
    return StepResult(*(_frozen(a) for a in (actions, active, sinr, service, energy, rssi)))
