"""Static centre-AP disk topologies.

Copied unchanged in behaviour from dsa04 (dca_marl.topology, scenario v1.3).
Topology records retain actual coordinates, gains and their source metadata.
Replay always uses saved coordinates and preserves the original NumPy version,
including when coordinates were supplied explicitly rather than sampled.
"""

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

_ARRAY_NAMES = ("ue_positions_m", "ue_ap_distances_m", "ue_ap_gain_linear")
_SAMPLING_METHOD = "r=radius_m*sqrt(U); phi=2*pi*V; independent U,V~Uniform[0,1)"
# Permit platform-level derivation rounding, while hashes verify recorded bytes
# exactly. No absolute tolerance may turn a tiny positive gain into zero.
_DERIVATION_RTOL = 1e-12


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _array(value: object, name: str) -> NDArray[np.float64]:
    try:
        raw = np.asarray(value)
        if not isinstance(value, np.ndarray):
            original = np.asarray(value, dtype=object)
            if any(
                isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
                for item in original.flat
            ):
                raise ValueError(f"{name} must contain real numbers")
        if raw.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain real numbers")
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real numbers") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite real numbers")
    return result


def _immutable(array: NDArray[np.float64]) -> NDArray[np.float64]:
    # An immutable bytes owner prevents setflags(write=True) from reopening the
    # array, unlike merely clearing a flag on caller-owned storage.
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(array.shape)


def _array_hash(array: NDArray[np.float64]) -> str:
    canonical = np.asarray(array, dtype="<f8", order="C")
    digest = hashlib.sha256()
    digest.update(f"float64-le:{canonical.shape}:".encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class SpatialConfig:
    """Required geometry and path-loss settings, independent of radio menus."""

    radius_m: float
    reference_distance_m: float
    reference_path_loss_db: float
    path_loss_exponent: float
    topology_seed: int

    def __post_init__(self) -> None:
        for name in (
            "radius_m",
            "reference_distance_m",
            "reference_path_loss_db",
            "path_loss_exponent",
        ):
            value = _finite_real(getattr(self, name), name)
            if value < 0 or (name != "reference_path_loss_db" and value == 0):
                qualifier = "nonnegative" if name == "reference_path_loss_db" else "positive"
                raise ValueError(f"{name} must be {qualifier}")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "topology_seed", _integer(self.topology_seed, "topology_seed", 0))


def _derive(
    spatial: SpatialConfig, positions: object
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if not isinstance(spatial, SpatialConfig):
        raise ValueError("spatial must be a SpatialConfig")
    coordinates = _array(positions, "ue_positions_m")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.shape[0] < 1:
        raise ValueError("ue_positions_m must have shape (N, 2) with N >= 1")
    # hypot avoids squaring overflow/underflow at extreme coordinate scales.
    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.hypot(coordinates[:, 0], coordinates[:, 1])
    if np.any(~np.isfinite(distances)) or np.any(distances > spatial.radius_m):
        raise ValueError("ue_positions_m must lie within the coverage disk")
    effective = np.maximum(distances, spatial.reference_distance_m)
    # Subtract logs instead of dividing distances, since the ratio can overflow
    # even when a small positive exponent produces a representable gain.
    log_ratio = np.maximum(0.0, np.log10(effective) - math.log10(spatial.reference_distance_m))
    log_ratio[distances <= spatial.reference_distance_m] = 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        loss_decades = (
            spatial.reference_path_loss_db / 10.0 + spatial.path_loss_exponent * log_ratio
        )
        gains = np.power(10.0, -loss_decades)
    if np.any(~np.isfinite(gains)) or np.any(gains <= 0):
        raise ValueError("ue_ap_gain_linear must be representable as positive finite floats")
    return coordinates, distances, gains


@dataclass(frozen=True)
class TopologySource:
    """Immutable origin, retained across record replay and reserialization."""

    kind: str = "explicit_positions"
    numpy_version: str = field(default_factory=lambda: np.__version__)
    bit_generator: str | None = None
    sampling_method: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("explicit_positions", "uniform_disk"):
            raise ValueError("topology source must be explicit_positions or uniform_disk")
        if not isinstance(self.numpy_version, str) or not self.numpy_version:
            raise ValueError("topology source numpy_version must be a nonempty string")
        if self.kind == "uniform_disk":
            if self.bit_generator != "PCG64" or self.sampling_method != _SAMPLING_METHOD:
                raise ValueError("uniform_disk topology source requires the PCG64 area sampler")
        elif self.bit_generator is not None or self.sampling_method is not None:
            raise ValueError("explicit_positions topology source must not claim RNG sampling")


@dataclass(frozen=True, eq=False)
class Topology:
    """Validated immutable snapshots, including when constructed directly."""

    spatial: SpatialConfig
    ue_positions_m: NDArray[np.float64]
    ue_ap_distances_m: NDArray[np.float64]
    ue_ap_gain_linear: NDArray[np.float64]
    source: TopologySource = field(default_factory=TopologySource)

    def __post_init__(self) -> None:
        if not isinstance(self.source, TopologySource):
            raise ValueError("source must be a TopologySource")
        positions, distances, gains = _derive(self.spatial, self.ue_positions_m)
        object.__setattr__(self, "ue_positions_m", _immutable(positions))
        for name, expected in (
            ("ue_ap_distances_m", distances),
            ("ue_ap_gain_linear", gains),
        ):
            actual = _array(getattr(self, name), name)
            if actual.shape != expected.shape or not np.allclose(
                actual, expected, rtol=_DERIVATION_RTOL, atol=0.0
            ):
                raise ValueError(f"{name} is inconsistent with the coordinates and spatial config")
            if name == "ue_ap_gain_linear" and np.any(actual <= 0):
                raise ValueError("ue_ap_gain_linear must be positive")
            object.__setattr__(self, name, _immutable(actual))

    @property
    def num_ues(self) -> int:
        return self.ue_positions_m.shape[0]

    def to_record(self) -> dict:
        """Return a JSON-compatible snapshot with canonical array SHA-256s.

        ``source`` distinguishes generated and explicitly supplied positions;
        ``rng`` retains their original sampling and NumPy metadata. Saved arrays
        remain the authority for replay, independently of the seed or version.
        """
        return {
            "schema_version": 1,
            "spatial": asdict(self.spatial),
            "source": self.source.kind,
            "rng": {
                "bit_generator": self.source.bit_generator,
                "numpy_version": self.source.numpy_version,
                "sampling_method": self.source.sampling_method,
            },
            **{name: getattr(self, name).tolist() for name in _ARRAY_NAMES},
            "array_sha256": {name: _array_hash(getattr(self, name)) for name in _ARRAY_NAMES},
        }


def generate_topology(spatial: SpatialConfig, num_ues: int) -> Topology:
    """Sample independent positions uniformly by area, once per topology."""
    if not isinstance(spatial, SpatialConfig):
        raise ValueError("spatial must be a SpatialConfig")
    count = _integer(num_ues, "num_ues", 1)
    rng = np.random.Generator(np.random.PCG64(spatial.topology_seed))
    radius = spatial.radius_m * np.sqrt(rng.random(count))
    angle = 2.0 * np.pi * rng.random(count)
    positions = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    coordinates, distances, gains = _derive(spatial, positions)
    source = TopologySource("uniform_disk", np.__version__, "PCG64", _SAMPLING_METHOD)
    return Topology(spatial, coordinates, distances, gains, source)


def topology_from_positions(spatial: SpatialConfig, ue_positions_m: object) -> Topology:
    """Derive distance and gain from actual positions without invoking an RNG."""
    positions, distances, gains = _derive(spatial, ue_positions_m)
    return Topology(spatial, positions, distances, gains)


def topology_from_record(record: Mapping) -> Topology:
    """Validate and replay saved arrays; never regenerate from a seed."""
    expected_keys = {"schema_version", "spatial", "source", "rng", "array_sha256", *_ARRAY_NAMES}
    if not isinstance(record, Mapping) or set(record) != expected_keys:
        raise ValueError("topology record has missing or unknown fields")
    if _integer(record["schema_version"], "schema_version", 1) != 1:
        raise ValueError("unsupported topology record schema_version")
    spatial_record = record["spatial"]
    if not isinstance(spatial_record, Mapping):
        raise ValueError("topology record spatial must be a mapping")
    try:
        spatial = SpatialConfig(**spatial_record)
    except TypeError as exc:
        raise ValueError("topology record spatial has missing or unknown fields") from exc
    rng_record = record["rng"]
    if not isinstance(rng_record, Mapping) or set(rng_record) != {
        "bit_generator",
        "numpy_version",
        "sampling_method",
    }:
        raise ValueError("topology record RNG metadata is invalid")
    source = TopologySource(kind=record["source"], **rng_record)
    hashes = record["array_sha256"]
    if not isinstance(hashes, Mapping) or set(hashes) != set(_ARRAY_NAMES):
        raise ValueError("topology record must hash all three arrays")
    arrays = {name: _array(record[name], name) for name in _ARRAY_NAMES}
    for name, array in arrays.items():
        if not isinstance(hashes[name], str) or hashes[name] != _array_hash(array):
            raise ValueError(f"topology record {name} hash mismatch")
    return Topology(spatial=spatial, source=source, **arrays)


# ----------------------------------------------------------------- scenario v2.0
_SHADOWED_SAMPLING = (
    "per UE in order: r=radius_m*sqrt(U), phi=2*pi*V, S_dB=sigma_db*Z (U,V~U[0,1), Z~N(0,1)); "
    "redraw (r, phi, S_dB) until the coverage predicate holds; one PCG64(topology_seed) stream"
)


@dataclass(frozen=True)
class ShadowingConfig:
    """Static log-normal shadowing and the coverage rule (scenario v2.0 §2.1)."""

    sigma_db: float
    coverage_window_fraction: float

    def __post_init__(self) -> None:
        sigma = _finite_real(self.sigma_db, "sigma_db")
        fraction = _finite_real(self.coverage_window_fraction, "coverage_window_fraction")
        if sigma < 0 or not 0 < fraction <= 1:
            raise ValueError("sigma_db must be >= 0 and coverage_window_fraction in (0, 1]")
        object.__setattr__(self, "sigma_db", sigma)
        object.__setattr__(self, "coverage_window_fraction", fraction)


@dataclass(frozen=True, eq=False)
class ShadowedTopology:
    """Path-loss topology plus per-UE static shadowing, identical on every subchannel.

    ``ue_ap_gain_linear`` is the mean channel gain PL(d_i) * S_i that the fading
    model multiplies; ``path.ue_ap_gain_linear`` is path loss alone. ``redraws``
    counts how often each UE was redrawn by the coverage rule.
    """

    path: Topology
    shadowing_db: NDArray[np.float64]
    redraws: NDArray[np.int64]

    def __post_init__(self) -> None:
        shadowing = _array(self.shadowing_db, "shadowing_db")
        redraws = np.asarray(self.redraws)
        if shadowing.shape != (self.path.num_ues,) or redraws.shape != (self.path.num_ues,):
            raise ValueError("shadowing_db and redraws need one entry per UE")
        if redraws.dtype.kind not in "iu" or np.any(redraws < 0):
            raise ValueError("redraws must be nonnegative integers")
        object.__setattr__(self, "shadowing_db", _immutable(shadowing))
        frozen = np.asarray(redraws, dtype=np.int64)
        object.__setattr__(
            self, "redraws", np.frombuffer(frozen.tobytes(), np.int64).reshape(frozen.shape)
        )

    @property
    def num_ues(self) -> int:
        return self.path.num_ues

    @property
    def ue_positions_m(self) -> NDArray[np.float64]:
        return self.path.ue_positions_m

    @property
    def ue_ap_distances_m(self) -> NDArray[np.float64]:
        return self.path.ue_ap_distances_m

    @property
    def ue_ap_gain_linear(self) -> NDArray[np.float64]:
        return self.path.ue_ap_gain_linear * 10 ** (self.shadowing_db / 10.0)

    def to_record(self) -> dict:
        return {
            "schema_version": 1,
            "sampling_method": _SHADOWED_SAMPLING,
            "numpy_version": np.__version__,
            "path": self.path.to_record(),
            "shadowing_db": self.shadowing_db.tolist(),
            "redraws": self.redraws.tolist(),
            "shadowing_sha256": _array_hash(self.shadowing_db),
        }


def generate_shadowed_topology(
    spatial: SpatialConfig,
    shadowing: ShadowingConfig,
    num_ues: int,
    covered,
    max_draws: int = 10_000,
) -> ShadowedTopology:
    """Sample UEs one at a time with shadowing; redraw a UE until ``covered(ue, gain)``.

    ``covered(ue, mean_gain_linear) -> bool`` is the coverage rule; it may depend
    on the UE's traffic class but not on other UEs. UE i consumes the stream only
    after UEs 0..i-1, so the first n UEs of a larger deployment equal the
    deployment with n UEs (deployments are nested across N).
    """
    if not isinstance(spatial, SpatialConfig) or not isinstance(shadowing, ShadowingConfig):
        raise ValueError("spatial and shadowing must be SpatialConfig and ShadowingConfig")
    count = _integer(num_ues, "num_ues", 1)
    rng = np.random.Generator(np.random.PCG64(spatial.topology_seed))
    positions = np.empty((count, 2))
    shadow = np.empty(count)
    redraws = np.zeros(count, dtype=np.int64)
    for ue in range(count):
        for draw in range(max_draws):
            radius = spatial.radius_m * np.sqrt(rng.random())
            angle = 2.0 * np.pi * rng.random()
            value = shadowing.sigma_db * rng.standard_normal()
            position = [radius * np.cos(angle), radius * np.sin(angle)]
            _, _, gain = _derive(spatial, [position])
            if covered(ue, float(gain[0]) * 10 ** (value / 10.0)):
                positions[ue], shadow[ue], redraws[ue] = position, value, draw
                break
        else:
            raise RuntimeError(f"UE {ue} not covered after {max_draws} draws")
    return ShadowedTopology(topology_from_positions(spatial, positions), shadow, redraws)
