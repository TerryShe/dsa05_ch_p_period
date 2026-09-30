"""Exploratory (not formal): channel, deadline and reference options for scenario v2.0.

Uses the v1.0 N=8 geometry of dev instances 0-99 (800 UEs). Everything is for a
single UE transmitting alone (no interference). Sections:

1. coverage under log-normal shadowing on the mean channel, and the coverage rule
   of the draft: a UE whose packet needs more than half its window at 23 dBm on
   the mean channel is redrawn (position and shadowing) until it does not;
2. fading autocorrelation against the Jakes model:
   * complex gain at lags of 10/20/40 slots for three generators, against
     J0(2 pi fD dt):
     - sum-of-sinusoids (SoS) Jakes: M sinusoids with arrival angles
       (2 pi n + theta) / M and independent uniform phases; its ensemble
       autocorrelation is exactly J0(2 pi fD dt);
     - AR(1) with rho = J0(2 pi fD tau) (lag-1 matching, as in the first draft);
     - AR(1) with rho matched at the lag where the Jakes correlation is 0.5;
   * power |h|^2 of the SoS generator: correlation against J0^2 (exact only for
     a Gaussian process) and its variance (1 for Rayleigh);
3. per packet: share of packets that cannot be delivered even at 23 dBm in every
   slot (best subchannel per slot), and the full-information (genie) minimum
   packet energy relative to the static v1.0 optimum. All settings of a packet
   share one fading seed (common random numbers), so rows of the same generator
   and subchannel count see the same SoS parameters;
4. per UE over one episode (deadlines in (200, 2000]): how much the 5% per-UE
   violation allowance lowers the genie lower bound when the most expensive
   floor(0.05 M) packets are dropped (L_eps / L_0). UE-episodes with more
   undeliverable packets than the allowance are infeasible and excluded.
5. capacity scale (reliability first, 23 dBm always): the genie minimum number of
   slots a packet needs (best subchannel per slot, best slots of its window) as a
   share of its window, i.e. the fraction of one subchannel a UE occupies, and the
   orthogonal capacity estimate K / mean share. Same packets and common fading
   seeds for every setting.

The genie minimum is an exact per-packet MILP: at most one power level per slot,
delivered mutual information >= packet size. "Undeliverable" (max power in every
slot does not suffice) and "solver failed" are counted separately. Dev only.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, brentq, milp
from scipy.special import j0

from dsa_period.central import optimal_packet_plan
from dsa_period.config import load_scenario
from dsa_period.physics import shannon_bits

SCENARIO = load_scenario("configs/scenario-v1-n8.toml")
CFG, MENU, SPATIAL = SCENARIO.physical, SCENARIO.menu, SCENARIO.spatial
POWER = np.asarray(MENU.power_w[1:])
ENERGY = CFG.slot_duration_s * POWER
TAU = CFG.slot_duration_s
X_HALF = brentq(lambda x: j0(x) - 0.5, 0.5, 2.4)  # J0(x) = 0.5 at x = 1.5211
OK, UNDELIVERABLE, SOLVER_FAILED = "ok", "undeliverable", "solver_failed"


def path_gain(distance_m):
    """v1.0 path loss (same formula as dsa_period.topology)."""
    d = np.maximum(distance_m, SPATIAL.reference_distance_m) / SPATIAL.reference_distance_m
    return 10 ** (-SPATIAL.reference_path_loss_db / 10) * d ** (-SPATIAL.path_loss_exponent)


def jakes_sos(rng, slots, channels, fd, sinusoids=32):
    """(slots, channels) complex gains, unit power, Jakes ensemble autocorrelation."""
    t = (rng.uniform(0, 1e3) + np.arange(slots)) * TAU  # random start: stationary
    theta = rng.uniform(0, 2 * np.pi, channels)
    alpha = (2 * np.pi * np.arange(sinusoids)[None, :] + theta[:, None]) / sinusoids
    phase = rng.uniform(0, 2 * np.pi, (channels, sinusoids))
    doppler = 2 * np.pi * fd * np.cos(alpha)  # (channels, sinusoids)
    arg = doppler[None, :, :] * t[:, None, None] + phase[None, :, :]
    return np.exp(1j * arg).sum(axis=2) / np.sqrt(sinusoids)


def ar1(rng, slots, channels, rho):
    h = (rng.standard_normal(channels) + 1j * rng.standard_normal(channels)) / np.sqrt(2)
    out = np.empty((slots, channels), dtype=complex)
    for t in range(slots):
        out[t] = h
        w = (rng.standard_normal(channels) + 1j * rng.standard_normal(channels)) / np.sqrt(2)
        h = rho * h + np.sqrt(1 - rho**2) * w
    return out


def fading(rng, model, slots, channels, fd):
    if model == "jakes":
        return jakes_sos(rng, slots, channels, fd)
    if model == "ar1-lag1":
        return ar1(rng, slots, channels, j0(2 * np.pi * fd * TAU))
    if model == "ar1-half":
        return ar1(rng, slots, channels, 0.5 ** (TAU * 2 * np.pi * fd / X_HALF))
    raise ValueError(model)


def genie_energy(gains_t, packet_bits):
    """(status, energy): exact minimum energy over one window.

    UNDELIVERABLE if even 23 dBm in every slot cannot finish; SOLVER_FAILED if the
    MILP reports anything but an optimal solution (energy is then None).
    """
    bits = np.array([shannon_bits(CFG, POWER * g / CFG.noise_power_w) for g in gains_t])
    if bits[:, -1].sum() < packet_bits:
        return UNDELIVERABLE, None
    slots, levels = bits.shape
    result = milp(
        np.tile(ENERGY, slots),
        constraints=[
            LinearConstraint(np.kron(np.eye(slots), np.ones(levels)), 0, 1),
            LinearConstraint(bits.reshape(1, -1), packet_bits, np.inf),
        ],
        integrality=np.ones(slots * levels),
        bounds=Bounds(0, 1),
        options={"disp": False},
    )
    return (OK, float(result.fun)) if result.status == 0 else (SOLVER_FAILED, None)


def packet_job(args):
    seed, gain, window, packet, model, fd, channels = args
    rng = np.random.default_rng(seed)
    power_gain = np.abs(fading(rng, model, window, channels, fd)) ** 2
    return genie_energy(gain * power_gain.max(axis=1), packet)


def episode_job(args):
    """Genie (status, energy) of all evaluated packets of one UE over one episode."""
    seed, gain, period, window, packet, fd, channels = args
    rng = np.random.default_rng(seed)
    horizon, warmup = 2000, 200
    power_gain = np.abs(jakes_sos(rng, horizon, channels, fd)) ** 2
    best = gain * power_gain.max(axis=1)
    phase = int(rng.integers(period))
    results = []
    for arrival in range(phase, horizon, period):
        deadline = arrival + window
        if warmup < deadline <= horizon:
            results.append(genie_energy(best[arrival:deadline], packet))
    return period, results


def min_slots_job(args):
    """Genie minimum number of 23 dBm slots for one packet (inf if undeliverable)."""
    seed, gain, window, packet, model, fd, channels = args
    rng = np.random.default_rng(seed)
    if fd == 0:
        power_gain = np.ones((window, channels))
    else:
        power_gain = np.abs(fading(rng, model, window, channels, fd)) ** 2
    bits = shannon_bits(CFG, POWER[-1] * gain * power_gain.max(axis=1) / CFG.noise_power_w)
    total = np.cumsum(np.sort(bits)[::-1])
    return float(np.searchsorted(total, packet) + 1) if total[-1] >= packet else np.inf


def covered_gains(rng, gain, period, packet, sigma):
    """Shadowed gains under the coverage rule of the draft (§2.1).

    Each UE keeps its position and draws a shadowing value; if its packet then
    needs more than half its window at 23 dBm on the mean channel, the UE is
    redrawn (a new position, uniform by area in the disk, and a new shadowing
    value) until it does not.
    """
    out = np.empty(gain.size)
    redrawn = draws = 0
    for u in range(gain.size):
        g = gain[u]
        first = True
        while True:
            candidate = g * 10 ** (rng.standard_normal() * sigma / 10)
            bits = shannon_bits(CFG, POWER[-1] * candidate / CFG.noise_power_w)
            if packet[u] / bits <= period[u] / 2:
                out[u] = candidate
                break
            if first:
                redrawn += 1
                first = False
            draws += 1
            g = path_gain(SPATIAL.radius_m * np.sqrt(rng.uniform()))
    return out, redrawn, draws


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, default=400, help="sampled packets per setting")
    parser.add_argument("--episode-ues", type=int, default=90, help="UEs for section 4")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sections", default="12345", help="which sections to run")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for index in range(100):
        topology, traffic = SCENARIO.instance(index)
        for ue in range(CFG.num_ues):
            rows.append(
                (
                    topology.ue_ap_gain_linear[ue],
                    int(traffic.period_slots[ue]),
                    float(traffic.packet_bits[ue]),
                )
            )
    gain = np.array([r[0] for r in rows])
    period = np.array([r[1] for r in rows])
    packet = np.array([r[2] for r in rows])

    if "1" in args.sections:
        print("1. shadowing on the mean channel, 23 dBm alone: share of UEs needing more than")
        for sigma in (0.0, 3.0, 4.0, 6.0, 8.0):
            shadowed = gain * 10 ** (rng.standard_normal(gain.size) * sigma / 10)
            need = packet / shannon_bits(CFG, POWER[-1] * shadowed / CFG.noise_power_w)
            print(
                f"  sigma {sigma:3.0f} dB: whole window {np.mean(need > period):.3f}, "
                f"3/4 {np.mean(need > 0.75 * period):.3f}, half {np.mean(need > 0.5 * period):.3f}"
            )
    covered, redrawn, draws = covered_gains(rng, gain, period, packet, 6.0)
    if "1" in args.sections:
        print(
            f"  coverage rule at sigma 6 dB (redraw position and shadowing): {redrawn} of "
            f"{gain.size} UEs redrawn ({redrawn / gain.size:.3f}), {draws} redraws in total"
        )

    if "2" in args.sections:
        print("2. complex-gain autocorrelation at lags 10/20/40 slots (theory: Jakes J0)")
        for fd in (2.0, 10.0, 50.0):
            theory = [j0(2 * np.pi * fd * lag * TAU) for lag in (10, 20, 40)]
            print(f"  fD {fd:4.0f} Hz  Jakes theory   " + "  ".join(f"{v:+.3f}" for v in theory))
            for model in ("jakes", "ar1-lag1", "ar1-half"):
                h = np.concatenate([fading(rng, model, 400, 1, fd) for _ in range(300)], axis=1)
                corr = [
                    np.mean(h[lag:] * np.conj(h[:-lag])).real / np.mean(np.abs(h) ** 2)
                    for lag in (10, 20, 40)
                ]
                print(f"             {model:<14} " + "  ".join(f"{v:+.3f}" for v in corr))
        print("   SoS power |h|^2 at fD 10 Hz (4000 processes): correlation vs J0^2")
        lags = (5, 10, 18, 24, 40)
        p = np.abs(np.concatenate([jakes_sos(rng, 60, 1, 10.0) for _ in range(4000)], axis=1)) ** 2
        mean, var = p.mean(), p.var()
        for lag in lags:
            c = np.mean((p[lag:] - mean) * (p[:-lag] - mean)) / var
            print(
                f"     lag {lag:>2}: SoS {c:+.3f}  J0^2 {j0(2 * np.pi * 10.0 * lag * TAU) ** 2:+.3f}"
            )
        print(f"     |h|^2 mean {mean:.3f}, variance {var:.3f} (Rayleigh: 1, 1)")

    pool = ProcessPoolExecutor(args.workers)
    if "3" in args.sections:
        print("3. single-UE genie packet energy / static v1 optimum (Jakes SoS unless noted)")
        sample = rng.choice(gain.size, args.packets, replace=False)
        static = {}
        for u in sample:
            bits = shannon_bits(CFG, POWER * gain[u] / CFG.noise_power_w)
            plan = optimal_packet_plan(bits, ENERGY, packet[u], int(period[u]))
            static[u] = plan.energy_j if plan is not None else np.nan
        settings = [
            ("flat, fD=10 Hz, AR(1) lag-1 (first draft)", "ar1-lag1", 10.0, 1, False, False),
            ("flat, fD=2 Hz", "jakes", 2.0, 1, False, False),
            ("flat, fD=10 Hz", "jakes", 10.0, 1, False, False),
            ("flat, fD=50 Hz", "jakes", 50.0, 1, False, False),
            (
                "selective K=4, fD=10 Hz, AR(1) lag-1 (first draft)",
                "ar1-lag1",
                10.0,
                4,
                False,
                False,
            ),
            ("selective K=4, fD=2 Hz", "jakes", 2.0, 4, False, False),
            ("selective K=4, fD=10 Hz", "jakes", 10.0, 4, False, False),
            ("selective K=4, fD=50 Hz", "jakes", 50.0, 4, False, False),
            ("selective K=4, fD=10 Hz, shadowing 6 dB + coverage", "jakes", 10.0, 4, True, False),
            ("  same, deadlines D=P-2J (8/18/36)", "jakes", 10.0, 4, True, True),
        ]
        seeds = rng.integers(1 << 62, size=sample.size)  # common random numbers
        for name, model, fd, channels, shadow, short in settings:
            jobs = []
            for n, u in enumerate(sample):
                window = int(period[u]) - (2 * int(np.ceil(0.05 * period[u])) if short else 0)
                g = covered[u] if shadow else gain[u]
                jobs.append((int(seeds[n]), g, window, packet[u], model, fd, channels))
            results = list(pool.map(packet_job, jobs, chunksize=8))
            undeliverable = sum(s == UNDELIVERABLE for s, _ in results)
            failed = sum(s == SOLVER_FAILED for s, _ in results)
            r = np.array(
                [
                    v / static[u]
                    for (s, v), u in zip(results, sample)
                    if s == OK and np.isfinite(static[u])
                ]
            )
            print(
                f"  {name:<54} undeliverable {undeliverable / sample.size:.3f}; "
                f"solver failed {failed}; ratio median {np.median(r):.3f} "
                f"p10 {np.percentile(r, 10):.3f} p90 {np.percentile(r, 90):.3f}"
            )

    if "4" in args.sections:
        print("4. per-UE episode: genie lower bound with the 5% allowance (drop the")
        print("   floor(0.05 M) most expensive packets) over the zero-loss bound;")
        print("   selective K=4, Jakes fD=10 Hz, shadowing 6 dB + coverage, D=P")
        chosen = rng.choice(gain.size, args.episode_ues, replace=False)
        jobs = [
            (
                int(rng.integers(1 << 62)),
                covered[u],
                int(period[u]),
                int(period[u]),
                packet[u],
                10.0,
                4,
            )
            for u in chosen
        ]
        by_period: dict[int, list[float]] = {}
        packets_total = undeliverable_total = failed_total = 0
        infeasible = solver_excluded = 0
        for p, results in pool.map(episode_job, jobs):
            packets_total += len(results)
            forced = sum(s == UNDELIVERABLE for s, _ in results)
            failed = sum(s == SOLVER_FAILED for s, _ in results)
            undeliverable_total += forced
            failed_total += failed
            allowed = int(np.floor(0.05 * len(results) + 1e-9))
            if failed:
                solver_excluded += 1  # a lower bound needs every deliverable packet's cost
                continue
            if forced > allowed:
                infeasible += 1  # no policy can meet the allowance: L_eps does not exist
                continue
            costs = np.sort([v for s, v in results if s == OK])
            keep = costs[: costs.size - (allowed - forced)]
            by_period.setdefault(p, []).append(keep.sum() / costs.sum())
        print(
            f"   packets {packets_total}: undeliverable {undeliverable_total}, "
            f"solver failed {failed_total}; UE-episodes infeasible for every policy "
            f"{infeasible}, excluded for solver failures {solver_excluded}"
        )
        for p in sorted(by_period):
            v = np.array(by_period[p])
            print(
                f"   period {p:>2}: {v.size} UEs, L_eps/L_0 median {np.median(v):.3f} "
                f"min {v.min():.3f} max {v.max():.3f}"
            )
    if "5" in args.sections:
        print("5. capacity scale at 23 dBm: genie minimum slots per packet / window")
        sample = rng.choice(gain.size, args.packets, replace=False)
        seeds = rng.integers(1 << 62, size=sample.size)  # common random numbers
        settings = [
            ("static, no shadowing (v1.0)", "jakes", 0.0, 1, False),
            ("static mean channel, shadowing 6 dB + coverage", "jakes", 0.0, 1, True),
            ("flat, fD=10 Hz, shadowing 6 dB + coverage", "jakes", 10.0, 1, True),
            ("selective K=4, fD=2 Hz, shadowing 6 dB + coverage", "jakes", 2.0, 4, True),
            ("selective K=4, fD=10 Hz, shadowing 6 dB + coverage", "jakes", 10.0, 4, True),
            ("selective K=4, fD=50 Hz, shadowing 6 dB + coverage", "jakes", 50.0, 4, True),
        ]
        for name, model, fd, channels, shadow in settings:
            jobs = [
                (
                    int(seeds[n]),
                    covered[u] if shadow else gain[u],
                    int(period[u]),
                    packet[u],
                    model,
                    fd,
                    channels,
                )
                for n, u in enumerate(sample)
            ]
            slots = np.array(list(pool.map(min_slots_job, jobs, chunksize=16)))
            share = slots / period[sample]
            finite = share[np.isfinite(share)]
            print(
                f"  {name:<52} undeliverable {np.mean(~np.isfinite(share)):.3f}; share mean "
                f"{finite.mean():.3f} p90 {np.percentile(finite, 90):.3f}; "
                f"K/mean share {4 / finite.mean():.1f} UEs"
            )
    pool.shutdown()


if __name__ == "__main__":
    main()
