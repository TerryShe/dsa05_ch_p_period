"""EXP-DSA05-011 post hoc analyses (descriptive, not pre-registered; report §6).

Run after the formal analysis on the same exp011-eval512 outputs:

* per-seed P2 of M, U and M+Y, and the catastrophic cells (one UE with
  w >= 0.5 while every other UE stays below 0.2), with the starved UE's
  distance, shadowing and strength rank (0 = strongest);
* near-AP starvation: per group and N, the (seed layer, deployment) cells whose
  UE within NEAR_M of the AP has a violation rate >= 0.5, in the evaluation
  and in the screening outputs of U seed 101's step-5M checkpoint;
* two robustness views of D_N: U without seed 101 (also against M+Y, O, L and
  U0 at N = 8, 10, 12), and every seed on the deployments without a UE within
  NEAR_M (a geometric subset chosen after seeing the failures); O - M and
  L - M (the design compares U with O and L);
* training-time summary statistics (share of flags read, marked share);
* a replay of one starved cell (``--replay N INSTANCE UE``) that counts the
  UE's actions under U seed 101, U seed 202 and M seed 101.
"""

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dsa_period.capacity import BOOTSTRAP_RESAMPLES, run_capacity_episode
from dsa_period.config import load_scenario
from dsa_period.marl.features import stack
from dsa_period.rules_v2 import make_policy_v2

ROOT = Path(__file__).parents[1]
OUT = ROOT / "outputs" / "exp011"
CONFIG = "configs/scenario-v2.toml"
GRID, CORE = (6, 8, 10, 12, 16), (8, 10, 12)
SEEDS = ("101", "202", "303")
NEAR_M = 4.0
STARVED, OTHERS_BELOW = 0.5, 0.2


def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = _module("analyze_exp011", "scripts/analyze_exp011.py")
driver = _module("run_exp011", "scripts/run_exp011.py")


def strength(episode: dict) -> np.ndarray:
    """A pathloss proxy: shadowing minus 30 log10(distance)."""
    return np.array(episode["shadowing_db"]) - 30.0 * np.log10(
        np.maximum(episode["distance_m"], 1.0)
    )


class ActionLog:
    """Wraps a policy and counts one UE's actions while it has a packet."""

    def __init__(self, inner, ue: int) -> None:
        self.inner, self.ue, self.name = inner, ue, getattr(inner, "name", "policy")

    def configure(self, *args) -> None:
        configure = getattr(self.inner, "configure", None)
        if configure is not None:
            configure(*args)

    def reset(self, rngs) -> None:
        self.inner.reset(rngs)
        self.power, self.channel = Counter(), Counter()

    def act(self, observations):
        obs = stack(observations)
        actions = np.asarray(self.inner.act(observations), dtype=np.int64)
        if obs.remaining[self.ue] > 0:
            self.power[int(actions[self.ue, 1])] += 1
            if actions[self.ue, 1] > 0:
                self.channel[int(actions[self.ue, 0])] += 1
        return actions


def replay(groups: dict, n: int, instance: int, ue: int) -> dict:
    out = {}
    for name, layer in (("U", 0), ("U", 1), ("M", 0)):
        spec = groups[name][layer][str(n)]
        scenario = load_scenario(CONFIG, n)
        log = ActionLog(make_policy_v2(spec, scenario.physical, scenario.menu), ue)
        outcome = run_capacity_episode(scenario, log, "dev", instance)
        pending = sum(log.power.values())
        out[f"{name} seed {SEEDS[layer]}"] = {
            "spec": spec,
            "violations": outcome["violations"][ue],
            "packets": outcome["packets"][ue],
            "pending_slots": pending,
            "send_share": 1.0 - log.power[0] / pending if pending else None,
            "power_levels": dict(sorted(log.power.items())),
            "channels_when_sending": dict(sorted(log.channel.items())),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=int, nargs=3, metavar=("N", "INSTANCE", "UE"))
    args = parser.parse_args(argv)
    plan = json.loads((OUT / "plan-exp011.json").read_text())
    groups = plan["groups"]
    outputs = [OUT / "eval" / driver.label_of(s, n) for s, n in driver.pairs_of(plan)]
    episodes, instances = analysis.load(outputs)

    def mat(name: str, n: int) -> np.ndarray:
        return analysis.matrix(episodes, groups[name], n, "w", instances)

    def rows(name: str, layer: int, n: int) -> list[dict]:
        spec = groups[name][layer][str(n)]
        by_instance = {e["instance"]: e for e in episodes[spec][n]}
        return [by_instance[i] for i in instances]

    def compare(a: np.ndarray, b: np.ndarray) -> dict:
        return analysis.compare(a, b, BOOTSTRAP_RESAMPLES)

    out: dict = {"near_m": NEAR_M, "starved": STARVED}
    out["P2_by_seed"] = {
        name: {n: dict(zip(SEEDS, mat(name, n).mean(axis=1).tolist())) for n in GRID}
        for name in ("M", "U", "M+Y")
    }

    catastrophic = []
    for n in GRID:
        for layer, seed in enumerate(SEEDS):
            for e in rows("U", layer, n):
                rates = np.array(e["violation_rate"])
                k = int(np.argmax(rates))
                if rates[k] >= STARVED and np.sort(rates)[-2] < OTHERS_BELOW:
                    s = strength(e)
                    catastrophic.append({
                        "N": n, "seed": seed, "instance": e["instance"], "ue": k,
                        "w": float(rates[k]), "distance_m": float(e["distance_m"][k]),
                        "shadowing_db": float(e["shadowing_db"][k]),
                        "strength_rank": int((s > s[k]).sum()),
                        "spec": groups["U"][layer][str(n)],
                    })  # fmt: skip
    out["catastrophic_U_cells"] = catastrophic

    near_starved: dict = {}
    for name in ("M", "U", "M+Y", "O", "L", "U0", "U-hide-all", "U-hide-nonurgent"):
        near_starved[name] = {}
        for n in plan["grids"][name]:
            count = 0
            for layer in range(len(groups[name])):
                for e in rows(name, layer, n):
                    near = np.array(e["distance_m"]) < NEAR_M
                    count += int((near & (np.array(e["violation_rate"]) >= STARVED)).any())
            near_starved[name][n] = count
    out["near_ue_starved_cells"] = near_starved
    has_near = {n: np.array([min(e["distance_m"]) < NEAR_M for e in rows("M", 0, n)]) for n in GRID}
    out["deployments_with_near_ue"] = {n: int(has_near[n].sum()) for n in GRID}

    screen = json.loads((OUT / "screen" / "u-seed101-step-005001472" / "episodes.json").read_text())
    screen_rows = next(iter(screen.values()))
    out["screen_u_seed101_step5M"] = {
        n: {
            "deployments_with_near_ue": sum(min(e["distance_m"]) < NEAR_M for e in r),
            "near_ue_starved": sum(
                e["violation_rate"][int(np.argmin(e["distance_m"]))] >= STARVED
                for e in r
                if min(e["distance_m"]) < NEAR_M
            ),
        }
        for n, r in ((int(k), v) for k, v in screen_rows.items())
    }

    keep = {n: ~has_near[n] for n in GRID}
    out["robustness"] = {
        n: {
            "U_without_seed101_minus_M": compare(mat("U", n)[1:], mat("M", n)),
            "without_near_ue_U_minus_M": compare(mat("U", n)[:, keep[n]], mat("M", n)[:, keep[n]]),
            **(
                {
                    "without_near_ue_U_minus_MY": compare(
                        mat("U", n)[:, keep[n]], mat("M+Y", n)[:, keep[n]]
                    ),
                    **{
                        f"U_without_seed101_minus_{other}": compare(mat("U", n)[1:], mat(other, n))
                        for other in ("M+Y", "O", "L", "U0")
                    },
                }  # fmt: skip
                if n in CORE
                else {}
            ),
            "deployments_kept": int(keep[n].sum()),
        }
        for n in GRID
    }
    out["content_vs_M"] = {
        name: {n: compare(mat(name, n), mat("M", n)) for n in GRID} for name in ("O", "L")
    }

    training = {}
    for key in ("u", "o", "l", "u0"):
        for seed in SEEDS:
            lines = (OUT / f"{key}-seed{seed}" / "training.jsonl").read_text().splitlines()
            last = [json.loads(line)["summary"] for line in lines[-200:]]
            sent = sum(s["bits_sent"] for s in last)
            training[f"{key}-seed{seed}"] = {
                "flags_read_share": sum(s["bits_read"] for s in last) / sent if sent else None,
                "marked_share": float(np.mean([s["marked_share"] for s in last])),
            }
    out["training_last_200_iterations"] = training

    if args.replay is not None:
        n, instance, ue = args.replay
        out["replay"] = {"N": n, "instance": instance, "ue": ue, **replay(groups, n, instance, ue)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    for n, row in out["robustness"].items():
        a, b = row["U_without_seed101_minus_M"], row["without_near_ue_U_minus_M"]
        print(f"N={n:<3} without seed 101 {a['difference']:+.5f} {a['ci95']}  "
              f"without near UEs {b['difference']:+.5f} {b['ci95']}")  # fmt: skip


if __name__ == "__main__":
    main()
