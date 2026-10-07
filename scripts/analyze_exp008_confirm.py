"""Confirmation analysis of EXP-DSA05-008/009 exactly as pre-registered.

EXP-008 design §7.2-§7.3; EXP-009 design §6 (same rules, its own set and
groups). Inputs: the formal ``evaluate_capacity --split <set>`` outputs and a
methods file frozen before the read. Each learned group lists its training
seeds; a seed is either one spec used at every N (a cross-N single policy) or a
seed layer {"N": spec} with one model per N (EXP-009, protocol v2 §4; N >= 24
map to the N = 20 model explicitly). A spec may serve several groups; its
episodes are evaluated once:

    {"primary": "M", "baseline": "A_near",
     "M": [...], "M_last": [...], "A_near": [...], "A_original": [...],
     "best_local": "best_local", "rules": ["calibrated:...", ...],
     "edf": ["central_edf", ...]}

``primary`` and ``baseline`` default to "M" and "A_near" (EXP-008); EXP-009
uses "A_mix" as its baseline. Every other list-valued key except rules and edf
is a descriptive learned group; keys starting with "_" are metadata.

* Learned groups pool their training seeds: F(N) is the mean over seeds and
  instances, N* the first-drop point of the unrounded F (not a mean of N*).
* H1: N*_0.01(primary) > N*_0.01(baseline).  H3: N*_0.01(primary) > N*_0.01(best_local).
* H2 (joint): the two-sided 95% bootstrap interval of F_0.01(8; primary) -
  F_0.01(8; baseline) has a lower bound > 0, and that of N = 6 a lower bound > -0.01.
* Bootstrap: 10,000 draws, seed 20260930; instances are paired across methods
  and resampled jointly over N; the training seeds of each learned group are
  resampled on their own (``evaluation._bootstrap``). The joint distribution of
  N* and of the capacity differences comes from the same draws.
* Descriptive: other learned groups, per seed, S_near / S_thr / S_006,
  eps = 0.05, failure composition at N = 4, 6, 8, the gap to N_EDF.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dsa_period import splits_v2
from dsa_period.capacity import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EPSILONS,
    allowance,
    capacity,
    capacity_label,
    eps_key,
    summarize,
)
from dsa_period.evaluation import _bootstrap, _interval

SET = "v2-confirm1"
NOT_LEARNED = ("primary", "baseline", "best_local", "rules", "edf")
SCORES = {"S_near": (6, 8), "S_thr": (6, 8, 10, 12), "S_006": (4, 6, 8, 10, 12, 16)}
FAILURE_GRID = (4, 6, 8)
CLASSES = ("fast", "medium", "slow")
H2_SUPERIORITY_N, H2_GUARD_N, H2_GUARD = 8, 6, -0.01


def load(
    paths: list[Path], formal: bool = True, confirm_set: str = SET
) -> tuple[dict, dict, list[int]]:
    """spec -> {N: episodes} over all evaluation outputs, their provenance, the instances."""
    episodes: dict[str, dict] = {}
    provenance, instances = {}, None
    for path in paths:
        record = json.loads((path / "summary.json").read_text())
        if record["split"] != confirm_set or record["protocol_id"] != splits_v2.PROTOCOL_ID:
            raise ValueError(f"{path}: not a {confirm_set} evaluation")
        if formal and not record["formal"]:
            raise ValueError(f"{path}: not a formal evaluation")
        if record["instances"] != list(splits_v2.indices(confirm_set)):
            raise ValueError(f"{path}: not the whole confirmation set")
        instances = record["instances"]
        provenance[str(path)] = record["provenance"]
        for spec, by_n in json.loads((path / "episodes.json").read_text()).items():
            for n, rows in by_n.items():
                if int(n) in episodes.get(spec, {}):
                    raise ValueError(f"{spec} at N = {n} was evaluated twice")
                episodes.setdefault(spec, {})[int(n)] = rows
    return episodes, provenance, instances


def spec_at(member, n: int) -> str:
    """The spec a seed uses at N: one spec for all N, or its layer's model for N."""
    return member if isinstance(member, str) else member[str(n)]


def grid_of(episodes: dict, member) -> list[int]:
    return sorted(episodes[member]) if isinstance(member, str) else sorted(int(n) for n in member)


def matrix(episodes: dict, members: list, n: int, key: str, instances: list[int]) -> np.ndarray:
    """[seeds, instances] feasibility at N for one eps key."""
    rows = []
    for member in members:
        cells = {e["instance"]: float(e["feasible"][key]) for e in episodes[spec_at(member, n)][n]}
        rows.append([cells[i] for i in instances])
    return np.array(rows)


def pooled_summary(episodes: dict, members: list, grid: list[int]) -> dict:
    return summarize(
        {n: {f"seed{i}": episodes[spec_at(m, n)][n] for i, m in enumerate(members)} for n in grid}
    )


def joint_capacity(
    mats: dict[str, dict[int, np.ndarray]], grid: list[int], draws: int, primary: str = "M"
) -> dict:
    """N* of every group and the differences primary - other from the same bootstrap draws."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    names = list(mats)
    num_instances = mats[names[0]][grid[0]].shape[1]
    stars = {name: [] for name in names}
    for _ in range(draws):
        inst = rng.integers(num_instances, size=num_instances)
        for name in names:
            seeds_n = mats[name][grid[0]].shape[0]
            seeds = rng.integers(seeds_n, size=seeds_n)
            point = capacity(grid, [mats[name][n][seeds][:, inst].mean() for n in grid])
            stars[name].append(point["n_star"])
    out: dict = {}
    for name in names:
        values, counts = np.unique(stars[name], return_counts=True)
        out[name] = {str(int(v)): c / draws for v, c in zip(values, counts)}
    for other in names:
        if other == primary:
            continue
        diff = np.array(stars[primary]) - np.array(stars[other])
        values, counts = np.unique(diff, return_counts=True)
        out[f"{primary}-{other}"] = {
            "distribution": {str(int(v)): c / draws for v, c in zip(values, counts)},
            "share_positive": float((diff > 0).mean()),
        }
    return out


def failures(episodes: dict, specs: dict[str, str], n: int) -> dict:
    rows = {}
    for label, spec in specs.items():
        failed = []
        for e in episodes[spec][n]:
            if e["feasible"]["0.01"]:
                continue
            ues = [
                {
                    "ue": u,
                    "class": CLASSES[e["class_index"][u]],
                    "distance_m": round(e["distance_m"][u], 1),
                    "undeliverable_share_of_allowance": round(
                        e["undeliverable"][u] / max(allowance(0.01, e["packets"][u]), 1), 3
                    ),
                    "violation_rate": round(rate, 4),
                }
                for u, rate in enumerate(e["violation_rate"])
                if rate > 0.01
            ]
            failed.append({"instance": e["instance"], "ues": ues})
        rows[label] = failed
    return rows


def analyze(episodes: dict, methods: dict, instances: list[int], draws: int) -> dict:
    primary = methods.get("primary", "M")
    baseline = methods.get("baseline", "A_near")
    learned = {
        g: list(v) for g, v in methods.items() if g not in NOT_LEARNED and not g.startswith("_")
    }
    groups = {**learned, "best_local": [methods["best_local"]]}
    grid = grid_of(episodes, learned[primary][0])
    result: dict = {"grid": grid, "instances": len(instances), "methods": methods, "main": {}}
    summaries = {g: pooled_summary(episodes, members, grid) for g, members in groups.items()}
    for spec in (*methods["rules"], *methods["edf"]):
        summaries[spec] = pooled_summary(episodes, [spec], sorted(episodes[spec]))
    for g, members in learned.items():
        for i, member in enumerate(members):
            summaries[f"{g} | seed {i}"] = pooled_summary(episodes, [member], grid)
    for eps in EPSILONS:
        key = eps_key(eps)
        result["main"][key] = {
            name: {
                "N_star": capacity_label(s["per_eps"][key]["capacity"], s["grid"]),
                "F": {r["num_ues"]: r["feasible_share"] for r in s["per_eps"][key]["rows"]},
                "ci95": {r["num_ues"]: r["ci95"] for r in s["per_eps"][key]["rows"]},
                "capacity_bootstrap": s["per_eps"][key]["capacity_bootstrap"],
            }
            for name, s in summaries.items()
        }
    others = [g for g in groups if g != primary and g != f"{primary}_last"]
    tests = {}
    for eps in EPSILONS:
        key = eps_key(eps)
        mats = {
            g: {n: matrix(episodes, members, n, key, instances) for n in grid}
            for g, members in groups.items()
        }
        point = {g: capacity(grid, [mats[g][n].mean() for n in grid]) for g in groups}
        h2 = {}
        for n, bound in ((H2_SUPERIORITY_N, 0.0), (H2_GUARD_N, H2_GUARD)):
            m, b = mats[primary][n], mats[baseline][n]
            interval = _interval(_bootstrap(m, b, draws, BOOTSTRAP_SEED))
            h2[str(n)] = {
                "difference": float(m.mean() - b.mean()),
                "ci95": interval,
                "bound": bound,
                "holds": interval[0] > bound,
            }
        scores = {}
        for score, score_grid in SCORES.items():
            m = np.mean([mats[primary][n] for n in score_grid], axis=0)
            row = {primary: float(m.mean())}
            for other in others:
                o = np.mean([mats[other][n] for n in score_grid], axis=0)
                row[other] = float(o.mean())
                row[f"{primary}-{other}"] = {
                    "difference": float(m.mean() - o.mean()),
                    "ci95": _interval(_bootstrap(m, o, draws, BOOTSTRAP_SEED)),
                }
            scores[score] = row
        edf = summaries[methods["edf"][0]]["per_eps"][key]["capacity"]["n_star"]
        tests[key] = {
            "N_star": {g: capacity_label(point[g], grid) for g in groups},
            "H1_primary_gt_baseline": point[primary]["n_star"] > point[baseline]["n_star"],
            "H3_primary_gt_best_local": point[primary]["n_star"] > point["best_local"]["n_star"],
            "H2": {**h2, "holds": all(row["holds"] for row in h2.values())},
            "joint_capacity_bootstrap": joint_capacity(
                {g: mats[g] for g in (primary, *others)}, grid, draws, primary
            ),
            "scores": scores,
            "gap_to_edf": {g: edf - point[g]["n_star"] for g in groups},
        }
    main = tests[eps_key(0.01)]
    result["tests"] = tests
    verdict = {
        "H1": main["H1_primary_gt_baseline"],
        "H2": main["H2"]["holds"],
        "H3": main["H3_primary_gt_best_local"],
    }
    result["verdict_eps_0.01"] = {**verdict, "all": all(verdict.values())}
    result["failure_composition"] = {}
    for n in FAILURE_GRID:
        labels = {f"{primary} {i}": spec_at(m, n) for i, m in enumerate(learned[primary])}
        labels.update({f"{baseline} {i}": spec_at(m, n) for i, m in enumerate(learned[baseline])})
        labels["best_local"] = methods["best_local"]
        labels["edf"] = methods["edf"][0]
        result["failure_composition"][str(n)] = failures(episodes, labels, n)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--methods", type=Path, required=True)
    parser.add_argument("--set", default=SET, choices=tuple(splits_v2.CONFIRMATION_SETS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--allow-informal", action="store_true", help="smoke tests only")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    if args.resamples != BOOTSTRAP_RESAMPLES and not args.allow_informal:
        parser.error("the formal analysis uses the protocol's bootstrap size")
    episodes, provenance, instances = load(args.evaluations, not args.allow_informal, args.set)
    methods = json.loads(args.methods.read_text())
    result = analyze(episodes, methods, instances, args.resamples)
    result["provenance"] = provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    verdict = result["verdict_eps_0.01"]
    t = result["tests"][eps_key(0.01)]
    print(f"eps=0.01 N*: {t['N_star']}")
    for n in (str(H2_SUPERIORITY_N), str(H2_GUARD_N)):
        h = t["H2"][n]
        print(f"  H2 N={n}: {h['difference']:+.4f} {h['ci95']} bound {h['bound']} -> {h['holds']}")
    print(f"H1 {verdict['H1']} H2 {verdict['H2']} H3 {verdict['H3']} -> all {verdict['all']}")


if __name__ == "__main__":
    main()
