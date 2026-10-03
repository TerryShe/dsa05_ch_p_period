"""EXP-DSA05-008 confirmation analysis exactly as pre-registered (design §7.2, §7.3).

Inputs: the formal ``evaluate_capacity --split v2-confirm1`` outputs and a
methods file frozen before the read, mapping each statistical group to its
specs (a spec may serve several groups; its episodes are evaluated once):

    {"M": [...3 selected EXP-008 checkpoints...], "M_last": [...],
     "A_near": [...], "A_original": [...], "best_local": "best_local",
     "rules": ["calibrated:...", ...], "edf": ["central_edf", ...]}

* Learned groups pool their training seeds: F(N) is the mean over seeds and
  instances, N* the first-drop point of the unrounded F (not a mean of N*).
* H1: N*_0.01(M) > N*_0.01(A_near).  H3: N*_0.01(M) > N*_0.01(best_local).
* H2 (joint): the two-sided 95% bootstrap interval of F_0.01(8; M) - F_0.01(8;
  A_near) has a lower bound > 0, and that of N = 6 has a lower bound > -0.01.
* Bootstrap: 10,000 draws, seed 20260930; instances are paired across methods
  and resampled jointly over N; the training seeds of each learned group are
  resampled on their own (``evaluation._bootstrap``). The joint distribution of
  N* and of the capacity differences comes from the same draws.
* Descriptive: A_original, last checkpoints, per seed, S_near / S_thr / S_006,
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
LEARNED = ("M", "M_last", "A_near", "A_original")
SCORES = {"S_near": (6, 8), "S_thr": (6, 8, 10, 12), "S_006": (4, 6, 8, 10, 12, 16)}
FAILURE_GRID = (4, 6, 8)
CLASSES = ("fast", "medium", "slow")
H2_SUPERIORITY_N, H2_GUARD_N, H2_GUARD = 8, 6, -0.01


def load(paths: list[Path], formal: bool = True) -> tuple[dict, dict, list[int]]:
    """spec -> {N: episodes} over all evaluation outputs, their provenance, the instances."""
    episodes: dict[str, dict] = {}
    provenance, instances = {}, None
    for path in paths:
        record = json.loads((path / "summary.json").read_text())
        if record["split"] != SET or record["protocol_id"] != splits_v2.PROTOCOL_ID:
            raise ValueError(f"{path}: not a {SET} evaluation")
        if formal and not record["formal"]:
            raise ValueError(f"{path}: not a formal evaluation")
        if record["instances"] != list(splits_v2.indices(SET)):
            raise ValueError(f"{path}: not the whole confirmation set")
        instances = record["instances"]
        provenance[str(path)] = record["provenance"]
        for spec, by_n in json.loads((path / "episodes.json").read_text()).items():
            if spec in episodes:
                raise ValueError(f"{spec} was evaluated twice")
            episodes[spec] = {int(n): rows for n, rows in by_n.items()}
    return episodes, provenance, instances


def matrix(episodes: dict, specs: list[str], n: int, key: str, instances: list[int]) -> np.ndarray:
    """[specs, instances] feasibility at N for one eps key."""
    rows = []
    for spec in specs:
        cells = {e["instance"]: float(e["feasible"][key]) for e in episodes[spec][n]}
        rows.append([cells[i] for i in instances])
    return np.array(rows)


def pooled_summary(episodes: dict, specs: list[str], grid: list[int]) -> dict:
    return summarize({n: {s: episodes[s][n] for s in specs} for n in grid})


def joint_capacity(mats: dict[str, dict[int, np.ndarray]], grid: list[int], draws: int) -> dict:
    """N* of every group and the differences M - other from the same bootstrap draws."""
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
        if other == "M":
            continue
        diff = np.array(stars["M"]) - np.array(stars[other])
        values, counts = np.unique(diff, return_counts=True)
        out[f"M-{other}"] = {
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
    learned = {g: list(methods[g]) for g in LEARNED}
    singles = {"best_local": [methods["best_local"]]}
    grid = sorted(episodes[learned["M"][0]])
    groups = {**learned, **singles}
    result: dict = {"grid": grid, "instances": len(instances), "methods": methods, "main": {}}
    summaries = {g: pooled_summary(episodes, specs, grid) for g, specs in groups.items()}
    for spec in (*methods["rules"], *methods["edf"]):
        summaries[spec] = pooled_summary(episodes, [spec], sorted(episodes[spec]))
    for g, specs in learned.items():
        for spec in specs:
            summaries[f"{g} | {spec}"] = pooled_summary(episodes, [spec], grid)
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
    tests = {}
    for eps in EPSILONS:
        key = eps_key(eps)
        mats = {
            g: {n: matrix(episodes, specs, n, key, instances) for n in grid}
            for g, specs in groups.items()
        }
        point = {g: capacity(grid, [mats[g][n].mean() for n in grid]) for g in groups}
        h2 = {}
        for n, bound in ((H2_SUPERIORITY_N, 0.0), (H2_GUARD_N, H2_GUARD)):
            interval = _interval(_bootstrap(mats["M"][n], mats["A_near"][n], draws, BOOTSTRAP_SEED))
            h2[str(n)] = {
                "difference": float(mats["M"][n].mean() - mats["A_near"][n].mean()),
                "ci95": interval,
                "bound": bound,
                "holds": interval[0] > bound,
            }
        scores = {}
        for score, score_grid in SCORES.items():
            m = np.mean([mats["M"][n] for n in score_grid], axis=0)
            row = {"M": float(m.mean())}
            for other in ("A_near", "A_original", "best_local"):
                o = np.mean([mats[other][n] for n in score_grid], axis=0)
                row[other] = float(o.mean())
                row[f"M-{other}"] = {
                    "difference": float(m.mean() - o.mean()),
                    "ci95": _interval(_bootstrap(m, o, draws, BOOTSTRAP_SEED)),
                }
            scores[score] = row
        edf = summaries[methods["edf"][0]]["per_eps"][key]["capacity"]["n_star"]
        tests[key] = {
            "N_star": {g: capacity_label(point[g], grid) for g in groups},
            "H1_M_greater_than_A_near": point["M"]["n_star"] > point["A_near"]["n_star"],
            "H3_M_greater_than_best_local": point["M"]["n_star"] > point["best_local"]["n_star"],
            "H2": {**h2, "holds": all(row["holds"] for row in h2.values())},
            "joint_capacity_bootstrap": joint_capacity(
                {g: mats[g] for g in ("M", "A_near", "A_original", "best_local")}, grid, draws
            ),
            "scores": scores,
            "gap_to_edf": {g: edf - point[g]["n_star"] for g in groups},
        }
    main = tests[eps_key(0.01)]
    result["tests"] = tests
    result["verdict_eps_0.01"] = {
        "H1": main["H1_M_greater_than_A_near"],
        "H2": main["H2"]["holds"],
        "H3": main["H3_M_greater_than_best_local"],
        "all": main["H1_M_greater_than_A_near"]
        and main["H2"]["holds"]
        and main["H3_M_greater_than_best_local"],
    }
    labels = {f"M {i}": spec for i, spec in enumerate(learned["M"])}
    labels.update({f"A_near {i}": spec for i, spec in enumerate(learned["A_near"])})
    labels["best_local"] = methods["best_local"]
    labels["edf"] = methods["edf"][0]
    result["failure_composition"] = {str(n): failures(episodes, labels, n) for n in FAILURE_GRID}
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--methods", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--allow-informal", action="store_true", help="smoke tests only")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output {args.output} already exists")
    if args.resamples != BOOTSTRAP_RESAMPLES and not args.allow_informal:
        parser.error("the formal analysis uses the protocol's bootstrap size")
    episodes, provenance, instances = load(args.evaluations, not args.allow_informal)
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
