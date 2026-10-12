"""Formal results of the shown experiments, read only (README §2.1, §4.1).

An experiment is shown only when its report exists on the gate ref. Its
evaluation chunks must be formal, clean-tree evaluations of the registered set;
anything else is refused rather than shown. Per-deployment values follow
``scripts/analyze_exp011.py``: a group's value at a deployment is the mean over
its seed layers, and P2 is the mean of those values.
"""

import json
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
from dsa_period import splits_v2

from server.registry import EXPERIMENTS, Experiment
from server.settings import Settings

FIELDS = {
    "w": lambda row: float(row["worst_violation_rate"]),
    "feasible": lambda row: float(row["feasible"]["0.01"]),
}


class Hidden(Exception):
    """The experiment's report is not committed on the gate ref."""


class Invalid(Exception):
    """The outputs are not formal evaluations of the registered set."""


def report_committed(settings: Settings, experiment: Experiment) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{settings.gate_ref}:{experiment.report}"],
        cwd=settings.main, capture_output=True,
    )  # fmt: skip
    return result.returncode == 0


@lru_cache(maxsize=48)
def _episodes(chunk: Path) -> dict:
    return json.loads((chunk / "episodes.json").read_text())


class Results:
    """The formal evaluations of one experiment."""

    def __init__(self, settings: Settings, experiment: Experiment) -> None:
        root = settings.outputs / experiment.outputs
        self.experiment = experiment
        self.plan = json.loads((root / experiment.plan).read_text())
        self.analysis = json.loads((root / experiment.analysis).read_text())
        self.instances = splits_v2.screening_instances(experiment.evaluation_set)
        self.chunks: dict[tuple[str, int], Path] = {}
        self.commits: set[str] = set()
        for path in sorted((root / "eval").glob("*/summary.json")):
            record = json.loads(path.read_text())
            ok = (
                record["protocol_id"] == splits_v2.PROTOCOL_ID
                and record["split"] == "dev"
                and record.get("screen") == experiment.evaluation_set
                and record["instances"] == self.instances
                and record["formal"]
                and not record["provenance"]["dirty"]
            )
            if not ok:
                raise Invalid(f"{path.parent}: not a formal {experiment.evaluation_set} evaluation")
            for spec in record["policies"]:
                for n in record["grid"]:
                    if (spec, int(n)) in self.chunks:
                        raise Invalid(f"{spec} at N = {n} was evaluated twice")
                    self.chunks[spec, int(n)] = path.parent
            self.commits.add(record["provenance"]["commit"])
        self.grids = {g: [int(n) for n in grid] for g, grid in self.plan["grids"].items()}

    def specs(self, group: str, n: int) -> list[str]:
        """One spec per seed layer of a group at N."""
        if n not in self.grids.get(group, []):
            raise KeyError(f"{group} is not evaluated at N = {n}")
        members = self.plan["groups"][group]
        members = members if isinstance(members, list) else [members]
        return [m if isinstance(m, str) else m[str(n)] for m in members]

    def rows(self, spec: str, n: int) -> dict[int, dict]:
        rows = {row["instance"]: row for row in _episodes(self.chunks[spec, n])[spec][str(n)]}
        if sorted(rows) != self.instances:
            raise Invalid(f"{spec} at N = {n} does not cover the registered instances once each")
        return rows

    def matrix(self, group: str, n: int, field: str) -> np.ndarray:
        """(seed layers, deployments) in registered instance order."""
        layers = [self.rows(spec, n) for spec in self.specs(group, n)]
        return np.array([[FIELDS[field](rows[i]) for i in self.instances] for rows in layers])

    def side(self, group: str, n: int) -> dict:
        w = self.matrix(group, n, "w")
        return {
            "group": group,
            "layers": len(w),
            "w": w.mean(axis=0).tolist(),
            "P2": float(w.mean()),
            "P2_analysis": self.analysis["P2"][group][str(n)],
            "P1": float(self.matrix(group, n, "feasible").mean()),
        }

    def paired(self, a: str, b: str, n: int) -> dict:
        return {"n": n, "instances": self.instances, "a": self.side(a, n), "b": self.side(b, n)}

    def deployment(self, instance: int, n: int, groups: list[str]) -> dict:
        """Per-UE records of one deployment; deployment fields must agree across all layers."""
        if instance not in self.instances:
            raise KeyError(f"instance {instance} is not in {self.experiment.evaluation_set}")
        common, per_group = None, {}
        for group in groups:
            layers = []
            for spec in self.specs(group, n):
                row = self.rows(spec, n)[instance]
                fixed = {
                    k: row[k] for k in ("packets", "class_index", "distance_m", "shadowing_db")
                }
                if common is None:
                    common = fixed
                elif fixed != common:
                    raise Invalid(f"instance {instance}: deployment fields differ in {spec}")
                layers.append(
                    {
                        "spec": spec,
                        "violation_rate": row["violation_rate"],
                        "violations": row["violations"],
                        "undeliverable": row["undeliverable"],
                        "w": row["worst_violation_rate"],
                    }
                )
            rates = np.array([layer["violation_rate"] for layer in layers])
            per_group[group] = {
                "layers": layers,
                "rate_mean": rates.mean(axis=0).tolist(),
                "w": float(np.mean([layer["w"] for layer in layers])),
            }
        return {"instance": instance, "n": n, "records": common, "groups": per_group}


class Catalog:
    def __init__(self, settings: Settings, experiments: tuple[Experiment, ...] = EXPERIMENTS):
        self.settings = settings
        self.experiments = {e.id: e for e in experiments}
        self._results: dict[str, Results] = {}

    def listing(self) -> list[dict]:
        return [
            {
                "id": e.id,
                "title": e.title,
                "report": e.report,
                "shown": report_committed(self.settings, e),
            }
            for e in self.experiments.values()
        ]

    def get(self, experiment_id: str) -> Results:
        experiment = self.experiments[experiment_id]
        if not report_committed(self.settings, experiment):
            raise Hidden(f"{experiment_id}: report {experiment.report} is not committed on "
                         f"{self.settings.gate_ref}")  # fmt: skip
        if experiment_id not in self._results:
            self._results[experiment_id] = Results(self.settings, experiment)
        return self._results[experiment_id]
