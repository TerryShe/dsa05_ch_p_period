"""Which deployments the app may touch (README §2.1).

Phase 1 runs no policy. It only regenerates a deployment's geometry, and only for
an instance and N that have a formal record in a shown experiment. The simulation
allowlist and the hard-coded refusals arrive with the trace backend (Phase 2).
"""

from server.catalog import Results


class Refused(Exception):
    """The request would touch an instance outside what the app may read."""


def check_geometry(results: Results, instance: int, n: int) -> None:
    if instance not in results.instances:
        raise Refused(
            f"instance {instance} has no formal record in {results.experiment.evaluation_set}"
        )
    if not any(n in grid for grid in results.grids.values()):
        raise Refused(f"N = {n} was not evaluated in {results.experiment.id}")


def check_records(geometry: dict, records: dict) -> None:
    """Regenerated geometry must equal the formal record (same config and code)."""
    for key in ("distance_m", "shadowing_db", "class_index"):
        if geometry[key] != records[key]:
            raise ValueError(f"regenerated {key} differs from the formal record")
