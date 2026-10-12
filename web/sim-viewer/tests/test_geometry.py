import pytest

from server.geometry import deployment_geometry
from server.guards import Refused, check_geometry, check_records
from server.registry import EXP011


@pytest.mark.parametrize(
    "instance",
    [
        30_000,  # v2-confirm1, the unread final confirmation set
        30_500,  # v2-confirm2
        6_000,  # exp011b-eval512
        7_000,  # block reserved for exp011b-stress
        2_000,  # exp008-screen256
        3_000,  # exp009-screen256
        4_000,  # exp010-eval512: read, but not a record of a shown experiment
        0,
    ],
)
def test_geometry_only_for_records_of_shown_experiments(results, instance):
    with pytest.raises(Refused):
        check_geometry(results, instance, 8)


def test_geometry_needs_an_evaluated_n(results):
    with pytest.raises(Refused):
        check_geometry(results, 5003, 20)
    check_geometry(results, 5003, 8)


@pytest.mark.parametrize("n", [6, 8, 10, 12, 16])
def test_regenerated_geometry_equals_the_formal_records(settings, results, n):
    rows = results.rows("best_local", n)
    for instance in results.instances[:32]:
        geometry = deployment_geometry(settings.repo / EXP011.config, n, instance)
        check_records(geometry, rows[instance])
        assert len(geometry["positions_m"]) == n


def test_positions_are_nested_across_n(settings):
    small = deployment_geometry(settings.repo / EXP011.config, 8, 5003)
    large = deployment_geometry(settings.repo / EXP011.config, 16, 5003)
    assert large["positions_m"][:8] == small["positions_m"]


def test_a_changed_record_is_detected(settings, results):
    geometry = deployment_geometry(settings.repo / EXP011.config, 8, 5003)
    record = dict(results.rows("best_local", 8)[5003])
    record["distance_m"] = [d + 1e-9 for d in record["distance_m"]]
    with pytest.raises(ValueError):
        check_records(geometry, record)
