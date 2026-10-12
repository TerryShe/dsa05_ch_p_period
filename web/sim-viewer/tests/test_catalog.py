import json
from dataclasses import replace

import pytest
from dsa_period import splits_v2

from server.catalog import Catalog, Hidden, Invalid, Results, report_committed
from server.registry import EXP011


def test_gate_shows_a_committed_report(settings):
    assert report_committed(settings, EXP011)
    assert Catalog(settings).listing() == [
        {"id": "exp011", "title": EXP011.title, "report": EXP011.report, "shown": True}
    ]


def test_gate_hides_an_experiment_before_its_report(before_report):
    catalog = Catalog(before_report)
    assert catalog.listing()[0]["shown"] is False
    with pytest.raises(Hidden):
        catalog.get("exp011")


def test_gate_hides_an_uncommitted_report(settings):
    running = replace(EXP011, id="running", report="docs/results/not-committed/report.md")
    catalog = Catalog(settings, (running,))
    assert catalog.listing()[0]["shown"] is False
    with pytest.raises(Hidden):
        catalog.get("running")


def test_recomputed_p2_and_p1_equal_the_analysis(results):
    """Seed-layer means per deployment reproduce every P2 and P1 of analysis.json."""
    for group, grid in results.grids.items():
        for n in grid:
            assert results.matrix(group, n, "w").mean() == pytest.approx(
                results.analysis["P2"][group][str(n)], rel=1e-12, abs=0
            )
            assert results.matrix(group, n, "feasible").mean() == pytest.approx(
                results.analysis["P1"][group][str(n)], rel=1e-12, abs=0
            )


def test_paired_sides_are_aligned_with_the_instances(results):
    body = results.paired("U", "M", 8)
    assert body["instances"] == splits_v2.screening_instances("exp011-eval512")
    assert len(body["a"]["w"]) == len(body["b"]["w"]) == 512
    assert body["a"]["layers"] == body["b"]["layers"] == 3
    assert body["a"]["P2"] == pytest.approx(body["a"]["P2_analysis"], rel=1e-12, abs=0)


def test_groups_outside_their_grid_are_refused(results):
    with pytest.raises(KeyError):
        results.specs("U0", 6)  # U0 is evaluated at N = 8, 10, 12 only


def test_deployment_records(results):
    body = results.deployment(5003, 8, ["U", "M"])
    assert len(body["records"]["distance_m"]) == 8
    u = body["groups"]["U"]
    assert len(u["layers"]) == 3
    assert u["w"] == pytest.approx(sum(layer["w"] for layer in u["layers"]) / 3)


def _fake_outputs(tmp_path, summary_changes):
    root = tmp_path / "exp011"
    chunk = root / "eval" / "x-n8"
    chunk.mkdir(parents=True)
    (root / "plan-exp011.json").write_text(json.dumps({"groups": {"X": "x"}, "grids": {"X": [8]}}))
    (root / "analysis.json").write_text("{}")
    summary = {
        "protocol_id": splits_v2.PROTOCOL_ID,
        "formal": True,
        "provenance": {"commit": "0" * 40, "dirty": False},
        "split": "dev",
        "screen": "exp011-eval512",
        "instances": splits_v2.screening_instances("exp011-eval512"),
        "grid": [8],
        "policies": ["x"],
    }
    (chunk / "summary.json").write_text(json.dumps({**summary, **summary_changes}))
    return tmp_path


@pytest.mark.parametrize(
    "changes",
    [
        {"formal": False},
        {"provenance": {"commit": "0" * 40, "dirty": True}},
        {"screen": "exp010-eval512"},
        {"split": "test"},
        {"instances": [5000]},
    ],
)
def test_non_formal_outputs_are_refused(settings, tmp_path, changes):
    outputs = _fake_outputs(tmp_path, changes)
    with pytest.raises(Invalid):
        Results(replace(settings, outputs=outputs), EXP011)
