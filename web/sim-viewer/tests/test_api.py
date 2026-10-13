import pytest
from fastapi.testclient import TestClient

from server.app import create_app


@pytest.fixture(scope="module")
def client(settings):
    return TestClient(create_app(settings))


def test_experiment_overview(client):
    body = client.get("/api/experiments/exp011").json()
    assert body["decision"] == "no evidence"
    assert body["instances"] == 512
    assert {g["name"] for g in body["groups"]} >= {"U", "M", "M+Y", "N_EDF", "CG-opt"}
    families = {f["id"]: f for f in body["comparisons"]}
    assert families["D_N"]["title"] == "U − M (main rule D_N)"
    assert [row["key"] for row in families["D_N"]["rows"]] == ["6", "8", "10", "12", "16"]


def test_paired_view(client):
    body = client.get("/api/experiments/exp011/paired", params={"a": "U", "b": "M", "n": 8}).json()
    assert body["a"]["group"] == "U" and body["b"]["group"] == "M"


def test_deployment_view(client):
    response = client.get(
        "/api/experiments/exp011/deployments/5003", params={"n": 8, "groups": "U,M+Y"}
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body["groups"]) == {"U", "M+Y"}
    assert body["geometry"]["distance_m"] == body["records"]["distance_m"]


@pytest.mark.parametrize(
    ("path", "params", "status"),
    [
        ("/api/experiments/nope", {}, 404),
        ("/api/experiments/exp011/paired", {"a": "U", "b": "nope", "n": 8}, 404),
        ("/api/experiments/exp011/paired", {"a": "U0", "b": "M", "n": 6}, 404),
        ("/api/experiments/exp011/deployments/30000", {"n": 8, "groups": "U"}, 403),
        ("/api/experiments/exp011/deployments/6000", {"n": 8, "groups": "U"}, 403),
    ],
)
def test_refusals(client, path, params, status):
    assert client.get(path, params=params).status_code == status


def test_hidden_experiment_is_refused(before_report):
    client = TestClient(create_app(before_report))
    assert client.get("/api/experiments").json()[0]["shown"] is False
    assert client.get("/api/experiments/exp011").status_code == 403
    assert client.get("/api/experiments/exp011/paired",
                      params={"a": "U", "b": "M", "n": 8}).status_code == 403  # fmt: skip
