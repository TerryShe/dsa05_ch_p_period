"""FastAPI app; run it on 127.0.0.1 only (README §1, §8)."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from server import exp011
from server.catalog import Catalog, Hidden, Invalid
from server.geometry import deployment_geometry
from server.guards import Refused, check_geometry, check_records
from server.settings import APP_DIR, Settings, load_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    catalog = Catalog(settings)
    app = FastAPI(title="sim-viewer")

    def results(experiment_id: str):
        try:
            return catalog.get(experiment_id)
        except KeyError:
            raise HTTPException(404, f"unknown experiment {experiment_id}") from None
        except Hidden as error:
            raise HTTPException(403, str(error)) from None
        except Invalid as error:
            raise HTTPException(500, str(error)) from None

    @app.get("/api/experiments")
    def experiments() -> list[dict]:
        return catalog.listing()

    @app.get("/api/experiments/{experiment_id}")
    def experiment(experiment_id: str) -> dict:
        r = results(experiment_id)
        e = r.experiment
        return {
            "id": e.id,
            "title": e.title,
            "report": e.report,
            "evaluation_set": e.evaluation_set,
            "instances": len(r.instances),
            "commits": sorted(r.commits),
            "groups": [
                {
                    "name": g,
                    "description": e.descriptions.get(g, ""),
                    "grid": r.grids[g],
                    "layers": len(r.specs(g, r.grids[g][0])),
                }
                for g in r.plan["groups"]
            ],
            "decision": r.analysis["decision"],
            "tables": exp011.tables(r.analysis),
            "comparisons": exp011.comparisons(r.analysis),
        }

    @app.get("/api/experiments/{experiment_id}/paired")
    def paired(experiment_id: str, a: str, b: str, n: int) -> dict:
        r = results(experiment_id)
        try:
            return r.paired(a, b, n)
        except KeyError as error:
            raise HTTPException(404, str(error)) from None

    @app.get("/api/experiments/{experiment_id}/deployments/{instance}")
    def deployment(
        experiment_id: str, instance: int, n: int, groups: str = Query(..., min_length=1)
    ) -> dict:
        r = results(experiment_id)
        try:
            check_geometry(r, instance, n)
            body = r.deployment(instance, n, groups.split(","))
        except Refused as error:
            raise HTTPException(403, str(error)) from None
        except KeyError as error:
            raise HTTPException(404, str(error)) from None
        geometry = deployment_geometry(settings.repo / r.experiment.config, n, instance)
        try:
            check_records(geometry, body["records"])
        except ValueError as error:
            raise HTTPException(500, str(error)) from None
        return {**body, "geometry": geometry}

    dist = APP_DIR / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


app = create_app()
