"""FastAPI transport for bounded, reproducible research experiments."""

import json
import logging
import threading
from dataclasses import asdict, replace
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from research.catalog import ALGORITHMS, GAMES, algorithms_for_game
from research.exports import export_archive, summary_rows
from research.simulation import ExperimentConfig, run_experiment
from ui.plots import diagnostics_figure, strategy_figure, trajectory_figure

from .guards import (
    RequestBodyLimitMiddleware,
    RequestBudgetMiddleware,
    ResourceConcurrencyMiddleware,
)
from .store import RequestLimiter, ResultStore, ResultTooLarge, ResultUnavailable

LOGGER = logging.getLogger(__name__)
LIMITS = {"max_steps": 200000, "max_runs": 5, "max_iterations": 100000}


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    game_name: str = Field(min_length=1, max_length=100)
    algorithm_names: list[Annotated[str, Field(min_length=1, max_length=100)]] = Field(
        min_length=1, max_length=9
    )
    iterations: int = Field(default=1000, ge=1, le=100000)
    runs: int = Field(default=3, ge=1, le=5)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    learning_rate: float = Field(default=0.2, gt=0)
    decay: float = -0.5
    temperature: float = Field(default=0.1, gt=0)
    exploration: float = Field(default=0.1, ge=0, le=1)
    initialization: Literal["random", "uniform", "biased"] = "random"


def _catalog():
    games = []
    for spec in GAMES.values():
        table = spec.factory().expected_payoff_matrix
        games.append(
            {
                "name": spec.name,
                "category": spec.category,
                "num_players": spec.num_players,
                "action_labels": list(spec.action_labels),
                "description": spec.description,
                "algorithms": list(algorithms_for_game(spec.name)),
                "equilibria": [profile.tolist() for profile in spec.equilibria],
                "payoffs": [
                    {"actions": list(actions), "values": table[actions].tolist()}
                    for actions in np.ndindex(*table.shape[:-1])
                ],
            }
        )
    return {
        "games": games,
        "algorithms": [
            {
                "name": algorithm.name,
                "description": algorithm.description,
                "feedback": algorithm.feedback,
            }
            for algorithm in ALGORITHMS.values()
        ],
        "limits": dict(LIMITS),
    }


def _summary(result):
    return {
        "run_count": len(result.runs),
        "runtime_seconds": result.runtime_seconds,
        "final_nash_gap": float(np.mean([run.nash_gap[-1] for run in result.runs])),
        "expected_regret": float(np.mean([run.average_regret[-1] for run in result.runs])),
        "mean_payoffs": np.mean([run.payoffs.mean(axis=0) for run in result.runs], axis=0).tolist(),
    }


def create_app(store=None, limiter=None, engine=None, static_dir=None):
    """Create an isolated application; injected clocks/engine support endpoint tests."""
    application = FastAPI(title="RegretMinDynamics", docs_url=None, redoc_url=None)
    application.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
    application.add_middleware(RequestBodyLimitMiddleware)
    application.state.store = ResultStore() if store is None else store
    application.state.limiter = RequestLimiter() if limiter is None else limiter
    application.state.engine = run_experiment if engine is None else engine
    application.state.simulation_slot = threading.BoundedSemaphore(1)
    application.state.render_slots = threading.BoundedSemaphore(2)
    application.state.archive_slot = threading.BoundedSemaphore(1)
    application.add_middleware(
        ResourceConcurrencyMiddleware,
        render_slots=application.state.render_slots,
        archive_slot=application.state.archive_slot,
    )
    application.add_middleware(RequestBudgetMiddleware, limiter=application.state.limiter)
    catalog_json = json.dumps(_catalog(), allow_nan=False)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request, error):
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Invalid experiment or chart settings.",
                "errors": [
                    {"field": ".".join(str(part) for part in item["loc"]), "message": item["msg"]}
                    for item in error.errors()
                ],
            },
        )

    def result_for(identifier):
        try:
            return application.state.store.get(identifier)
        except ResultUnavailable:
            raise HTTPException(
                410, "This temporary experiment expired or was evicted. Run it again."
            ) from None

    @application.get("/health")
    def health():
        return {"status": "ok"}

    @application.get("/api/catalog")
    def catalog():
        return Response(catalog_json, media_type="application/json")

    @application.post("/api/experiments")
    def experiment(settings: ExperimentRequest):
        config = ExperimentConfig(
            **{**settings.model_dump(), "algorithm_names": tuple(settings.algorithm_names)}
        )
        try:
            config.validate()
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        if config.iterations * config.runs * len(config.algorithm_names) > LIMITS["max_steps"]:
            raise HTTPException(
                422, "Experiments support at most 200,000 total iteration/run/algorithm steps."
            )
        if not application.state.simulation_slot.acquire(blocking=False):
            raise HTTPException(
                429,
                "Another experiment is running. Try again shortly.",
                headers={"Retry-After": "2"},
            )
        try:
            result = application.state.engine(config)
            identifier = application.state.store.add(result)
            return {
                "id": identifier,
                "config": asdict(result.config),
                "summary": _summary(result),
                "empirical_available": any(
                    run.empirical_strategies is not None for run in result.runs
                ),
                "comparison": [{**row, "seed": str(row["seed"])} for row in summary_rows(result)],
            }
        except ResultTooLarge:
            raise HTTPException(
                413, "The result exceeds temporary storage capacity. Reduce the experiment size."
            ) from None
        except ValueError:
            LOGGER.warning("Experiment could not complete with supplied settings", exc_info=True)
            raise HTTPException(
                422,
                "The experiment could not complete with these settings. Try a lower learning rate.",
            ) from None
        except Exception:
            LOGGER.exception("Experiment execution failed")
            raise HTTPException(
                500, "The experiment could not complete. Please try again."
            ) from None
        finally:
            application.state.simulation_slot.release()

    @application.get("/api/experiments/{identifier}/figure")
    def figure(
        identifier: str,
        kind: Literal["trajectory", "diagnostics", "strategies"] = "trajectory",
        view: Literal[
            "Current policy", "Time-average policy", "Empirical frequencies"
        ] = "Current policy",
        animate: bool = False,
        metric: Literal["nash_gap", "average_regret", "payoffs"] = "nash_gap",
    ):
        result = result_for(identifier)
        if kind == "trajectory" and view == "Empirical frequencies":
            eligible = tuple(run for run in result.runs if run.empirical_strategies is not None)
            if not eligible:
                raise HTTPException(422, "This experiment has no recorded empirical frequencies.")
            result = replace(result, runs=eligible)
        try:
            if kind == "trajectory":
                chart = trajectory_figure(
                    result, GAMES[result.config.game_name], view=view, animate=animate
                )
            elif kind == "diagnostics":
                chart = diagnostics_figure(result, metric=metric)
            else:
                chart = strategy_figure(result, GAMES[result.config.game_name])
            return Response(
                chart.to_json(),
                media_type="application/json",
                headers={"Cache-Control": "no-store"},
            )
        except Exception:
            LOGGER.exception("Experiment figure rendering failed")
            raise HTTPException(
                500, "The figure could not be prepared. Please try again."
            ) from None

    @application.get("/api/experiments/{identifier}/download")
    def download(identifier: str):
        result = result_for(identifier)
        try:
            return Response(
                export_archive(result),
                media_type="application/zip",
                headers={
                    "Content-Disposition": 'attachment; filename="regretmin-experiment.zip"',
                    "Cache-Control": "no-store",
                    # The archive is already compressed; skip redundant gzip buffering.
                    "Content-Encoding": "identity",
                },
            )
        except Exception:
            LOGGER.exception("Experiment archive creation failed")
            raise HTTPException(
                500, "The archive could not be prepared. Please try again."
            ) from None

    @application.api_route(
        "/api", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
    )
    @application.api_route(
        "/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
    )
    def api_not_found(path: str = ""):
        raise HTTPException(404, "API endpoint not found.")

    assets = (
        Path(static_dir)
        if static_dir is not None
        else Path(__file__).resolve().parents[1] / "frontend" / "dist"
    )
    if assets.is_dir() and (assets / "index.html").is_file():
        application.mount("/", StaticFiles(directory=assets, html=True), name="frontend")
    else:

        @application.get("/")
        def frontend_not_built():
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Frontend assets are not built. Use the Vite dev server, or build frontend/ and restart the API.",
                },
            )

    return application


app = create_app()
