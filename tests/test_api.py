import asyncio
import io
import json
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.server import create_app
from api.store import (
    RateLimitExceeded,
    RequestLimiter,
    ResultStore,
    ResultTooLarge,
    ResultUnavailable,
    result_size,
)
from research.simulation import ExperimentConfig, run_experiment


def settings(**overrides):
    return {
        "game_name": "Rock Paper Scissors",
        "algorithm_names": ["Hedge", "Fictitious Play"],
        "iterations": 12,
        "runs": 1,
        "seed": 19,
        **overrides,
    }


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(static_dir=tmp_path / "missing")) as connection:
        yield connection


def test_uvicorn_entrypoint_exposes_health_endpoint():
    from app import app as entrypoint

    with TestClient(entrypoint) as connection:
        response = connection.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_catalog_contains_certified_games_limits_and_action_tables(client):
    response = client.get("/api/catalog")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["games"]) == 21
    assert len(payload["algorithms"]) == 9
    assert payload["limits"] == {"max_steps": 200000, "max_runs": 5, "max_iterations": 100000}
    game = next(item for item in payload["games"] if item["name"] == "Rock Paper Scissors")
    assert game["num_players"] == 2
    assert game["action_labels"] == ["Rock", "Paper", "Scissors"]
    assert len(game["payoffs"]) == 9
    assert game["payoffs"][1] == {"actions": [0, 1], "values": [-1.0, 1.0]}
    np.testing.assert_allclose(game["equilibria"][0], 1 / 3)
    assert response.headers["content-encoding"] == "gzip"


def test_seeded_experiment_summary_and_full_archive_are_consistent(client):
    response = client.post("/api/experiments", json=settings())
    assert response.status_code == 200
    created = response.json()
    assert len(created["id"]) >= 32
    assert created["summary"]["run_count"] == 2
    assert created["empirical_available"] is True
    assert len(created["comparison"]) == 2
    archived = client.get(f"/api/experiments/{created['id']}/download")
    assert archived.status_code == 200
    assert archived.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(archived.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        arrays = np.load(io.BytesIO(archive.read("results.npz")), allow_pickle=False)
        config = ExperimentConfig(
            **{
                **manifest["config"],
                "algorithm_names": tuple(manifest["config"]["algorithm_names"]),
            }
        )
        replay = run_experiment(config)
        for index, result in enumerate(replay.runs):
            saved = arrays[manifest["runs"][index]["arrays"]["strategies"]]
            np.testing.assert_array_equal(saved, result.strategies)
            np.testing.assert_allclose(saved.sum(axis=-1), 1)
    assert created["config"] == json.loads(json.dumps(asdict(replay.config)))
    np.testing.assert_allclose(
        created["summary"]["mean_payoffs"],
        np.mean([run.payoffs.mean(axis=0) for run in replay.runs], axis=0),
    )


def test_comparison_seed_survives_javascript_numeric_json_parsing(tmp_path):
    exact_seed = 8346550854979276149

    def seeded_engine(config):
        result = run_experiment(config)
        return replace(result, runs=(replace(result.runs[0], seed=exact_seed),))

    with TestClient(create_app(engine=seeded_engine, static_dir=tmp_path)) as connection:
        response = connection.post("/api/experiments", json=settings(algorithm_names=["Hedge"]))
        assert response.status_code == 200
        # JavaScript parses JSON numbers as IEEE 754 doubles, including integer literals.
        browser_payload = json.loads(response.text, parse_int=float)
        assert browser_payload["comparison"][0]["seed"] == str(exact_seed)
        assert browser_payload["config"]["seed"] == 19
        archived = connection.get(f"/api/experiments/{response.json()['id']}/download")
        with zipfile.ZipFile(io.BytesIO(archived.content)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            assert manifest["runs"][0]["seed"] == exact_seed
            assert isinstance(manifest["runs"][0]["seed"], int)
            assert str(exact_seed) in archive.read("summary.csv").decode()


@pytest.mark.parametrize(
    "kind,parameters",
    [
        ("trajectory", {"view": "Current policy", "animate": "true"}),
        ("trajectory", {"view": "Time-average policy"}),
        ("trajectory", {"view": "Empirical frequencies"}),
        ("diagnostics", {"metric": "average_regret"}),
        ("strategies", {}),
    ],
)
def test_all_figure_views_return_plotly_json(client, kind, parameters):
    identifier = client.post("/api/experiments", json=settings()).json()["id"]
    response = client.get(
        f"/api/experiments/{identifier}/figure", params={"kind": kind, **parameters}
    )
    assert response.status_code == 200
    figure = response.json()
    assert figure["data"]
    assert "layout" in figure
    if parameters.get("animate") == "true":
        assert len(figure["frames"]) <= 80
    if parameters.get("view") == "Empirical frequencies":
        algorithms = {trace.get("meta", {}).get("algorithm") for trace in figure["data"]}
        assert "Hedge" not in algorithms


def test_unsupported_empirical_view_and_invalid_chart_parameters_are_clear(client):
    identifier = client.post("/api/experiments", json=settings(algorithm_names=["Hedge"])).json()[
        "id"
    ]
    assert (
        client.get(
            f"/api/experiments/{identifier}/figure", params={"view": "Empirical frequencies"}
        ).status_code
        == 422
    )
    assert (
        client.get(f"/api/experiments/{identifier}/figure", params={"kind": "unknown"}).status_code
        == 422
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"iterations": 100001},
        {"iterations": True},
        {"runs": 6},
        {"seed": -1},
        {"seed": 2**32},
        {"algorithm_names": []},
        {"iterations": 100000, "runs": 3},
        {"learning_rate": "NaN"},
        {"temperature": 0},
        {"exploration": 2},
        {"game_name": "unknown"},
        {"extra": "unexpected"},
    ],
)
def test_public_settings_validation_prevents_invalid_or_excess_work(client, overrides):
    response = client.post("/api/experiments", json=settings(**overrides))
    assert response.status_code == 422
    assert "detail" in response.json()


def test_busy_simulation_returns_429_without_waiting(tmp_path):
    started, finish = threading.Event(), threading.Event()

    def slow_engine(config):
        started.set()
        assert finish.wait(5)
        return run_experiment(config)

    with TestClient(create_app(engine=slow_engine, static_dir=tmp_path)) as connection:
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(connection.post, "/api/experiments", json=settings())
            assert started.wait(5)
            busy = connection.post("/api/experiments", json=settings())
            finish.set()
            assert busy.status_code == 429
            assert int(busy.headers["retry-after"]) >= 1
            assert first.result().status_code == 200


@pytest.mark.parametrize("chunked", [False, True])
def test_oversized_body_is_rejected_before_json_parsing(client, chunked):
    content = iter([b"x" * 32768] * 3) if chunked else b"x" * 65537
    response = client.post(
        "/api/experiments", content=content, headers={"content-type": "application/json"}
    )
    assert response.status_code == 413
    assert "64 KiB" in response.json()["detail"]


def test_figure_concurrency_is_bounded_to_two(client, monkeypatch):
    import api.server as server

    original = server.trajectory_figure
    started, finish = threading.Event(), threading.Event()
    lock = threading.Lock()
    active = 0

    def slow_figure(*args, **kwargs):
        nonlocal active
        with lock:
            active += 1
            if active == 2:
                started.set()
        assert finish.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(server, "trajectory_figure", slow_figure)
    identifier = client.post("/api/experiments", json=settings()).json()["id"]
    path = f"/api/experiments/{identifier}/figure"
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = [executor.submit(client.get, path) for _ in range(2)]
        assert started.wait(5)
        busy = client.get(path)
        finish.set()
        assert busy.status_code == 429
        assert int(busy.headers["retry-after"]) >= 1
        assert all(response.result().status_code == 200 for response in pending)


def test_archive_concurrency_is_bounded_to_one(client, monkeypatch):
    import api.server as server

    original = server.export_archive
    started, finish = threading.Event(), threading.Event()

    def slow_archive(*args):
        started.set()
        assert finish.wait(5)
        return original(*args)

    monkeypatch.setattr(server, "export_archive", slow_archive)
    identifier = client.post("/api/experiments", json=settings()).json()["id"]
    path = f"/api/experiments/{identifier}/download"
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(client.get, path)
        assert started.wait(5)
        busy = client.get(path)
        finish.set()
        assert busy.status_code == 429
        assert int(busy.headers["retry-after"]) >= 1
        assert pending.result().status_code == 200


@pytest.mark.parametrize("suffix,slots", [("figure", 2), ("download", 1)])
def test_response_memory_slots_remain_held_until_body_is_sent(tmp_path, suffix, slots):
    application = create_app(static_dir=tmp_path)
    with TestClient(application) as connection:
        identifier = connection.post("/api/experiments", json=settings()).json()["id"]
        path = f"/api/experiments/{identifier}/{suffix}"

        async def slow_clients():
            release = asyncio.Event()
            blocked = [asyncio.Event() for _ in range(slots)]

            async def request(index):
                consumed = False

                async def receive():
                    nonlocal consumed
                    if not consumed:
                        consumed = True
                        return {"type": "http.request", "body": b"", "more_body": False}
                    await asyncio.Event().wait()

                async def send(message):
                    if message["type"] == "http.response.body" and message.get("body"):
                        blocked[index].set()
                        await release.wait()

                scope = {
                    "type": "http",
                    "http_version": "1.1",
                    "method": "GET",
                    "scheme": "http",
                    "path": path,
                    "raw_path": path.encode(),
                    "query_string": b"",
                    "headers": [],
                    "client": ("testclient", 50000),
                    "server": ("testserver", 80),
                    "root_path": "",
                }
                await application(scope, receive, send)

            pending = [asyncio.create_task(request(index)) for index in range(slots)]
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(event.wait() for event in blocked)), timeout=5
                )
                assert connection.get(path).status_code == 429
            finally:
                release.set()
                await asyncio.gather(*pending)

        asyncio.run(slow_clients())


@pytest.mark.parametrize("suffix,slots", [("figure", 2), ("download", 1)])
def test_client_disconnect_releases_response_memory_slots(tmp_path, suffix, slots):
    application = create_app(static_dir=tmp_path)
    with TestClient(application) as connection:
        identifier = connection.post("/api/experiments", json=settings()).json()["id"]
        path = f"/api/experiments/{identifier}/{suffix}"

        async def disconnected_request():
            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                if message["type"] == "http.response.body":
                    raise OSError("client disconnected")

            scope = {
                "type": "http",
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": path,
                "raw_path": path.encode(),
                "query_string": b"",
                "headers": [],
                "client": ("testclient", 50000),
                "server": ("testserver", 80),
                "root_path": "",
            }
            await application(scope, receive, send)

        for _ in range(slots):
            with pytest.raises(OSError):
                asyncio.run(disconnected_request())
        assert connection.get(path).status_code == 200


@pytest.mark.parametrize(
    "target,path,failures",
    [
        ("trajectory_figure", "figure", 2),
        ("export_archive", "download", 1),
    ],
)
def test_render_and_archive_failures_release_slots(client, monkeypatch, target, path, failures):
    import api.server as server

    original = getattr(server, target)
    count = 0

    def flaky(*args, **kwargs):
        nonlocal count
        count += 1
        if count <= failures:
            raise RuntimeError("private internal failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(server, target, flaky)
    identifier = client.post("/api/experiments", json=settings()).json()["id"]
    for _ in range(failures):
        response = client.get(f"/api/experiments/{identifier}/{path}")
        assert response.status_code == 500
        assert "private" not in response.text
    assert client.get(f"/api/experiments/{identifier}/{path}").status_code == 200


def test_engine_failure_releases_compute_slot_and_hides_details(tmp_path):
    count = 0

    def flaky_engine(config):
        nonlocal count
        count += 1
        if count == 1:
            raise RuntimeError("internal secret filesystem detail")
        return run_experiment(config)

    with TestClient(create_app(engine=flaky_engine, static_dir=tmp_path)) as connection:
        failure = connection.post("/api/experiments", json=settings())
        assert failure.status_code == 500
        assert "secret" not in failure.text
        assert connection.post("/api/experiments", json=settings()).status_code == 200


def test_store_ttl_eviction_and_memory_budget_are_enforced():
    now = [0.0]
    store = ResultStore(max_results=2, max_bytes=100000, ttl_seconds=10, clock=lambda: now[0])
    result = run_experiment(
        ExperimentConfig("Rock Paper Scissors", ("Hedge",), iterations=4, runs=1)
    )
    first, second = store.add(result), store.add(result)
    store.add(result)
    with pytest.raises(ResultUnavailable):
        store.get(first)
    assert store.count == 2
    assert store.memory_bytes <= 100000
    now[0] = 11
    with pytest.raises(ResultUnavailable):
        store.get(second)
    assert store.count == 0
    assert store.memory_bytes == 0


def test_store_evicts_by_bytes_and_rejects_an_oversized_result():
    result = run_experiment(
        ExperimentConfig("Rock Paper Scissors", ("Hedge",), iterations=4, runs=1)
    )
    store = ResultStore(max_bytes=result_size(result) + 1)
    first = store.add(result)
    second = store.add(result)
    with pytest.raises(ResultUnavailable):
        store.get(first)
    assert store.get(second) is result
    assert store.count == 1
    with pytest.raises(ResultTooLarge):
        ResultStore(max_bytes=1).add(result)


def test_storage_rejection_releases_compute_slot(tmp_path):
    with TestClient(create_app(store=ResultStore(max_bytes=1), static_dir=tmp_path)) as connection:
        assert connection.post("/api/experiments", json=settings()).status_code == 413
        assert connection.post("/api/experiments", json=settings()).status_code == 413


def test_expired_experiment_is_410_for_figures_and_downloads(tmp_path):
    now = [0.0]
    store = ResultStore(ttl_seconds=10, clock=lambda: now[0])
    with TestClient(create_app(store=store, static_dir=tmp_path)) as connection:
        identifier = connection.post("/api/experiments", json=settings()).json()["id"]
        now[0] = 11
        assert connection.get(f"/api/experiments/{identifier}/figure").status_code == 410
        assert connection.get(f"/api/experiments/{identifier}/download").status_code == 410


def test_request_limits_are_bounded_and_return_retry_after(tmp_path):
    now = [0.0]
    limiter = RequestLimiter(clock=lambda: now[0], max_clients=2)
    for _ in range(120):
        limiter.check("one", "read")
    with pytest.raises(RateLimitExceeded) as denied:
        limiter.check("one", "read")
    assert denied.value.retry_after == 60
    for _ in range(6):
        limiter.check("two", "run")
    with pytest.raises(RateLimitExceeded):
        limiter.check("two", "run")
    limiter.check("three", "read")
    assert limiter.client_count == 2
    small = RequestLimiter(read_limit=1, run_limit=1)
    with TestClient(create_app(limiter=small, static_dir=tmp_path)) as connection:
        assert connection.get("/api/catalog").status_code == 200
        response = connection.get("/api/catalog")
        assert response.status_code == 429
        assert int(response.headers["retry-after"]) >= 1


def test_invalid_run_requests_are_also_rate_limited(tmp_path):
    with TestClient(
        create_app(limiter=RequestLimiter(run_limit=1), static_dir=tmp_path)
    ) as connection:
        assert connection.post("/api/experiments", json=settings(iterations=0)).status_code == 422
        assert connection.post("/api/experiments", json=settings()).status_code == 429


def test_health_static_assets_and_api_errors_remain_separate(tmp_path):
    (tmp_path / "index.html").write_text("<html>React workspace</html>")
    with TestClient(create_app(static_dir=tmp_path)) as connection:
        assert connection.get("/healthz").json() == {"status": "ok"}
        assert "React workspace" in connection.get("/").text
        missing = connection.get("/api/missing")
        assert missing.status_code == 404
        assert "React workspace" not in missing.text
        assert missing.headers["content-type"].startswith("application/json")
    with TestClient(create_app(static_dir=tmp_path / "missing")) as connection:
        missing_assets = connection.get("/")
        assert missing_assets.status_code == 503
        assert "frontend" in missing_assets.text.lower()
