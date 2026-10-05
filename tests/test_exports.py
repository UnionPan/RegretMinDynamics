"""Portable exports retain unsampled research data and explicit metric meaning."""

import csv
import io
import json
import zipfile
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest


@dataclass(frozen=True)
class ExportConfig:
    game_name: str = "Rock Paper Scissors"
    algorithm_names: tuple[str, ...] = ("Hedge",)
    iterations: int = 1250
    runs: int = 2
    seed: int = 7


def make_result(iterations=1250, actions=3, players=2):
    game_name = (
        "Pure Coordination" if players == 3 else
        "Minority Game" if actions == 2 else
        "Rock Paper Scissors Lizard Spock" if actions == 5 else
        "Rock Paper Scissors"
    )
    runs = tuple(
        SimpleNamespace(
            algorithm="Hedge", run=run, seed=100 + run,
            strategies=np.full((iterations, players, actions), 1 / actions),
            empirical_strategies=np.full((iterations, players, actions), 1 / actions) if run else None,
            payoffs=np.arange(iterations * players, dtype=float).reshape(iterations, players),
            average_regret=np.full((iterations, players), 0.25),
            nash_gap=np.linspace(0.5, 0.0, iterations),
        )
        for run in range(2)
    )
    return SimpleNamespace(config=ExportConfig(game_name=game_name, iterations=iterations), runs=runs, runtime_seconds=0.12)


def test_archive_roundtrips_full_arrays_without_pickle():
    from research.exports import export_archive

    result = make_result()
    with zipfile.ZipFile(io.BytesIO(export_archive(result))) as archive:
        assert set(archive.namelist()) == {"manifest.json", "results.npz", "summary.csv"}
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["config"]["seed"] == 7
        assert manifest["seeds"] == [100, 101]
        assert {"python", "numpy"}.issubset(manifest["versions"])
        assert "expected" in manifest["metric_definitions"]["average_regret"].lower()
        assert "deviation" in manifest["metric_definitions"]["nash_gap"].lower()
        with np.load(io.BytesIO(archive.read("results.npz")), allow_pickle=False) as arrays:
            for index, run in enumerate(result.runs):
                mapping = manifest["runs"][index]["arrays"]
                for field in ("strategies", "payoffs", "average_regret", "nash_gap"):
                    np.testing.assert_array_equal(arrays[mapping[field]], getattr(run, field))
                assert arrays[mapping["strategies"]].shape[0] == 1250
            assert "run_0000_empirical_strategies" not in arrays
            np.testing.assert_array_equal(arrays["run_0001_empirical_strategies"], result.runs[1].empirical_strategies)
        rows = list(csv.DictReader(io.StringIO(archive.read("summary.csv").decode())))
        assert len(rows) == 2
        assert "average_expected_regret_final_p1" in rows[0]
        assert "nash_gap_final" in rows[0]


def test_summary_values_distinguish_payoff_regret_and_nash_gap():
    from research.exports import summary_rows

    result = make_result(iterations=3)
    rows = summary_rows(result)
    assert rows[0]["payoff_mean_p1"] == 2.0
    assert rows[0]["payoff_final_p2"] == 5.0
    assert rows[0]["average_expected_regret_final_p1"] == 0.25
    assert rows[0]["nash_gap_final"] == 0.0
    assert rows[0]["nash_gap_mean"] == 0.25
    assert all(np.isscalar(value) for value in rows[0].values())


@pytest.mark.parametrize("iterations,actions,players", [(10, 3, 2), (100_000, 2, 3), (10, 2, 2), (10, 5, 2)])
def test_gif_frame_and_path_work_are_bounded(iterations, actions, players):
    from plotting.exports import frame_indices, path_indices

    frames = frame_indices(iterations)
    path = path_indices(iterations)
    assert len(frames) <= 60
    assert len(path) <= 1000
    assert frames[0] == path[0] == 0
    assert frames[-1] == path[-1] == iterations - 1


def test_sampled_paths_include_every_animated_policy_timestamp():
    from plotting.exports import frame_indices, path_indices

    assert set(frame_indices(100_000)).issubset(path_indices(100_000))


@pytest.mark.parametrize("actions,players", [(3, 2), (2, 3), (2, 2), (5, 2)])
def test_gif_is_rendered_in_memory_and_leaves_no_shared_frames(tmp_path, monkeypatch, actions, players):
    import imageio.v2 as imageio

    from plotting.exports import render_gif

    monkeypatch.chdir(tmp_path)
    data = render_gif(make_result(iterations=3, actions=actions, players=players))
    assert data[:6] in (b"GIF87a", b"GIF89a")
    assert len(imageio.mimread(io.BytesIO(data), format="GIF")) <= 3
    assert not list(tmp_path.iterdir())
