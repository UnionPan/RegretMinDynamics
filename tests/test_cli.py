"""End-to-end checks of the documented research entry points."""

import io
import json
import runpy
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def invoke(script, *arguments):
    return subprocess.run(
        [sys.executable, str(ROOT / script), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize(
    "script,game,algorithm,canonical_game,canonical_algorithm",
    [
        ("main.py", "PureCoordination", "OptFTRL", "Pure Coordination", "Optimistic FTRL"),
        ("rps_experiments.py", "RPS", "FictitiousPlay", "Rock Paper Scissors", "Fictitious Play"),
        ("rps_experiments.py", "RPS", "SmoothFP_T0.5", "Rock Paper Scissors", "Smooth Fictitious Play"),
    ],
)
def test_cli_runs_shared_seeded_engine_and_exports(
    tmp_path, script, game, algorithm, canonical_game, canonical_algorithm
):
    completed = invoke(
        script, "--game", game, "--algorithm", algorithm,
        "--iterations", "12", "--runs", "2", "--seed", "19",
        "--output-dir", str(tmp_path), "--no-gif",
    )
    assert completed.returncode == 0, completed.stderr
    archives = list(tmp_path.glob("*.zip"))
    assert len(archives) == 1
    assert not list(tmp_path.glob("*.gif"))
    with zipfile.ZipFile(archives[0]) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["config"]["game_name"] == canonical_game
        assert manifest["config"]["iterations"] == 12
        assert manifest["config"]["seed"] == 19
        if algorithm == "SmoothFP_T0.5":
            assert manifest["config"]["temperature"] == 0.5
        assert len(manifest["runs"]) == 2
        with np.load(io.BytesIO(archive.read("results.npz")), allow_pickle=False) as arrays:
            assert arrays["run_0000_strategies"].shape[0] == 12
            from research.simulation import ExperimentConfig, run_experiment

            result = run_experiment(ExperimentConfig(**{
                **manifest["config"], "algorithm_names": tuple(manifest["config"]["algorithm_names"])
            }))
            np.testing.assert_array_equal(arrays["run_0000_strategies"], result.runs[0].strategies)
            assert result.runs[0].algorithm == canonical_algorithm


@pytest.mark.parametrize("script", ["main.py", "rps_experiments.py"])
@pytest.mark.parametrize("argument,value", [("--game", "missing"), ("--algorithm", "missing"), ("--iterations", "0")])
def test_cli_invalid_inputs_fail_cleanly(script, argument, value, tmp_path):
    completed = invoke(script, argument, value, "--output-dir", str(tmp_path), "--no-gif")
    assert completed.returncode != 0
    assert "error:" in completed.stderr.lower()
    assert "Traceback" not in completed.stderr
    assert not list(tmp_path.iterdir())


def test_legacy_iterations_flag_and_default_gif(tmp_path):
    completed = invoke(
        "rps_experiments.py", "--game", "RPS", "--algorithm", "Hedge",
        "--num_iterations", "3", "--runs", "1", "--output-dir", str(tmp_path),
    )
    assert completed.returncode == 0, completed.stderr
    assert len(list(tmp_path.glob("*.gif"))) == 1
    assert not list(tmp_path.glob("frame_*.png"))


@pytest.mark.parametrize("script", ["main.py", "rps_experiments.py"])
def test_all_selection_uses_only_compatible_algorithms(script, tmp_path):
    completed = invoke(
        script, "--game", "all", "--algorithm", "all", "--iterations", "2",
        "--runs", "1", "--output-dir", str(tmp_path), "--no-gif",
    )
    assert completed.returncode == 0, completed.stderr
    assert len(list(tmp_path.glob("*.zip"))) == 4


def test_programmatic_rps_entrypoint_preserves_legacy_signature(tmp_path):
    from rps_experiments import run_rps_experiments

    results = run_rps_experiments(
        "RPS", "Hedge", 3, runs=1, seed=11, output_dir=tmp_path, no_gif=True
    )
    assert len(results) == 1
    assert results[0].config.iterations == 3
    assert results[0].config.seed == 11


def test_in_process_cli_writes_archive_and_gif(tmp_path):
    from main import run_cli

    code = run_cli([
        "--game", "PureCoordination", "--algorithm", "RegretMatching",
        "--iterations", "2", "--runs", "1", "--output-dir", str(tmp_path),
    ])
    assert code == 0
    assert len(list(tmp_path.glob("*.zip"))) == 1
    assert len(list(tmp_path.glob("*.gif"))) == 1
    assert not list(tmp_path.glob(".experiment-*"))


@pytest.mark.parametrize(
    "arguments",
    [["--game", "missing"], ["--algorithm", "missing"], ["--iterations", "0"],
     ["--game", "RPS"], ["--algorithm", "FictitiousPlay"]],
)
def test_cli_parser_rejects_invalid_or_incompatible_selections(arguments, tmp_path, capsys):
    from main import run_cli

    with pytest.raises(SystemExit) as error:
        run_cli([*arguments, "--output-dir", str(tmp_path), "--no-gif"])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_in_process_all_selection_is_category_compatible(tmp_path):
    from main import run_selected_experiments
    from research.catalog import algorithms_for_game

    results = run_selected_experiments(
        "all", "all", iterations=2, runs=1, output_dir=tmp_path, no_gif=True
    )
    assert len(results) == 4
    for result in results:
        assert result.config.algorithm_names == algorithms_for_game(result.config.game_name)


@pytest.mark.parametrize("script,game", [("main.py", "PureCoordination"), ("rps_experiments.py", "RPS")])
def test_script_entry_points_report_success(script, game, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        script, "--game", game, "--algorithm", "Hedge", "--iterations", "2",
        "--runs", "1", "--output-dir", str(tmp_path), "--no-gif",
    ])
    with pytest.raises(SystemExit) as completion:
        runpy.run_path(str(ROOT / script), run_name="__main__")
    assert completion.value.code == 0
    assert len(list(tmp_path.glob("*.zip"))) == 1
