"""Portable, full-resolution experiment archives with explicit metric definitions."""

import csv
import io
import json
import platform
import zipfile
from dataclasses import asdict
from importlib import metadata
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from research.simulation import ExperimentResult


METRIC_DEFINITIONS = {
    "strategies": "Decision policies actually used to sample actions, including exploration.",
    "empirical_strategies": "Historical action frequencies when recorded; distinct from decision policies.",
    "payoffs": "Expected per-player utilities under the current policy profile and base payoff table.",
    "average_regret": (
        "Average expected external regret: the best fixed action's cumulative expected utility "
        "against the sequence of opponent policies minus the learner's cumulative expected "
        "utility, divided by elapsed iterations. This is not realized bandit regret."
    ),
    "nash_gap": "Sum of nonnegative unilateral expected utility gains from a best-response deviation.",
}


def summary_rows(result: "ExperimentResult") -> list[dict]:
    """Return scalar summaries, with per-player metrics named explicitly."""
    rows = []
    for run in result.runs:
        row = {"algorithm": run.algorithm, "run": int(run.run), "seed": int(run.seed)}
        for player in range(run.payoffs.shape[1]):
            suffix = f"p{player + 1}"
            row[f"payoff_mean_{suffix}"] = float(np.mean(run.payoffs[:, player]))
            row[f"payoff_final_{suffix}"] = float(run.payoffs[-1, player])
            row[f"average_expected_regret_final_{suffix}"] = float(run.average_regret[-1, player])
        row["nash_gap_final"] = float(run.nash_gap[-1])
        row["nash_gap_mean"] = float(np.mean(run.nash_gap))
        rows.append(row)
    return rows


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("fastapi", "uvicorn", "plotly", "matplotlib", "imageio"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _summary_csv(rows: list[dict]) -> str:
    output = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output.getvalue()


def export_archive(result: "ExperimentResult") -> bytes:
    """Create a ZIP containing JSON provenance, plain-array NPZ data, and CSV summaries.

    No visualization downsampling is applied to exported research histories.
    NumPy arrays can be loaded with ``allow_pickle=False``.
    """
    arrays = {}
    runs = []
    fields = ("strategies", "empirical_strategies", "payoffs", "average_regret", "nash_gap")
    for index, run in enumerate(result.runs):
        mapping = {}
        for field in fields:
            values = getattr(run, field)
            if values is None:
                continue
            if np.asarray(values).dtype.hasobject:
                raise ValueError("Experiment arrays must not contain Python objects.")
            key = f"run_{index:04d}_{field}"
            arrays[key] = values
            mapping[field] = key
        runs.append({
            "algorithm": run.algorithm,
            "run": int(run.run),
            "seed": int(run.seed),
            "arrays": mapping,
        })
    manifest = {
        "schema_version": 1,
        "config": asdict(result.config),
        "runtime_seconds": float(result.runtime_seconds),
        "seeds": [int(run.seed) for run in result.runs],
        "versions": _versions(),
        "metric_definitions": METRIC_DEFINITIONS,
        "array_dimensions": {
            "strategies": ["iteration", "player", "action"],
            "empirical_strategies": ["iteration", "player", "action"],
            "payoffs": ["iteration", "player"],
            "average_regret": ["iteration", "player"],
            "nash_gap": ["iteration"],
        },
        "runs": runs,
    }
    data = io.BytesIO()
    np.savez(data, **arrays)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("manifest.json", json.dumps(manifest, indent=2, allow_nan=False))
        output.writestr("results.npz", data.getvalue())
        output.writestr("summary.csv", _summary_csv(summary_rows(result)))
    return archive.getvalue()
