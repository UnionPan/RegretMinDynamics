"""Seeded, portable command-line experiments for the Box Games catalog."""

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

from research.catalog import ALGORITHMS, GAMES, algorithms_for_game, games_for_category
from research.exports import export_archive
from research.simulation import ExperimentConfig, run_experiment

GAME_ALIASES = {
    "PureCoordination": "Pure Coordination",
    "CoordinationWithSpectator": "Coordination with Spectator",
    "MatchingPenniesWithTwist": "Matching Pennies with Twist",
    "MatchingPenniesWithOutsideOption": "Matching Pennies with Outside Option",
    "RPS": "Rock Paper Scissors",
    "BiasedRPS": "Biased Rock Paper Scissors",
    "AsymmetricRPS": "Asymmetric Rock Paper Scissors",
    "NoisyRPS": "Noisy Rock Paper Scissors",
}
ALGORITHM_ALIASES = {
    "OptFTRL": "Optimistic FTRL",
    "PGA": "Projected Gradient Ascent",
    "RegretMatching": "Regret Matching",
    "RegretMatchingSoftmax": "Regret Matching Softmax",
    "BFTL_EXP3": "BFTL-EXP3",
    "FictitiousPlay": "Fictitious Play",
    "SmoothFP_T0.1": "Smooth Fictitious Play",
    "SmoothFP_T0.5": "Smooth Fictitious Play",
    "Smooth FP (T=0.1)": "Smooth Fictitious Play",
}
DEFAULT_GAMES = {
    "Box Games": tuple(GAME_ALIASES[name] for name in (
        "PureCoordination", "CoordinationWithSpectator",
        "MatchingPenniesWithTwist", "MatchingPenniesWithOutsideOption",
    )),
    "RPS Games": tuple(GAME_ALIASES[name] for name in (
        "RPS", "BiasedRPS", "AsymmetricRPS", "NoisyRPS",
    )),
}


def _selections(game_name, algorithm_name, category):
    games = DEFAULT_GAMES[category] if game_name == "all" else (
        GAME_ALIASES.get(game_name, game_name),
    )
    available_games = games_for_category(category)
    for name in games:
        if name not in GAMES or name not in available_games:
            raise ValueError(f"Unknown game for {category}: {game_name}. Available: {', '.join(available_games)}")
    algorithm = ALGORITHM_ALIASES.get(algorithm_name, algorithm_name)
    selections = []
    for game in games:
        available = algorithms_for_game(game)
        algorithms = available if algorithm_name == "all" else (algorithm,)
        if any(name not in ALGORITHMS or name not in available for name in algorithms):
            raise ValueError(f"Unknown or incompatible algorithm: {algorithm_name}. Available: {', '.join(available)}")
        selections.append((game, algorithms))
    return selections


def _write_output(path: Path, data: bytes) -> None:
    """Replace a finished output atomically, using a unique temporary file."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".experiment-", delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _output_stem(config: ExperimentConfig) -> str:
    serialized = json.dumps(asdict(config), sort_keys=True).encode()
    digest = hashlib.sha256(serialized).hexdigest()[:10]
    game = re.sub(r"[^A-Za-z0-9]+", "_", config.game_name).strip("_")
    return f"{game}_seed_{config.seed}_{digest}"


def run_selected_experiments(
    game_name="all", algorithm_name="all", *, category="Box Games",
    iterations=1000, runs=3, seed=42, output_dir="outputs", no_gif=False,
    learning_rate=0.2, decay=-0.5, temperature=None, exploration=0.1,
    initialization="random",
):
    """Execute catalog-compatible configurations and save full-resolution archives."""
    selected = _selections(game_name, algorithm_name, category)
    if temperature is None:
        temperature = 0.5 if algorithm_name == "SmoothFP_T0.5" else 0.1
    configs = tuple(ExperimentConfig(
        game_name=game, algorithm_names=tuple(algorithms), iterations=iterations,
        runs=runs, seed=seed, learning_rate=learning_rate, decay=decay,
        temperature=temperature, exploration=exploration, initialization=initialization,
    ) for game, algorithms in selected)
    # Validate every configuration before creating files or starting a batch.
    for config in configs:
        config.validate()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for config in configs:
        print(f"Running {config.game_name}: {len(config.algorithm_names)} algorithm(s), {runs} run(s), seed {seed}")
        result = run_experiment(config)
        stem = _output_stem(config)
        archive = destination / f"{stem}.zip"
        _write_output(archive, export_archive(result))
        print(f"Saved full histories: {archive}")
        if not no_gif:
            from plotting.exports import render_gif

            animation = destination / f"{stem}.gif"
            _write_output(animation, render_gif(result))
            print(f"Saved bounded animation: {animation}")
        results.append(result)
    return tuple(results)


def run_cli(argv=None, *, category="Box Games") -> int:
    parser = argparse.ArgumentParser(description=f"Reproducible {category} learning experiments")
    parser.add_argument("--game", default="all", help="Readable catalog name, legacy alias, or all (the four original games)")
    parser.add_argument("--algorithm", default="all", help="Readable catalog name, legacy alias, or all compatible algorithms")
    parser.add_argument("--iterations", "--num_iterations", dest="iterations", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=3, help="Independent seeded repeats per algorithm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs", help="Archive and GIF output directory")
    parser.add_argument("--no-gif", action="store_true", help="Save research data without animation rendering")
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--decay", type=float, default=-0.5)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--exploration", type=float, default=0.1)
    parser.add_argument("--initialization", default="random", help="random, uniform, or biased")
    args = parser.parse_args(argv)
    try:
        run_selected_experiments(
            args.game, args.algorithm, category=category, iterations=args.iterations,
            runs=args.runs, seed=args.seed, output_dir=args.output_dir, no_gif=args.no_gif,
            learning_rate=args.learning_rate, decay=args.decay, temperature=args.temperature,
            exploration=args.exploration, initialization=args.initialization,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
