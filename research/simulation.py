"""Seeded experiment execution with explicit initialization and immutable outputs."""

import hashlib
import time
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np

from .catalog import ALGORITHMS, GAMES, algorithms_for_game, canonical_algorithm_name
from .diagnostics import compute_diagnostics


@dataclass(frozen=True)
class ExperimentConfig:
    game_name: str
    algorithm_names: tuple[str, ...]
    iterations: int = 1000
    runs: int = 3
    seed: int = 42
    learning_rate: float = 0.2
    decay: float = -0.5
    temperature: float = 0.1
    exploration: float = 0.1
    initialization: str = "random"

    def validate(self):
        """Validate at execution time so an unfinished UI form remains representable."""
        if self.game_name not in GAMES:
            raise ValueError("Choose a supported game.")
        if not isinstance(self.algorithm_names, tuple) or not self.algorithm_names:
            raise ValueError("Choose at least one algorithm.")
        names = tuple(canonical_algorithm_name(name) for name in self.algorithm_names)
        if len(set(names)) != len(names):
            raise ValueError("Choose each algorithm only once.")
        if any(name not in algorithms_for_game(self.game_name) for name in names):
            raise ValueError("Choose algorithms supported by the selected game.")
        for label, value in (("Iterations", self.iterations), ("Repeats", self.runs)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{label} must be a positive integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer.")
        for label, value in (
            ("Learning rate", self.learning_rate),
            ("Temperature", self.temperature),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be finite and positive.")
        if not np.isfinite(self.decay):
            raise ValueError("Learning-rate exponent must be finite.")
        if not np.isfinite(self.exploration) or not 0 <= self.exploration <= 1:
            raise ValueError("Exploration must be between zero and one.")
        if self.initialization not in ("random", "uniform", "biased"):
            raise ValueError("Initialization must be random, uniform, or biased.")


@dataclass(frozen=True)
class RunResult:
    algorithm: str
    run: int
    seed: int
    strategies: np.ndarray
    empirical_strategies: np.ndarray | None
    payoffs: np.ndarray
    average_regret: np.ndarray
    nash_gap: np.ndarray


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    runs: tuple[RunResult, ...]
    runtime_seconds: float


def _seed(*parts):
    encoded = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(encoded, digest_size=8, person=b"RegretMin").digest(), "little"
    )


def _starting_profile(config, game, run):
    generator = np.random.default_rng(_seed(config.seed, config.game_name, run, "initialization"))
    profiles = []
    for player, actions in enumerate(game.num_actions):
        if config.initialization == "random":
            probabilities = generator.dirichlet(np.ones(actions))
        elif config.initialization == "uniform":
            probabilities = np.full(actions, 1 / actions)
        else:
            probabilities = np.full(actions, 0.25 / (actions - 1))
            probabilities[(player + run) % actions] = 0.75
        profiles.append(probabilities)
    return profiles


def _initial_conditions(name, profiles, learning_rate):
    """Convert one shared starting profile to each algorithm's state semantics.

    Exponential-score methods use log probabilities; softmax regret uses
    log probabilities divided by the initial learning rate; PGA starts with
    simplex coordinates and regret matching starts with nonnegative regret
    priors. Fictitious-play methods receive five total pseudocounts per player,
    which initialize beliefs, not the resulting best-response decision policy.
    BFTL's initial sampled policy additionally includes its exploration mixture.
    """
    if name in ("Fictitious Play", "Smooth Fictitious Play"):
        return [np.array(profile * 5, copy=True) for profile in profiles]
    if name in ("Projected Gradient Ascent", "Regret Matching"):
        return [np.array(profile, copy=True) for profile in profiles]
    scores = [np.log(np.maximum(profile, np.finfo(float).tiny)) for profile in profiles]
    if name == "Regret Matching Softmax":
        return [score / learning_rate for score in scores]
    return scores


def _algorithm(name, game, config, generator):
    factory = ALGORITHMS[name].factory
    eta = {"initial_eta": config.learning_rate, "decay_rate": config.decay}
    if name == "BFTL-EXP3":
        delta = {"initial_delta": config.exploration, "decay_rate": -0.15}
        return factory(game, config.iterations, eta, delta, rng=generator)
    if name in ("Regret Matching", "Fictitious Play"):
        return factory(game, config.iterations, rng=generator)
    if name == "Smooth Fictitious Play":
        return factory(game, config.iterations, temperature=config.temperature, rng=generator)
    return factory(game, config.iterations, eta, rng=generator)


def _readonly(values):
    copied = np.array(values, dtype=float, copy=True)
    copied.setflags(write=False)
    return copied


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Execute independent streams, stable under algorithm selection and ordering."""
    config.validate()
    names = tuple(canonical_algorithm_name(name) for name in config.algorithm_names)
    config = replace(config, algorithm_names=names)
    start = time.perf_counter()
    results = []
    for name in names:
        for run in range(config.runs):
            run_seed = _seed(config.seed, config.game_name, name, run, "actions")
            game = GAMES[config.game_name].factory()
            if hasattr(game, "rng"):
                game.rng = np.random.default_rng(
                    _seed(config.seed, config.game_name, name, run, "payoffs")
                )
            profiles = _starting_profile(config, game, run)
            algorithm = _algorithm(name, game, config, np.random.default_rng(run_seed))
            algorithm.run(initial_scores=_initial_conditions(name, profiles, config.learning_rate))
            strategies = _readonly(algorithm.policy_history)
            empirical = (
                _readonly(algorithm.strategies)
                if name in ("Fictitious Play", "Smooth Fictitious Play")
                else None
            )
            payoffs, regret, gap = compute_diagnostics(game.expected_payoff_matrix, strategies)
            results.append(
                RunResult(
                    name,
                    run,
                    run_seed,
                    strategies,
                    empirical,
                    _readonly(payoffs),
                    _readonly(regret),
                    _readonly(gap),
                )
            )
            del algorithm
    return ExperimentResult(config, tuple(results), time.perf_counter() - start)
