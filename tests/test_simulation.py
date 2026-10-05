from dataclasses import replace

import numpy as np
import pytest

from research.catalog import GAMES, algorithms_for_game
from research.simulation import ExperimentConfig, run_experiment


def config(**kwargs):
    return ExperimentConfig(
        game_name="Rock Paper Scissors",
        algorithm_names=("EXP3", "Hedge"),
        iterations=24,
        runs=2,
        seed=17,
        **kwargs,
    )


def test_seeded_results_repeat_and_do_not_change_global_rng():
    before = np.random.get_state()
    left, right = run_experiment(config()), run_experiment(config())
    after = np.random.get_state()
    for x, y in zip(before, after):
        np.testing.assert_array_equal(x, y)
    for a, b in zip(left.runs, right.runs):
        for field in ("strategies", "payoffs", "average_regret", "nash_gap"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
            assert not getattr(a, field).flags.writeable
    assert not np.array_equal(left.runs[0].strategies, left.runs[1].strategies)


def test_algorithm_selection_order_does_not_change_a_seeded_run():
    left = run_experiment(config())
    right = run_experiment(replace(config(), algorithm_names=("Hedge", "EXP3")))
    lookup = {(r.algorithm, r.run): r for r in right.runs}
    for result in left.runs:
        np.testing.assert_array_equal(
            result.strategies, lookup[result.algorithm, result.run].strategies
        )


@pytest.mark.parametrize("initialization", ["random", "uniform", "biased"])
def test_score_algorithms_share_starting_profiles_by_repeat(initialization):
    result = run_experiment(config(initialization=initialization))
    by_run = {(run.algorithm, run.run): run for run in result.runs}
    for repeat in range(2):
        np.testing.assert_array_equal(
            by_run[("EXP3", repeat)].strategies[0], by_run[("Hedge", repeat)].strategies[0]
        )
    if initialization == "uniform":
        np.testing.assert_allclose(result.runs[0].strategies[0], 1 / 3)
    elif initialization == "biased":
        np.testing.assert_allclose(
            result.runs[0].strategies[0], [[0.75, 0.125, 0.125], [0.125, 0.75, 0.125]]
        )


def test_config_can_represent_empty_form_until_execution():
    settings = replace(config(), algorithm_names=())
    assert settings.algorithm_names == ()
    with pytest.raises(ValueError, match="at least one"):
        settings.validate()


def test_legacy_smooth_fictitious_play_alias_is_canonicalized():
    settings = replace(config(), algorithm_names=("Smooth FP (T=0.1)",), runs=1)
    result = run_experiment(settings)
    assert result.config.algorithm_names == ("Smooth Fictitious Play",)
    assert result.runs[0].algorithm == "Smooth Fictitious Play"


@pytest.mark.parametrize("game_name", list(GAMES))
def test_every_supported_game_algorithm_pair_has_valid_policy_history(game_name):
    result = run_experiment(
        ExperimentConfig(
            game_name=game_name,
            algorithm_names=algorithms_for_game(game_name),
            iterations=8,
            runs=1,
        )
    )
    for run in result.runs:
        assert run.strategies.shape[0] == 8
        assert np.isfinite(run.strategies).all()
        assert (run.strategies >= 0).all()
        np.testing.assert_allclose(run.strategies.sum(axis=-1), 1)
        assert run.payoffs.shape == (8, run.strategies.shape[1])
        assert run.average_regret.shape == run.payoffs.shape
        assert run.nash_gap.shape == (8,)


def test_noisy_simulation_is_seeded_and_diagnostics_use_expected_table():
    settings = replace(config(), game_name="Noisy Rock Paper Scissors", runs=1)
    first, second = run_experiment(settings), run_experiment(settings)
    for a, b in zip(first.runs, second.runs):
        np.testing.assert_array_equal(a.strategies, b.strategies)
    from games.rps_games import RockPaperScissors
    from research.diagnostics import expected_utilities

    for run in first.runs:
        np.testing.assert_allclose(
            run.payoffs, expected_utilities(RockPaperScissors().payoff_matrix, run.strategies)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0},
        {"runs": 0},
        {"seed": -1},
        {"learning_rate": float("nan")},
        {"temperature": 0},
        {"exploration": 1.1},
        {"initialization": "invalid"},
        {"algorithm_names": ()},
        {"game_name": "invalid"},
        {"algorithm_names": ("Hedge", "Hedge")},
        {"algorithm_names": ("unknown",)},
        {"iterations": 1.5},
        {"seed": True},
        {"decay": float("inf")},
    ],
)
def test_invalid_experiment_configuration_has_clear_validation(kwargs):
    with pytest.raises(ValueError):
        run_experiment(replace(config(), **kwargs))
