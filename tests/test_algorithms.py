import numpy as np
import pytest

from algorithms.b_ftrl_exp3 import BFTL_EXP3
from algorithms.exp3 import EXP3
from algorithms.fictitious_play import FictitiousPlay, SmoothFictitiousPlay
from algorithms.hedge import Hedge
from algorithms.opt_ftrl import OptimisticFTRL
from algorithms.pga import ProjectedGradientAscent
from algorithms.regret_matching import RegretMatching
from algorithms.regret_matching_softmax import RegretMatchingSoftmax
from games.implemented_games import CoordinationWithSpectator
from games.rps_games import RockPaperScissors

ETA = {"initial_eta": 0.2, "decay_rate": -0.5}
CLASSES = [
    BFTL_EXP3,
    EXP3,
    Hedge,
    OptimisticFTRL,
    ProjectedGradientAscent,
    RegretMatching,
    RegretMatchingSoftmax,
    FictitiousPlay,
    SmoothFictitiousPlay,
]


def make_algorithm(cls, iterations=12, **kwargs):
    game = (
        RockPaperScissors()
        if cls in (FictitiousPlay, SmoothFictitiousPlay)
        else CoordinationWithSpectator()
    )
    if cls is BFTL_EXP3:
        return cls(game, iterations, ETA, {"initial_delta": 0.1, "decay_rate": -0.15}, **kwargs)
    if cls in (RegretMatching, FictitiousPlay, SmoothFictitiousPlay):
        return cls(game, iterations, **kwargs)
    return cls(game, iterations, ETA, **kwargs)


@pytest.mark.parametrize("cls", CLASSES)
def test_supplied_initial_conditions_are_not_modified(cls):
    algorithm = make_algorithm(cls)
    initial = [np.linspace(1.0, 2.0, n) for n in algorithm.num_actions]
    before = [x.copy() for x in initial]
    algorithm.run(initial_scores=initial)
    for actual, expected in zip(initial, before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("cls", CLASSES)
def test_reusing_algorithm_resets_all_history(cls):
    algorithm = make_algorithm(cls)
    algorithm.run()
    algorithm.run()
    assert len(algorithm.strategies) == algorithm.num_iterations


@pytest.mark.parametrize("cls", CLASSES)
def test_local_rng_and_actual_policy_are_valid(cls):
    left = make_algorithm(cls, rng=np.random.default_rng(17))
    right = make_algorithm(cls, rng=np.random.default_rng(17))
    left.run()
    right.run()
    np.testing.assert_array_equal(left.policy_history, right.policy_history)
    policies = np.asarray(left.policy_history)
    assert policies.shape == (12, left.num_players, left.num_actions[0])
    assert np.isfinite(policies).all()
    assert (policies >= 0).all()
    np.testing.assert_allclose(policies.sum(axis=-1), 1)


def test_bftl_history_contains_exploration_distribution():
    algorithm = make_algorithm(BFTL_EXP3, rng=np.random.default_rng(12))
    algorithm.run([np.array([2.0, -2.0])] * 3)
    np.testing.assert_allclose(
        algorithm.policy_history[0], 0.9 * np.asarray(algorithm.strategies[0]) + 0.05
    )


def test_fictitious_play_exposes_decision_and_empirical_histories():
    algorithm = make_algorithm(FictitiousPlay, rng=np.random.default_rng(12))
    algorithm.run([np.array([5.0, 1.0, 1.0]), np.array([1.0, 5.0, 1.0])])
    np.testing.assert_array_equal(algorithm.policy_history[0][0], [0, 0, 1])
    assert not np.array_equal(algorithm.policy_history, algorithm.strategies)


def test_smooth_response_is_stable_at_small_temperature():
    algorithm = make_algorithm(
        SmoothFictitiousPlay, temperature=1e-6, rng=np.random.default_rng(12)
    )
    algorithm.run([np.array([5.0, 1.0, 1.0]), np.array([1.0, 5.0, 1.0])])
    assert np.isfinite(algorithm.policy_history).all()
