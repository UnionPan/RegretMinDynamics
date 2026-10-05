import numpy as np

from games.implemented_games import MatchingPenniesWithOutsideOption
from games.rps_games import RockPaperScissors
from research.diagnostics import expected_external_regret, expected_utilities, nash_gap


def test_uniform_rps_has_zero_expected_payoffs_and_nash_gap():
    table = RockPaperScissors().payoff_matrix
    policies = np.full((4, 2, 3), 1 / 3)
    np.testing.assert_allclose(expected_utilities(table, policies), 0, atol=1e-12)
    np.testing.assert_allclose(nash_gap(table, policies), 0, atol=1e-12)
    np.testing.assert_allclose(expected_external_regret(table, policies), 0, atol=1e-12)


def test_incorrect_outside_option_marker_has_certifiable_deviation_gain():
    table = MatchingPenniesWithOutsideOption().payoff_matrix
    policies = np.full((1, 3, 2), 0.5)
    np.testing.assert_allclose(nash_gap(table, policies), [0.25])


def test_expected_regret_compares_one_fixed_action_across_rounds():
    table = np.array([[[0, 0], [2, 0]], [[1, 0], [0, 0]]], dtype=float)
    policies = np.array([[[1, 0], [1, 0]], [[0, 1], [0, 1]]], dtype=float)
    # Actual reward is zero each round. Fixed actions earn (0, 2) and (1, 0).
    np.testing.assert_allclose(expected_external_regret(table, policies)[:, 0], [1, 1])


def test_expected_external_regret_remains_signed():
    table = np.array([[[2, 0], [0, 0]], [[0, 0], [2, 0]]], dtype=float)
    policies = np.array([[[1, 0], [1, 0]], [[0, 1], [0, 1]]], dtype=float)
    np.testing.assert_allclose(expected_external_regret(table, policies)[:, 0], [0, -1])
