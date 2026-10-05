import numpy as np
import pytest

from research.catalog import ALGORITHMS, GAMES, algorithms_for_game, games_for_category
from research.diagnostics import nash_gap


@pytest.mark.parametrize("name", list(GAMES))
def test_advertised_equilibria_are_certified(name):
    spec = GAMES[name]
    game = spec.factory()
    table = getattr(game, "payoff_matrix", getattr(game, "base_payoff_matrix", None))
    assert len(spec.action_labels) == game.num_actions[0]
    for equilibrium in spec.equilibria:
        np.testing.assert_allclose(equilibrium.sum(axis=-1), 1)
        assert np.max(nash_gap(table, equilibrium[None])) < 1e-9
        assert not equilibrium.flags.writeable


def test_anti_coordination_games_include_actual_pure_equilibria():
    assert len(GAMES["Minority Game"].equilibria) == 3
    assert len(GAMES["Dispersion Game"].equilibria) == 7


def test_category_and_algorithm_filtering_matches_game_arity():
    assert "Pure Coordination" in games_for_category("Box Games")
    assert "Rock Paper Scissors" in games_for_category("RPS Games")
    assert "Fictitious Play" not in algorithms_for_game("Pure Coordination")
    assert "Smooth Fictitious Play" in algorithms_for_game("Rock Paper Scissors")
    assert ALGORITHMS["BFTL-EXP3"].feedback == "Full information"


def test_public_goods_parameter_is_honored_and_default_is_preserved():
    from games.implemented_games import PublicGoodsGame

    np.testing.assert_allclose(PublicGoodsGame().get_payoff((1, 0, 0)), [-0.5, 0.5, 0.5])
    np.testing.assert_allclose(PublicGoodsGame(multiplier=3).get_payoff((1, 0, 0)), [0, 1, 1])
