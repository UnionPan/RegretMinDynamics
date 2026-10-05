"""Shared numerical plumbing; algorithm update rules remain in their modules."""

import numpy as np


def initial_vectors(values, num_actions, default=0.0, nonnegative=False):
    if values is None:
        return [np.full(count, default, dtype=float) for count in num_actions]
    if len(values) != len(num_actions):
        raise ValueError("Initial conditions must include every player.")
    copied = [np.array(value, dtype=float, copy=True) for value in values]
    for value, count in zip(copied, num_actions):
        if value.shape != (count,) or not np.isfinite(value).all():
            raise ValueError(
                "Initial conditions must be finite vectors matching each action space."
            )
        if nonnegative and (value < 0).any():
            raise ValueError("Initial action counts or regret priors must be nonnegative.")
    return copied


def softmax(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "Scores became non-finite; reduce the learning rate or initialization magnitude."
        )
    weights = np.exp(values - np.max(values))
    return weights / weights.sum()


def full_information_payoffs(game, actions):
    """Slice deterministic tables; retain noisy feedback calls and their order."""
    table = getattr(game, "payoff_matrix", None)
    if table is not None:
        vectors = []
        for player in range(game.num_players):
            index = tuple(
                slice(None) if other == player else action for other, action in enumerate(actions)
            )
            vectors.append(np.array(table[index][:, player], copy=True))
        return vectors
    vectors = []
    for player in range(game.num_players):
        values = np.zeros(game.num_actions[player])
        for action in range(game.num_actions[player]):
            profile = list(actions)
            profile[player] = action
            values[action] = game.get_payoff(tuple(profile))[player]
        vectors.append(values)
    return vectors
