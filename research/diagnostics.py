"""Diagnostics on expected payoff tables, without sampling additional rewards.

External regret compares one fixed action across rounds with the learner's
expected rewards against the sequence of opponents' mixed policies.
It may be negative and is separate from realized or importance-weighted regret.
"""

import numpy as np


def counterfactual_utilities(table, policies):
    """Return expected payoff for each fixed action, shaped (round, player, action)."""
    table = np.asarray(table, dtype=float)
    policies = np.asarray(policies, dtype=float)
    players = table.shape[-1] if table.ndim else 0
    if table.ndim != players + 1 or players not in (2, 3):
        raise ValueError(
            "A payoff table must have one action axis per player and a final player axis."
        )
    if policies.ndim != 3 or policies.shape[1] != players or policies.shape[0] == 0:
        raise ValueError(
            "Policies must have shape (round, player, action) with at least one round."
        )
    if any(count != policies.shape[2] for count in table.shape[:-1]):
        raise ValueError("Policy action counts must match the payoff table.")
    if not np.isfinite(table).all() or not np.isfinite(policies).all():
        raise ValueError("Payoffs and policies must be finite.")
    if (policies < 0).any() or not np.allclose(policies.sum(axis=-1), 1, atol=1e-10, rtol=0):
        raise ValueError("Each player's policy must be a probability distribution.")
    action_axes = "abc"[:players]
    values = []
    for player in range(players):
        opponents = tuple(other for other in range(players) if other != player)
        expression = ",".join((action_axes, *("t" + action_axes[other] for other in opponents)))
        expression += "->t" + action_axes[player]
        values.append(
            np.einsum(
                expression,
                table[..., player],
                *(policies[:, other, :] for other in opponents),
                optimize=True,
            )
        )
    return np.stack(values, axis=1)


def expected_utilities(table, policies):
    """Expected stage payoffs under independent mixed policies, shaped (round, player)."""
    return np.sum(counterfactual_utilities(table, policies) * policies, axis=-1)


def unilateral_deviation_gains(table, policies):
    """Best unilateral improvement for each player, shaped (round, player)."""
    values = counterfactual_utilities(table, policies)
    rewards = np.sum(values * policies, axis=-1)
    return np.maximum(values.max(axis=-1) - rewards, 0)


def nash_gap(table, policies):
    """Sum of nonnegative unilateral improvements, shaped (round,)."""
    return unilateral_deviation_gains(table, policies).sum(axis=-1)


def expected_external_regret(table, policies):
    """Signed average regret through each round, shaped (round, player)."""
    values = counterfactual_utilities(table, policies)
    rewards = np.sum(values * policies, axis=-1)
    best_fixed = np.cumsum(values, axis=0).max(axis=-1)
    return (best_fixed - np.cumsum(rewards, axis=0)) / np.arange(1, len(policies) + 1)[:, None]


def compute_diagnostics(table, policies):
    """Compute all displayed histories with a single payoff contraction."""
    values = counterfactual_utilities(table, policies)
    rewards = np.sum(values * policies, axis=-1)
    gaps = np.maximum(values.max(axis=-1) - rewards, 0).sum(axis=-1)
    regret = np.cumsum(values, axis=0).max(axis=-1) - np.cumsum(rewards, axis=0)
    regret /= np.arange(1, len(policies) + 1)[:, None]
    return rewards, regret, gaps


def pure_equilibria(table):
    """Enumerate pure equilibria of the small supported finite games."""
    table = np.asarray(table, dtype=float)
    profiles = np.asarray(
        [
            [np.eye(table.shape[player])[action] for player, action in enumerate(actions)]
            for actions in np.ndindex(*table.shape[:-1])
        ]
    )
    gaps = nash_gap(table, profiles)
    equilibria = []
    for profile in profiles[gaps < 1e-10]:
        copied = np.array(profile, copy=True)
        copied.setflags(write=False)
        equilibria.append(copied)
    return tuple(equilibria)
