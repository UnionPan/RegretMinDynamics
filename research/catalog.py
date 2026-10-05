"""Shared game metadata and certified equilibrium examples.

All pure equilibria are enumerated from the supplied payoff tables.
Mixed profiles are certified examples; this registry does not enumerate all
mixed equilibria or continuous equilibrium sets.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable

import numpy as np

from algorithms.b_ftrl_exp3 import BFTL_EXP3
from algorithms.exp3 import EXP3
from algorithms.fictitious_play import FictitiousPlay, SmoothFictitiousPlay
from algorithms.hedge import Hedge
from algorithms.opt_ftrl import OptimisticFTRL
from algorithms.pga import ProjectedGradientAscent
from algorithms.regret_matching import RegretMatching
from algorithms.regret_matching_softmax import RegretMatchingSoftmax
from games.implemented_games import (
    CoordinationWithSpectator,
    MajorityGame,
    MatchingPenniesWithOutsideOption,
    MatchingPenniesWithTwist,
    PublicGoodsGame,
    PureCoordination,
    StagHunt,
    ThreePlayerHawkDove,
    ThreePlayerPrisonersDilemma,
    VolunteersDilemma,
)
from games.rps_games import (
    AsymmetricRockPaperScissors,
    BiasedRockPaperScissors,
    CyclicGame,
    DispersionGame,
    MinorityGame,
    RockPaperScissors,
    RockPaperScissorsLizardSpock,
    RockPaperScissorsWell,
    RockPaperScissorsWithNoise,
)

from .diagnostics import nash_gap, pure_equilibria


@dataclass(frozen=True)
class GameSpec:
    name: str
    category: str
    action_labels: tuple[str, ...]
    description: str
    equilibria: tuple[np.ndarray, ...]
    factory: Callable
    num_players: int


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    description: str
    feedback: str
    categories: tuple[str, ...]
    factory: Callable


def _game(name, category, labels, description, factory, mixed=()):
    game = factory()
    table = game.expected_payoff_matrix
    candidates = tuple(np.array(profile, dtype=float, copy=True) for profile in mixed)
    for profile in candidates:
        if np.max(nash_gap(table, profile[None])) > 1e-9:
            raise ValueError(f"Invalid equilibrium metadata for {name}.")
        profile.setflags(write=False)
    return GameSpec(
        name,
        category,
        tuple(labels),
        description,
        pure_equilibria(table) + candidates,
        factory,
        game.num_players,
    )


def _binary_profile(probability_zero):
    return np.asarray([[q, 1 - q] for q in probability_zero])


def _uniform(players, actions):
    return np.full((players, actions), 1 / actions)


_BOX = "Box Games"
_RPS = "RPS Games"
_game_specs = (
    _game(
        "Pure Coordination",
        _BOX,
        ("Action 0", "Action 1"),
        "A three-player parity coordination game. Everyone earns 1 when an even number choose action 1; otherwise everyone earns 0.",
        PureCoordination,
        (_uniform(3, 2),),
    ),
    _game(
        "Coordination with Spectator",
        _BOX,
        ("Action 0", "Action 1"),
        "Players 1 and 2 earn 1 when their actions match. Player 3 always earns 0. Equilibrium sets include spectator-independent edges.",
        CoordinationWithSpectator,
        (_uniform(3, 2),),
    ),
    _game(
        "Matching Pennies with Twist",
        _BOX,
        ("Action 0", "Action 1"),
        "Player 1 earns 0.1 for action 1 and 0 for action 0. Players 2 and 3 have opposed matching incentives that reverse with Player 1's action.",
        MatchingPenniesWithTwist,
        (_binary_profile((0, 0.5, 0.5)),),
    ),
    _game(
        "Matching Pennies with Outside Option",
        _BOX,
        ("Action 0", "Action 1"),
        "A mixed-motive three-player game. The pure profile (1, 1, 0) is a strict Nash equilibrium with payoff (1, 1, 1). Uniform mixing is not an equilibrium.",
        MatchingPenniesWithOutsideOption,
        (_binary_profile((1 / 3, 0.5, 0.5)),),
    ),
    _game(
        "3-Player Prisoner's Dilemma",
        _BOX,
        ("Defect", "Cooperate"),
        "Defect is action 0 and strictly dominates cooperate. All defect is the unique Nash equilibrium; all cooperate gives everyone a higher payoff.",
        ThreePlayerPrisonersDilemma,
    ),
    _game(
        "Public Goods Game",
        _BOX,
        ("Free-ride", "Contribute"),
        "A contribution costs 1 and produces 1.5 shared equally. Free-riding strictly dominates contributing, while joint contribution raises total welfare.",
        PublicGoodsGame,
    ),
    _game(
        "Volunteer's Dilemma",
        _BOX,
        ("Don't volunteer", "Volunteer"),
        "One volunteer supplies benefit 3 to everyone and pays cost 2. Each profile with exactly one volunteer is a pure Nash equilibrium.",
        VolunteersDilemma,
        (_binary_profile((np.sqrt(2 / 3),) * 3),),
    ),
    _game(
        "Majority Game",
        _BOX,
        ("Action A", "Action B"),
        "Members of the majority earn 2; the minority player loses 1. Both unanimous action profiles are pure Nash equilibria.",
        MajorityGame,
        (_uniform(3, 2),),
    ),
    _game(
        "Hawk-Dove Game",
        _BOX,
        ("Dove", "Hawk"),
        "Doves share value 6 peacefully. A lone Hawk takes all; multiple Hawks share value minus fighting cost 10. Pure equilibria have one Hawk.",
        ThreePlayerHawkDove,
        (_binary_profile(((1 + np.sqrt(21)) / 10,) * 3),),
    ),
    _game(
        "Stag Hunt",
        _BOX,
        ("Hare", "Stag"),
        "Hare guarantees 2. Stag yields 5 only when everyone chooses Stag, and 0 otherwise. Both unanimous profiles are pure equilibria.",
        StagHunt,
        (_binary_profile((1 - np.sqrt(2 / 5),) * 3),),
    ),
    _game(
        "Rock Paper Scissors",
        _RPS,
        ("Rock", "Paper", "Scissors"),
        "The standard zero-sum cyclic game. Uniform mixing is a Nash equilibrium.",
        RockPaperScissors,
        (_uniform(2, 3),),
    ),
    _game(
        "Biased Rock Paper Scissors",
        _RPS,
        ("Rock", "Paper", "Scissors"),
        "Zero-sum RPS with Rock-versus-Scissors payoffs scaled to 1.2. Equilibrium places more weight on Paper.",
        BiasedRockPaperScissors,
        (np.asarray([[5 / 16, 6 / 16, 5 / 16]] * 2),),
    ),
    _game(
        "Asymmetric Rock Paper Scissors",
        _RPS,
        ("Rock", "Paper", "Scissors"),
        "Standard RPS for Player 1; Player 2's payoffs are multiplied by 0.7. Positive payoff scaling preserves best responses and the uniform equilibrium.",
        AsymmetricRockPaperScissors,
        (_uniform(2, 3),),
    ),
    _game(
        "Noisy Rock Paper Scissors",
        _RPS,
        ("Rock", "Paper", "Scissors"),
        "Independent Gaussian noise with standard deviation 0.1 is added to payoff queries, including Fictitious Play's response estimates. Displayed diagnostics and equilibria use the noise-free expected table.",
        RockPaperScissorsWithNoise,
        (_uniform(2, 3),),
    ),
    _game(
        "Rock Paper Scissors Lizard Spock",
        _RPS,
        ("Rock", "Paper", "Scissors", "Lizard", "Spock"),
        "Five-action zero-sum RPS: each action beats two others and loses to two. Uniform mixing is a certified equilibrium.",
        RockPaperScissorsLizardSpock,
        (_uniform(2, 5),),
    ),
    _game(
        "Rock Paper Scissors Well",
        _RPS,
        ("Rock", "Paper", "Scissors", "Well"),
        "Well beats Rock and Scissors and loses to Paper. A certified equilibrium assigns no mass to Rock and mixes Paper, Scissors, and Well equally.",
        RockPaperScissorsWell,
        (np.asarray([[0, 1 / 3, 1 / 3, 1 / 3]] * 2),),
    ),
    *(
        _game(
            f"Cyclic Game ({actions} actions)",
            _RPS,
            tuple(f"Action {i}" for i in range(actions)),
            "Each action loses to its next neighbor and beats its previous neighbor; other matchups tie. Uniform mixing is a certified equilibrium.",
            lambda count=actions: CyclicGame(count),
            (_uniform(2, actions),),
        )
        for actions in (4, 5, 6)
    ),
    _game(
        "Minority Game",
        _RPS,
        ("Option A", "Option B"),
        "Different actions reward both players with 1; matching costs both 1. Two off-diagonal pure equilibria and uniform mixed equilibrium exist.",
        MinorityGame,
        (_uniform(2, 2),),
    ),
    _game(
        "Dispersion Game",
        _RPS,
        ("Action 0", "Action 1", "Action 2"),
        "A three-action anti-coordination game. Different actions reward both players with 1; matching costs both 1. All six off-diagonal profiles are pure equilibria.",
        DispersionGame,
        (_uniform(2, 3),),
    ),
)
GAMES = MappingProxyType({game.name: game for game in _game_specs})

_algorithm_specs = (
    AlgorithmSpec(
        "BFTL-EXP3",
        "Samples an exploratory exponential policy and updates scores with full counterfactual feedback. The existing update contains no bandit estimator or optimistic predictor.",
        "Full information",
        (_BOX,),
        BFTL_EXP3,
    ),
    AlgorithmSpec(
        "EXP3",
        "Importance-weighted selected-action rewards update exponential scores. This implementation has no explicit exploration mixture or reward normalization.",
        "Bandit",
        (_BOX, _RPS),
        EXP3,
    ),
    AlgorithmSpec(
        "Hedge",
        "Exponential policy from accumulated learning-rate-scaled counterfactual payoffs. The learning-rate schedule controls each score increment.",
        "Full information",
        (_BOX, _RPS),
        Hedge,
    ),
    AlgorithmSpec(
        "Optimistic FTRL",
        "Exponential policy with the preserved score increment eta times (twice the current counterfactual payoff minus its previous value).",
        "Full information",
        (_BOX,),
        OptimisticFTRL,
    ),
    AlgorithmSpec(
        "Projected Gradient Ascent",
        "Projects accumulated payoff scores onto the probability simplex. This implementation preserves the cumulative-score recurrence.",
        "Full information",
        (_BOX,),
        ProjectedGradientAscent,
    ),
    AlgorithmSpec(
        "Regret Matching",
        "Policies are proportional to positive external cumulative regrets. Optional nonnegative initialization values are regret priors; default priors are zero.",
        "Full information",
        (_BOX,),
        RegretMatching,
    ),
    AlgorithmSpec(
        "Regret Matching Softmax",
        "Applies a learning-rate-scaled softmax to external cumulative regrets. It maintains one regret value per action, without a swap-regret matrix.",
        "Full information",
        (_BOX,),
        RegretMatchingSoftmax,
    ),
    AlgorithmSpec(
        "Fictitious Play",
        "Best responds to opponents' empirical action counts. Pseudocounts initialize beliefs; the decision policy differs from empirical frequencies. In noisy games, action-payoff estimates use fresh noise samples.",
        "Payoff table / beliefs",
        (_RPS,),
        FictitiousPlay,
    ),
    AlgorithmSpec(
        "Smooth Fictitious Play",
        "Softmax response to expected payoffs against empirical opponent beliefs. Temperature smooths the response; pseudocounts initialize beliefs. In noisy games, action-payoff estimates use fresh noise samples.",
        "Payoff table / beliefs",
        (_RPS,),
        SmoothFictitiousPlay,
    ),
)
ALGORITHMS = MappingProxyType({algorithm.name: algorithm for algorithm in _algorithm_specs})


def games_for_category(category):
    return tuple(name for name, game in GAMES.items() if game.category == category)


def algorithms_for_game(game_name):
    return tuple(
        name
        for name, algorithm in ALGORITHMS.items()
        if GAMES[game_name].category in algorithm.categories
    )


def canonical_algorithm_name(name):
    return "Smooth Fictitious Play" if name == "Smooth FP (T=0.1)" else name
