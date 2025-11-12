import numpy as np


class RockPaperScissors:
    """
    Standard Rock Paper Scissors game.
    Actions: 0=Rock, 1=Paper, 2=Scissors
    """
    def __init__(self):
        self.num_players = 2
        self.num_actions = [3, 3]
        self.name = "RockPaperScissors"

        # Payoff matrix: [p1_action, p2_action, player]
        self.payoff_matrix = np.zeros((3, 3, 2))

        # Player 1 payoffs (rows=P1 actions, cols=P2 actions)
        p1_payoffs = np.array([
            [ 0, -1,  1],  # Rock
            [ 1,  0, -1],  # Paper
            [-1,  1,  0]   # Scissors
        ])

        self.payoff_matrix[:, :, 0] = p1_payoffs
        self.payoff_matrix[:, :, 1] = -p1_payoffs  # Zero-sum game

    def get_payoff(self, actions):
        """Returns payoffs for both players given their actions."""
        return self.payoff_matrix[actions[0], actions[1]]


class BiasedRockPaperScissors:
    """
    Biased RPS where Rock is slightly stronger.
    Actions: 0=Rock, 1=Paper, 2=Scissors
    """
    def __init__(self, rock_bonus=0.2):
        self.num_players = 2
        self.num_actions = [3, 3]
        self.rock_bonus = rock_bonus
        self.name = f"BiasedRPS_bonus{rock_bonus}"

        # Payoff matrix: [p1_action, p2_action, player]
        self.payoff_matrix = np.zeros((3, 3, 2))

        # Base payoffs for Player 1
        p1_payoffs = np.array([
            [ 0, -1,  1 + rock_bonus],   # Rock (bonus when beating Scissors)
            [ 1,  0, -1],                # Paper
            [-1 - rock_bonus,  1,  0]    # Scissors (loses more to Rock)
        ])

        # Player 2 payoffs (zero-sum, symmetric game)
        p2_payoffs = np.array([
            [ 0,  1, -1 - rock_bonus],   # When P1 plays Rock
            [-1,  0,  1],                # When P1 plays Paper
            [ 1 + rock_bonus, -1,  0]    # When P1 plays Scissors (Rock bonus applies)
        ])

        self.payoff_matrix[:, :, 0] = p1_payoffs
        self.payoff_matrix[:, :, 1] = p2_payoffs

    def get_payoff(self, actions):
        """Returns payoffs for both players given their actions."""
        return self.payoff_matrix[actions[0], actions[1]]


class RockPaperScissorsLizardSpock:
    """
    Extended RPS with 5 options (Rock Paper Scissors Lizard Spock).
    Actions: 0=Rock, 1=Paper, 2=Scissors, 3=Lizard, 4=Spock

    Rules:
    - Rock crushes Scissors and Lizard
    - Paper covers Rock and disproves Spock
    - Scissors cuts Paper and decapitates Lizard
    - Lizard eats Paper and poisons Spock
    - Spock vaporizes Rock and smashes Scissors
    """
    def __init__(self):
        self.num_players = 2
        self.num_actions = [5, 5]
        self.name = "RPSLS"

        # Payoff matrix: [p1_action, p2_action, player]
        self.payoff_matrix = np.zeros((5, 5, 2))

        # Player 1 payoffs - each action beats 2 others and loses to 2 others
        p1_payoffs = np.array([
            [ 0, -1,  1, 1, -1],  # Rock: beats Scissors, Lizard
            [ 1,  0, -1, -1, 1],  # Paper: beats Rock, Spock
            [-1,  1,  0, 1, -1],  # Scissors: beats Paper, Lizard
            [-1,  1, -1, 0, 1],   # Lizard: beats Paper, Spock
            [ 1, -1,  1, -1, 0]   # Spock: beats Rock, Scissors
        ])

        self.payoff_matrix[:, :, 0] = p1_payoffs
        self.payoff_matrix[:, :, 1] = -p1_payoffs  # Zero-sum game

    def get_payoff(self, actions):
        """Returns payoffs for both players given their actions."""
        return self.payoff_matrix[actions[0], actions[1]]


class AsymmetricRockPaperScissors:
    """
    Asymmetric RPS where players have different payoff structures.
    Player 1 gets standard payoffs, Player 2 gets modified payoffs.
    """
    def __init__(self, p2_scale=0.7):
        self.num_players = 2
        self.num_actions = [3, 3]
        self.p2_scale = p2_scale
        self.name = f"AsymmetricRPS_scale{p2_scale}"

        # Payoff matrix: [p1_action, p2_action, player]
        self.payoff_matrix = np.zeros((3, 3, 2))

        # Standard payoff matrix for player 1
        p1_payoffs = np.array([
            [ 0, -1,  1],  # Rock
            [ 1,  0, -1],  # Paper
            [-1,  1,  0]   # Scissors
        ])

        # Scaled payoff matrix for player 2 (not quite zero-sum)
        p2_payoffs = -p1_payoffs * p2_scale

        self.payoff_matrix[:, :, 0] = p1_payoffs
        self.payoff_matrix[:, :, 1] = p2_payoffs

    def get_payoff(self, actions):
        """Returns payoffs for both players given their actions."""
        return self.payoff_matrix[actions[0], actions[1]]


class RockPaperScissorsWithNoise:
    """
    RPS where outcomes have some randomness (not fully deterministic).
    Win/loss probabilities are noisy.
    """
    def __init__(self, noise_level=0.1):
        self.num_players = 2
        self.num_actions = [3, 3]
        self.noise_level = noise_level
        self.name = f"NoisyRPS_noise{noise_level}"

        # Base payoff matrix: [p1_action, p2_action, player]
        self.base_payoff_matrix = np.zeros((3, 3, 2))

        # Base payoffs for player 1
        p1_payoffs = np.array([
            [ 0, -1,  1],  # Rock
            [ 1,  0, -1],  # Paper
            [-1,  1,  0]   # Scissors
        ])

        self.base_payoff_matrix[:, :, 0] = p1_payoffs
        self.base_payoff_matrix[:, :, 1] = -p1_payoffs  # Zero-sum base

    def get_payoff(self, actions):
        """
        Returns noisy payoffs for both players.
        Noise is added independently to each player's payoff.
        """
        base_payoffs = self.base_payoff_matrix[actions[0], actions[1]].copy()

        # Add independent noise to each player
        noise = np.random.normal(0, self.noise_level, size=2)
        return base_payoffs + noise


class RockPaperScissorsWell:
    """
    Rock-Paper-Scissors-Well (4 actions)
    Actions: 0=Rock, 1=Paper, 2=Scissors, 3=Well

    Rules:
    - Standard RPS for Rock, Paper, Scissors
    - Well beats Rock and Scissors (contains them)
    - Paper beats Well (covers it)
    """
    def __init__(self):
        self.num_players = 2
        self.num_actions = [4, 4]
        self.name = "RPSW"

        # Payoff matrix: [p1_action, p2_action, player]
        self.payoff_matrix = np.zeros((4, 4, 2))

        # Player 1 payoffs
        p1_payoffs = np.array([
            [ 0, -1,  1, -1],  # Rock: beats Scissors, loses to Paper, Well
            [ 1,  0, -1,  1],  # Paper: beats Rock, Well, loses to Scissors
            [-1,  1,  0, -1],  # Scissors: beats Paper, loses to Rock, Well
            [ 1, -1,  1,  0]   # Well: beats Rock, Scissors, loses to Paper
        ])

        self.payoff_matrix[:, :, 0] = p1_payoffs
        self.payoff_matrix[:, :, 1] = -p1_payoffs  # Zero-sum game

    def get_payoff(self, actions):
        """Returns payoffs for both players given their actions."""
        return self.payoff_matrix[actions[0], actions[1]]


class CyclicGame:
    """
    Generalized Cyclic Game with n actions.
    Action i beats action (i+1) mod n and loses to action (i-1) mod n.
    Creates a perfect cycle.
    """
    def __init__(self, n_actions=4):
        self.num_players = 2
        self.num_actions = [n_actions, n_actions]
        self.n_actions = n_actions
        self.name = f"Cyclic{n_actions}"

        # Payoff matrix: [p1_action, p2_action, player]
        self.payoff_matrix = np.zeros((n_actions, n_actions, 2))

        # Player 1 payoffs - cyclic structure
        p1_payoffs = np.zeros((n_actions, n_actions))
        for i in range(n_actions):
            for j in range(n_actions):
                if i == j:
                    p1_payoffs[i, j] = 0  # Tie
                elif (i + 1) % n_actions == j:
                    p1_payoffs[i, j] = -1  # i loses to i+1
                elif (j + 1) % n_actions == i:
                    p1_payoffs[i, j] = 1  # i beats j+1 (i.e., j loses to j+1)

        self.payoff_matrix[:, :, 0] = p1_payoffs
        self.payoff_matrix[:, :, 1] = -p1_payoffs  # Zero-sum game

    def get_payoff(self, actions):
        """Returns payoffs for both players given their actions."""
        return self.payoff_matrix[actions[0], actions[1]]


class MinorityGame:
    """
    Minority Game (El Farol Bar Problem variant)
    2-player, 2-action game where players want to be in the minority.

    Actions: 0=Option A, 1=Option B
    Payoff: +1 if in minority, -1 if in majority, 0 if tied

    This game has no pure or mixed Nash equilibrium in the traditional sense,
    leading to perpetual oscillations and potentially chaotic dynamics.
    """
    def __init__(self):
        self.num_players = 2
        self.num_actions = [2, 2]
        self.name = "MinorityGame"

        # Payoff matrix: [p1_action, p2_action, player]
        self.payoff_matrix = np.zeros((2, 2, 2))

        # Different choices - both in minority/majority (doesn't apply in 2-player)
        # In 2-player version: different = good, same = bad
        self.payoff_matrix[0, 0] = [-1, -1]  # Both choose A - both penalized
        self.payoff_matrix[1, 1] = [-1, -1]  # Both choose B - both penalized
        self.payoff_matrix[0, 1] = [1, 1]    # Different - both rewarded
        self.payoff_matrix[1, 0] = [1, 1]    # Different - both rewarded

    def get_payoff(self, actions):
        """Returns payoffs for both players given their actions."""
        return self.payoff_matrix[actions[0], actions[1]]


class DispersionGame:
    """
    Dispersion Game (3 actions)
    Players prefer to choose different actions from opponent.
    Like Minority Game but with 3 options.

    Actions: 0, 1, 2
    Payoff: Higher reward for choosing less popular option
    """
    def __init__(self):
        self.num_players = 2
        self.num_actions = [3, 3]
        self.name = "DispersionGame"

        # Payoff matrix: [p1_action, p2_action, player]
        self.payoff_matrix = np.zeros((3, 3, 2))

        # Diagonal (same choice) - both get -1
        for i in range(3):
            self.payoff_matrix[i, i] = [-1, -1]

        # Off-diagonal (different choice) - both get +1
        for i in range(3):
            for j in range(3):
                if i != j:
                    self.payoff_matrix[i, j] = [1, 1]

    def get_payoff(self, actions):
        """Returns payoffs for both players given their actions."""
        return self.payoff_matrix[actions[0], actions[1]]
