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

        # Base payoffs
        p1_payoffs = np.array([
            [ 0, -1,  1 + rock_bonus],  # Rock (bonus when beating Scissors)
            [ 1,  0, -1],                # Paper
            [-1,  1,  0]                 # Scissors
        ])

        # Player 2 sees the opposite (with Rock bonus)
        p2_payoffs = np.array([
            [ 0,  1, -1],                # Rock
            [-1,  0,  1],                # Paper
            [-1 - rock_bonus, -1, 0]     # Scissors (loses more to Rock)
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
