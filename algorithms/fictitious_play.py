import numpy as np


class FictitiousPlay:
    """
    Fictitious Play algorithm for 2-player games.

    Each player maintains beliefs about opponent's strategy based on
    historical action frequencies, and best-responds to these beliefs.
    """
    def __init__(self, game, num_iterations):
        self.game = game
        self.num_iterations = num_iterations
        self.num_players = game.num_players
        self.num_actions = game.num_actions

        assert self.num_players == 2, "Fictitious Play only supports 2-player games"

        # History tracking
        self.strategies = []
        self.beliefs = []  # Track belief history
        self.action_counts = None

    def get_belief(self, player, action_counts):
        """
        Compute belief (empirical frequency) about opponent's strategy.
        """
        opponent = 1 - player
        opponent_counts = action_counts[opponent]
        total = np.sum(opponent_counts)

        if total == 0:
            # Uniform belief if no history
            return np.ones(self.num_actions[opponent]) / self.num_actions[opponent]

        return opponent_counts / total

    def compute_expected_payoff(self, player, action, opponent_belief):
        """
        Compute expected payoff for a player's action given belief about opponent.
        """
        opponent = 1 - player
        expected_payoff = 0.0

        for opp_action in range(self.num_actions[opponent]):
            actions = [None, None]
            actions[player] = action
            actions[opponent] = opp_action

            payoffs = self.game.get_payoff(actions)
            expected_payoff += payoffs[player] * opponent_belief[opp_action]

        return expected_payoff

    def best_response(self, player, opponent_belief):
        """
        Find best response action(s) to opponent's believed strategy.
        Returns a probability distribution (uniform over best responses).
        """
        expected_payoffs = np.zeros(self.num_actions[player])

        for action in range(self.num_actions[player]):
            expected_payoffs[action] = self.compute_expected_payoff(
                player, action, opponent_belief
            )

        # Find all actions that maximize expected payoff
        max_payoff = np.max(expected_payoffs)
        best_actions = np.where(np.abs(expected_payoffs - max_payoff) < 1e-10)[0]

        # Return uniform distribution over best responses
        strategy = np.zeros(self.num_actions[player])
        strategy[best_actions] = 1.0 / len(best_actions)

        return strategy

    def run(self, initial_scores=None):
        """
        Run Fictitious Play.

        initial_scores: Can be used to set initial action counts.
                       Format: [p1_counts, p2_counts] where each is an array
                       of pseudo-counts for each action.
        """
        # Initialize action counts
        if initial_scores is not None:
            self.action_counts = [
                np.array(initial_scores[0], dtype=float),
                np.array(initial_scores[1], dtype=float)
            ]
        else:
            # Start with small uniform pseudo-counts to avoid division by zero
            self.action_counts = [
                np.ones(self.num_actions[0]) * 0.1,
                np.ones(self.num_actions[1]) * 0.1
            ]

        self.strategies = []
        self.beliefs = []

        for t in range(self.num_iterations):
            # Compute beliefs about opponent for each player
            beliefs = [
                self.get_belief(0, self.action_counts),
                self.get_belief(1, self.action_counts)
            ]

            # Compute best response strategies
            strategies = [
                self.best_response(0, beliefs[0]),
                self.best_response(1, beliefs[1])
            ]

            # Sample actions from best response strategies
            actions = [
                np.random.choice(self.num_actions[0], p=strategies[0]),
                np.random.choice(self.num_actions[1], p=strategies[1])
            ]

            # Update action counts
            self.action_counts[0][actions[0]] += 1
            self.action_counts[1][actions[1]] += 1

            # Store current strategies (empirical frequencies)
            empirical_strategies = [
                self.action_counts[0] / np.sum(self.action_counts[0]),
                self.action_counts[1] / np.sum(self.action_counts[1])
            ]

            self.strategies.append(empirical_strategies)
            self.beliefs.append(beliefs)

        self.strategies = np.array(self.strategies)
        self.beliefs = np.array(self.beliefs)

        return self.strategies


class SmoothFictitiousPlay:
    """
    Smooth Fictitious Play - a variant where players don't always play
    pure best responses, but instead use a smoothed response.
    """
    def __init__(self, game, num_iterations, temperature=0.1):
        self.game = game
        self.num_iterations = num_iterations
        self.num_players = game.num_players
        self.num_actions = game.num_actions
        self.temperature = temperature

        assert self.num_players == 2, "Fictitious Play only supports 2-player games"

        self.strategies = []
        self.beliefs = []
        self.action_counts = None

    def get_belief(self, player, action_counts):
        """Compute belief about opponent's strategy."""
        opponent = 1 - player
        opponent_counts = action_counts[opponent]
        total = np.sum(opponent_counts)

        if total == 0:
            return np.ones(self.num_actions[opponent]) / self.num_actions[opponent]

        return opponent_counts / total

    def compute_expected_payoff(self, player, action, opponent_belief):
        """Compute expected payoff for a player's action given belief."""
        opponent = 1 - player
        expected_payoff = 0.0

        for opp_action in range(self.num_actions[opponent]):
            actions = [None, None]
            actions[player] = action
            actions[opponent] = opp_action

            payoffs = self.game.get_payoff(actions)
            expected_payoff += payoffs[player] * opponent_belief[opp_action]

        return expected_payoff

    def smooth_response(self, player, opponent_belief):
        """
        Compute smoothed response using softmax over expected payoffs.
        """
        expected_payoffs = np.zeros(self.num_actions[player])

        for action in range(self.num_actions[player]):
            expected_payoffs[action] = self.compute_expected_payoff(
                player, action, opponent_belief
            )

        # Apply softmax with temperature
        exp_payoffs = np.exp(expected_payoffs / self.temperature)
        strategy = exp_payoffs / np.sum(exp_payoffs)

        return strategy

    def run(self, initial_scores=None):
        """Run Smooth Fictitious Play."""
        if initial_scores is not None:
            self.action_counts = [
                np.array(initial_scores[0], dtype=float),
                np.array(initial_scores[1], dtype=float)
            ]
        else:
            self.action_counts = [
                np.ones(self.num_actions[0]) * 0.1,
                np.ones(self.num_actions[1]) * 0.1
            ]

        self.strategies = []
        self.beliefs = []

        for t in range(self.num_iterations):
            # Compute beliefs
            beliefs = [
                self.get_belief(0, self.action_counts),
                self.get_belief(1, self.action_counts)
            ]

            # Compute smooth response strategies
            strategies = [
                self.smooth_response(0, beliefs[0]),
                self.smooth_response(1, beliefs[1])
            ]

            # Sample actions
            actions = [
                np.random.choice(self.num_actions[0], p=strategies[0]),
                np.random.choice(self.num_actions[1], p=strategies[1])
            ]

            # Update counts
            self.action_counts[0][actions[0]] += 1
            self.action_counts[1][actions[1]] += 1

            # Store empirical strategies
            empirical_strategies = [
                self.action_counts[0] / np.sum(self.action_counts[0]),
                self.action_counts[1] / np.sum(self.action_counts[1])
            ]

            self.strategies.append(empirical_strategies)
            self.beliefs.append(beliefs)

        self.strategies = np.array(self.strategies)
        self.beliefs = np.array(self.beliefs)

        return self.strategies
