import numpy as np

from .common import initial_vectors, softmax


class EXP3:
    def __init__(self, game, num_iterations, eta_config, rng=None):
        self.game = game
        self.num_iterations = num_iterations
        self.eta_config = eta_config
        self.num_players = game.num_players
        self.num_actions = game.num_actions
        self.scores = [np.zeros(self.num_actions[i]) for i in range(self.num_players)]
        self.strategies = []
        self.policy_history = self.strategies
        self.rng = np.random.default_rng() if rng is None else rng

    def run(self, initial_scores=None):
        self.scores = initial_vectors(initial_scores, self.num_actions)
        self.strategies = []
        self.policy_history = self.strategies

        for n in range(self.num_iterations):
            eta = self.eta_config["initial_eta"] * (n + 1) ** self.eta_config["decay_rate"]

            # Calculate mixed strategy from scores
            strategy_profile = []
            for i in range(self.num_players):
                player_scores = self.scores[i]
                strategy = softmax(player_scores)
                strategy_profile.append(strategy)
            self.strategies.append(strategy_profile)

            # Sample actions from the strategy profile
            action_profile = []
            for i in range(self.num_players):
                action = self.rng.choice(self.num_actions[i], p=strategy_profile[i])
                action_profile.append(action)
            action_profile = tuple(action_profile)

            # Get payoffs for the chosen actions
            payoffs = self.game.get_payoff(action_profile)

            # Estimate payoffs
            estimated_payoffs = []
            for i in range(self.num_players):
                estimated_payoff_vector = np.zeros(self.num_actions[i])
                chosen_action = action_profile[i]
                prob = strategy_profile[i][chosen_action]
                estimated_payoff_vector[chosen_action] = payoffs[i] / prob
                estimated_payoffs.append(estimated_payoff_vector)

            # Update scores
            for i in range(self.num_players):
                self.scores[i] += eta * estimated_payoffs[i]
