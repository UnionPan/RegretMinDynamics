
import numpy as np

class BFTL_EXP3:
    def __init__(self, game, num_iterations, eta_config, delta_config):
        self.game = game
        self.num_iterations = num_iterations
        self.eta_config = eta_config
        self.delta_config = delta_config
        self.num_players = game.num_players
        self.num_actions = game.num_actions
        self.scores = [np.zeros(self.num_actions[i]) for i in range(self.num_players)]
        self.strategies = []

    def run(self, initial_scores=None):
        if initial_scores:
            self.scores = initial_scores

        for n in range(self.num_iterations):
            eta = self.eta_config['initial_eta'] * (n + 1) ** self.eta_config['decay_rate']
            delta = self.delta_config['initial_delta'] * (n + 1) ** self.delta_config['decay_rate']

            # Calculate mixed strategy from scores
            strategy_profile = []
            for i in range(self.num_players):
                player_scores = self.scores[i]
                exp_scores = np.exp(player_scores - np.max(player_scores))
                strategy = exp_scores / np.sum(exp_scores)
                strategy_profile.append(strategy)
            self.strategies.append(strategy_profile)

            # Sample actions with exploration
            action_profile = []
            explored_strategy_profile = []
            for i in range(self.num_players):
                uniform_strategy = np.ones(self.num_actions[i]) / self.num_actions[i]
                explored_strategy = (1 - delta) * strategy_profile[i] + delta * uniform_strategy
                explored_strategy_profile.append(explored_strategy)
                action = np.random.choice(self.num_actions[i], p=explored_strategy)
                action_profile.append(action)
            action_profile = tuple(action_profile)

            # Get full payoff feedback for all actions (no importance sampling)
            full_payoffs = []
            for i in range(self.num_players):
                payoff_vector = np.zeros(self.num_actions[i])
                for a in range(self.num_actions[i]):
                    # Create counterfactual action profile
                    counterfactual_profile = list(action_profile)
                    counterfactual_profile[i] = a
                    payoff_vector[a] = self.game.get_payoff(tuple(counterfactual_profile))[i]
                full_payoffs.append(payoff_vector)

            # Update scores
            for i in range(self.num_players):
                self.scores[i] += eta * full_payoffs[i]
