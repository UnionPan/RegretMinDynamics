
import numpy as np

class EXP3:
    def __init__(self, game, num_iterations, eta_config):
        self.game = game
        self.num_iterations = num_iterations
        self.eta_config = eta_config
        self.num_players = game.num_players
        self.num_actions = game.num_actions
        self.scores = [np.zeros(self.num_actions[i]) for i in range(self.num_players)]
        self.strategies = []

    def run(self, initial_scores=None):
        if initial_scores:
            self.scores = initial_scores
        for n in range(self.num_iterations):
            eta = self.eta_config['initial_eta'] * (n + 1) ** self.eta_config['decay_rate']
            
            # Calculate mixed strategy from scores
            strategy_profile = []
            for i in range(self.num_players):
                player_scores = self.scores[i]
                exp_scores = np.exp(player_scores - np.max(player_scores))
                strategy = exp_scores / np.sum(exp_scores)
                strategy_profile.append(strategy)
            self.strategies.append(strategy_profile)

            # Sample actions from the strategy profile
            action_profile = []
            for i in range(self.num_players):
                action = np.random.choice(self.num_actions[i], p=strategy_profile[i])
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
