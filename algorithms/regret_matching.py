
import numpy as np

class RegretMatching:
    def __init__(self, game, num_iterations):
        self.game = game
        self.num_iterations = num_iterations
        self.num_players = game.num_players
        self.num_actions = game.num_actions
        self.regret_sum = [np.zeros(self.num_actions[i]) for i in range(self.num_players)]
        self.strategy_sum = [np.zeros(self.num_actions[i]) for i in range(self.num_players)]
        self.strategies = []

    def run(self, initial_scores=None): # initial_scores is not used here, but kept for compatibility
        for n in range(self.num_iterations):
            # Get current strategy
            strategy_profile = self._get_strategy()
            self.strategies.append(strategy_profile)

            # Sample actions from the strategy profile
            action_profile = []
            for i in range(self.num_players):
                action = np.random.choice(self.num_actions[i], p=strategy_profile[i])
                action_profile.append(action)
            action_profile = tuple(action_profile)

            # Get payoffs for the chosen actions
            payoffs = self.game.get_payoff(action_profile)

            # Update regrets
            for i in range(self.num_players):
                # For each action of player i, calculate the counterfactual payoff
                counterfactual_payoffs = np.zeros(self.num_actions[i])
                for j in range(self.num_actions[i]):
                    if j == action_profile[i]:
                        counterfactual_payoffs[j] = payoffs[i]
                    else:
                        counterfactual_action_profile = list(action_profile)
                        counterfactual_action_profile[i] = j
                        counterfactual_payoffs[j] = self.game.get_payoff(tuple(counterfactual_action_profile))[i]
                
                self.regret_sum[i] += counterfactual_payoffs - payoffs[i]

            # Update strategy sum
            for i in range(self.num_players):
                self.strategy_sum[i] += strategy_profile[i]

    def _get_strategy(self):
        strategy_profile = []
        for i in range(self.num_players):
            positive_regret = np.maximum(self.regret_sum[i], 0)
            sum_positive_regret = np.sum(positive_regret)
            if sum_positive_regret > 0:
                strategy = positive_regret / sum_positive_regret
            else:
                strategy = np.ones(self.num_actions[i]) / self.num_actions[i]
            strategy_profile.append(strategy)
        return strategy_profile

    def get_average_strategy(self):
        avg_strategy = []
        for i in range(self.num_players):
            avg_strategy.append(self.strategy_sum[i] / self.num_iterations)
        return avg_strategy
