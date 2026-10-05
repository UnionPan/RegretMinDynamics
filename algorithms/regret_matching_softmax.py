import numpy as np

from .common import initial_vectors, softmax


class RegretMatchingSoftmax:
    """Softmax over external cumulative regrets; no swap-regret matrix is maintained."""

    def __init__(self, game, num_iterations, eta_config, rng=None):
        self.game = game
        self.num_iterations = num_iterations
        self.eta_config = eta_config
        self.num_players = game.num_players
        self.num_actions = game.num_actions

        # External regret: regret_sum[i][a] compares fixed action a with actual play
        # (counterfactual regret comparing each action to the action actually taken)
        self.regret_sum = [np.zeros(self.num_actions[i]) for i in range(self.num_players)]
        self.strategies = []
        self.policy_history = self.strategies
        self.rng = np.random.default_rng() if rng is None else rng

    def run(self, initial_scores=None):
        self.regret_sum = initial_vectors(initial_scores, self.num_actions)
        self.strategies = []
        self.policy_history = self.strategies

        for n in range(self.num_iterations):
            eta = self.eta_config["initial_eta"] * (n + 1) ** self.eta_config["decay_rate"]

            # Get current strategy using softmax over cumulative regrets
            strategy_profile = self._get_softmax_strategy(eta)
            self.strategies.append(strategy_profile)

            # Sample actions from the strategy profile
            action_profile = []
            for i in range(self.num_players):
                action = self.rng.choice(self.num_actions[i], p=strategy_profile[i])
                action_profile.append(action)
            action_profile = tuple(action_profile)

            # Get payoffs for the chosen actions
            payoffs = self.game.get_payoff(action_profile)

            # Update external regrets
            for i in range(self.num_players):
                # For each action of player i, calculate the counterfactual payoff
                counterfactual_payoffs = np.zeros(self.num_actions[i])
                for a in range(self.num_actions[i]):
                    # Create counterfactual action profile where player i plays action a
                    counterfactual_action_profile = list(action_profile)
                    counterfactual_action_profile[i] = a
                    counterfactual_payoffs[a] = self.game.get_payoff(
                        tuple(counterfactual_action_profile)
                    )[i]

                # Update regret: how much better each action would have been
                self.regret_sum[i] += counterfactual_payoffs - payoffs[i]

    def _get_softmax_strategy(self, eta):
        """
        Convert cumulative regrets to strategy using softmax (exponential weights).

        Unlike standard regret matching (which uses positive regrets proportionally),
        this uses exponential weighting similar to Hedge.
        """
        strategy_profile = []
        for i in range(self.num_players):
            # Apply softmax to cumulative regrets with learning rate eta
            scaled_regrets = eta * self.regret_sum[i]
            # Numerical stability: subtract max before exp
            strategy = softmax(scaled_regrets)
            strategy_profile.append(strategy)
        return strategy_profile

    def get_average_regret(self):
        """Returns the average regret per iteration."""
        avg_regret = []
        for i in range(self.num_players):
            avg_regret.append(self.regret_sum[i] / self.num_iterations)
        return avg_regret
