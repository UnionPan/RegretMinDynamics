import numpy as np

from .common import full_information_payoffs, initial_vectors


class ProjectedGradientAscent:
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

            # Calculate mixed strategy from scores (projection onto the simplex)
            strategy_profile = []
            for i in range(self.num_players):
                strategy = self._project_to_simplex(self.scores[i])
                strategy_profile.append(strategy)
            self.strategies.append(strategy_profile)

            # Sample actions from the strategy profile
            action_profile = []
            for i in range(self.num_players):
                action = self.rng.choice(self.num_actions[i], p=strategy_profile[i])
                action_profile.append(action)
            action_profile = tuple(action_profile)

            # Get full payoff feedback for all actions (no importance sampling)
            full_payoffs = full_information_payoffs(self.game, action_profile)

            # Update scores
            for i in range(self.num_players):
                self.scores[i] += eta * full_payoffs[i]

    def _project_to_simplex(self, v):
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w
