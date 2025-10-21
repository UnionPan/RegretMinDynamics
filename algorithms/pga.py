
import numpy as np

class ProjectedGradientAscent:
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
            
            # Calculate mixed strategy from scores (projection onto the simplex)
            strategy_profile = []
            for i in range(self.num_players):
                strategy = self._project_to_simplex(self.scores[i])
                strategy_profile.append(strategy)
            self.strategies.append(strategy_profile)

            # Sample actions from the strategy profile
            action_profile = []
            for i in range(self.num_players):
                action = np.random.choice(self.num_actions[i], p=strategy_profile[i])
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
