
import numpy as np

class Game:
    def __init__(self, payoff_matrix, num_players):
        self.payoff_matrix = payoff_matrix
        self.num_players = num_players
        self.num_actions = self.payoff_matrix.shape[:-1]

    def get_payoff(self, actions):
        return self.payoff_matrix[actions]
