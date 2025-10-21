
import numpy as np
from .base import Game

class PureCoordination(Game):
    def __init__(self):
        payoff_matrix = np.zeros((2, 2, 2, 3))
        payoff_matrix[0, 0, 0] = [1, 1, 1]
        payoff_matrix[0, 0, 1] = [0, 0, 0]
        payoff_matrix[0, 1, 0] = [0, 0, 0]
        payoff_matrix[0, 1, 1] = [1, 1, 1]
        payoff_matrix[1, 0, 0] = [0, 0, 0]
        payoff_matrix[1, 0, 1] = [1, 1, 1]
        payoff_matrix[1, 1, 0] = [1, 1, 1]
        payoff_matrix[1, 1, 1] = [0, 0, 0]
        super().__init__(payoff_matrix, 3)

class CoordinationWithSpectator(Game):
    def __init__(self):
        payoff_matrix = np.zeros((2, 2, 2, 3))
        payoff_matrix[0, 0, 0] = [1, 1, 0]
        payoff_matrix[0, 0, 1] = [1, 1, 0]
        payoff_matrix[0, 1, 0] = [0, 0, 0]
        payoff_matrix[0, 1, 1] = [0, 0, 0]
        payoff_matrix[1, 0, 0] = [0, 0, 0]
        payoff_matrix[1, 0, 1] = [0, 0, 0]
        payoff_matrix[1, 1, 0] = [1, 1, 0]
        payoff_matrix[1, 1, 1] = [1, 1, 0]
        super().__init__(payoff_matrix, 3)

class MatchingPenniesWithTwist(Game):
    def __init__(self):
        payoff_matrix = np.zeros((2, 2, 2, 3))
        payoff_matrix[0, 0, 0] = [0, 1, 0]
        payoff_matrix[0, 0, 1] = [0, 0, 1]
        payoff_matrix[0, 1, 0] = [0, 0, 1]
        payoff_matrix[0, 1, 1] = [0, 1, 0]
        payoff_matrix[1, 0, 0] = [0.1, 0, 1]
        payoff_matrix[1, 0, 1] = [0.1, 1, 0]
        payoff_matrix[1, 1, 0] = [0.1, 1, 0]
        payoff_matrix[1, 1, 1] = [0.1, 0, 1]
        super().__init__(payoff_matrix, 3)

class MatchingPenniesWithOutsideOption(Game):
    def __init__(self):
        payoff_matrix = np.zeros((2, 2, 2, 3))
        payoff_matrix[0, 0, 0] = [-1, 1, 1]
        payoff_matrix[0, 0, 1] = [1, 1, -1]
        payoff_matrix[0, 1, 0] = [-1, -1, -1]
        payoff_matrix[0, 1, 1] = [1, -1, 1]
        payoff_matrix[1, 0, 0] = [1, -1, -1]
        payoff_matrix[1, 0, 1] = [-1, 1, 1]
        payoff_matrix[1, 1, 0] = [1, 1, 1]
        payoff_matrix[1, 1, 1] = [-1, 1, -1]
        super().__init__(payoff_matrix, 3)
