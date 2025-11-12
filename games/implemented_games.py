
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

class ThreePlayerPrisonersDilemma(Game):
    """
    3-Player Prisoner's Dilemma
    Actions: 0=Defect, 1=Cooperate
    Defecting is a dominant strategy, but all better off if all cooperate.
    True PD: T > R > P > S (Temptation > Reward > Punishment > Sucker)
    """
    def __init__(self):
        payoff_matrix = np.zeros((2, 2, 2, 3))

        # All defect (0,0,0) - Punishment
        payoff_matrix[0, 0, 0] = [1, 1, 1]

        # Two defect, one cooperates - Sucker gets 0, Defectors get Temptation
        payoff_matrix[0, 0, 1] = [2.5, 2.5, 0]  # P3 cooperates (sucker)
        payoff_matrix[0, 1, 0] = [2.5, 0, 2.5]  # P2 cooperates (sucker)
        payoff_matrix[1, 0, 0] = [0, 2.5, 2.5]  # P1 cooperates (sucker)

        # One defects, two cooperate - Defector gets highest, Cooperators get Reward
        payoff_matrix[0, 1, 1] = [4, 2, 2]  # P1 defects (temptation)
        payoff_matrix[1, 0, 1] = [2, 4, 2]  # P2 defects (temptation)
        payoff_matrix[1, 1, 0] = [2, 2, 4]  # P3 defects (temptation)

        # All cooperate (1,1,1) - Reward (but T > R, so not Nash equilibrium)
        payoff_matrix[1, 1, 1] = [3, 3, 3]

        super().__init__(payoff_matrix, 3)

class PublicGoodsGame(Game):
    """
    Public Goods Game
    Actions: 0=Free-ride, 1=Contribute
    Contributions are multiplied (by 1.5) and shared equally.
    Individual contribution costs 1, but multiplied benefit is 0.5 per person.
    """
    def __init__(self, multiplier=1.5):
        payoff_matrix = np.zeros((2, 2, 2, 3))

        # Nobody contributes
        payoff_matrix[0, 0, 0] = [0, 0, 0]

        # One contributes (costs 1, generates 1.5 shared = 0.5 each)
        payoff_matrix[0, 0, 1] = [0.5, 0.5, -0.5]
        payoff_matrix[0, 1, 0] = [0.5, -0.5, 0.5]
        payoff_matrix[1, 0, 0] = [-0.5, 0.5, 0.5]

        # Two contribute (cost 2, generate 3 shared = 1 each)
        payoff_matrix[0, 1, 1] = [1, 0, 0]  # P1 free-rides
        payoff_matrix[1, 0, 1] = [0, 1, 0]  # P2 free-rides
        payoff_matrix[1, 1, 0] = [0, 0, 1]  # P3 free-rides

        # All contribute (cost 3, generate 4.5 shared = 1.5 each, net = 0.5 each)
        payoff_matrix[1, 1, 1] = [0.5, 0.5, 0.5]

        super().__init__(payoff_matrix, 3)

class VolunteersDilemma(Game):
    """
    Volunteer's Dilemma
    Actions: 0=Don't volunteer, 1=Volunteer
    If at least one volunteers (pays cost 2), everyone gets benefit 3.
    Otherwise everyone gets 0.
    """
    def __init__(self):
        payoff_matrix = np.zeros((2, 2, 2, 3))

        # Nobody volunteers - everyone gets 0
        payoff_matrix[0, 0, 0] = [0, 0, 0]

        # One volunteers - that person pays cost
        payoff_matrix[0, 0, 1] = [3, 3, 1]  # P3 volunteers
        payoff_matrix[0, 1, 0] = [3, 1, 3]  # P2 volunteers
        payoff_matrix[1, 0, 0] = [1, 3, 3]  # P1 volunteers

        # Two volunteer - both pay cost
        payoff_matrix[0, 1, 1] = [3, 1, 1]
        payoff_matrix[1, 0, 1] = [1, 3, 1]
        payoff_matrix[1, 1, 0] = [1, 1, 3]

        # All volunteer - all pay cost
        payoff_matrix[1, 1, 1] = [1, 1, 1]

        super().__init__(payoff_matrix, 3)

class MajorityGame(Game):
    """
    Majority Game
    Actions: 0=Action A, 1=Action B
    Players prefer to be in the majority.
    Payoff = 2 if in majority, 0 if tied, -1 if in minority.
    """
    def __init__(self):
        payoff_matrix = np.zeros((2, 2, 2, 3))

        # All choose same action - all get 2
        payoff_matrix[0, 0, 0] = [2, 2, 2]  # All A
        payoff_matrix[1, 1, 1] = [2, 2, 2]  # All B

        # Two vs one - majority gets 2, minority gets -1
        payoff_matrix[0, 0, 1] = [2, 2, -1]  # P1, P2 in majority (A)
        payoff_matrix[0, 1, 0] = [2, -1, 2]  # P1, P3 in majority (A)
        payoff_matrix[1, 0, 0] = [-1, 2, 2]  # P2, P3 in majority (A)
        payoff_matrix[1, 1, 0] = [2, 2, -1]  # P1, P2 in majority (B)
        payoff_matrix[1, 0, 1] = [2, -1, 2]  # P1, P3 in majority (B)
        payoff_matrix[0, 1, 1] = [-1, 2, 2]  # P2, P3 in majority (B)

        super().__init__(payoff_matrix, 3)

class ThreePlayerHawkDove(Game):
    """
    3-Player Hawk-Dove Game
    Actions: 0=Dove, 1=Hawk
    Resource value = 6, fighting cost = 10
    - All Dove: share equally (2 each)
    - All Hawk: fight, get negative payoff (-4 each)
    - Mixed: Hawks split resource, Doves get nothing
    """
    def __init__(self):
        payoff_matrix = np.zeros((2, 2, 2, 3))
        V = 6  # Resource value
        C = 10  # Fighting cost

        # All Dove - share peacefully
        payoff_matrix[0, 0, 0] = [V/3, V/3, V/3]

        # All Hawk - fight
        payoff_matrix[1, 1, 1] = [(V-C)/3, (V-C)/3, (V-C)/3]

        # Two Doves, one Hawk - Hawk takes all
        payoff_matrix[0, 0, 1] = [0, 0, V]
        payoff_matrix[0, 1, 0] = [0, V, 0]
        payoff_matrix[1, 0, 0] = [V, 0, 0]

        # One Dove, two Hawks - Hawks fight, Dove gets nothing
        payoff_matrix[0, 1, 1] = [0, (V-C)/2, (V-C)/2]
        payoff_matrix[1, 0, 1] = [(V-C)/2, 0, (V-C)/2]
        payoff_matrix[1, 1, 0] = [(V-C)/2, (V-C)/2, 0]

        super().__init__(payoff_matrix, 3)

class StagHunt(Game):
    """
    3-Player Stag Hunt
    Actions: 0=Hunt Hare (safe), 1=Hunt Stag (risky but rewarding)
    - All hunt stag: everyone gets 5
    - Hunt hare: guaranteed 2
    - Stag hunt fails if anyone defects
    """
    def __init__(self):
        payoff_matrix = np.zeros((2, 2, 2, 3))

        # All hunt hare - safe option
        payoff_matrix[0, 0, 0] = [2, 2, 2]

        # All hunt stag - best outcome but risky
        payoff_matrix[1, 1, 1] = [5, 5, 5]

        # Mixed: Stag hunters get 0, hare hunters get 2
        payoff_matrix[0, 0, 1] = [2, 2, 0]
        payoff_matrix[0, 1, 0] = [2, 0, 2]
        payoff_matrix[1, 0, 0] = [0, 2, 2]
        payoff_matrix[0, 1, 1] = [2, 0, 0]
        payoff_matrix[1, 0, 1] = [0, 2, 0]
        payoff_matrix[1, 1, 0] = [0, 0, 2]

        super().__init__(payoff_matrix, 3)
