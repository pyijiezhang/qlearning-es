import numpy as np
from gym.envs.toy_text import discrete


LEFT = 0
RIGHT = 1


class RiverswimEnv6(discrete.DiscreteEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, nS=6):

        # defining the number of actions
        nA = 2

        # defining the reward system and dynamics of RiverSwim environment
        P, isd = self.__init_dynamics(nS, nA)

        super(RiverswimEnv6, self).__init__(nS, nA, P, isd)

    def __init_dynamics(self, nS, nA):

        # P[s][a] == [(probability,nextstate,reward,done),...]
        P = {}
        for s in range(nS):
            P[s] = {a: [] for a in range(nA)}

        # reward transitions
        P[0][LEFT] = [(1.0, 0, 5 / 100, 0)]
        P[nS - 1][RIGHT] = [(0.6, nS - 1, 1, 0), (0.4, nS - 2, 1, 0)]

        # left transitions
        for s in range(1, nS):
            P[s][LEFT] = [(1.0, max(0, s - 1), 0, 0)]

        # right transitions
        for s in range(1, nS - 1):
            P[s][RIGHT] = [
                (0.05, max(0, s - 1), 0, 0),
                (0.55, s, 0, 0),
                (0.4, min(nS - 1, s + 1), 0, 0),
            ]
        P[0][RIGHT] = [(0.6, 0, 0, 0), (0.4, 1, 0, 0)]

        # initial state distribution
        isd = np.zeros(nS)
        isd[0] = 1.0

        return P, isd

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class RiverswimEnv6Kappa1(discrete.DiscreteEnv):
    """ kappa-equivalent riverswim #1
    for (s_1,right),the transition is (0.05,0.45,0.5) instead of (0.05,0.55,0.4)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, nS=6):

        # defining the number of actions
        nA = 2

        # defining the reward system and dynamics of RiverSwim environment
        P, isd = self.__init_dynamics(nS, nA)

        super(RiverswimEnv6Kappa1, self).__init__(nS, nA, P, isd)

    def __init_dynamics(self, nS, nA):

        # P[s][a] == [(probability,nextstate,reward,done),...]
        P = {}
        for s in range(nS):
            P[s] = {a: [] for a in range(nA)}

        # reward transitions
        P[0][LEFT] = [(1.0, 0, 5 / 100, 0)]
        P[nS - 1][RIGHT] = [(0.6, nS - 1, 1, 0), (0.4, nS - 2, 1, 0)]

        # left transitions
        for s in range(1, nS):
            P[s][LEFT] = [(1.0, max(0, s - 1), 0, 0)]

        # right transitions
        for s in range(1, nS - 1):
            if s == 1:
                P[s][RIGHT] = [
                    (0.05, max(0, s - 1), 0, 0),
                    (0.45, s, 0, 0),
                    (0.5, min(nS - 1, s + 1), 0, 0),
                ]
            else:
                P[s][RIGHT] = [
                    (0.05, max(0, s - 1), 0, 0),
                    (0.55, s, 0, 0),
                    (0.4, min(nS - 1, s + 1), 0, 0),
                ]
        P[0][RIGHT] = [(0.6, 0, 0, 0), (0.4, 1, 0, 0)]

        # initial state distribution
        isd = np.zeros(nS)
        isd[0] = 1.0

        return P, isd

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class RiverswimEnv6Kappa2(discrete.DiscreteEnv):
    """ kappa-equivalent riverswim #2
    for (s_1,right),the transition is (0.15,0.3,0.55) instead of (0.05,0.55,0.4)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, nS=6):

        # defining the number of actions
        nA = 2

        # defining the reward system and dynamics of RiverSwim environment
        P, isd = self.__init_dynamics(nS, nA)

        super(RiverswimEnv6Kappa2, self).__init__(nS, nA, P, isd)

    def __init_dynamics(self, nS, nA):

        # P[s][a] == [(probability,nextstate,reward,done),...]
        P = {}
        for s in range(nS):
            P[s] = {a: [] for a in range(nA)}

        # reward transitions
        P[0][LEFT] = [(1.0, 0, 5 / 100, 0)]
        P[nS - 1][RIGHT] = [(0.6, nS - 1, 1, 0), (0.4, nS - 2, 1, 0)]

        # left transitions
        for s in range(1, nS):
            P[s][LEFT] = [(1.0, max(0, s - 1), 0, 0)]

        # right transitions
        for s in range(1, nS - 1):
            if s == 1:
                P[s][RIGHT] = [
                    (0.15, max(0, s - 1), 0, 0),
                    (0.3, s, 0, 0),
                    (0.55, min(nS - 1, s + 1), 0, 0),
                ]
            else:
                P[s][RIGHT] = [
                    (0.05, max(0, s - 1), 0, 0),
                    (0.55, s, 0, 0),
                    (0.4, min(nS - 1, s + 1), 0, 0),
                ]
        P[0][RIGHT] = [(0.6, 0, 0, 0), (0.4, 1, 0, 0)]

        # initial state distribution
        isd = np.zeros(nS)
        isd[0] = 1.0

        return P, isd

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class RiverswimEnv6Kappa3(discrete.DiscreteEnv):
    """ kappa-equivalent riverswim #3
    for (s_1,right) and (s_3,right),the transition is (0.15,0.3,0.55) instead of (0.05,0.55,0.4)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, nS=6):

        # defining the number of actions
        nA = 2

        # defining the reward system and dynamics of RiverSwim environment
        P, isd = self.__init_dynamics(nS, nA)

        super(RiverswimEnv6Kappa3, self).__init__(nS, nA, P, isd)

    def __init_dynamics(self, nS, nA):

        # P[s][a] == [(probability,nextstate,reward,done),...]
        P = {}
        for s in range(nS):
            P[s] = {a: [] for a in range(nA)}

        # reward transitions
        P[0][LEFT] = [(1.0, 0, 5 / 100, 0)]
        P[nS - 1][RIGHT] = [(0.6, nS - 1, 1, 0), (0.4, nS - 2, 1, 0)]

        # left transitions
        for s in range(1, nS):
            P[s][LEFT] = [(1.0, max(0, s - 1), 0, 0)]

        # right transitions
        for s in range(1, nS - 1):
            if s == 1 or s == 3:
                P[s][RIGHT] = [
                    (0.15, max(0, s - 1), 0, 0),
                    (0.3, s, 0, 0),
                    (0.55, min(nS - 1, s + 1), 0, 0),
                ]
            else:
                P[s][RIGHT] = [
                    (0.05, max(0, s - 1), 0, 0),
                    (0.55, s, 0, 0),
                    (0.4, min(nS - 1, s + 1), 0, 0),
                ]
        P[0][RIGHT] = [(0.6, 0, 0, 0), (0.4, 1, 0, 0)]

        # initial state distribution
        isd = np.zeros(nS)
        isd[0] = 1.0

        return P, isd

    def render(self, mode="human"):
        pass

    def close(self):
        pass


sigma_s_a_rs6 = {}

sigma_s_a_rs6[(0, 0, 0)] = [
    ((1, 0), 0, 0),
    ((2, 0), 1, 0),
    ((3, 0), 2, 0),
    ((4, 0), 3, 0),
    ((5, 0), 4, 0),
]
sigma_s_a_rs6[(1, 0, 0)] = [
    ((0, 0), 0, 0.05),
    ((2, 0), 1, 0),
    ((3, 0), 2, 0),
    ((4, 0), 3, 0),
    ((5, 0), 4, 0),
]
sigma_s_a_rs6[(2, 0, 1)] = [
    ((0, 0), 0, 0.05),
    ((1, 0), 0, 0),
    ((3, 0), 2, 0),
    ((4, 0), 3, 0),
    ((5, 0), 4, 0),
]
sigma_s_a_rs6[(3, 0, 2)] = [
    ((0, 0), 0, 0.05),
    ((1, 0), 0, 0),
    ((2, 0), 1, 0),
    ((4, 0), 3, 0),
    ((5, 0), 4, 0),
]
sigma_s_a_rs6[(4, 0, 3)] = [
    ((0, 0), 0, 0.05),
    ((1, 0), 0, 0),
    ((2, 0), 1, 0),
    ((3, 0), 2, 0),
    ((5, 0), 4, 0),
]
sigma_s_a_rs6[(5, 0, 4)] = [
    ((0, 0), 0, 0.05),
    ((1, 0), 0, 0),
    ((2, 0), 1, 0),
    ((3, 0), 2, 0),
    ((4, 0), 3, 0),
]
sigma_s_a_rs6[(0, 1, 0)] = [((5, 1), 5, 1.0)]
sigma_s_a_rs6[(0, 1, 1)] = [((5, 1), 4, 1.0)]
sigma_s_a_rs6[(5, 1, 5)] = [((0, 1), 0, 0)]
sigma_s_a_rs6[(5, 1, 4)] = [((0, 1), 1, 0)]
sigma_s_a_rs6[(1, 1, 0)] = [((2, 1), 1, 0), ((3, 1), 2, 0), ((4, 1), 3, 0)]
sigma_s_a_rs6[(1, 1, 1)] = [((2, 1), 2, 0), ((3, 1), 3, 0), ((4, 1), 4, 0)]
sigma_s_a_rs6[(1, 1, 2)] = [((2, 1), 3, 0), ((3, 1), 4, 0), ((4, 1), 5, 0)]
sigma_s_a_rs6[(2, 1, 1)] = [((1, 1), 0, 0), ((3, 1), 2, 0), ((4, 1), 3, 0)]
sigma_s_a_rs6[(2, 1, 2)] = [((1, 1), 1, 0), ((3, 1), 3, 0), ((4, 1), 4, 0)]
sigma_s_a_rs6[(2, 1, 3)] = [((1, 1), 2, 0), ((3, 1), 4, 0), ((4, 1), 5, 0)]
sigma_s_a_rs6[(3, 1, 2)] = [((1, 1), 0, 0), ((2, 1), 1, 0), ((4, 1), 3, 0)]
sigma_s_a_rs6[(3, 1, 3)] = [((1, 1), 1, 0), ((2, 1), 2, 0), ((4, 1), 4, 0)]
sigma_s_a_rs6[(3, 1, 4)] = [((1, 1), 2, 0), ((2, 1), 3, 0), ((4, 1), 5, 0)]
sigma_s_a_rs6[(4, 1, 3)] = [((1, 1), 0, 0), ((2, 1), 1, 0), ((3, 1), 2, 0)]
sigma_s_a_rs6[(4, 1, 4)] = [((1, 1), 1, 0), ((2, 1), 2, 0), ((3, 1), 3, 0)]
sigma_s_a_rs6[(4, 1, 5)] = [((1, 1), 2, 0), ((2, 1), 3, 0), ((3, 1), 4, 0)]

# define the equivalent classes for (s,a) for riverswim
C_1_rs6 = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
C_2_rs6 = [(0, 1), (5, 1)]
C_3_rs6 = [(1, 1), (2, 1), (3, 1), (4, 1)]
C_s_a_rs6 = [C_1_rs6, C_2_rs6, C_3_rs6]

