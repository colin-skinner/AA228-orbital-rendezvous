import numpy as np
from HCW import RolloutParams

class POMDP:
    def __init__(self, gamma, states, actions, observations, transition, reward, observation, tro):
        self.gamma = gamma
        self.S = states
        self.A = actions
        self.O = observations
        self.T = transition
        self.R = reward
        self.O_fn = observation
        self.TRO = tro


class MCTS:
    def __init__(self, problem: POMDP, depth=10, num_sims=100, c=1.0, rollout=None, rollout_depth=10,
                 rollout_params: RolloutParams = None):
        self.P = problem
        self.N = {}  # visit counts
        self.Q = {}  # action-value estimates
        self.d = depth
        self.m = num_sims
        self.c = c
        self.rollout = rollout if rollout else greedy_rollout
        self.rollout_depth = rollout_depth
        self.rollout_params: RolloutParams = rollout_params

    def explore(self, h):
        Nh = sum(self.N.get((h, a), 0) for a in self.P.A)
        return max(self.P.A, key=lambda a: self.Q.get((h, a), 0.0) + self.c * bonus(self.N.get((h, a), 0), Nh))

    def simulate(self, s, h, d):
        """Added extra params"""
        if d <= 0:
            return self.rollout(self.P, s, self.rollout_depth, self.params)

        if (h, self.P.A[0]) not in self.N:
            for a in self.P.A:
                self.N[(h, a)] = 0
                self.Q[(h, a)] = 0.0
            return self.rollout(self.P, s, self.rollout_depth, self.params)

        a = self.explore(h)
        s_next, r, o = self.P.TRO(s, a, self.params)
        q = r + self.P.gamma * self.simulate(s_next, h + ((a, o),), d - 1)

        self.N[(h, a)] += 1
        self.Q[(h, a)] += (q - self.Q[(h, a)]) / self.N[(h, a)]
        return q

    def __call__(self, b, h=()):
        for _ in range(self.m):
            s = sample_state(self.P.S, b)
            self.simulate(s, h, self.d)
        return max(self.P.A, key=lambda a: self.Q.get((h, a), 0.0))


def bonus(n, N):
    return np.sqrt(np.log(N + 1) / (n + 1))


def sample_state(states, belief):
    """Sample from EKF belief (Gaussian). belief is an ekf.State with mean x and covariance P."""
    return np.random.multivariate_normal(belief.x, belief.P)


def greedy_rollout(problem: POMDP, s, depth, params:RolloutParams = None):
    """Greedy rollout: at each step, pick action that maximizes immediate reward.
    Added extra params"""
    total = 0.0
    for i in range(depth):
        # Evaluate each action and pick the one with best immediate reward
        best_a, best_r, best_s = None, -np.inf, None
        for a in problem.A:
            s_next, r, _ = problem.TRO(s, a, params)
            if r > best_r:
                best_a, best_r, best_s = a, r, s_next
        total += (problem.gamma ** i) * best_r
        s = best_s
    return total
