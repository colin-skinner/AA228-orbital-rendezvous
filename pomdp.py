import numpy as np

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
    def __init__(self, problem, depth=10, num_sims=100, c=1.0, rollout=None, rollout_depth=10):
        self.P = problem
        self.N = {}  # visit counts
        self.Q = {}  # action-value estimates
        self.d = depth
        self.m = num_sims
        self.c = c
        self.rollout = rollout if rollout else greedy_rollout
        self.rollout_depth = rollout_depth

    def _key(self, a):
        """Return a hashable representation of action `a` for use as dict keys."""
        import numpy as _np
        if isinstance(a, _np.ndarray):
            return tuple(a.tolist())
        try:
            return a.item()
        except Exception:
            return a

    def _hkey(self, h):
        """Normalize a history tuple `h` (sequence of (a,o) pairs) into a fully hashable form."""
        if not h:
            return ()
        return tuple((self._key(a), self._key(o)) for (a, o) in h)

    def explore(self, h):
        hkey = self._hkey(h)
        Nh = sum(self.N.get((hkey, self._key(a)), 0) for a in self.P.A)
        return max(self.P.A, key=lambda a: self.Q.get((hkey, self._key(a)), 0.0) + self.c * bonus(self.N.get((hkey, self._key(a)), 0), Nh))

    def simulate(self, s, h, d):
        if d <= 0:
            return self.rollout(self.P, s, self.rollout_depth)

        # initialize counts/Q for this history if not present
        # actions in self.P.A may be plain ints or numpy scalars; treat them uniformly
        hkey = self._hkey(h)
        if (hkey, self._key(self.P.A[0])) not in self.N:
            for a in self.P.A:
                ak = self._key(a)
                self.N[(hkey, ak)] = 0
                self.Q[(hkey, ak)] = 0.0
            return self.rollout(self.P, s, self.rollout_depth)

        a = self.explore(h)
        s_next, r, o = self.P.TRO(s, a)
        q = r + self.P.gamma * self.simulate(s_next, h + ((a, o),), d - 1)

        ak = self._key(a)
        self.N[(hkey, ak)] += 1
        self.Q[(hkey, ak)] += (q - self.Q[(hkey, ak)]) / self.N[(hkey, ak)]
        return q

    def __call__(self, b, h=()):
        for _ in range(self.m):
            s = sample_state(self.P.S, b)
            self.simulate(s, h, self.d)
        hkey = self._hkey(h)
        return max(self.P.A, key=lambda a: self.Q.get((hkey, self._key(a)), 0.0))


def bonus(n, N):
    return np.sqrt(np.log(N + 1) / (n + 1))


def sample_state(states, belief):
    """Sample from EKF belief (Gaussian). belief is an ekf.State with mean x and covariance P."""
    return np.random.multivariate_normal(belief.x, belief.P)


def greedy_rollout(problem, s, depth):
    """Greedy rollout: at each step, pick action that maximizes immediate reward."""
    total = 0.0
    for i in range(depth):
        # Evaluate each action and pick the one with best immediate reward
        best_a, best_r, best_s = None, -np.inf, None
        for a in problem.A:
            s_next, r, _ = problem.TRO(s, a)
            if r > best_r:
                best_a, best_r, best_s = a, r, s_next
        total += (problem.gamma ** i) * best_r
        s = best_s
    return total
