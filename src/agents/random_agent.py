"""M0: uniform random policy."""
import numpy as np


class RandomAgent:
    def __init__(self, num_actions, seed=0):
        self.num_actions = num_actions
        self._rng = np.random.default_rng(seed)

    def act(self, obs, deterministic=False):
        return int(self._rng.integers(self.num_actions))
