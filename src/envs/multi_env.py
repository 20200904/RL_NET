"""One environment that plays a different network each episode."""
import gymnasium as gym
import numpy as np

from src.envs.generate import make_env, verified_scenario


class MultiNetEnv(gym.Env):
    """Samples a network uniformly from ``specs`` at every reset.

    ``reset(options={"network": i})`` selects network ``i`` instead (used by
    evaluation). All networks share one observation and action space; this is
    checked when the splits are generated.

    All networks in one process must have the same shape: NASim keeps the
    host-vector layout in class attributes of ``HostVector``.
    """

    def __init__(self, specs, step_limit, seed=0):
        self.specs = list(specs)
        self.step_limit = step_limit
        self._rng = np.random.default_rng(seed)
        # scenario generation reseeds numpy's global RNG, so do it all up front
        self._scenarios = [verified_scenario(s) for s in self.specs]
        self._envs = [None] * len(self.specs)
        self.network = 0
        self._env = self._get(0)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def _get(self, idx):
        if self._envs[idx] is None:
            self._envs[idx] = make_env(
                self.specs[idx], self.step_limit, self._scenarios[idx]
            )
        return self._envs[idx]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options and "network" in options:
            self.network = int(options["network"])
        else:
            self.network = int(self._rng.integers(len(self.specs)))
        self._env = self._get(self.network)
        obs, info = self._env.reset()
        info["network"] = self.network
        return obs, info

    def step(self, action):
        return self._env.step(action)
