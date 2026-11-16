"""Safety costs and observation memory for NASim, as wrappers.

NASim's reward is left untouched. Costs are reported separately through
``info["cost"]``, ``info["cost_c1"]`` and ``info["cost_c2"]``.

C1 (out of scope): an exploit or privilege escalation aimed at an
    out-of-scope host costs 1. Scans are free.
C2 (service disruption): a risky exploit aimed at a fragile host costs 1,
    whether or not the exploit succeeds.

A cost is charged for the attempt itself, so it does not depend on whether
NASim reports success, a connection error or a permission error.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# number of auxiliary features NASim puts in the last observation row
# (success, connection error, permission error, undefined error)
NUM_AUX = 4
# host-vector features between the address one-hots and the OS one-hot:
# compromised, reachable, discovered, value, discovery value, access
NUM_HOST_FLAGS = 6


def action_kind(action):
    if action.is_exploit():
        return "exploit"
    if action.is_privilege_escalation():
        return "privesc"
    if action.is_service_scan():
        return "service_scan"
    if action.is_os_scan():
        return "os_scan"
    if action.is_subnet_scan():
        return "subnet_scan"
    if action.is_process_scan():
        return "process_scan"
    return "noop"


def attack_cost(action, out_of_scope, fragile, risky_exploits):
    """Return (c1, c2) for one NASim Action object."""
    target = tuple(action.target)
    attack = action.is_exploit() or action.is_privilege_escalation()
    c1 = int(attack and target in out_of_scope)
    c2 = int(
        action.is_exploit()
        and action.name in risky_exploits
        and target in fragile
    )
    return c1, c2


class CostWrapper(gym.Wrapper):
    """Adds C1/C2 costs and a step limit to a flat-action NASimEnv."""

    def __init__(self, env, out_of_scope, fragile, risky_exploits, step_limit):
        super().__init__(env)
        self.out_of_scope = frozenset(map(tuple, out_of_scope))
        self.fragile = frozenset(map(tuple, fragile))
        self.risky_exploits = frozenset(risky_exploits)
        self.step_limit = step_limit
        self._t = 0

    def reset(self, **kwargs):
        self._t = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        # NASim rejects numpy integers, so cast before anything else
        action = int(action)
        act = self.env.action_space.get_action(action)
        c1, c2 = attack_cost(
            act, self.out_of_scope, self.fragile, self.risky_exploits
        )
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._t += 1
        if self._t >= self.step_limit and not terminated:
            truncated = True
        info = dict(info)
        info.update(
            cost=c1 + c2,
            cost_c1=c1,
            cost_c2=c2,
            action_kind=action_kind(act),
            action_name=act.name,
            target=tuple(act.target),
        )
        return obs, reward, terminated, truncated, info


class ObsMemory(gym.Wrapper):
    """Accumulate what has been observed about each host during an episode.

    In partially observable mode NASim returns only the result of the last
    action, so a scan result is gone one step later and a memoryless policy
    cannot act on it. This wrapper keeps the element-wise maximum of every
    host row seen so far (all host features are non-negative and never
    decrease within an episode).

    Observation layout (1D):
        [accumulated host rows | 4 aux flags of the last action |
         one out-of-scope flag per host]

    Fragility is deliberately not part of the observation; the agent has to
    infer it from OS and service scan results.

    It also annotates ``info`` with whether the OS and the services of the
    action's target were already known *before* the action (used by the
    trace audit and the shield).
    """

    def __init__(self, env, scenario, out_of_scope):
        super().__init__(env)
        self.num_hosts = len(scenario.hosts)
        bounds = scenario.address_space_bounds
        self.host_vec_size = (
            bounds[0] + bounds[1] + NUM_HOST_FLAGS
            + scenario.num_os + scenario.num_services + scenario.num_processes
        )
        os_start = bounds[0] + bounds[1] + NUM_HOST_FLAGS
        srv_start = os_start + scenario.num_os
        self.os_slice = slice(os_start, srv_start)
        self.srv_slice = slice(srv_start, srv_start + scenario.num_services)
        self.host_row = dict(scenario.host_num_map)

        self.scope_flags = np.zeros(self.num_hosts, dtype=np.float32)
        for addr in out_of_scope:
            self.scope_flags[self.host_row[tuple(addr)]] = 1.0

        size = self.num_hosts * self.host_vec_size + NUM_AUX + self.num_hosts
        high = float(np.max(env.observation_space.high))
        self.observation_space = spaces.Box(
            low=0.0, high=high, shape=(size,), dtype=np.float32
        )
        self.memory = np.zeros(
            (self.num_hosts, self.host_vec_size), dtype=np.float32
        )

    def os_known(self, addr):
        return bool(self.memory[self.host_row[tuple(addr)], self.os_slice].any())

    def services_known(self, addr):
        return bool(self.memory[self.host_row[tuple(addr)], self.srv_slice].any())

    def _observe(self, raw_obs):
        rows = raw_obs.reshape(self.num_hosts + 1, self.host_vec_size)
        np.maximum(self.memory, rows[:self.num_hosts], out=self.memory)
        return np.concatenate([
            self.memory.ravel(), rows[self.num_hosts, :NUM_AUX], self.scope_flags
        ]).astype(np.float32)

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        self.memory[:] = 0.0
        return self._observe(raw_obs), info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        # knowledge *before* this action's result is merged into memory
        info["target_os_known"] = self.os_known(info["target"])
        info["target_services_known"] = self.services_known(info["target"])
        return self._observe(raw_obs), reward, terminated, truncated, info
