"""Small compatibility helpers around NASim (NASim itself is never modified)."""
import gymnasium as gym
import nasim

# Generator settings used for the Phase 0 measurements. With a fixed host
# count the NASim generator always produces the same subnets and topology;
# only host configs, exploits, privescs and firewall change with the seed.
GEN_DEFAULTS = dict(
    num_services=3,
    num_os=2,
    num_processes=2,
    exploit_probs=1.0,
    privesc_probs=1.0,
)


class IntAction(gym.Wrapper):
    """Cast actions to a Python int before they reach NASim.

    NASim 0.12.0 asserts ``isinstance(action, int)`` for flat action spaces
    (nasim/envs/action.py, FlatActionSpace.get_action), which rejects the
    ``numpy.int64`` values produced by gymnasium spaces, SB3 and torch.
    """

    def step(self, action):
        return self.env.step(int(action))


def make_benchmark(name, fully_obs=False, seed=None):
    env = nasim.make_benchmark(
        name, seed=seed, fully_obs=fully_obs, flat_actions=True, flat_obs=True
    )
    return IntAction(env)


def make_generated(num_hosts, seed, fully_obs=False, step_limit=400, **overrides):
    params = dict(GEN_DEFAULTS)
    params.update(overrides)
    num_services = params.pop("num_services")
    env = nasim.generate(
        num_hosts,
        num_services,
        fully_obs=fully_obs,
        flat_actions=True,
        flat_obs=True,
        seed=seed,
        step_limit=step_limit,
        **params,
    )
    return IntAction(env)
