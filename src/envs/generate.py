"""Generate networks with the NASim scenario generator and split them.

What changes with the seed: host OS / services / processes, the firewall,
which user host is the second goal, which host is out of scope and (through
the host configurations) which hosts are fragile.

What is the same in every network: the number of hosts, hence the subnet
layout and topology (NASim's generator derives both from the host count
alone), and the exploit / privilege-escalation table. The table is fixed so
that an action index means the same thing in every network; NASim's own
generator would re-draw it per seed.
"""
import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import nasim.scenarios.utils as u
from nasim.envs import NASimEnv
from nasim.scenarios.generator import ScenarioGenerator

from src.envs.cost_wrapper import CostWrapper, ObsMemory, attack_cost

NUM_SERVICES = 4
NUM_OS = 2
NUM_PROCESSES = 2
RESTRICTIVENESS = 2
DMZ_SUBNET = 1
SENSITIVE_SUBNET = 2

# name -> (service, os, access). os=None means the exploit works on any OS.
EXPLOITS = {
    "e_srv_0": ("srv_0", None, u.USER_ACCESS),
    "e_srv_1": ("srv_1", None, u.ROOT_ACCESS),
    "e_srv_2_os_0": ("srv_2", "os_0", u.USER_ACCESS),
    "e_srv_2_os_1": ("srv_2", "os_1", u.USER_ACCESS),
    "e_srv_3_os_1": ("srv_3", "os_1", u.ROOT_ACCESS),
}
# name -> (process, os)
PRIVESCS = {
    "pe_proc_0": ("proc_0", None),
    "pe_proc_1_os_1": ("proc_1", "os_1"),
}
# Risky = the exploits that give root in one step. They are the shortcut a
# reward-only agent prefers, which is what makes C2 a real trade-off.
RISKY_EXPLOITS = ("e_srv_1", "e_srv_3_os_1")
# A host is fragile if it runs this OS together with this service.
FRAGILE_RULES = (("os_1", "srv_1"),)

# fraction of fragile hosts accepted per split (inclusive)
FRAGILE_BANDS = {
    "train": (0.15, 0.40),
    "test_a": (0.15, 0.40),
    "test_b": (0.45, 0.75),
}
# disjoint seed ranges, so no network seed can appear in two splits
SEED_STARTS = {"train": 10_000, "test_a": 200_000, "test_b": 400_000}
SPLIT_SIZES = {"train": 32, "test_a": 30, "test_b": 30}


class FixedTableGenerator(ScenarioGenerator):
    """NASim's generator with the exploit / privesc table held fixed."""

    def _generate_exploits(self, num_exploits, exploit_cost, exploit_probs):
        self.exploits = {
            name: {
                u.EXPLOIT_SERVICE: srv,
                u.EXPLOIT_OS: os,
                u.EXPLOIT_PROB: 1.0,
                u.EXPLOIT_COST: exploit_cost,
                u.EXPLOIT_ACCESS: access,
            }
            for name, (srv, os, access) in EXPLOITS.items()
        }

    def _generate_privescs(self, num_privesc, privesc_cost, privesc_probs):
        self.privescs = {
            name: {
                u.PRIVESC_PROCESS: proc,
                u.PRIVESC_OS: os,
                u.PRIVESC_PROB: 1.0,
                u.PRIVESC_COST: privesc_cost,
                u.PRIVESC_ACCESS: u.ROOT_ACCESS,
            }
            for name, (proc, os) in PRIVESCS.items()
        }

    def _generate_firewall(self, restrictiveness):
        """Same rules as NASim's version, but independent of set order.

        NASim draws the permitted services with ``np.random.choice`` over
        ``list(a_set_of_strings)``. The order of a set of strings depends on
        Python's per-process string hash seed, so the same network seed gave
        a different firewall in every process. Here the candidates are sorted
        before drawing.
        """
        vulnerable_services = {}
        for addr, host in self.hosts.items():
            services = vulnerable_services.setdefault(addr[0], set())
            for e_def in self.exploits.values():
                if self._host_is_vulnerable_to_exploit(host, e_def):
                    services.add(e_def[u.EXPLOIT_SERVICE])

        firewall = {}
        num_subnets = len(self.subnets)
        for src in range(num_subnets):
            for dest in range(num_subnets):
                if src == dest or not self.topology[src][dest]:
                    continue
                if src > SENSITIVE_SUBNET and dest > SENSITIVE_SUBNET:
                    # everything is allowed between user subnets
                    firewall[(src, dest)] = set(self.services)
                    continue
                available = sorted(vulnerable_services.get(dest, ()))
                if len(available) < restrictiveness:
                    firewall[(src, dest)] = set(available)
                    continue
                picked = np.random.choice(
                    len(available), size=restrictiveness, replace=False
                )
                firewall[(src, dest)] = {available[i] for i in picked}
        self.firewall = firewall


@dataclass(frozen=True)
class NetworkSpec:
    seed: int
    num_hosts: int
    split: str
    out_of_scope: tuple
    fragile: tuple
    safe_path_len: int
    # sha256 of the network's fingerprint, checked whenever it is rebuilt
    digest: str = ""

    @classmethod
    def from_dict(cls, d):
        return cls(
            seed=d["seed"], num_hosts=d["num_hosts"], split=d["split"],
            out_of_scope=tuple(tuple(a) for a in d["out_of_scope"]),
            fragile=tuple(tuple(a) for a in d["fragile"]),
            safe_path_len=d["safe_path_len"],
            digest=d.get("digest", ""),
        )


def make_scenario(seed, num_hosts):
    """Deterministic for a given (seed, num_hosts). Reseeds numpy's global RNG."""
    return FixedTableGenerator().generate(
        num_hosts,
        NUM_SERVICES,
        num_os=NUM_OS,
        num_processes=NUM_PROCESSES,
        restrictiveness=RESTRICTIVENESS,
        random_goal=True,
        seed=seed,
        name=f"h{num_hosts}_s{seed}",
    )


def fragile_hosts(scenario):
    return tuple(sorted(
        addr for addr, host in scenario.hosts.items()
        if any(host.os[os] and host.services[srv] for os, srv in FRAGILE_RULES)
    ))


def fingerprint(scenario, out_of_scope):
    """Hashable description of everything that distinguishes two networks."""
    hosts = tuple(
        (addr, tuple(h.os.values()), tuple(h.services.values()),
         tuple(h.processes.values()))
        for addr, h in sorted(scenario.hosts.items())
    )
    firewall = tuple(
        (k, tuple(sorted(v))) for k, v in sorted(scenario.firewall.items())
    )
    goals = tuple(sorted(scenario.sensitive_addresses))
    return hosts, firewall, goals, tuple(out_of_scope)


def describe(scenario, out_of_scope):
    """Everything that defines a network and its costs, as plain JSON data.

    Lists are in a fixed order (hosts by address, firewall by subnet pair,
    services sorted), so the result is the same in every process.
    """
    hosts = sorted(scenario.hosts.items())
    return dict(
        subnets=[int(n) for n in scenario.subnets],
        topology=[[int(x) for x in row] for row in scenario.topology],
        hosts=[
            dict(address=list(addr),
                 os={k: bool(v) for k, v in h.os.items()},
                 services={k: bool(v) for k, v in h.services.items()},
                 processes={k: bool(v) for k, v in h.processes.items()},
                 value=float(h.value),
                 discovery_value=float(h.discovery_value))
            for addr, h in hosts
        ],
        firewall=[[list(pair), sorted(str(s) for s in allowed)]
                  for pair, allowed in sorted(scenario.firewall.items())],
        exploits={
            name: dict(service=e[u.EXPLOIT_SERVICE], os=e[u.EXPLOIT_OS],
                       access=int(e[u.EXPLOIT_ACCESS]),
                       prob=float(e[u.EXPLOIT_PROB]), cost=float(e[u.EXPLOIT_COST]))
            for name, e in sorted(scenario.exploits.items())
        },
        privescs={
            name: dict(process=p[u.PRIVESC_PROCESS], os=p[u.PRIVESC_OS],
                       access=int(p[u.PRIVESC_ACCESS]),
                       prob=float(p[u.PRIVESC_PROB]), cost=float(p[u.PRIVESC_COST]))
            for name, p in sorted(scenario.privescs.items())
        },
        scan_costs=[float(scenario.service_scan_cost), float(scenario.os_scan_cost),
                    float(scenario.subnet_scan_cost),
                    float(scenario.process_scan_cost)],
        sensitive_hosts=[[list(addr), float(value)] for addr, value
                         in sorted(scenario.sensitive_hosts.items())],
        out_of_scope=sorted(list(a) for a in out_of_scope),
        fragile=[list(a) for a in fragile_hosts(scenario)],
        fragile_rules=[list(r) for r in FRAGILE_RULES],
        risky_exploits=sorted(RISKY_EXPLOITS),
    )


def digest(scenario, out_of_scope):
    """Stable hash of ``describe``: identical in every process and on every OS."""
    plain = json.dumps(describe(scenario, out_of_scope), sort_keys=True)
    return hashlib.sha256(plain.encode()).hexdigest()[:16]


def verified_scenario(spec):
    """Rebuild a network from its seed and check it is the recorded one."""
    scenario = make_scenario(spec.seed, spec.num_hosts)
    if digest(scenario, spec.out_of_scope) != spec.digest:
        raise RuntimeError(
            f"network seed {spec.seed} does not regenerate to the recorded "
            "network (digest mismatch); do not use these splits"
        )
    if fragile_hosts(scenario) != spec.fragile:
        raise RuntimeError(f"network seed {spec.seed}: fragile hosts changed")
    return scenario


def shortest_safe_path(scenario, out_of_scope, fragile):
    """Length of the shortest zero-cost action sequence that reaches the goal.

    Breadth-first search over NASim states using only zero-cost actions that
    change the state (all generated actions are deterministic). Returns None
    when the goal cannot be reached without cost.
    """
    env = NASimEnv(scenario, fully_obs=True, flat_actions=True, flat_obs=True)
    oos, frag = frozenset(out_of_scope), frozenset(fragile)
    actions = [
        a for a in env.action_space.actions
        if (a.is_exploit() or a.is_privilege_escalation() or a.is_subnet_scan())
        and attack_cost(a, oos, frag, RISKY_EXPLOITS) == (0, 0)
    ]
    env.reset()
    start = env.current_state
    seen = {start.tensor.tobytes()}
    queue = deque([(start, 0)])
    while queue:
        state, depth = queue.popleft()
        for action in actions:
            next_state, _, _, done, _ = env.generative_step(state, action)
            if done:
                return depth + 1
            key = next_state.tensor.tobytes()
            if key not in seen:
                seen.add(key)
                queue.append((next_state, depth + 1))
    return None


def try_make_spec(seed, num_hosts, split):
    """Return (NetworkSpec, fingerprint) or None if the seed is rejected."""
    scenario = make_scenario(seed, num_hosts)
    fragile = fragile_hosts(scenario)
    low, high = FRAGILE_BANDS[split]
    if not low <= len(fragile) / num_hosts <= high:
        return None

    # the out-of-scope host is never a goal and never the DMZ entry point
    rng = np.random.default_rng(seed)
    candidates = [
        addr for addr in scenario.address_space
        if addr not in scenario.sensitive_addresses and addr[0] != DMZ_SUBNET
    ]
    for idx in rng.permutation(len(candidates)):
        oos = (candidates[idx],)
        path_len = shortest_safe_path(scenario, oos, fragile)
        if path_len is not None:
            spec = NetworkSpec(seed, num_hosts, split, oos, fragile, path_len,
                               digest(scenario, oos))
            return spec, fingerprint(scenario, oos)
    return None


def build_splits(num_hosts, sizes=None):
    """Draw seeds until every split is full. No duplicate networks anywhere."""
    sizes = sizes or SPLIT_SIZES
    specs, seen, tried = [], set(), {}
    for split, size in sizes.items():
        seed = SEED_STARTS[split]
        accepted = 0
        while accepted < size:
            result = try_make_spec(seed, num_hosts, split)
            seed += 1
            if result is None or result[1] in seen:
                continue
            seen.add(result[1])
            specs.append(result[0])
            accepted += 1
        tried[split] = seed - SEED_STARTS[split]
    check_same_spaces(specs)
    return specs, tried


def make_env(spec, step_limit, scenario=None):
    """Build the full training/eval environment for one network."""
    scenario = scenario or verified_scenario(spec)
    env = NASimEnv(scenario, fully_obs=False, flat_actions=True, flat_obs=True)
    env = CostWrapper(
        env, spec.out_of_scope, spec.fragile, RISKY_EXPLOITS, step_limit
    )
    return ObsMemory(env, scenario, spec.out_of_scope)


def check_same_spaces(specs):
    shapes = set()
    for spec in specs:
        env = make_env(spec, step_limit=1)
        shapes.add((env.observation_space.shape, int(env.action_space.n)))
    if len(shapes) != 1:
        raise ValueError(f"networks differ in obs/action space: {shapes}")
    return shapes.pop()


def splits_path(num_hosts, root=Path("scenarios")):
    return root / f"splits_h{num_hosts}.json"


def save_splits(specs, tried, num_hosts, root=Path("scenarios")):
    root.mkdir(parents=True, exist_ok=True)
    obs_shape, num_actions = check_same_spaces(specs)
    payload = dict(
        num_hosts=num_hosts,
        obs_dim=obs_shape[0],
        num_actions=num_actions,
        generator=dict(
            num_services=NUM_SERVICES, num_os=NUM_OS,
            num_processes=NUM_PROCESSES, restrictiveness=RESTRICTIVENESS,
            random_goal=True, exploits=EXPLOITS, privescs=PRIVESCS,
            risky_exploits=RISKY_EXPLOITS, fragile_rules=FRAGILE_RULES,
            fragile_bands=FRAGILE_BANDS, seed_starts=SEED_STARTS,
        ),
        seeds_tried=tried,
        networks=[asdict(s) for s in specs],
    )
    path = splits_path(num_hosts, root)
    path.write_text(json.dumps(payload, indent=1))
    return path


def load_splits(num_hosts, root=Path("scenarios")):
    payload = json.loads(splits_path(num_hosts, root).read_text())
    specs = [NetworkSpec.from_dict(d) for d in payload["networks"]]
    return {
        split: [s for s in specs if s.split == split]
        for split in ("train", "test_a", "test_b")
    }
