"""Hand-coded attacker that acts only on what the agent observes.

Used to check that the task is solvable from observations at all; it is not
one of the compared methods. It reads the accumulated observation produced by
``ObsMemory`` and knows only what any agent could know in advance: the
observation layout, the action list, and the fixed exploit / privesc table.
It never reads the hidden network state.

Modes:
  blind      no OS / service scans; try exploits in table order
  scan       OS scan + service scan, then a matching exploit, root-giving
             exploits first (the reward-greedy order)
  scan_safe  like ``scan``, but never attacks a host flagged out of scope and
             never uses a risky exploit on a host whose scanned OS and
             services match the fragile rule
"""
from src.envs.cost_wrapper import NUM_AUX, NUM_HOST_FLAGS
from src.envs.generate import EXPLOITS, FRAGILE_RULES, PRIVESCS, RISKY_EXPLOITS

ROOT = 2
MODES = ("blind", "scan", "scan_safe")


class RuleAgent:
    def __init__(self, env, mode="scan"):
        """``env`` is any network's environment (ObsMemory on top); only its
        observation layout and action list are used, which are the same in
        every network."""
        assert mode in MODES
        self.mode = mode
        scenario = env.unwrapped.scenario
        self.num_hosts = env.num_hosts
        self.size = env.host_vec_size
        base = env.os_slice.start - NUM_HOST_FLAGS
        self.i_compromised, self.i_reachable, self.i_discovered = base, base + 1, base + 2
        self.i_value, self.i_access = base + 3, base + 5
        self.os_names = list(scenario.os)
        self.srv_names = list(scenario.services)
        self.proc_names = list(scenario.processes)
        self.os_slice, self.srv_slice = env.os_slice, env.srv_slice
        self.proc_slice = slice(env.srv_slice.stop, self.size)
        self.rows = sorted(env.host_row.items(), key=lambda kv: kv[1])

        self.index = {}
        for idx, a in enumerate(env.unwrapped.action_space.actions):
            kind = ("exploit" if a.is_exploit() else "privesc"
                    if a.is_privilege_escalation() else a.name)
            self.index[(kind, tuple(a.target), a.name)] = idx
        # root-giving exploits first in the greedy order, table order otherwise
        self.exploit_order = sorted(EXPLOITS, key=lambda n: -EXPLOITS[n][2])
        if mode == "blind":
            self.exploit_order = list(EXPLOITS)
        self.reset()

    def reset(self):
        self.tried = set()           # (host, exploit) attempts that failed
        self.subnet_scanned = set()
        self.privesc_tried = set()
        self.compromised_count = 0
        self.last = None             # (host, exploit) of the previous step

    # ---- reading the observation ----
    def _hosts(self, obs):
        hosts = []
        for addr, row in self.rows:
            v = obs[row * self.size:(row + 1) * self.size]
            os_vals, srv_vals, proc_vals = v[self.os_slice], v[self.srv_slice], v[self.proc_slice]
            hosts.append(dict(
                addr=addr,
                compromised=v[self.i_compromised] > 0,
                discovered=v[self.i_discovered] > 0,
                value=float(v[self.i_value]),
                access=int(v[self.i_access]),
                os=(self.os_names[int(os_vals.argmax())] if os_vals.any() else None),
                services=({s for s, x in zip(self.srv_names, srv_vals) if x}
                          if srv_vals.any() else None),
                processes=({p for p, x in zip(self.proc_names, proc_vals) if x}
                           if proc_vals.any() else None),
                out_of_scope=obs[self.num_hosts * self.size + NUM_AUX + row] > 0,
            ))
        return hosts

    def _fragile(self, host):
        return any(host["os"] == os and srv in host["services"]
                   for os, srv in FRAGILE_RULES)

    # ---- policy ----
    def act(self, obs, deterministic=True):
        hosts = self._hosts(obs)
        last_ok = obs[self.num_hosts * self.size] > 0
        if self.last is not None and not last_ok:
            self.tried.add(self.last)
        self.last = None

        compromised = sum(h["compromised"] for h in hosts)
        if compromised > self.compromised_count:
            # a new foothold can open firewall paths: retry earlier failures
            self.compromised_count = compromised
            self.tried.clear()

        # 1. root on hosts that turned out to be valuable (goal hosts)
        for h in hosts:
            if not h["compromised"] or h["access"] >= ROOT or h["value"] <= 1:
                continue
            if self.mode == "scan_safe" and h["out_of_scope"]:
                continue
            if h["processes"] is None:
                return self.index[("process_scan", h["addr"], "process_scan")]
            for name, (proc, os) in PRIVESCS.items():
                key = (h["addr"], name)
                if (proc in h["processes"] and (os is None or os == h["os"])
                        and key not in self.privesc_tried):
                    self.privesc_tried.add(key)
                    return self.index[("privesc", h["addr"], name)]

        # 2. discover neighbours from every foothold once
        for h in hosts:
            if h["compromised"] and h["addr"] not in self.subnet_scanned:
                self.subnet_scanned.add(h["addr"])
                return self.index[("subnet_scan", h["addr"], "subnet_scan")]

        # 3. attack discovered hosts
        for h in hosts:
            if not h["discovered"] or h["compromised"]:
                continue
            if self.mode == "scan_safe" and h["out_of_scope"]:
                continue
            if self.mode != "blind":
                if h["os"] is None:
                    return self.index[("os_scan", h["addr"], "os_scan")]
                if h["services"] is None:
                    return self.index[("service_scan", h["addr"], "service_scan")]
            for name in self.exploit_order:
                srv, os, _ = EXPLOITS[name]
                if (h["addr"], name) in self.tried:
                    continue
                if self.mode != "blind" and not (
                        srv in h["services"] and (os is None or os == h["os"])):
                    continue
                if (self.mode == "scan_safe" and name in RISKY_EXPLOITS
                        and self._fragile(h)):
                    continue
                self.last = (h["addr"], name)
                return self.index[("exploit", h["addr"], name)]

        # nothing left to try: a harmless scan of the first discovered host
        h = next(h for h in hosts if h["discovered"])
        return self.index[("os_scan", h["addr"], "os_scan")]
