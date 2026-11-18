# Draft issue for NASim (not filed)

**Title:** Generated scenarios are not reproducible across processes: firewall depends on `PYTHONHASHSEED`

**Body:**

### Summary

`ScenarioGenerator.generate(..., seed=s)` can return a different firewall for
the same `seed` in different Python processes. Hosts, exploits and privilege
escalations come out the same, but the firewall can differ. String hashing is
randomised per process by default (`PYTHONHASHSEED`), and the firewall code
draws from a list built from a `set` of strings, so the draw follows that
set's iteration order.

### Version

- nasim 0.12.0 (from PyPI)
- Python 3.11.17, numpy 2.4.6, Windows 10. The cause is not OS-specific.

### Where

`nasim/scenarios/generator.py`, `ScenarioGenerator._generate_firewall`:

```python
dest_avail = subnet_services[dest].copy()          # a set of service names
...
dest_allowed = np.random.choice(list(dest_avail))  # order of list(set) varies per process
...
    dest_allowed = np.random.choice(list(dest_avail))
```

`subnet_services[...]` is a `set` of strings. `list(dest_avail)` has an order
that depends on the string hash seed, so `np.random.choice` picks different
services from the same RNG state.

This only matters when a zone has at least `restrictiveness` vulnerable
services. With the default `restrictiveness=5`, small scenarios often have
fewer than that, so every service is allowed and nothing random happens.

### Reproduce

```python
import os, subprocess, sys
import nasim

if len(sys.argv) > 1:  # child process: generate once and print the firewall
    s = nasim.scenarios.generate_scenario(8, 4, restrictiveness=1, seed=0)
    print(sorted((k, sorted(map(str, v))) for k, v in s.firewall.items()))
    sys.exit()

outputs = {
    h: subprocess.run([sys.executable, __file__, "child"], capture_output=True, text=True,
                      env=dict(os.environ, PYTHONHASHSEED=h)).stdout
    for h in ("1", "2", "3", "4")
}
print("nasim", nasim.__version__, "| distinct firewalls for seed=0:", len(set(outputs.values())))
```

Expected: `distinct firewalls for seed=0: 1`. Observed with nasim 0.12.0:
`distinct firewalls for seed=0: 2`.

### Impact

Two processes given the same seed can train and evaluate on different
networks without any error, for example parallel workers, or a later
re-evaluation of a saved model. Within one process, results are consistent,
so the problem is easy to miss.

### Suggested fix

Sort before drawing, e.g. `np.random.choice(sorted(dest_avail))` in both
places. This keeps the RNG call sequence the same, so the set of possible
firewalls is unchanged; only the mapping from seed to firewall becomes
stable. Setting `PYTHONHASHSEED` is a workaround for users.

I did not find other set- or hash-order dependence on the generation path
(`generate` → `_generate_*` → `Scenario`), nor in `NASimEnv.step`.
