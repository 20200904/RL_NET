"""Same seed, different firewall: NASim 0.12.0 scenario generation depends on PYTHONHASHSEED."""
import os
import subprocess
import sys

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
