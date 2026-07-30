#!/usr/bin/env python3
"""Flag in-flight folds whose elapsed time is out of family for their target size.

Three folds have stalled the same way (22ps protenix, 9i5n boltz2, 9iar boltz2): one
dispatch thread spinning at 100% CPU, no output written, deaf to SIGINT and SIGTERM. Wall
time alone cannot distinguish that from a big target -- a 36-minute fold looks merely slow
until you know the model does 0.85 s/residue, at which point it is a 7x outlier. This
compares each running fold against the measured per-model ceiling and names the outliers.

Self-contained on purpose: it reads the target YAMLs directly rather than importing
abag_xm_generate, so it also runs on a host whose worktree predates that helper.

    python3 scripts/abag_xm_stall_scan.py            # scan, exit 1 if anything is flagged
    python3 scripts/abag_xm_stall_scan.py --dump PID # py-spy/gdb stack of a suspect worker

Recovery for a confirmed stall (in order, verifying the cmdline before every kill):
    kill -INT -<supervisor pgid>     # usually ignored -- that is the signature
    kill -TERM -<supervisor pgid>    # takes the supervisor, not the worker
    kill -9 <worker pid>             # explicit pid only, never a pattern
The chip is released immediately and the next fold opens it; no tt-smi reset was needed in
either recovery so far -- the process holds the chip, it does not wedge it.
"""
import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
YAML_DIR = ROOT / "examples" / "abag_xm"
# Max s/residue observed on qb1, the slower host (turn 37/41 measurements).
CEILING = {"protenix-v2": 4.0, "opendde-abag": 4.0, "boltz2": 0.9}
BOLTZ2_RAGGED_CEILING = 2.6   # mps not dividing diffusion_samples: measured 2.23 s/res
FLAG_RATIO = 1.5
DIFFUSION_SAMPLES = 50

_RES = {}


def residues(target):
    if target not in _RES:
        try:
            d = yaml.safe_load((YAML_DIR / f"{target}.yaml").open())
            _RES[target] = sum(len(v.get("sequence", ""))
                               for e in d.get("sequences", []) for k, v in e.items()
                               if k == "protein")
        except Exception:
            _RES[target] = 0
    return _RES[target]


def dump(pid):
    """Best-effort stack of a suspect worker: py-spy if present, else gdb, else threads."""
    # py-spy lives in a dedicated diag venv rather than the shared tt-bio env, so
    # installing a debugging tool cannot perturb the environment folds run in.
    pyspy = Path.home() / ".abag_xm_diag_venv" / "bin" / "py-spy"
    for cmd in ([str(pyspy), "dump", "--pid", str(pid)],
                ["py-spy", "dump", "--pid", str(pid)],
                ["gdb", "-p", str(pid), "-batch", "-ex", "thread apply all bt 12"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if r.returncode == 0 and r.stdout.strip():
                print(f"--- {cmd[0]} dump of {pid} ---\n{r.stdout}")
                return 0
        except Exception:
            continue
    print(f"neither py-spy nor gdb available; per-thread CPU for {pid}:")
    for t in sorted(Path(f"/proc/{pid}/task").glob("*")):
        try:
            f = (t / "stat").read_text().split()
            print(f"  tid {t.name} ticks={int(f[13]) + int(f[14])} "
                  f"wchan={(t / 'wchan').read_text()}")
        except Exception:
            pass
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=int, metavar="PID", help="stack-dump a suspect worker")
    a = ap.parse_args()
    if a.dump:
        return dump(a.dump)

    ps = subprocess.run(["ps", "-eo", "pid,etimes,args"], capture_output=True, text=True).stdout
    flagged = 0
    seen = set()
    for line in ps.splitlines():
        m = re.search(r"abag_xm/([0-9a-z]+)\.yaml --model ([a-z0-9-]+)", line)
        if not m or "spawn_main" in line:
            continue
        parts = line.split()
        pid, elapsed = parts[0], int(parts[1])
        target, model = m.group(1), m.group(2)
        if (target, model) in seen:      # supervisor + its duplicate ps line
            continue
        seen.add((target, model))
        mps = int(re.search(r"--max_parallel_samples (\d+)", line).group(1)) \
            if "--max_parallel_samples" in line else 5
        rate = CEILING.get(model, 4.0)
        if model == "boltz2" and DIFFUSION_SAMPLES % mps:
            rate = BOLTZ2_RAGGED_CEILING
        ceiling = rate * residues(target)
        ratio = elapsed / ceiling if ceiling else 0.0
        note = ""
        if ratio > FLAG_RATIO:
            note = "   <== OUT OF FAMILY, suspect stall"
            flagged += 1
        print(f"  pid {pid:>7s}  {target:6s} {model:14s} mps={mps}  {residues(target):4d} res  "
              f"elapsed {elapsed:5d}s  ceiling {ceiling:6.0f}s  {ratio:4.1f}x{note}")
    if not seen:
        print("  no folds in flight")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
