#!/usr/bin/env python3
"""Two full OF3T training steps in ONE card visit: exactness ON, then exactness OFF.

`of3t-stepfloor`'s `fullstep.py` is imported, not forked, and not edited: this file only
chooses the process it runs in. It gives the campaign the one number nobody has -- a full
taped training step with `exact_training` ON, the configuration whose gradients clear the
accuracy bar -- and the same step with the knob OFF, on the same board, the same clock and
the same capture, so the exactness price at STEP scope is a difference of two measurements
rather than a scaling of a trunk-cycle A/B.

WHY ONE PROCESS. `capture()` runs a real `predict_one` to the sampler and costs ~9 minutes;
two processes would pay it twice and would compare two captures. `S.capture` is memoised
here, so arm 2 rehydrates arm 1's snapshot exactly as rep 2 rehydrates rep 1's.

ARM ORDER IS ON FIRST, deliberately. Rep 0 is not a compile penalty in this harness
(`step_rekey_b_384.json`: cold 449.4 s against a warm median of 468.9 s), so putting the
headline arm first costs it nothing and banks it first if the launch is cut short.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perf.of3t_perf import step as S                                  # noqa: E402
from perf.of3t_stepfloor import fullstep as F                         # noqa: E402

HOST_QUIET = REPO / "perf" / "c12_orchestrator" / "pair_guard" / "host_quiet.py"
OUT = REPO / "perf" / "of3t_restep" / "out"

_CAPTURED: dict = {}
_real_capture = S.capture


def capture_once(tokens, out):
    """One capture for both arms. The fields `capture()` writes into its caller's `out` are
    replayed into the second arm's `out` so each artifact carries its own provenance."""
    if tokens not in _CAPTURED:
        scratch: dict = {}
        held, meta = _real_capture(tokens, scratch)
        _CAPTURED[tokens] = (held, meta, scratch, 0)
    held, meta, fields, used = _CAPTURED[tokens]
    out.update(fields)
    out["capture_reused_from_arm"] = None if used == 0 else 1
    _CAPTURED[tokens] = (held, meta, fields, used + 1)
    return held, meta


S.capture = capture_once
F.S = S


def quiet(label):
    p = subprocess.run([sys.executable, str(HOST_QUIET)], capture_output=True, text=True)
    rec = {"when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rc": p.returncode, "green": p.returncode == 0,
           "loadavg": list(os.getloadavg()),
           "report": p.stdout.strip().splitlines()[-6:]}
    print(f"[host_quiet {label}] rc={p.returncode} loadavg={rec['loadavg']}", flush=True)
    return rec


def arm(name, exact, argv, out_path):
    from tt_bio import autograd as ag
    pre = quiet(f"{name}/pre")
    sys.argv = ["fullstep.py"] + argv + ["--out", str(out_path)]
    t0 = time.perf_counter()
    if exact:
        rc = F.main()
    else:
        with ag.exact_training(False):
            rc = F.main()
    wall = round(time.perf_counter() - t0, 3)
    post = quiet(f"{name}/post")
    # STAMP. fullstep writes env.commit and config.cycles_pinned itself; what it does not
    # know is which arm it was, so that is written here beside them rather than inferred.
    d = json.loads(out_path.read_text())
    d["arm"] = {"name": name, "exact_training": bool(exact),
                "exact_training_ops": list(ag.exact_training_ops()),
                "exact_softmax_installed_now": ag.exact_softmax_installed(),
                "wall_s": wall, "rc": rc,
                "host_quiet_pre": pre, "host_quiet_post": post}
    d["config"]["cycles"] = d["config"]["cycles_pinned"]
    d["scope"] = {
        "contains": ["trunk no_grad recycle prefix", "one taped trunk cycle",
                     "diffusion module on N noised structures inside the same tape",
                     "af3_loss heads on host", "backward over the whole tape",
                     "AdamW optimizer step"],
        "omits": ["upstream's 48 diffusion samples (this runs --samples)",
                  "upstream's per-step U{1..4} recycle draw (this PINS --cycles)"],
        "note": "trunk_nograd_prefix_s is reported per rep, so the step with and without "
                "the no_grad prefix are both readable off this artifact.",
    }
    out_path.write_text(json.dumps(d, indent=1, default=str))
    print(f"[arm {name}] rc={rc} wall={wall}s -> {out_path}", flush=True)
    return rc


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--reps-on", type=int, default=1)
    ap.add_argument("--reps-off", type=int, default=1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base = [f"--tokens", str(a.tokens), "--cycles", str(a.cycles),
            "--samples", str(a.samples)]
    tag = a.tag or f"{a.tokens}"
    rc = 0
    rc |= arm("exact_on", True, base + ["--reps", str(a.reps_on)],
              OUT / f"step_exact_on_{tag}.json")
    rc |= arm("exact_off", False, base + ["--reps", str(a.reps_off)],
              OUT / f"step_exact_off_{tag}.json")
    return rc


if __name__ == "__main__":
    sys.exit(main())
