#!/usr/bin/env python3
"""Which routes the BACKWARD runs on, and whether the tape gate reaches it.

`of3t-tapedfwd` was briefed on the forward. The forward is 3.415 s of a 466.702 s step
(`of3t-stepfloor` step_rekey_b_384 rep 2) and the backward is 456.668 s, so the obvious next
question is whether the same tape gate that switches eleven fused kernels and three L1
residency levers off in the forward also switches them off in the backward -- which would put
the identical defect on 97.9 % of the step instead of 0.73 %.

Read from source the answer is no: every `with ag.tape():` in `tt_bio/train/openfold3.py`
(:458, :541, :593) and in `perf/of3t_stepfloor/fullstep.py:402` closes before the loss and the
backward, and `ops.taping()` is `grad_hook() is not None`, which `taped_ttnn.tape` restores on
the way out. But this row's whole premise is that the code says one thing and the run does
another, so it is counted instead: the same `ops.taping` substitution records its caller
during the backward, and the served/declined counters are read across it.

    bwroutes.py --tokens 384 --cycles 1 --out out/bwroutes_384.json
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

from perf.clocksample import during                                   # noqa: E402
from perf.of3t_perf import step as S                                  # noqa: E402
from perf.of3t_tapedfwd.fires import (TapingProbe, counter_attrs,     # noqa: E402
                                      delta, read_counters)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--out", type=Path,
                    default=Path("perf/of3t_tapedfwd/out/bwroutes.json"))
    a = ap.parse_args()
    out = {"doc": "Which routes the OF3 backward runs on, and whether ops.taping() reaches it.",
           "argv": sys.argv[1:],
           "env": {"host": socket.gethostname(),
                   "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
                   "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                            capture_output=True, text=True).stdout.strip(),
                   "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "loadavg": os.getloadavg()}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    clk = during(period=2.0)
    try:
        with clk:
            import ttnn
            from tt_bio import autograd as ag
            from tt_bio.tenstorrent import get_device
            held, _ = S.capture(a.tokens, out)
            trunk = held["trunk"][0]
            dev = get_device()
            attrs = counter_attrs()
            probe = TapingProbe().install()

            # One untaped forward first, to burn JIT off the inference routes so the taped
            # forward below is not paying for a compile the backward will be blamed for.
            S.cycle_once(trunk, held, a.cycles, taped=False)
            ttnn.synchronize_device(dev)

            snap_args, snap_kwargs = held["trunk_snap"]
            args = S._rehydrate(snap_args, dev)
            kwargs = {k: v for k, v in S._rehydrate(snap_kwargs, dev).items()
                      if k != "progress_fn"}
            trunk.num_cycles = a.cycles
            probe.take()
            fwd_before = read_counters(attrs)
            with ag.tape():
                t0 = time.perf_counter()
                s_tr, z_tr = trunk(*args, **kwargs)
                ttnn.synchronize_device(dev)
                out["taped_forward_s"] = round(time.perf_counter() - t0, 4)
                out["forward_counters"] = delta(fwd_before, read_counters(attrs))
                out["forward_taping_sites"] = probe.take()
            out["tape_exit_s"] = None      # the tape is closed by the `with`, as the loop does

            # The roots the step seeds. Only the pair track here: this row is pricing routes,
            # not reproducing the objective, and a seed of ones on z reaches the same trunk.
            roots = [r for r in (s_tr, z_tr) if isinstance(r, ag.Tensor)]
            out["roots_taped"] = len(roots)
            out["tape_nodes"] = len(ag._reverse_topo(roots)) if roots else 0
            seeds = [ttnn.ones_like(ag._unwrap(r)) for r in roots]

            bw_before = read_counters(attrs)
            probe.take()
            t0 = time.perf_counter()
            ag.backward(roots, seeds)
            ttnn.synchronize_device(dev)
            out["backward_s"] = round(time.perf_counter() - t0, 4)
            out["backward_counters"] = delta(bw_before, read_counters(attrs))
            out["backward_taping_sites"] = probe.take()
            out["backward_asked_the_tape_gate"] = bool(out["backward_taping_sites"])
            probe.restore()
            out["verdict"] = (
                "the backward asked the tape gate "
                f"{sum(out['backward_taping_sites'].values())} times"
                if out["backward_taping_sites"] else
                "the backward never asked the tape gate at all")
    except Exception:
        out["error"] = traceback.format_exc()
        print(out["error"], file=sys.stderr, flush=True)
    out["clock"] = clk.summary()
    out["clock_line"] = clk.line(0)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("forward_counters", "backward_counters")}, indent=1)[:2500])
    print(f"wrote {a.out}", flush=True)
    return 1 if "error" in out else 0


if __name__ == "__main__":
    raise SystemExit(main())
