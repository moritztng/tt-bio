"""AdamW, before and after the arena rewrite, alternating inside ONE process.

AdamW is the only phase of the OF3 training step that is host-bound, so it is graded on host
load the way every other phase is graded on AICLK. Two runs cannot supply that: the baseline
arm of `perf/of3t_p10optim/out/base_384.json` ran at 0.54 load/core and the arena arm at 0.38,
and a 1.23x read off that pair is measuring the box as much as the change.

So both arms run here, alternating, over the same parameter census and the same gradients,
with the host load stamped at each round. The BASELINE IS THE REAL PRE-CHANGE MODULE, read out
of git at the ref given and imported beside the current one -- not a reconstruction of it, which
would be a baseline this row wrote for itself.

AdamW's work is crop-independent: it updates parameter tensors, and the census is the same
3,152 tensors / 381,302,188 elements at every crop. The capture still runs at the crop named so
the parameter set is the shipped one rather than a synthesised shape list.

    adamw_ab.py --tokens 384 --rounds 8 --out <json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

from perf.clocksample import during                                   # noqa: E402
from perf.of3t_perf import step as S                                  # noqa: E402
from perf.of3t_stepfloor.fullstep import declare_all                  # noqa: E402

SEED = 20260926


def baseline_module(ref):
    """`tt_bio/train/optim.py` as of ``ref``, importable beside the current one.

    Read from git rather than kept as a copy in this tree: a checked-in baseline drifts from
    the commit it claims to be, and then the ratio is against a file nobody ran.
    """
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:tt_bio/train/optim.py"],
                         capture_output=True, text=True, check=True).stdout
    # The module's two relative imports are the only thing that stops it loading under a name
    # outside the package. Absolute forms of the same two modules, which are unchanged here.
    src = (src.replace("from .mesh import", "from tt_bio.train.mesh import")
              .replace("from .tensors import", "from tt_bio.train.tensors import"))
    path = Path(tempfile.mkdtemp(prefix="optim_baseline_")) / "optim_baseline.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("optim_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["optim_baseline"] = mod
    spec.loader.exec_module(mod)
    return mod, src


def clone_params(params, dev):
    """A second, independent device copy of every parameter.

    Both optimizers must step a set of their own: `step()` writes `t.value`, so one shared set
    would have each arm updating weights the other already moved and the two would stop being
    the same measurement after round one.
    """
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.train.tensors import to_device, to_host
    out = {}
    for n, t in params.items():
        out[n] = ag.Tensor(to_device(to_host(t.value), dev, dtype=t.value.dtype),
                           requires_grad=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--baseline-ref", default="7d26d8f4f",
                    help="the commit whose tt_bio/train/optim.py is the A arm")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "ncpu": os.cpu_count(),
        "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg_start": [round(x, 2) for x in os.getloadavg()]}}

    _clk = [None]

    def dump():
        if _clk[0] is not None:
            out["env"]["aiclk_during"] = _clk[0].summary()
        out["env"]["loadavg_now"] = [round(x, 2) for x in os.getloadavg()]
        a.out.write_text(json.dumps(out, indent=1, default=str))

    dump()
    with during() as clk:
        _clk[0] = clk
        import numpy as np
        import ttnn
        from tt_bio.tenstorrent import get_device
        from tt_bio.train.optim import AdamW as NewAdamW
        from tt_bio.train.tensors import to_device

        base_mod, base_src = baseline_module(a.baseline_ref)
        out["baseline"] = {"ref": a.baseline_ref, "bytes": len(base_src),
                           "has_arena": "_Arena" in base_src}
        if out["baseline"]["has_arena"]:
            raise SystemExit(f"{a.baseline_ref} already carries the arena; it is not a baseline")

        held, _meta = S.capture(a.tokens, out)
        trunk = held["trunk"][0]
        sampler = held["sampler"][0]
        dev = get_device()
        out["env"]["arch"] = str(dev.arch())
        params_new = declare_all(trunk, sampler, out)
        params_old = clone_params(params_new, dev)
        dump()

        arms = {"baseline": base_mod.AdamW(params_old, lr=a.lr),
                "arena": NewAdamW(params_new, lr=a.lr)}
        # Shapes once. Every round draws a fresh gradient but both arms get the SAME one, and
        # the draw is outside the timed region.
        shapes = {n: tuple(int(d) for d in arms["arena"].master[n].shape) for n in params_new}
        out["census"] = {"params": len(shapes),
                         "elements": int(sum(int(np.prod(s)) for s in shapes.values())),
                         "largest": int(max(int(np.prod(s)) for s in shapes.values()))}
        dump()

        rng = np.random.default_rng(SEED)
        rounds = []
        out["rounds"] = rounds
        for r in range(a.rounds):
            g = {n: (rng.standard_normal(s) * 1e-3).astype(np.float32) for n, s in shapes.items()}
            for pset in (params_old, params_new):
                for n, t in pset.items():
                    t.grad = to_device(g[n], dev, dtype=t.value.dtype)
            ttnn.synchronize_device(dev)
            # Order flipped each round so a systematic first/second effect -- a cold page
            # cache, a neighbour's burst -- lands on both arms equally.
            order = ("baseline", "arena") if r % 2 == 0 else ("arena", "baseline")
            row = {"round": r, "order": list(order),
                   "loadavg": [round(x, 2) for x in os.getloadavg()]}
            for name in order:
                opt = arms[name]
                t0 = time.perf_counter()
                opt.step()
                ttnn.synchronize_device(dev)
                row[name] = {"phase_s": round(time.perf_counter() - t0, 4),
                             "split": opt.last_phase_s,
                             "writes_skipped": opt.last_writes_skipped}
            rounds.append(row)
            print(f"[round {r}] order {order[0]:8s} first | "
                  f"baseline {row['baseline']['split']['total']:6.3f}s  "
                  f"arena {row['arena']['split']['total']:6.3f}s  "
                  f"load {row['loadavg'][0]:5.2f}/{os.cpu_count()}", flush=True)
            dump()

        # Round 0 builds both moment sets from nothing, which is an allocation both arms pay
        # once and neither pays again. It is reported and excluded.
        warm = rounds[1:] or rounds
        summary = {}
        for name in ("baseline", "arena"):
            tot = [x[name]["split"]["total"] for x in warm]
            summary[name] = {
                "median_s": round(statistics.median(tot), 4),
                "min_s": round(min(tot), 4), "max_s": round(max(tot), 4), "n": len(tot),
                "split_median": {k: round(statistics.median(
                    [x[name]["split"][k] for x in warm]), 4)
                    for k in ("grad_read", "arith", "device_cast", "device_write")}}
        summary["speedup"] = round(summary["baseline"]["median_s"]
                                   / summary["arena"]["median_s"], 4)
        summary["saved_s"] = round(summary["baseline"]["median_s"]
                                   - summary["arena"]["median_s"], 4)
        summary["cold_round_excluded"] = {n: rounds[0][n]["split"]["total"]
                                          for n in ("baseline", "arena")}
        summary["arena_peak_elements"] = arms["arena"]._arena.peak_elements
        out["summary"] = summary
        out["env"]["loadavg_end"] = [round(x, 2) for x in os.getloadavg()]
        dump()
        print(json.dumps(summary, indent=1))
    dump()
    print(f"WROTE {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
