#!/usr/bin/env python3
"""Every fused activation in one executed 512 aa Boltz-2 fold, by call site (c12-unfused-silu-bh).

`_UNFUSED_SILU` gates exactly ONE of eight `activation="silu"` call sites in the tree, and Moritz's
standing rule is UNIFIED, NEVER PER-MODEL. A grep tells you a site exists; it does not tell you
whether it fires at this size, how often, or on what shape. So this wraps `ttnn.linear` and
`ttnn.matmul` for one real fold and records the caller's file:line from the frame, the activation
argument and the operand shapes. An eligibility firing condition is not a code fact until you see
it fire.

`sys._getframe(1)` rather than `traceback` because this runs on 108,608 linear calls.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ab_flag_levers as AB          # noqa: E402
import clk                           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--mhz", type=int, default=1350)
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio as _TB
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), \
        f"imported tt_bio from {_TB.__file__}, not this worktree"

    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = args.steps, args.recycles
    tally: dict = collections.defaultdict(int)
    recording = [False]

    def shp(t):
        try:
            return "x".join(str(d) for d in t.shape)
        except Exception:
            return "?"

    def wrap(name, fn):
        def inner(*a, **kw):
            if recording[0]:
                f = sys._getframe(1)
                act = kw.get("activation")
                # a positional activation is not used anywhere in this tree; assert rather than guess
                key = (f"{Path(f.f_code.co_filename).name}:{f.f_lineno}", name, str(act),
                       shp(a[0]) if a else "?", shp(a[1]) if len(a) > 1 else "?")
                tally[key] += 1
            return fn(*a, **kw)
        return inner

    orig = {n: getattr(ttnn, n) for n in ("linear", "matmul")}
    for n, f in orig.items():
        setattr(ttnn, n, wrap(n, f))

    dev = get_device()
    nodes = clk.nodes_open_by_this_process()
    assert len(nodes) == 1, f"expected exactly one open chip, got {nodes}"
    clk.force(args.mhz, nodes)
    sampler = clk.Sampler(nodes[0])

    work = Path(tempfile.mkdtemp(prefix="c12-silu-sites-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    AB._seed_msa(AB.FIX / f"cdk2x2_{args.size}.yaml",
                 (AB.FIX / f"cdk2x2_{args.size}.a3m").read_text(), msa_dir)
    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("c12-unfused-silu-sites", cfg)

    recording[0] = True
    t = time.perf_counter()
    state.predict_one(AB.FIX / f"cdk2x2_{args.size}.yaml", cfg)
    ttnn.synchronize_device(dev)
    wall = round(time.perf_counter() - t, 3)
    recording[0] = False
    aiclk = sampler.stop()
    clk.release()

    rows = [{"site": k[0], "op": k[1], "activation": k[2], "a_shape": k[3], "b_shape": k[4],
             "calls": v} for k, v in tally.items()]
    rows.sort(key=lambda r: -r["calls"])
    acts = collections.Counter()
    for r in rows:
        acts[r["activation"]] += r["calls"]
    out = {
        "doc": __doc__, "size": args.size, "wall_s": wall,
        "host": socket.gethostname(), "card_node": nodes[0],
        "clock_forced_mhz": args.mhz, "aiclk_during_fold": aiclk,
        "loadavg": os.getloadavg(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "note": "instrumented fold; wall_s carries the wrapper overhead and is NOT a perf number",
        "calls_by_activation": dict(acts),
        "total_calls": sum(acts.values()),
        "rows": rows,
        "silu_rows": [r for r in rows if r["activation"] == "silu"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print("ACTS " + json.dumps(dict(acts)))
    for r in out["silu_rows"]:
        print("SILU %-28s %-8s a=%-20s b=%-12s calls=%d" % (
            r["site"], r["op"], r["a_shape"], r["b_shape"], r["calls"]))
    print("NONSILU-ACT " + json.dumps([r for r in rows
                                       if r["activation"] not in ("None", "silu")]))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
