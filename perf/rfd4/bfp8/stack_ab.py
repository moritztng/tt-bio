"""The bfp8 stack as a stack, timed paired: two RFD4 trunks in one process on one card, one built on the
default lever set and one on another, each trunk pass captured as a trace and the two replayed
alternately (A B A B ...), so host load and clock drift land on both arms alike.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/bfp8/stack_ab.py --weights <bundle> --out <file.json> \
        [--n 128] [--reps 40] [--base LEVERS] [--levers LEVERS]

A design runs 180 passes (60 network calls x 3), so `trunk_s_per_design` is 180 x the median pass.
Inputs are seeded normals at the released widths (pass time does not depend on values); the two
arms' outputs on the same input are compared too (rel RMS of z), as a sanity check, not the grade.
AICLK is sampled for the life of the run (`perf/clocksample.py`).
"""
import argparse
import contextlib
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/ for clocksample

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402
import ttnn  # noqa: E402

PASSES_PER_DESIGN = 180


@contextlib.contextmanager
def held(spec):
    """`MODULE.NAME=EXPR` set on tt_bio.MODULE for the duration; nothing when empty."""
    if not spec:
        yield
        return
    name, expr = spec.split("=", 1)
    mod = importlib.import_module("tt_bio." + name.rsplit(".", 1)[0])
    attr = name.rsplit(".", 1)[1]
    prev = getattr(mod, attr)
    setattr(mod, attr, eval(expr))
    try:
        yield
    finally:
        setattr(mod, attr, prev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--base", default=None, help="lever set of the base arm (default rfd4.trunk.NORMAL_LEVERS)")
    ap.add_argument("--levers", default=None, help="lever set of the stack arm (default rfd4.trunk.FAST_LEVERS)")
    ap.add_argument("--base-set", default="", help="MODULE.NAME=EXPR (a tt_bio module attribute) held while the base "
                    "arm builds and captures its programs, e.g. 'trimul_tail.T1_NB={}': a knob A/B under one lever set")
    a = ap.parse_args()

    from tt_bio import tenstorrent as T
    from tt_bio.rfd4 import network as N
    from tt_bio.rfd4 import trunk as TR
    base = ",".join(sorted(TR.NORMAL_LEVERS)) if a.base is None else a.base
    stack = ",".join(sorted(TR.FAST_LEVERS)) if a.levers is None else a.levers

    dev = T.get_device(trace="rf3")
    state, config = N.load_checkpoint(a.weights)
    cfg = config["nn"]
    sd = {k: v.float() for k, v in N._sub(state, "nn.").items()}

    def build(spec):
        os.environ["TT_BIO_LEVERS"] = spec
        try:
            return TR.Trunk(T.WeightScope(sd), TR.compute_kernel_config(), nlayers=cfg["nlayers"],
                            n_heads=cfg["nheads"], every_n=cfg["update_pair_repr_every_n"],
                            residual_dtype=ttnn.float32)
        finally:
            os.environ.pop("TT_BIO_LEVERS", None)

    arms = {"base": build(base), "stack": build(stack)}
    n, tok, pair, dc = a.n, cfg["token_dim"], cfg["pair_repr_dim"], cfg["dim_cond"]
    g = torch.Generator().manual_seed(0)
    s0, z0 = torch.randn(1, n, tok, generator=g), torch.randn(1, n, n, pair, generator=g)
    c0 = torch.randn(1, n, dc, generator=g)
    up = lambda t, dt: ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    masks = TR.trunk_masks(torch.ones(1, n), dev, residual_dtype=ttnn.float32)
    s, z, cond = up(s0, ttnn.float32), up(z0, ttnn.float32), up(c0, ttnn.bfloat16)
    sync = lambda: ttnn.synchronize_device(dev)  # noqa: E731

    def free(ts):
        for t in ts:
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)

    rec = {"n": n, "levers": {"base": sorted(arms["base"]._levers), "stack": sorted(arms["stack"]._levers)},
           "base_set": a.base_set,
           "tt_bio": str(Path(TR.__file__).resolve().parents[2]), "load_start": os.getloadavg()[0]}
    with clocksample.during(period=0.5) as clk:
        traces, outs = {}, {}
        for k, tr in arms.items():
            fn = lambda tr=tr: tr.run_pass(ttnn.clone(s), ttnn.clone(z), cond, masks)  # noqa: E731
            with T.lever_scope(tr._levers), held(a.base_set if k == "base" else ""):  # run_pass is not scoped itself
                free(fn())  # compile
                sync()
                out = fn()
                outs[k] = ttnn.to_torch(out[1]).double()
                free(out)
                traces[k] = T.capture_trace(dev, fn)
            sync()
        ms = {k: [] for k in arms}
        for _ in range(a.reps):
            for k in arms:
                t = time.perf_counter()
                ttnn.execute_trace(dev, traces[k][0], cq_id=0, blocking=True)
                ms[k].append(1e3 * (time.perf_counter() - t))
        for tid, res in traces.values():
            ttnn.release_trace(dev, tid)
            free(res)
    d = outs["stack"] - outs["base"]
    rec["z_rel_rms_stack_vs_base"] = float(d.pow(2).mean().sqrt() / outs["base"].pow(2).mean().sqrt())
    ratios = [b / s_ for b, s_ in zip(ms["base"], ms["stack"])]
    for k, v in ms.items():
        med = statistics.median(v)
        rec[k] = {"pass_ms_median": round(med, 3), "pass_ms_min": round(min(v), 3),
                  "trunk_s_per_design": round(PASSES_PER_DESIGN * med / 1e3, 3)}
    rec["paired_speedup_median"] = round(statistics.median(ratios), 4)
    rec["reps"] = a.reps
    rec["load_end"] = os.getloadavg()[0]
    rec["aiclk"] = clk.summary()
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()
