#!/usr/bin/env python3
"""The reachable bar for GO condition 3: upstream bf16-mixed against upstream float64.

`of3t-trajwide` measured our twenty-step trajectory against upstream's float64 one and read
`rel_d` 2.564253e-01 at k = 20 with a sub-linear growth exponent of -0.27724. The SHAPE passed.
The MAGNITUDE had no comparator, and A26's 1.0495450e-01 is not one: it is a single-step GRADIENT
bar, and Adam normalises by sqrt(v), so there is no factor that carries a per-step gradient error
into a twenty-step displacement error.

So the comparator is measured one rung up in k, and it is upstream against upstream: their
bf16-mixed loop against their float64 loop, same boundary capture, same noise-level partition,
same twenty steps, same `d_k = w_k - w_0`, same log-log fit over k = 2..20. That is what upstream
actually ships, bf16-mixed with an fp32 master, so the resulting curve is the REACHABLE
trajectory.

THE HARNESS IS `of3t-trajwide`'s, NOT A NEW ONE. `run_theirs` already assembles upstream's own
module, their `PerSampleGradManager`, their `torch.optim.Adam` and their `AlphaFoldLRScheduler` in
`runner.py:449-470` order, already resolves the reference tree in-process, already dumps `w_k` per
rung and already resumes. It takes the dtype as an argument. The whole of the bf16-mixed arm is:

    build at float32 (the fp32 master), and wrap the module's `forward` in
    `torch.autocast("cpu", bfloat16)`, upcasting the returned tensor back to float32 so the
    cotangent and the backward run at master precision.

That is the mixed-precision recipe: parameters and the optimizer in fp32, the matmul-class ops in
bf16, the reduction-class ops (softmax, the normalisations, the loss) kept in fp32 by autocast's
own policy, gradients accumulated in fp32. It is eight lines here and nothing else in the step
changes, which is the answer to the brief's "if that is more than a flag, say how much more".

SCOPE. `trajwide.do_score` builds its name set as the intersection of the two sides' dumps. Both
sides here are upstream, so that intersection is all 761 reference tensors -- WIDER than the 573
our side shares with upstream by name. A bar measured on a different scope than the reading it
bounds is not a bar, so this scores BOTH: the natural 761 and the 573 restricted to
`of3t-trajwide`'s own scored set, and the 573 figure is the one that may be quoted beside
2.564253e-01.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TWDIR = os.path.join(os.path.dirname(HERE), "of3t_trajwide")
# One slice assignment, not a loop of inserts at a fixed index. Inserting N paths at one index
# leaves them in the REVERSE of the order written, which is the mechanism behind D149; it was
# harmless here (neither directory carries a package under test) and it is still the shape the
# ratchet refuses, so it goes.
sys.path[:0] = [p for p in (TWDIR, HERE) if p not in sys.path]

import numpy as np                                                       # noqa: E402
import trajwide as TW                                                    # noqa: E402

#: the tree `openfold3` actually resolved from, read back in THIS process by `resolve_ref()`.
#: `None` until it has been read, so an artifact written without one is visibly missing it
#: rather than quietly carrying a constant.
REF_TREE = None

RUNS = "/home/ttuser/of3t_runs/trajbar"
TRAJWIDE_RUNS = "/home/ttuser/of3t_runs/trajwide"     # READ ONLY. `of3t-trajwide`'s row may be live.


# --------------------------------------------------------------------- the bf16-mixed reference

def _mixed_forward(orig):
    """Upstream's shipped recipe: bf16 compute over an fp32 master.

    The output is upcast to float32 on the way out, which is where the loss lives under mixed
    precision, so the captured cotangent is applied at master precision and the backward that
    follows runs outside the autocast context exactly as PyTorch's own guidance says it should.
    """
    import torch

    def up(x):
        return x.float() if torch.is_tensor(x) and x.is_floating_point() else x

    def fwd(*a, **k):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = orig(*a, **k)
        if isinstance(out, (tuple, list)):
            return type(out)(up(x) for x in out)
        if isinstance(out, dict):
            return {kk: up(v) for kk, v in out.items()}
        return up(out)

    return fwd


def install_mixed():
    """Patch `trajwide.build_theirs` so `trajwide.run_theirs` builds the mixed arm instead.

    Deliberately a patch of the ONE constructor rather than a forked `run_theirs`: the step order,
    the clipping, the accumulation, the scheduler and the resume path must be the same code as the
    float64 arm they are being compared against, or the difference between the two curves is not
    the dtype.
    """
    import torch
    orig_build = TW.build_theirs

    def build_mixed(dtype):
        m, own, inc, moved = orig_build(torch.float32)
        m.forward = _mixed_forward(m.forward)
        return m, own, inc, moved

    TW.build_theirs = build_mixed
    return orig_build


def cast_tree(x, dtype):
    """Cast every floating tensor in the captured boundary, at ANY depth.

    `trajwide.main` casts the top level only, which is a no-op there because the capture is
    already float64 and so is its target. It is not a no-op here. The capture's `feats` entry is
    a dict of 30-odd float64 tensors, and leaving them float64 while the parameters are float32
    makes upstream's own LayerNorm raise `mixed dtype (CPU)` inside the first
    `diffusion_conditioning` transition. So the bf16-mixed arm is the autocast wrapper PLUS this
    walk, and that is the whole of the delta from the float64 arm.
    """
    import torch
    if torch.is_tensor(x):
        return x.to(dtype) if x.is_floating_point() else x
    if isinstance(x, dict):
        return {k: cast_tree(v, dtype) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(cast_tree(v, dtype) for v in x)
    return x


def count_dtypes(x, acc=None):
    import torch
    acc = {} if acc is None else acc
    if torch.is_tensor(x):
        acc[str(x.dtype)] = acc.get(str(x.dtype), 0) + 1
    elif isinstance(x, dict):
        for v in x.values():
            count_dtypes(v, acc)
    elif isinstance(x, (list, tuple)):
        for v in x:
            count_dtypes(v, acc)
    return dict(sorted(acc.items()))


def dtype_probe():
    """What the arm actually ran at, read off live tensors rather than asserted from a flag.

    A26's bf16 floor and this bar are only comparable if the bf16 here is the same bf16 upstream
    ships, so the fp32 ISLANDS autocast keeps are enumerated rather than assumed: the matmul class
    goes to bf16, softmax and the reductions stay fp32, and upstream's own `LayerNorm` has an
    explicit bf16 branch (`normalization.py:64-75`) that casts its affine parameters down to the
    activation dtype instead of upcasting the activation.
    """
    import torch
    seen = {}
    a = torch.randn(64, 64, dtype=torch.float32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        seen["matmul"] = str((a @ a).dtype)
        seen["linear"] = str(torch.nn.functional.linear(a, a).dtype)
        seen["softmax"] = str(torch.softmax(a, -1).dtype)
        seen["layer_norm_fp32_in"] = str(torch.nn.functional.layer_norm(a, (64,)).dtype)
        b = a @ a
        seen["layer_norm_bf16_in"] = str(torch.nn.functional.layer_norm(b, (64,)).dtype)
        seen["sum"] = str(a.sum().dtype)
        seen["mean"] = str(a.mean().dtype)
        seen["rsqrt"] = str(torch.rsqrt(a.abs()).dtype)
        seen["silu_bf16_in"] = str(torch.nn.functional.silu(b).dtype)
    return seen


# ------------------------------------------------------------------------- the reference tree

def resolve_ref():
    """Install the reference tree and READ BACK the one `openfold3` came from.

    `trajwide.build_theirs` already calls `refpath.assert_resolved()`, so the bar's arms were
    never at risk -- but this file recorded `TW.REF_TREE`, a value it never read, and a static
    reader cannot tell that apart from a constant. It is also one edit away from being one: any
    path to `write_steplog` that does not go through `build_theirs` would record `None`, and
    nothing would refuse. So the resolution is taken here, in this file, before the arm starts,
    and its RETURN is what the steplog carries.
    """
    tree = TW.refpath.install()
    got = TW.refpath.assert_resolved(tree)
    import openfold3
    print(f"REF_TREE resolved: {got}", flush=True)
    print(f"openfold3.__file__ = {openfold3.__file__}", flush=True)
    return got


# ------------------------------------------------------------------------------------ the arms

def run_arm(a):
    global REF_TREE
    import torch
    t0 = time.time()
    load0 = os.getloadavg()
    os.makedirs(a.out_dir, exist_ok=True)
    REF_TREE = resolve_ref()

    D = torch.load(TW.DIFFCAP, map_location="cpu", weights_only=False)
    kw, cot = D["kwargs"], D["cot"]
    n_sample = int(kw["xl_noisy"].shape[1])
    blocks = TW.partition(n_sample, a.per_step, a.steps)
    print(f"[{time.time()-t0:.0f}s] boundary {TW.DIFFCAP}: {n_sample} noise levels, "
          f"{a.per_step} accumulation samples/step, blocks of {[len(b) for b in blocks[1]]}",
          flush=True)

    dtype = {"bf16mixed": torch.float32, "f64": torch.float64}[a.mode]
    if a.mode == "bf16mixed":
        install_mixed()
    kwv = cast_tree(kw, dtype)
    print(f"boundary cast to {dtype}: {count_dtypes(kwv)}", flush=True)

    log = []
    d_out = TW.wdir(a.out_dir, a.arm)

    def write_steplog(evidence=None):
        sl = os.path.join(a.out_dir, f"steplog_{a.arm}.json")
        tmp = sl + ".part"
        json.dump({"arm": a.arm, "side": "theirs", "mode": a.mode, "ref_tree": REF_TREE,
                   "steps": log, "evidence": evidence, "n_steps": len(log),
                   "complete": evidence is not None, "threads": a.threads,
                   "loadavg_start": load0, "loadavg_now": os.getloadavg(),
                   "wall_s": time.time() - t0}, open(tmp, "w"), indent=1, default=str)
        os.replace(tmp, sl)
        return sl

    TW.STEPLOG_SINK = write_steplog
    TW.run_theirs(dtype, blocks, cot, kwv, steps=a.steps, warmup=a.warmup, log=log,
                  d_out=d_out, also_second=a.aa_in_process)
    ev = dict(TW.run_theirs.last)
    ev["mode"] = a.mode
    ev["boundary_dtypes"] = count_dtypes(kwv)
    ev["autocast_policy"] = dtype_probe() if a.mode == "bf16mixed" else None
    ev["cotangent_dtype"] = str(cot.dtype)
    ev["loadavg_start"], ev["loadavg_end"] = list(load0), list(os.getloadavg())
    ev["wall_s"] = time.time() - t0
    ev["threads"] = a.threads
    TW.STEPLOG_SINK = None
    sl = write_steplog(ev)
    print(f"wrote {sl} ({len(log)} steps, {time.time()-t0:.0f}s, "
          f"loadavg {load0} -> {os.getloadavg()})", flush=True)
    return 0


# ----------------------------------------------------------------------------------- the zero arm

def make_zero_arm(out_dir, ref_dir, steps=20):
    """A16: a side that never moves. `d_k = w_k - w_0` is exactly 0 at every rung, so
    `rel_d = ||0 - d_theirs|| / ||d_theirs||` must read exactly 1.000000e+00.

    Built as symlinks to the reference's own k01 dump, so it costs no disk and, unlike an
    analytic 1.0, it goes through the real loader and the real scorer. It is a detector for a
    dead SCORER, and it does not claim to be a detector for a dead RUNNER -- the runner's own
    dead-arm evidence is `of3t-trajwide`'s `zero` arm, which ran on device.
    """
    d = TW.wdir(out_dir, "zero")
    src = os.path.join(ref_dir, "k01.npz")
    if not os.path.exists(src):
        raise SystemExit(f"no reference k01 at {src}")
    for k in range(1, steps + 1):
        dst = os.path.join(d, f"k{k:02d}.npz")
        if os.path.islink(dst) or os.path.exists(dst):
            os.unlink(dst)
        os.symlink(src, dst)
    return d


# ---------------------------------------------------------------------------------- the scoring

def names_for(ref_dir, arm_dir, sd, restrict_dir=None):
    """`trajwide.do_score`'s name rule verbatim, plus an optional third dump set to intersect
    with, which is how the bar is held to the SAME 573 tensors the reading it bounds was scored
    on."""
    w1t, w1o = TW.load_w(ref_dir, min(TW.have_steps(ref_dir))), TW.load_w(
        arm_dir, min(TW.have_steps(arm_dir)))
    names = sorted(n for n in w1t
                   if n in w1o and TW.PREFIX + n in sd
                   and tuple(sd[TW.PREFIX + n].shape) == tuple(w1t[n].shape))
    if restrict_dir:
        keep = set(TW.load_w(restrict_dir, min(TW.have_steps(restrict_dir))).keys())
        names = [n for n in names if n in keep]
    return names


def score_pair(ref_dir, arm_dir, names, sd, w0="own"):
    import torch
    ks = sorted(set(TW.have_steps(ref_dir)) & set(TW.have_steps(arm_dir)))
    if not ks:
        raise SystemExit(f"nothing to score: ref {TW.have_steps(ref_dir)}, "
                         f"arm {TW.have_steps(arm_dir)}")
    W0 = {n: sd[TW.PREFIX + n].to(torch.float64).numpy().astype(np.float32) for n in names}
    nw0 = math.sqrt(sum(float(W0[n].astype(np.float64).ravel()
                              @ W0[n].astype(np.float64).ravel()) for n in names))
    W0o = TW.load_w(arm_dir, ks[0]) if w0 == "own" else None
    rows = []
    for k in ks:
        wt, wo = TW.load_w(ref_dir, k), TW.load_w(arm_dir, k)
        r = TW.score_step(names, k, wo, wt, W0, nw0, W0o)
        rows.append(r)
        print(f"  k={r['k']:2d} rel_d={r['rel_d']:.6e} "
              f"floor={r['fp32_differencing_floor']:.3e} "
              f"|d_arm|={r['d_ours_norm']:.4e} |d_ref|={r['d_theirs_norm']:.4e} "
              f"worst={r['worst_per_tensor']:.3e} ({r['worst_tensor']})", flush=True)
    return rows, ks, nw0


def load_ckpt_sd():
    import torch
    sd = torch.load(TW.CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    return {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}


def do_score(a):
    """Every pair scored against ONE load of the reference per rung.

    A rung is 813 MB of reference dump and up to 800 MB of arm dump, and there are five pairs to
    score against the same twenty reference rungs. Loading the reference once per rung instead of
    once per pair is the difference between one pass over 16 GB and five.
    """
    sd = load_ckpt_sd()
    ref_dir = a.ref_dir
    pairs = []
    for spec in a.pair:
        parts = spec.split(":")
        tag, arm_dir = parts[0], parts[1]
        restrict = parts[2] if len(parts) > 2 and parts[2] else None
        names = names_for(ref_dir, arm_dir, sd, restrict_dir=restrict)
        pairs.append(dict(tag=tag, arm_dir=arm_dir, restrict=restrict, names=names))
        print(f"pair {tag}: {len(names)} tensors, arm {arm_dir}"
              + (f", restricted to {restrict}" if restrict else ""), flush=True)

    import torch
    for P in pairs:
        W0 = {n: sd[TW.PREFIX + n].to(torch.float64).numpy().astype(np.float32)
              for n in P["names"]}
        P["W0"] = W0
        P["nw0"] = math.sqrt(sum(float(W0[n].astype(np.float64).ravel()
                                       @ W0[n].astype(np.float64).ravel()) for n in P["names"]))
        P["W0o"] = TW.load_w(P["arm_dir"], min(TW.have_steps(P["arm_dir"]))) \
            if a.w0 == "own" else None
        P["rows"] = []

    # Per PAIR, not one global intersection. A 3-rung determinism arm scored beside a 20-rung
    # treatment used to drag the treatment down to 3 rungs, and the summary line still said
    # "k=20" because it was a literal. Both halves of that are fixed here: each pair gets its own
    # rung list, and every summary prints the k it actually reached.
    for P in pairs:
        P["ks"] = sorted(set(TW.have_steps(ref_dir)) & set(TW.have_steps(P["arm_dir"])))
        print(f"pair {P['tag']}: scoring k = {P['ks']}", flush=True)
    ks = sorted({k for P in pairs for k in P["ks"]})
    for k in ks:
        wt = TW.load_w(ref_dir, k)
        for P in pairs:
            if k not in P["ks"]:
                continue
            wo = TW.load_w(P["arm_dir"], k)
            r = TW.score_step(P["names"], k, wo, wt, P["W0"], P["nw0"], P["W0o"])
            P["rows"].append(r)
            print(f"  k={k:2d} {P['tag']:16s} rel_d={r['rel_d']:.6e} "
                  f"floor={r['fp32_differencing_floor']:.3e} "
                  f"|d_arm|={r['d_ours_norm']:.4e} |d_ref|={r['d_theirs_norm']:.4e} "
                  f"worst={r['worst_per_tensor']:.3e} ({r['worst_tensor']})", flush=True)
            del wo
        del wt

    out = {"ref_dir": ref_dir, "w0_baseline": a.w0, "steps_scored": ks,
           "accumulate_grad_batches": a.per_step, "warmup_no_steps": a.warmup,
           "model_sq_norm": TW.MODEL_SQ_NORM, "scored": {}}
    for P in pairs:
        rows = P["rows"]
        out["scored"][P["tag"]] = {
            "arm_dir": P["arm_dir"], "restrict_dir": P["restrict"],
            "n_tensors": len(P["names"]), "w0_norm": P["nw0"], "steps_scored": P["ks"],
            "scope": TW.scope_share(P["names"]),
            "d1": {"arm_norm": rows[0]["d_ours_norm"], "ref_norm": rows[0]["d_theirs_norm"],
                   "rel_d": rows[0]["rel_d"],
                   "zero_both_sides": rows[0]["d_ours_norm"] == 0.0 == rows[0]["d_theirs_norm"]},
            "growth_k2_20": TW.growth(rows),
            "per_step": rows,
        }
        g = out["scored"][P["tag"]]["growth_k2_20"]
        print(f"== {P['tag']:28s} {len(P['names'])} tensors  rungs {rows[0]['k']}..{rows[-1]['k']}"
              f"  rel_d(k={rows[-1]['k']})={rows[-1]['rel_d']:.6e}  "
              f"exponent={g.get('exponent')}  intercept={g.get('intercept')}  r2={g.get('r2')}  "
              f"{g.get('shape')}  fitted_over_k={g.get('fitted_over_k')}", flush=True)

    for tag, arm in (("bf16mixed", "bf16mixed"), ("bf16mixed_aa2", "bf16mixed_aa2")):
        sl = os.path.join(a.out_dir, f"steplog_{arm}.json")
        if os.path.exists(sl):
            out.setdefault("steplogs", {})[arm] = json.load(open(sl))
    out["sha256"] = {"diffusion_boundary": TW.sha256(TW.DIFFCAP),
                     "checkpoint": TW.sha256(TW.CKPT),
                     "grad_manager.py":
                         TW.sha256("/home/ttuser/of3t_traj20/upstream/grad_manager.py"),
                     "lr_schedulers.py":
                         TW.sha256("/home/ttuser/of3t_traj20/upstream/lr_schedulers.py")}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    print(f"wrote {a.out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="bf16mixed", choices=["bf16mixed", "f64"])
    ap.add_argument("--arm", default="bf16mixed")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--make-zero", action="store_true", dest="make_zero")
    ap.add_argument("--ref-dir", default=os.path.join(TRAJWIDE_RUNS, "w", "theirs"),
                    dest="ref_dir")
    ap.add_argument("--pair", action="append", default=[],
                    help="tag:arm_dump_dir[:restrict_dump_dir], repeatable. The restrict dir is "
                         "a third dump set to intersect the name list with -- of3t-trajwide's "
                         "w/shipped is what holds the bar to its 573 tensors")
    ap.add_argument("--per-step", type=int, default=4, dest="per_step")
    ap.add_argument("--steps", type=int, default=TW.STEPS)
    ap.add_argument("--warmup", type=int, default=TW.SCHED["warmup_no_steps"])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out-dir", default=RUNS, dest="out_dir")
    ap.add_argument("--w0", default="own", choices=["ckpt", "own"])
    ap.add_argument("--aa-in-process", action="store_true", dest="aa_in_process")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    os.environ["MKL_NUM_THREADS"] = str(a.threads)
    a.out = a.out or "perf/of3t_trajbar/BAR.json"

    if a.make_zero:
        print(make_zero_arm(a.out_dir, a.ref_dir, a.steps))
    if a.run:
        run_arm(a)
    if a.score:
        return do_score(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
