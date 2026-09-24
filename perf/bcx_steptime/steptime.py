#!/usr/bin/env python3
"""bcx-steptime: the whole 4 extra-MSA + 48 Evoformer checkpointed gradient step at n=256 on the
mask-fixed tree, against shipped main, arms interleaved in ONE process.

Main refuses a masked fold (its `evoformer_stack` asserts an all-ones mask), so it can only run
the all-ones program. Three arms:

  main       origin/main's tt_bio, all-ones mask (its tape has no backward for a fused
             activation the trunk uses, so it is recorded as refused, not timed)
  ckpt       wk/bcx-ckpt b5097866c, the tree the 13.40 s headline was measured on, all-ones
  tree       this branch's tt_bio, all-ones mask: the same function as main
  tree+mask  this branch, `--pad` residues masked, which is the program BindCraft 2 runs when it
             pads a design to the 32 bucket (the five rejected trajectories padded 19)

Two copies of `tt_bio` share one device: this tree is imported first and opens the card, its
modules are stashed, then main's copy is imported from `--main-tree` with its device handle set
to the open one. Each arm swaps its copy back into `sys.modules`, which is what `taped_ttnn`'s
shim walks, so each arm tapes its own modules.

Per rep: taped forward + backward (the gradient step), an untaped forward of the same stack (the
one stop-gradient recycle BindCraft 2 runs first), process CPU seconds, loadavg, AICLK sampled
from the card's own sysfs node inside the window, and whether the logit gradient is finite.

  run    the device arms; saves timings and each rep's logit gradient
  ref    the float64 chain on CPU for the same draws, unmasked and masked; no device
  grade  joins the two
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_afgrad import afgrad as A  # noqa: E402
from perf.bcx_stack.stack import Clock, dist  # noqa: E402

OUT = ROOT / "perf" / "bcx_steptime"


def draws(n, seed, pad):
    """`afgrad stack`'s draws in its order, plus the pad mask. The masked arm's loss reads real
    residues only, as BindCraft 2's losses do; the pad rows are the ones AF2 leaves unread."""
    torch.manual_seed(seed)
    logits = torch.randn(n, 20) * 2.0
    wm = torch.randn(1, n, 256, dtype=torch.float64) / (n * 256) ** 0.5
    wz = torch.randn(n, n, 128, dtype=torch.float64) / (n * n * 128) ** 0.5
    seq = torch.ones(n)
    seq[n - pad:] = 0
    wm_m, wz_m = wm.clone(), wz.clone()
    wm_m[:, n - pad:] = 0
    wz_m[n - pad:] = 0
    wz_m[:, n - pad:] = 0
    return logits, torch.arange(n), (wm, wz), (wm_m, wz_m), seq


def _pkg():
    return {k: v for k, v in sys.modules.items() if k == "tt_bio" or k.startswith("tt_bio.")}


def _activate(mods):
    for k in list(_pkg()):
        del sys.modules[k]
    sys.modules.update(mods)


class Copy:
    """One tt_bio copy: its modules, its device model and its Dev wrapper."""

    def __init__(self, name, mods, dm):
        self.name, self.mods, self.dm = name, mods, dm
        _activate(mods)
        self.dev = A.Dev(dm)


def load(args):
    """This tree first (it opens the card), then each `--copies name=path` with its device handle
    set to the open one. Returns {name: Copy} and the bf16 reference used for the embedding."""
    A._float64_layernorm()
    from tt_bio.af2 import load_af2_device_model
    from tt_bio.af2_reference import load_af2_model
    from tt_bio.af2_weights import load_af2_state_dict
    state = load_af2_state_dict(args.params)
    dm_tree = load_af2_device_model(state, template=False, trunk_dtype=torch.bfloat16)
    ref_bf16 = load_af2_model(state, template=False, trunk_dtype=torch.bfloat16)
    for p in ref_bf16.parameters():
        p.requires_grad_(False)
    tree_mods = _pkg()
    tn = tree_mods["tt_bio.tenstorrent"]
    copies = {}
    for spec in filter(None, args.copies.split(",")):
        name, path = spec.split("=")
        for k in list(_pkg()):
            del sys.modules[k]
        sys.path.insert(0, path)
        import tt_bio.tenstorrent as mtn
        assert pathlib.Path(mtn.__file__).resolve().is_relative_to(pathlib.Path(path).resolve()), mtn.__file__
        mtn._device, mtn._device_lease = tn._device, tn._device_lease
        mtn._trace_region_size = getattr(tn, "_trace_region_size", 0)
        from tt_bio.af2 import load_af2_device_model as load_other
        dm = load_other(state, template=False, trunk_dtype=torch.bfloat16)
        mods = _pkg()
        sys.path.remove(path)
        copies[name] = Copy(name, mods, dm)
    copies["tree"] = Copy("tree", tree_mods, dm_tree)
    _activate(tree_mods)
    return copies, ref_bf16


RECYCLE = [True]


def step(c, ref_bf16, logits, ridx, w, masks, ke, kv):
    """One taped gradient step and one untaped forward on copy `c`. `masks` is None or
    (msa_mask_tt, pair_masks)."""
    _activate(c.mods)
    dev, ag, dm = c.dev, c.dev.ag, c.dm
    wm, wz = w
    msa_mask, pm = masks if masks else (None, (None, None))

    def extra(i, z):
        blk = dm.device_extra_msa[i]
        const = dm._up(dm.opm_constant[i].reshape(1, 1, -1))
        return blk(blk._residual(z, const), *pm)

    def evo(i, m, z):
        return dm.device_evoformer[i](m, z, msa_mask, *pm)

    def stack(m, z, ckpt):
        for i in range(ke):
            z = ag.checkpoint(lambda t, i=i: extra(i, t), z) if ckpt else extra(i, z)
        for i in range(kv):
            m, z = (ag.checkpoint(lambda a, b, i=i: evo(i, a, b), m, z) if ckpt
                    else evo(i, m, z))
        return m, z

    gc.collect()
    r0 = r1 = rc0 = rc1 = 0.0
    if RECYCLE[0]:
        # the recycle: the same stack untaped, as BindCraft 2 runs it under stop_gradient
        with torch.no_grad():
            m0, z0 = A.embed(ref_bf16, logits.float(), ridx)
        mr, zr = dev.up(m0), dev.up(z0)
        dev.sync()
        r0, rc0 = time.time(), time.process_time()
        mr, zr = stack(mr, zr, ckpt=False)
        dev.sync()
        r1, rc1 = time.time(), time.process_time()
        import ttnn
        ttnn.deallocate(mr)
        ttnn.deallocate(zr)

    lgt = logits.clone().float().requires_grad_(True)
    m0, z0 = A.embed(ref_bf16, lgt, ridx)
    ml, zl = dev.leaf(m0), dev.leaf(z0)
    dev.sync()
    t0, c0 = time.time(), time.process_time()
    with dev.tt.tape():
        mo, zo = stack(ml, zl, ckpt=True)
    dev.sync()
    t1 = time.time()
    seeds = [dev.seed(wm, mo), dev.seed(wz, zo)]
    dev.sync()
    t2, c2 = time.time(), time.process_time()
    ag.backward([mo, zo], seeds)
    dev.sync()
    t3, c3 = time.time(), time.process_time()
    gm0, gz0 = dev.grad(ml, m0.shape), dev.grad(zl, z0.shape)
    torch.autograd.backward([m0, z0], [gm0.to(m0.dtype), gz0.to(z0.dtype)])
    ag.release_pins()
    del mo, zo, ml, zl, seeds
    gc.collect()
    g = lgt.grad.double()
    return g, {"fwd": t1 - t0, "bwd": t3 - t2, "step": t3 - t0, "step_cpu": c3 - c0,
               "bwd_cpu": c3 - c2, "recycle_fwd": r1 - r0, "recycle_cpu": rc1 - rc0,
               "load1": os.getloadavg()[0], "finite": bool(torch.isfinite(g).all()),
               "finite_mo_zo_grads": bool(torch.isfinite(gm0).all() and torch.isfinite(gz0).all()),
               "spans": [(r0, r1), (t0, t3)]}


def cmd_run(args):
    copies, ref_bf16 = load(args)
    clock = Clock()
    n, ke, kv = args.n, args.extra, args.evo
    logits, ridx, w_full, w_mask, seq = draws(n, args.seed, args.pad)
    tree = copies["tree"]
    _activate(tree.mods)
    from tt_bio.af2 import af2_pair_masks
    mask_2d = seq[:, None] * seq[None, :]
    masks = (tree.dev.up(seq[None]), af2_pair_masks(mask_2d, tree.dm._device))
    assert masks[1][0] is not None, "the pad mask took the all-ones path"
    arms = {c: (copies[c], w_full, None) for c in copies}
    arms["tree+mask"] = (tree, w_mask, masks)
    names = args.arms.split(",")
    blob = {"stamp": A.stamp(args.card), "aiclk_node": clock.path, "pci": clock.pci,
            "copies": args.copies, "copy_commits": args.copy_commits,
            "n": n, "pad": args.pad, "k_extra": ke, "k_evo": kv, "ckpt": True, "seed": args.seed,
            "arms": names, "reps": args.reps, "warm": {}, "per_arm": {}, "runs": {}}
    grads, runs = {a: [] for a in names}, {a: [] for a in names}
    blob["refused"] = {}
    for a in list(names):                             # warm: JIT + program cache, discarded
        c, w, mk = arms[a]
        try:
            _, r = step(c, ref_bf16, logits, ridx, w, mk, ke, kv)
        except Exception as exc:  # noqa: BLE001 -- an arm that cannot take the step is a result
            import traceback
            traceback.print_exc()
            blob["refused"][a] = f"{type(exc).__name__}: {exc}"
            print("refused", a, blob["refused"][a][:300], flush=True)
            names.remove(a)
            del grads[a], runs[a]
            gc.collect()
            continue
        r["aiclk"] = clock.window(r.pop("spans"))
        blob["warm"][a] = r
        print("warm", a, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()},
              flush=True)
    for rep in range(args.reps):
        order = names[rep % len(names):] + names[:rep % len(names)]
        if rep % 2:
            order = order[::-1]
        for a in order:
            c, w, mk = arms[a]
            g, r = step(c, ref_bf16, logits, ridx, w, mk, ke, kv)
            r["aiclk"] = clock.window(r["spans"])
            r["order"] = order.index(a)
            grads[a].append(g)
            runs[a].append(r)
            print(a, rep, {k: (round(v, 3) if isinstance(v, float) else v)
                           for k, v in r.items() if k != "spans"}, flush=True)
        save(args, blob, grads, runs, clock, partial=True)
    save(args, blob, grads, runs, clock, partial=False)
    clock.stop()


def save(args, blob, grads, runs, clock, partial):
    for a, rs in runs.items():
        if not rs:
            continue
        ok = [r for r in rs if r["finite"]]
        pa = {"reps_finite": [r["finite"] for r in rs], "n_finite": len(ok),
              "load1": [r["load1"] for r in rs],
              "aiclk": clock.window([s for r in rs for s in r["spans"]]),
              "reps_bit_identical": [bool(torch.equal(grads[a][0], g)) for g in grads[a]]}
        for key in ("step", "fwd", "bwd", "step_cpu", "bwd_cpu", "recycle_fwd", "recycle_cpu"):
            pa[key] = dist([r[key] for r in ok]) if ok else None
            pa[key + "_all"] = [r[key] for r in rs]
        blob["per_arm"][a] = pa
    blob["runs"] = {a: [{k: v for k, v in r.items()} for r in rs] for a, rs in runs.items()}
    blob["partial"] = partial
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.out or f"run_n{args.n}_s{args.seed}"
    (OUT / f"{tag}.json").write_text(json.dumps(blob, indent=1, default=str))
    torch.save({a: gs for a, gs in grads.items()}, OUT / f"{tag}_grads.pt")
    print(f"wrote {OUT / tag}.json", flush=True)


def cmd_ref(args):
    """float64 logit gradients for the same draws: the all-ones function and the masked one."""
    torch.set_num_threads(args.threads)
    A._float64_layernorm()
    from tt_bio.af2_reference import load_af2_model
    from tt_bio.af2_weights import load_af2_state_dict
    state = load_af2_state_dict(args.params)
    ref = load_af2_model(state, template=False, trunk_dtype=torch.float64).double()
    for p in ref.parameters():
        p.requires_grad_(False)
    n, ke, kv = args.n, args.extra, args.evo
    logits, ridx, w_full, w_mask, seq = draws(n, args.seed, args.pad)
    out = {}
    for name in args.ref_arms.split(","):
        masked = name == "mask"
        wm, wz = w_mask if masked else w_full
        sm = seq.double() if masked else torch.ones(n, dtype=torch.float64)
        m2 = sm[:, None] * sm[None, :]
        lg = logits.double().clone().requires_grad_(True)
        t0 = time.time()
        m, z = A.embed(ref, lg, ridx)
        for i in range(ke):
            z = ref.extra_msa[i](torch.zeros(1, n, 64, dtype=torch.float64), z,
                                 torch.zeros(1, n, dtype=torch.float64), m2)[1]
        for i in range(kv):
            m, z = ref.evoformer[i](m, z, sm[None], m2)
        ((wm * m).sum() + (wz * z).sum()).backward()
        out[name] = lg.grad.detach()
        print(name, "seconds", round(time.time() - t0, 1), "norm", float(out[name].norm()), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT / f"ref_n{n}_s{args.seed}_p{args.pad}.pt")


def cmd_grade(args):
    tag = args.out or f"run_n{args.n}_s{args.seed}"
    blob = json.loads((OUT / f"{tag}.json").read_text())
    grads = torch.load(OUT / f"{tag}_grads.pt")
    ref = torch.load(OUT / f"ref_n{args.n}_s{args.seed}_p{args.pad}.pt")
    real = slice(0, args.n - args.pad)
    grade = {}
    for a, gs in grads.items():
        r64 = ref["mask"] if a.endswith("+mask") else ref["ones"]
        rows = real if a.endswith("+mask") else slice(None)
        grade[a] = [None if not torch.isfinite(g).all() else A.cmp(g[rows], r64[rows]) for g in gs]
        if a.endswith("+mask"):
            grade[a + ":pad_rows_grad_norm"] = [float(g[args.n - args.pad:].norm()) for g in gs]
    for a in grads:
        if a not in ("tree", "tree+mask") and "tree" in grads:
            grade[f"tree_vs_{a}_rep0"] = A.cmp(grads["tree"][0], grads[a][0])
    blob["grade_vs_f64"] = grade
    (OUT / f"{tag}_graded.json").write_text(json.dumps(blob, indent=1, default=str))
    print(json.dumps(grade, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "ref", "grade"])
    ap.add_argument("--params", default=A.DEFAULT_PARAMS)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES") or 0))
    ap.add_argument("--copies", default="main=/tmp/bcx-steptime-main,ckpt=/tmp/bcx-steptime-ckpt",
                    help="name=path of other tt_bio trees loaded beside this one")
    ap.add_argument("--copy-commits", default=None)
    ap.add_argument("--arms", default="main,ckpt,tree,tree+mask")
    ap.add_argument("--ref-arms", default="ones,mask")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--pad", type=int, default=19)
    ap.add_argument("--extra", type=int, default=4)
    ap.add_argument("--evo", type=int, default=48)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-recycle", action="store_true")
    args = ap.parse_args()
    RECYCLE[0] = not args.no_recycle
    {"run": cmd_run, "ref": cmd_ref, "grade": cmd_grade}[args.cmd](args)


if __name__ == "__main__":
    main()
