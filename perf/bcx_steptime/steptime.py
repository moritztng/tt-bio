#!/usr/bin/env python3
"""bcx-steptime: the whole 4 extra-MSA + 48 Evoformer checkpointed gradient step at n=256 on the
mask-fixed tree, arms interleaved in ONE process against one open card.

Shipped main cannot be an arm: its tape has no backward for the `MUL_UNARY_SFPU` attention scale
the trunk fuses into `_fp32_softmax_tail`, so its first taped backward raises. Every arm is this
tree, which is what the design loop runs:

  tree              all-ones mask, masks passed as None (the path `evoformer_stack` takes when
                    nothing is padded)
  tree+ones         all-ones mask with EVERY masked site forced on: the MSA mask as a tensor of
                    ones and the pair masks as ones and a zero key bias. It is the unmasked
                    function, so it shares `tree`'s float64 reference: the neutrality control
  tree+ones@nofix   the same, with `_mask_biases` as it was before `bcx-nan` `e5cf74790` (no
                    `fill_implicit_tile_padding`), patched in-process
  tree+mask         `--pad` residues masked, the program BindCraft 2 runs when it pads a design
                    to the 32 bucket (the five rejected trajectories padded 19)

Per rep: an untaped forward of the stack (the stop-gradient recycle BindCraft 2 runs first), the
taped forward + backward (the gradient step), process CPU seconds, loadavg, AICLK sampled from the
card's own sysfs node inside the window, and whether the logit gradient is finite.

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

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_afgrad import afgrad as A  # noqa: E402
from perf.bcx_stack.stack import Clock, dist  # noqa: E402

OUT = ROOT / "perf" / "bcx_steptime"

# `AF2EvoformerBlock._mask_biases` before `bcx-nan`, compiled into af2's own namespace so it
# resolves `ttnn` through the same module attribute `taped_ttnn._swap` shims.
_NOFIX = '''
def _mask_biases(self, msa_mask):
    if len(msa_mask.shape) == 3:
        msa_mask = ttnn.reshape(msa_mask, tuple(msa_mask.shape)[1:])
    rows, n = (int(d) for d in msa_mask.shape)
    flat = ttnn.multiply(ttnn.subtract(msa_mask, 1.0), MASK_LOGIT_BIAS)
    row_bias = ttnn.reshape(flat, (rows, 1, 1, n))
    transposed = ttnn.permute(flat, (1, 0))
    col_bias = ttnn.reshape(transposed, (n, 1, 1, rows))
    return row_bias, col_bias
'''


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


class Arms:
    """The four arms' masks and the `_mask_biases` each one runs."""

    def __init__(self, dm, dev, seq, w_full, w_mask):
        import ttnn
        from tt_bio import af2
        self.af2, self.fixed = af2, af2.AF2EvoformerBlock._mask_biases
        ns = {}
        exec(_NOFIX, af2.__dict__, ns)
        self.nofix = ns["_mask_biases"]
        n = seq.numel()
        up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                                       device=dm._device, dtype=ttnn.bfloat16)
        ones = (dev.up(torch.ones(1, n)), (up(torch.ones(1, n, n)), up(torch.zeros(1, 1, 1, n))))
        padded = (dev.up(seq[None]), af2.af2_pair_masks(seq[:, None] * seq[None, :], dm._device))
        assert padded[1][0] is not None, "the pad mask took the all-ones path"
        self.spec = {"tree": (w_full, None), "tree+ones": (w_full, ones),
                     "tree+ones@nofix": (w_full, ones), "tree+mask": (w_mask, padded)}

    def use(self, name):
        self.af2.AF2EvoformerBlock._mask_biases = (self.nofix if name.endswith("@nofix")
                                                   else self.fixed)
        return self.spec[name]


RECYCLE = [True]


def step(dm, dev, ref_bf16, logits, ridx, w, masks, ke, kv):
    """One untaped forward and one taped gradient step. `masks` is None or
    (msa_mask_tt, pair_masks)."""
    ag = dev.ag
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
        import ttnn
        with torch.no_grad():
            m0, z0 = A.embed(ref_bf16, logits.float(), ridx)
        mr, zr = dev.up(m0), dev.up(z0)
        dev.sync()
        r0, rc0 = time.time(), time.process_time()
        mr, zr = stack(mr, zr, ckpt=False)
        dev.sync()
        r1, rc1 = time.time(), time.process_time()
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
    A._float64_layernorm()
    from tt_bio.af2 import load_af2_device_model
    from tt_bio.af2_reference import load_af2_model
    from tt_bio.af2_weights import load_af2_state_dict
    state = load_af2_state_dict(args.params)
    dm = load_af2_device_model(state, template=False, trunk_dtype=torch.bfloat16)
    ref_bf16 = load_af2_model(state, template=False, trunk_dtype=torch.bfloat16)
    for p in ref_bf16.parameters():
        p.requires_grad_(False)
    dev = A.Dev(dm)
    clock = Clock()
    n, ke, kv = args.n, args.extra, args.evo
    logits, ridx, w_full, w_mask, seq = draws(n, args.seed, args.pad)
    arms = Arms(dm, dev, seq, w_full, w_mask)
    names = args.arms.split(",")
    blob = {"stamp": A.stamp(args.card), "aiclk_node": clock.path, "pci": clock.pci,
            "n": n, "pad": args.pad, "k_extra": ke, "k_evo": kv, "ckpt": True, "seed": args.seed,
            "arms": names, "reps": args.reps, "warm": {}, "per_arm": {}, "runs": {}}
    grads, runs = {a: [] for a in names}, {a: [] for a in names}
    for a in names:                                   # warm: JIT + program cache, discarded
        _, r = step(dm, dev, ref_bf16, logits, ridx, *arms.use(a), ke, kv)
        r["aiclk"] = clock.window(r.pop("spans"))
        blob["warm"][a] = r
        print("warm", a, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()},
              flush=True)
    for rep in range(args.reps):
        order = names[rep % len(names):] + names[:rep % len(names)]
        if rep % 2:
            order = order[::-1]
        for a in order:
            g, r = step(dm, dev, ref_bf16, logits, ridx, *arms.use(a), ke, kv)
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
    blob["runs"] = runs
    blob["partial"] = partial
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.out or f"run_n{args.n}_s{args.seed}"
    (OUT / f"{tag}.json").write_text(json.dumps(blob, indent=1, default=str))
    torch.save(grads, OUT / f"{tag}_grads.pt")
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
    """Each arm against ITS OWN float64 function: the all-ones arms against `ones`, the padded
    arm against `mask` on its real rows. Device arms are compared to each other only as a
    diagnostic, never as the grade."""
    tag = args.out or f"run_n{args.n}_s{args.seed}"
    blob = json.loads((OUT / f"{tag}.json").read_text())
    grads = torch.load(OUT / f"{tag}_grads.pt")
    ref = torch.load(OUT / f"ref_n{args.n}_s{args.seed}_p{args.pad}.pt")
    real = slice(0, args.n - args.pad)
    grade = {}
    for a, gs in grads.items():
        masked = a == "tree+mask"
        r64, rows = (ref["mask"], real) if masked else (ref["ones"], slice(None))
        grade[a] = [A.cmp(g[rows], r64[rows]) if torch.isfinite(g).all() else None for g in gs]
        if masked:
            grade[a + ":pad_rows_grad_norm"] = [float(g[args.n - args.pad:].norm()) for g in gs]
    for a in ("tree+ones", "tree+ones@nofix"):
        if a in grads and "tree" in grads:
            grade[f"{a}_vs_tree_rep0"] = A.cmp(grads[a][0], grads["tree"][0])
    grade["f64_ones_vs_mask_real_rows"] = A.cmp(ref["mask"][real], ref["ones"][real])
    blob["grade_vs_f64"] = grade
    (OUT / f"{tag}_graded.json").write_text(json.dumps(blob, indent=1, default=str))
    print(json.dumps(grade, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "ref", "grade"])
    ap.add_argument("--params", default=A.DEFAULT_PARAMS)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES") or 0))
    ap.add_argument("--arms", default="tree,tree+ones,tree+ones@nofix,tree+mask")
    ap.add_argument("--ref-arms", default="ones,mask")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--pad", type=int, default=19)
    ap.add_argument("--extra", type=int, default=4)
    ap.add_argument("--evo", type=int, default=48)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-recycle", action="store_true")
    args = ap.parse_args()
    RECYCLE[0] = not args.no_recycle
    {"run": cmd_run, "ref": cmd_ref, "grade": cmd_grade}[args.cmd](args)


if __name__ == "__main__":
    main()
