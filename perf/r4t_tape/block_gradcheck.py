#!/usr/bin/env python3
"""Gradcheck of a taped RFD4 TrunkBlock in UPSTREAM parameter space, against float64.

The device block runs taped under the production lever set; its weights are built by
`train.folds.Folds`, so the gradient read back is the one the optimizer owns: per upstream parameter
(`to_qkv.weight` whole, `norm_cond.weight` through the AdaLN fold), plus the inputs x, z, cond. The
reference is `tests/rfd4_reference.trunk_block` (upstream's `MultiheadAttnAndTransition.forward`) in
float64 at the same bf16-rounded point, validated against central differences by `checks.gradcheck`.
`--break` drops the Q slice of the `to_qkv` split from the fold backward, which must FAIL.

Modules: `trunk_block` (default) and `pair_update` (`--module`, upstream `PairReprUpdate`; `--break` drops the xi split of `linear_x`).

Arms: `prod` = the shipped lever set (bfp8 token weights under `tok_w8`/`transition_b8`); `bf16` = the same
without the levers that STORE a trainable weight below bf16 (AdamW refuses a bfp8 leaf, T9); `fp32` = that,
with the block in float32, which must show the error collapse.

The device block takes the residual stream already masked and does not re-mask its output (the sublayers
mask their own updates), where upstream masks x on entry and exit. The two VJPs agree for a seed that is
zero on padded tokens, which a masked loss gives; so x and the seed are masked here, as RFD4 trains.
"""
from __future__ import annotations

import argparse, contextlib, json, sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tests")]
from tt_bio.train import checks as CK  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--real", type=int, default=96)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--probe", type=int, default=24, help="central-difference coords per tensor")
    ap.add_argument("--break", dest="brk", action="store_true")
    ap.add_argument("--scale-grad", metavar="PARAM",
                    help="subtle broken arm: multiply this parameter's device gradient by 1.10 (D28)")
    ap.add_argument("--arm", choices=("prod", "bf16", "fp32"), default="prod")
    ap.add_argument("--module", choices=("trunk_block", "pair_update", "trunk"), default="trunk_block")
    ap.add_argument("--layers", type=int, default=3, help="trunk: blocks (a pair update after block 0)")
    ap.add_argument("--cycles", type=int, default=2, help="trunk: recycles; only the last is differentiated")
    ap.add_argument("--sdpa-own", action="store_true",
                    help="tape SDPA with its own chunked forward (fp32-capable) instead of the fused kernel")
    ap.add_argument("--levers", choices=("arm", "none"), default="arm",
                    help="none: every mode lever off (the plain ttnn path), to locate a forward disagreement")
    ap.add_argument("--fwd-only", action="store_true", help="stop after the forward-vs-reference line")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--floor-only", type=Path, metavar="JSON",
                    help="host only: write the bf16-emulated floor column (D39) for this config and stop")
    ap.add_argument("--floor", type=Path, metavar="JSON",
                    help="grade FLOOR_SITES on 1.5x this committed floor column (refused if it does not reproduce)")
    a = ap.parse_args()
    if a.sdpa_own or a.arm == "fp32":  # the fused kernel refuses fp32 operands
        import os
        os.environ["TT_BIO_SDPA_OWN_FORWARD"] = "1"
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    import rfd4_reference as REF
    from tt_bio import autograd as ag, taped_ttnn as T, tenstorrent as tt
    from tt_bio.rfd4 import trunk as TR
    from tt_bio.train.folds import Folds
    DIM, DP, DC, H = 768, 384, 256, 12
    bf = lambda t: t.to(torch.bfloat16).to(torch.float64)  # noqa: E731

    W8 = {"tok_w8", "transition_b8", "trimul_gin_w8", "trimul_t1w8", "pair_bf16z", "pair_bfp8z"}
    levers = set(TR.mode_levers())
    if a.arm != "prod":
        levers = {v for v in levers if v not in W8 and not v.startswith("pair_x8")}
    if a.levers == "none":
        levers = set()
    dt = ttnn.float32 if a.arm == "fp32" else ttnn.bfloat16
    if a.arm == "fp32":
        bf = lambda t: t  # noqa: E731

    def up(t):
        return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    x, z, cond, mask = (bf(t) for t in REF.block_inputs(1, a.n, DIM, DP, DC, lengths=[a.real], seed=1))
    pair = mask[:, :, None] * mask[:, None, :]
    bias_mask = torch.where(pair.bool(), 0.0, REF.NEG_INF)[:, None]
    x = x * mask[..., None]
    ckc = TR.compute_kernel_config()
    if a.module == "trunk_block":
        w64 = REF.block_weights(DIM, DP, DC, H, seed=0)
        build = lambda sd: TR.TrunkBlock(tt.WeightScope(sd), ckc, H, dtype=dt)  # noqa: E731
        keys, ins, out_t, cut = ("x", "z", "cond", "mask", "bias"), ("x", "z", "cond"), x, "q_w"
        ref_fn = lambda w, I, r=REF.ident: REF.trunk_block(w, I["x"], I["z"], I["cond"], mask, n_heads=H, rnd=r)  # noqa: E731
        out_mask = mask[..., None]
    elif a.module == "trunk":
        # The whole trunk as training runs it: every recycle executes, only the last is differentiated
        # (`ops.recycle_region`; upstream `grad: last`). Seeded on s, which every parameter reaches.
        w64 = REF.trunk_weights(a.layers, DIM, DP, DC, H, seed=11)
        gc = torch.Generator().manual_seed(13)
        early = [bf(torch.randn(1, a.n, DC, generator=gc, dtype=torch.float64)) for _ in range(a.cycles - 1)]
        cond = bf(torch.randn(1, a.n, DC, generator=gc, dtype=torch.float64))
        build = lambda sd: TR.Trunk(tt.WeightScope(sd), ckc, a.layers, H, dtype=dt)  # noqa: E731
        keys, ins, out_t, cut = ("x", "z", "cond"), ("x", "z", "cond"), x, "q_w"
        # The early cycles run once at the base point and are frozen: that is what `grad: last`
        # differentiates, and a central difference sees exactly it (through `grad_last` alone it would
        # also see every earlier cycle move, 0.98 off at n=64).
        frozen = REF.trunk({k: bf(v.double()) for k, v in w64.items()}, x, z * pair[..., None], early, mask,
                           a.layers, n_heads=H)
        ref_fn = lambda w, I, r=REF.ident: REF.trunk(w, I["x"], I["z"], [I["cond"]], mask, a.layers,  # noqa: E731
                                                      n_heads=H, state=frozen, rnd=r)[0]
        out_mask = mask[..., None]
        z = z * pair[..., None]
    else:
        w64 = REF.pair_update_weights(DIM, DP, 256, seed=5, prefix="pu")
        build = lambda sd: TR.PairUpdate(tt.WeightScope(sd).child("pu"), ckc, dtype=dt)  # noqa: E731
        keys, ins, out_t, cut = ("x", "z", "mask", "pm", "pmc"), ("x", "z"), z, "xi_w"
        ref_fn = lambda w, I, r=REF.ident: REF.pair_update(w, "pu", I["x"], I["z"], mask, r)  # noqa: E731
        out_mask = pair[..., None]
        z = z * out_mask  # the pair stream arrives masked, as the trunk passes it
    w64 = {k: bf(v.double()) for k, v in w64.items()}
    seed = torch.randn(tuple(out_t.shape), generator=torch.Generator().manual_seed(a.seed), dtype=torch.float64)
    seed = seed * out_mask
    I0 = {"x": x, "z": z, "cond": cond}
    ins_ = [k for k in ins if k in I0]
    cfg = {"module": a.module, "n": a.n, "real": a.real, "seed": a.seed, "layers": a.layers, "cycles": a.cycles}

    def bf16_floor():
        """{tensor: rel, cos} of the float64 gradient of the reference with every op's output rounded to bf16
        (backward exact) against the plain float64 gradient: the floor any bf16 forward sits on."""
        ref = {k: v.clone().requires_grad_(True) for k, v in w64.items()}
        ref.update({k: I0[k].clone().requires_grad_(True) for k in ins_})
        wref, names = {k: ref[k] for k in w64}, list(ref)
        with torch.enable_grad():
            g64 = torch.autograd.grad((ref_fn(wref, ref) * seed).sum(), [ref[k] for k in names], allow_unused=True)
            gem = torch.autograd.grad((ref_fn(wref, ref, REF.bf16) * seed).sum(), [ref[k] for k in names],
                                      allow_unused=True)
        out = {}
        for k, u, v in zip(names, g64, gem):
            if u is not None and v is not None and float(u.norm()) > 0:
                m = CK.metrics(v.detach().numpy(), u.detach().numpy())
                out[k] = {"rel": float(m["rel_l2"]), "cos": float(m["cos"])}
        return out

    def digest(fl):
        import hashlib
        return hashlib.sha256(json.dumps(fl, sort_keys=True).encode()).hexdigest()[:16]

    if a.floor_only:
        fl = bf16_floor()
        a.floor_only.parent.mkdir(parents=True, exist_ok=True)
        a.floor_only.write_text(json.dumps({"config": cfg, "digest": digest(fl), "floor": fl}, indent=1, sort_keys=True))
        sites = {k: v for k, v in fl.items() if CK.floor_site(k)}
        print(f"floor column {a.floor_only} digest {digest(fl)}: {len(fl)} tensors; sites {sites}")
        return 0
    committed = None
    if a.floor:
        doc = json.loads(a.floor.read_text())
        if doc["config"] != cfg or doc["digest"] != digest(doc["floor"]):
            print(f"REFUSED: {a.floor} is for {doc['config']} digest {doc['digest']}, not this run {cfg}")
            return 2
        committed = doc["floor"]
    dev = tt.get_device()

    f = Folds({k: v.float() for k, v in w64.items()})
    # float32 everywhere: the shared trimul computes in the process dtype (`_dtype()`), not the operand
    # dtype, so without the override the fp32 arm ran its triangle multiplications in bf16 on fp32 inputs
    wide = tt.device_dtype_override(ttnn.float32) if a.arm == "fp32" else contextlib.nullcontext()
    with TR.lever_scope(levers), wide:
        def inputs():
            return {"x": up(x), "z": up(z), "cond": up(cond), "mask": up(mask[..., None]), "bias": up(bias_mask),
                    "pm": up(pair), "pmc": up(pair[..., None])}
        raw = inputs()
        if a.module == "trunk":
            tmasks = TR.trunk_masks(mask, dev, dtype=dt, residual_dtype=dt)
            econds = [up(c) for c in early]
            call = lambda m, A: m(A[0], A[1], econds + [A[2]], tmasks)[0]  # noqa: E731
        else:
            call = lambda m, A: m(*A)  # noqa: E731
        with f.recording() as sd:
            m = build(sd)
            # lazy uploads, as served. The fused SDPA refuses fp32, so the fp32 arm warms on the tape
            # (a verb dispatches to its tape entry only when an operand is a taped Tensor).
            if a.arm == "fp32":
                with T.tape():
                    warm = call(m, [ag.Tensor(raw[k], requires_grad=True) if k in ins else raw[k] for k in keys]).value
            else:
                warm = call(m, [raw[k] for k in keys])
            warm = ttnn.to_torch(warm).double()   # now: the graded call may reuse its buffers
            # the warm call may write its pair stream in place (pair_tail): the graded call reads fresh inputs
            warm_moved = {k: float((ttnn.to_torch(raw[k]).double().reshape(v.shape) - v).abs().max())
                          for k, v in (("x", x), ("z", z)) if k in keys}
            raw = inputs()
            L = {k: (ag.Tensor(v, requires_grad=True) if k in ins else v) for k, v in raw.items()}
            for _p, _o, _k, t in tt.walk_device_weights(m):
                ag.parameter(t)
            with T.tape():
                out = call(m, [L[k] for k in keys])
            for _p, _o, _k, t in tt.walk_device_weights(m):  # a declined fused path can upload more
                ag.parameter(t)
        walked = {p: t for p, _o, _k, t in tt.walk_device_weights(m)}
        pre = ttnn.to_torch(out.value).double()   # the forward as it stood before any backward ran
        out.backward(seed=up(seed))
        dropped = None
        if a.brk:
            dropped = next(p for p in walked if p.endswith(cut))
            f.sources.pop(id(walked[dropped]))
        f.tape_backward()
    with torch.no_grad():
        want = ref_fn(w64, I0)
    have = ttnn.to_torch(out.value).double().reshape(want.shape)
    fwd_rel = float((have - want).mul(out_mask).norm() / want.mul(out_mask).norm())
    moved = float((have - pre.reshape(want.shape)).abs().max())
    print(f"backward moved the forward output by {moved:.3e} (nonzero = a backward wrote into a live forward buffer)", flush=True)
    served = warm.reshape(want.shape)
    served_rel = float((served - want).mul(out_mask).norm() / want.mul(out_mask).norm())
    print(f"forward vs float64 reference: taped rel_l2 {fwd_rel:.3e}, warm call {served_rel:.3e}; "
          f"warm call moved its inputs by {warm_moved}; levers {sorted(levers)}", flush=True)
    if a.fwd_only:
        return 0
    # a one-cycle trunk never reads the recycle projections (rms(0) = 0): no gradient, named not graded
    unread = sorted(n for n, v in f.leaves.items() if v.grad is None)
    if unread:
        print(f"no gradient (unread by this config): {unread}", flush=True)
    got = {n: v.grad.double().numpy() for n, v in f.leaves.items() if v.grad is not None}
    if a.scale_grad:
        got[a.scale_grad] = got[a.scale_grad] * 1.10
    for k in ins:
        got[k] = ttnn.to_torch(L[k].grad).double().reshape(I0[k].shape).numpy()
    ref = {k: v.clone().requires_grad_(True) for k, v in w64.items()}
    ref.update({k: I0[k].clone().requires_grad_(True) for k in ins})
    names = list(got)
    wref = {k: ref[k] for k in w64}
    dname = "float32" if a.arm == "fp32" else "bfloat16"
    rep = CK.gradcheck(f"{a.module}/{a.arm}" + ("/own" if a.sdpa_own else "") + ("/broken" if a.brk else "") + (f"/x1.10:{a.scale_grad}" if a.scale_grad else ""),
                       ref_loss=lambda: (ref_fn(wref, ref) * seed).sum(),
                       ref_params=[ref[k] for k in names], tt_grads={k: got[k] for k in names},
                       op_class="reduction", dtype=dname, n_probe=a.probe, seed=a.seed)
    print(rep, flush=True)
    fresh = bf16_floor() if committed is not None else {}
    if any(abs(f["rel"] - committed.get(k, {"rel": -1.0})["rel"]) > 1e-9 * max(f["rel"], 1e-12) for k, f in fresh.items()):
        print(f"REFUSED: the committed floor does not reproduce (digest {digest(fresh)} vs {digest(committed)})")
        return 2
    comp = CK.composed_verdict(rep, committed)
    # The floor any bf16 forward sits on: the float64 gradient of the same reference with every op's output
    # rounded to bf16 (backward exact). Informational; at a ReLU it is the mask flips, not the VJP.
    floor = {}
    if a.arm != "fp32":
        floor = {k: f["rel"] for k, f in (fresh or bf16_floor()).items() if f["rel"] > 0}
        worst = sorted(((rep.grads[k]["rel_l2"] / floor[k], k) for k in floor if k in rep.grads), reverse=True)[:5]
        print("bf16-forward floor (float64 grad of a bf16-rounded forward): max %.3e median %.3e; device/floor worst %s"
              % (max(floor.values()), sorted(floor.values())[len(floor) // 2], [(k, round(r, 2)) for r, k in worst]), flush=True)
    res = {"module": a.module, "arm": a.arm, "sdpa_own": a.sdpa_own or a.arm == "fp32", "levers": sorted(levers),
           "break": a.brk, "scale_grad": a.scale_grad, "dropped": dropped, "seed": a.seed, "n": a.n, "real": a.real,
           "fwd_rel": fwd_rel, "bf16_floor": floor, "warm_moved": warm_moved, "pass_g1": rep.passed, "composed": comp, "fd_worst": rep.fd_worst, "fd_kinked": rep.fd_kinked,
           "grads": {k: {q: g[q] for q in ("rel_l2", "cos", "pass")} for k, g in rep.grads.items()},
           "output_digest": __import__("hashlib").sha256(ttnn.to_torch(out.value).float().numpy().tobytes()).hexdigest()[:16]}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("module", "arm", "pass_g1", "composed", "fd_worst", "fd_kinked", "dropped",
                                          "scale_grad", "output_digest")}))
    print("COMPOSED_PASS" if comp["pass"] else "COMPOSED_FAIL")
    return 0 if comp["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
