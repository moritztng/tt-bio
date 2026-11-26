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

import argparse, json, sys
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
    ap.add_argument("--break", dest="brk", action="store_true")
    ap.add_argument("--arm", choices=("prod", "bf16", "fp32"), default="prod")
    ap.add_argument("--module", choices=("trunk_block", "pair_update"), default="trunk_block")
    ap.add_argument("--sdpa-own", action="store_true",
                    help="tape SDPA with its own chunked forward (fp32-capable) instead of the fused kernel")
    ap.add_argument("--out", type=Path)
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
    dev = tt.get_device()
    DIM, DP, DC, H = 768, 384, 256, 12
    bf = lambda t: t.to(torch.bfloat16).to(torch.float64)  # noqa: E731

    W8 = {"tok_w8", "transition_b8", "trimul_gin_w8", "trimul_t1w8", "pair_bf16z", "pair_bfp8z"}
    levers = set(TR.mode_levers())
    if a.arm != "prod":
        levers = {v for v in levers if v not in W8 and not v.startswith("pair_x8")}
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
        ref_fn = lambda w, I: REF.trunk_block(w, I["x"], I["z"], I["cond"], mask, n_heads=H)  # noqa: E731
        out_mask = mask[..., None]
    else:
        w64 = REF.pair_update_weights(DIM, DP, 256, seed=5, prefix="pu")
        build = lambda sd: TR.PairUpdate(tt.WeightScope(sd).child("pu"), ckc, dtype=dt)  # noqa: E731
        keys, ins, out_t, cut = ("x", "z", "mask", "pm", "pmc"), ("x", "z"), z, "xi_w"
        ref_fn = lambda w, I: REF.pair_update(w, "pu", I["x"], I["z"], mask)  # noqa: E731
        out_mask = pair[..., None]
        z = z * out_mask  # the pair stream arrives masked, as the trunk passes it
    w64 = {k: bf(v.double()) for k, v in w64.items()}
    seed = torch.randn(tuple(out_t.shape), generator=torch.Generator().manual_seed(a.seed), dtype=torch.float64)
    seed = seed * out_mask
    I0 = {"x": x, "z": z, "cond": cond}

    f = Folds({k: v.float() for k, v in w64.items()})
    with TR.lever_scope(levers):
        raw = {"x": up(x), "z": up(z), "cond": up(cond), "mask": up(mask[..., None]), "bias": up(bias_mask),
               "pm": up(pair), "pmc": up(pair[..., None])}
        with f.recording() as sd:
            m = build(sd)
            # lazy uploads, as served. The fused SDPA refuses fp32, so the fp32 arm warms on the tape
            # (a verb dispatches to its tape entry only when an operand is a taped Tensor).
            if a.arm == "fp32":
                with T.tape():
                    m(*(ag.Tensor(raw[k], requires_grad=True) if k in ins else raw[k] for k in keys))
            else:
                m(*(raw[k] for k in keys))
            L = {k: (ag.Tensor(v, requires_grad=True) if k in ins else v) for k, v in raw.items()}
            for _p, _o, _k, t in tt.walk_device_weights(m):
                ag.parameter(t)
            with T.tape():
                out = m(*(L[k] for k in keys))
            for _p, _o, _k, t in tt.walk_device_weights(m):  # a declined fused path can upload more
                ag.parameter(t)
        walked = {p: t for p, _o, _k, t in tt.walk_device_weights(m)}
        out.backward(seed=up(seed))
        dropped = None
        if a.brk:
            dropped = next(p for p in walked if p.endswith(cut))
            f.sources.pop(id(walked[dropped]))
        f.tape_backward()
    got = {n: v.grad.double().numpy() for n, v in f.leaves.items()}
    for k in ins:
        got[k] = ttnn.to_torch(L[k].grad).double().reshape(I0[k].shape).numpy()
    ref = {k: v.clone().requires_grad_(True) for k, v in w64.items()}
    ref.update({k: I0[k].clone().requires_grad_(True) for k in ins})
    names = list(got)
    wref = {k: ref[k] for k in w64}
    dname = "float32" if a.arm == "fp32" else "bfloat16"
    rep = CK.gradcheck(f"{a.module}/{a.arm}" + ("/own" if a.sdpa_own else "") + ("/broken" if a.brk else ""),
                       ref_loss=lambda: (ref_fn(wref, ref) * seed).sum(),
                       ref_params=[ref[k] for k in names], tt_grads={k: got[k] for k in names},
                       op_class="reduction", dtype=dname, n_probe=24, seed=a.seed)
    print(rep, flush=True)
    worst = max(rep.grads.items(), key=lambda kv: kv[1]["rel_l2"])
    res = {"module": a.module, "arm": a.arm, "sdpa_own": a.sdpa_own or a.arm == "fp32", "levers": sorted(levers), "break": a.brk, "dropped": dropped, "seed": a.seed, "n": a.n, "real": a.real, "pass": rep.passed,
           "fd_worst": rep.fd_worst, "worst": [worst[0], worst[1]["rel_l2"]],
           "grads": {k: {q: g[q] for q in ("rel_l2", "cos", "pass")} for k, g in rep.grads.items()},
           "output_digest": __import__("hashlib").sha256(ttnn.to_torch(out.value).float().numpy().tobytes()).hexdigest()[:16]}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("module", "arm", "pass", "worst", "fd_worst", "dropped", "output_digest")}))
    print("ALL_PASS" if rep.passed else "SOME_FAIL")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
