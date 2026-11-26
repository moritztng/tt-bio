#!/usr/bin/env python3
"""Gradcheck of a taped RFD4 TrunkBlock in UPSTREAM parameter space, against float64.

The device block runs taped under the production lever set; its weights are built by
`train.folds.Folds`, so the gradient read back is the one the optimizer owns: per upstream parameter
(`to_qkv.weight` whole, `norm_cond.weight` through the AdaLN fold), plus the inputs x, z, cond. The
reference is `tests/rfd4_reference.trunk_block` (upstream's `MultiheadAttnAndTransition.forward`) in
float64 at the same bf16-rounded point, validated against central differences by `checks.gradcheck`.
`--break` drops the Q slice of the `to_qkv` split from the fold backward, which must FAIL.
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
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
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

    def up(t):
        return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    w64 = {k: bf(v.double()) for k, v in REF.block_weights(DIM, DP, DC, H, seed=0).items()}
    x, z, cond, mask = (bf(t) for t in REF.block_inputs(1, a.n, DIM, DP, DC, lengths=[a.real], seed=1))
    pair = mask[:, :, None] * mask[:, None, :]
    bias_mask = torch.where(pair.bool(), 0.0, REF.NEG_INF)[:, None]
    seed = torch.randn(1, a.n, DIM, generator=torch.Generator().manual_seed(a.seed), dtype=torch.float64)

    f = Folds({k: v.float() for k, v in w64.items()})
    with TR.lever_scope(TR.mode_levers()):
        raw = {"x": up(x), "z": up(z), "cond": up(cond), "mask": up(mask[..., None]), "bias": up(bias_mask)}
        with f.recording() as sd:
            m = TR.TrunkBlock(tt.WeightScope(sd), TR.compute_kernel_config(), H)
            m(raw["x"], raw["z"], raw["cond"], raw["mask"], raw["bias"])  # lazy uploads, as served
            L = {k: (ag.Tensor(v, requires_grad=True) if k in ("x", "z", "cond") else v) for k, v in raw.items()}
            for _p, _o, _k, t in tt.walk_device_weights(m):
                ag.parameter(t)
            with T.tape():
                out = m(L["x"], L["z"], L["cond"], L["mask"], L["bias"])
        walked = {p: t for p, _o, _k, t in tt.walk_device_weights(m)}
        out.backward(seed=up(seed))
        dropped = None
        if a.brk:
            dropped = next(p for p in walked if p.endswith("q_w"))
            f.sources.pop(id(walked[dropped]))
        f.tape_backward()
    got = {n: v.grad.double().numpy() for n, v in f.leaves.items()}
    for k, t in (("x", x), ("z", z), ("cond", cond)):
        got[k] = ttnn.to_torch(L[k].grad).double().reshape(t.shape).numpy()
    ref = {k: v.clone().requires_grad_(True) for k, v in w64.items()}
    ref.update({k: t.clone().requires_grad_(True) for k, t in (("x", x), ("z", z), ("cond", cond))})
    names = list(got)
    wref = {k: ref[k] for k in w64}
    rep = CK.gradcheck("trunk_block/bfloat16" + ("/broken" if a.brk else ""),
                       ref_loss=lambda: (REF.trunk_block(wref, ref["x"], ref["z"], ref["cond"], mask, n_heads=H)
                                         * seed).sum(),
                       ref_params=[ref[k] for k in names], tt_grads={k: got[k] for k in names},
                       op_class="reduction", dtype="bfloat16", n_probe=24, seed=a.seed)
    print(rep, flush=True)
    worst = max(rep.grads.items(), key=lambda kv: kv[1]["rel_l2"])
    res = {"break": a.brk, "dropped": dropped, "seed": a.seed, "n": a.n, "real": a.real, "pass": rep.passed,
           "fd_worst": rep.fd_worst, "worst": [worst[0], worst[1]["rel_l2"]],
           "grads": {k: {q: g[q] for q in ("rel_l2", "cos", "pass")} for k, g in rep.grads.items()},
           "output_digest": __import__("hashlib").sha256(ttnn.to_torch(out.value).float().numpy().tobytes()).hexdigest()[:16]}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("pass", "worst", "fd_worst", "dropped", "output_digest")}))
    print("ALL_PASS" if rep.passed else "SOME_FAIL")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
