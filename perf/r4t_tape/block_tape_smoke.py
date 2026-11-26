#!/usr/bin/env python3
"""Does a taped RFD4 TrunkBlock / PairUpdate run forward AND backward, and does every weight get a gradient?

The audit that finds the next missing VJP: a verb or kernel with no tape entry raises (loud by design),
so the first exception names it. Runs under the production lever set (`trunk.mode_levers()`), so every
fused kernel inference uses is reached and has to decline or carry an entry. Weights are registered with
`autograd.parameter` off `walk_device_weights`, the way the OF3 adapter registers them. Not a gradcheck:
the per-tensor grade against float64 is `block_gradcheck.py`, once this runs clean.
"""
from __future__ import annotations

import argparse, json, sys, traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tests")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--real", type=int, default=96)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    import rfd4_reference as REF
    from tt_bio import autograd as ag, taped_ttnn as T, tenstorrent as tt
    from tt_bio.rfd4 import trunk as TR
    dev = tt.get_device()
    DIM, DP, DC, H = 768, 384, 256, 12

    def up(t):
        return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def scope(w, p=""):
        s = tt.WeightScope({k: v.to(torch.float32) for k, v in w.items()})
        return s.child(p) if p else s

    x, z, cond, mask = REF.block_inputs(1, a.n, DIM, DP, DC, lengths=[a.real], seed=1)
    pair = mask[:, :, None] * mask[:, None, :]
    bias_mask = torch.where(pair.bool(), 0.0, REF.NEG_INF)[:, None]
    g = torch.Generator().manual_seed(11)
    rep = {"n": a.n, "real": a.real, "levers": sorted(TR.mode_levers()), "modules": {}}
    builds = {
        "trunk_block": (lambda: TR.TrunkBlock(scope(REF.block_weights(DIM, DP, DC, H, seed=0)),
                                              TR.compute_kernel_config(), H),
                        lambda m, L: m(L["x"], L["z"], L["cond"], L["mask"], L["bias"]), ("x", "z", "cond")),
        "pair_update": (lambda: TR.PairUpdate(scope(REF.pair_update_weights(DIM, DP, 256, seed=5, prefix="pu"), "pu"),
                                              TR.compute_kernel_config()),
                        lambda m, L: m(L["x"], L["z"], L["mask"], L["pm"], L["pmc"]), ("x", "z")),
    }
    for name, (build, call, diff) in builds.items():
        r = rep["modules"][name] = {}
        try:
            with TR.lever_scope(TR.mode_levers()):
                m = build()
                raw = {"x": up(x), "z": up(z), "cond": up(cond), "mask": up(mask[..., None]), "bias": up(bias_mask),
                       "pm": up(pair), "pmc": up(pair[..., None])}
                ref_out = ttnn.to_torch(call(m, raw)).double()
                walked = {p: t for p, _o, _k, t in tt.walk_device_weights(m)}
                leaves = {p: ag.parameter(t) for p, t in walked.items()}
                L = {k: (ag.Tensor(v, requires_grad=True) if k in diff else v) for k, v in raw.items()}
                with T.tape():
                    out = call(m, L)
                got = ttnn.to_torch(out.value).double()
                r["fwd_taped_vs_untaped_bit_exact"] = bool(torch.equal(got, ref_out))
                r["fwd_taped_vs_untaped_max_abs"] = float((got - ref_out).abs().max())
                seed = torch.randn(tuple(got.shape), generator=g)
                out.backward(seed=up(seed))
                zero = [p for p, l in leaves.items()
                        if l.grad is None or float(ttnn.to_torch(l.grad).double().abs().max()) == 0.0]
                r["params"] = len(leaves)
                r["params_zero_or_no_grad"] = zero
                r["inputs_with_grad"] = [k for k in diff if L[k].grad is not None]
                r["ok"] = not zero and len(r["inputs_with_grad"]) == len(diff)
        except Exception as e:  # the audit's whole point: the first raise names the gap
            r["ok"] = False
            r["error"] = f"{type(e).__name__}: {e}"[:2000]
            r["trace"] = traceback.format_exc()[-3000:]
        print(name, json.dumps({k: v for k, v in r.items() if k != "trace"})[:1500], flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=1))
    return 0 if all(r.get("ok") for r in rep["modules"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
