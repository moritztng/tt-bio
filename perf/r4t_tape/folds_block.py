#!/usr/bin/env python3
"""Fold coverage on a taped RFD4 TrunkBlock / PairUpdate: does every device weight trace back to upstream
parameters, and does every upstream parameter get a gradient through the folds?

Builds each module under `train.folds.Folds.recording` (the build, the untaped first forward that makes the
lazy uploads, and the taped forward, which can upload more on a declined path), registers every walked
device weight as a tape leaf, runs one backward and pushes the device gradients through the recorded folds
with `Folds.tape_backward`. Reports orphans (device weights with no upstream source), unreached upstream
parameters, device weights the taped forward never read (duplicates for a declined fused kernel) and
upstream parameters left without a gradient. Not a gradcheck: the float64 grade is the block gradcheck.
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
    from tt_bio.train.folds import Folds
    dev = tt.get_device()
    DIM, DP, DC, H = 768, 384, 256, 12

    def up(t):
        return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    x, z, cond, mask = REF.block_inputs(1, a.n, DIM, DP, DC, lengths=[a.real], seed=1)
    pair = mask[:, :, None] * mask[:, None, :]
    bias_mask = torch.where(pair.bool(), 0.0, REF.NEG_INF)[:, None]
    g = torch.Generator().manual_seed(11)
    rep = {"n": a.n, "real": a.real, "levers": sorted(TR.mode_levers()), "modules": {}}
    mods = {
        "trunk_block": (REF.block_weights(DIM, DP, DC, H, seed=0), "",
                        lambda s: TR.TrunkBlock(s, TR.compute_kernel_config(), H),
                        lambda m, L: m(L["x"], L["z"], L["cond"], L["mask"], L["bias"])),
        "pair_update": (REF.pair_update_weights(DIM, DP, 256, seed=5, prefix="pu"), "pu",
                        lambda s: TR.PairUpdate(s, TR.compute_kernel_config()),
                        lambda m, L: m(L["x"], L["z"], L["mask"], L["pm"], L["pmc"])),
    }
    for name, (weights, prefix, build, call) in mods.items():
        r = rep["modules"][name] = {}
        try:
            f = Folds(weights)
            with TR.lever_scope(TR.mode_levers()):
                raw = {"x": up(x), "z": up(z), "cond": up(cond), "mask": up(mask[..., None]), "bias": up(bias_mask),
                       "pm": up(pair), "pmc": up(pair[..., None])}
                with f.recording() as sd:
                    s = tt.WeightScope(sd)
                    m = build(s.child(prefix) if prefix else s)
                    call(m, raw)
                    walked = {p: t for p, _o, _k, t in tt.walk_device_weights(m)}
                    leaves = {p: ag.parameter(t) for p, t in walked.items()}
                    L = {k: (ag.Tensor(v, requires_grad=True) if k in ("x", "z", "cond") else v)
                         for k, v in raw.items()}
                    with T.tape():
                        out = call(m, L)
                    walked = {p: t for p, _o, _k, t in tt.walk_device_weights(m)}
                    for p, t in walked.items():
                        leaves.setdefault(p, ag.parameter(t))
                got = ttnn.to_torch(out.value)
                out.backward(seed=up(torch.randn(tuple(got.shape), generator=g)))
                idle = {id(h) for h in f.tape_backward()}
                r["device_weights"] = len(walked)
                r["recorded_sources"] = len(f.sources)
                r["orphans"] = f.orphans(walked)
                r["constants"] = f.constants(walked)
                r["unread_under_tape"] = sorted(p for p, t in walked.items() if id(t) in idle)
                r["upstream_params"] = len(f.leaves)
                r["unreached_upstream"] = f.unreached()
                r["upstream_without_grad"] = sorted(n for n, v in f.leaves.items()
                                                    if v.grad is None or float(v.grad.abs().max()) == 0.0)
                r["ok"] = not r["orphans"] and not r["unreached_upstream"] and not r["upstream_without_grad"]
        except Exception as e:
            r["ok"] = False
            r["error"] = f"{type(e).__name__}: {e}"[:2000]
            r["trace"] = traceback.format_exc()[-3000:]
        print(name, json.dumps({k: v for k, v in r.items() if k != "trace"})[:3000], flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=1))
    return 0 if all(r.get("ok") for r in rep["modules"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
