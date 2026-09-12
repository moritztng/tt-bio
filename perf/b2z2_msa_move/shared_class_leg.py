#!/usr/bin/env python3
"""`PairWeightedAveraging` under the L1 row block, at every shape its THREE callers present.

The row block ships default on and the class is shared: boltz2 builds it in `MSALayer`,
protenix-v2 at `protenix.py:2499` and openfold3 at `openfold3_msa_embedder.py:79`, all three
with the same `head_dim=8, n_heads=8, c_m=64` and only depth, tokens and c_z differing. The
Boltz-2 parity in `parity_default_wh_c9.json` therefore covers one point of a three-caller
surface, and the gate rule is that a lever is gated on a SHAPE property and so has to be shown
on the shapes, not on the model that happened to be measured.

Each case runs the class twice from cloned inputs, `_PWA_L1_ROWS = -1` (the pre-lever single
pass) against `0` (derive), and compares with `torch.equal`. It also records the block the rule
picks, so a case where the rule declines to block is visible as such rather than passing as a
vacuous bit-exact match -- a check that cannot fail is not a check.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

# (label, depth, tokens, c_z). Depth is padded MSA rows, tokens is the padded token axis.
# The boltz2 row is the one shape already covered end to end, kept as the positive control:
# if it does not reproduce 1024 -> 512 here, this harness is not exercising the same rule.
CASES = [
    ("boltz2-512aa-control", 1024, 512, 128),
    ("protenix-v2-cz128",     512, 512, 128),
    ("protenix-v2-cz256",    1024, 384, 256),
    ("protenix-v2-cz384",     256, 512, 384),
    ("openfold3-512aa",      1024, 512, 128),
    ("openfold3-wide",        512, 768, 128),
    ("small-no-block",         64, 256, 128),
]
C_M, C_H, N_HEADS = 64, 8, 8


def weights(torch, c_z):
    """One deterministic set of PWA weights. Values do not matter; shapes and exactness do."""
    g = torch.Generator().manual_seed(0)

    def r(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32) * 0.05
    return {
        "norm_m.weight": torch.ones(C_M), "norm_m.bias": torch.zeros(C_M),
        "norm_z.weight": torch.ones(c_z), "norm_z.bias": torch.zeros(c_z),
        "proj_m.weight": r(C_H * N_HEADS, C_M),
        "proj_g.weight": r(C_H * N_HEADS, C_M),
        "proj_z.weight": r(N_HEADS, c_z),
        "proj_o.weight": r(C_M, C_H * N_HEADS),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T

    dev = T.get_device(trace_region_size=512 << 20)
    kernel_cls = (ttnn.types.WormholeComputeKernelConfig
                  if dev.arch() == ttnn.Arch.WORMHOLE_B0
                  else ttnn.types.BlackholeComputeKernelConfig)
    ckc = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                     fp32_dest_acc_en=True, packer_l1_acc=True)

    out = {"env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "arch": str(dev.arch()), "grid": list(T.COMPUTE_GRID_MAIN),
                   "l1_budget_bytes": T._l1_budget_bytes(T._PAIR_L1_CONSUMER_RESERVE)},
           "cases": []}

    for label, depth, tokens, c_z in CASES:
        blk = T.pwa_l1_row_block(depth, tokens, C_M)
        rec = {"label": label, "depth": depth, "tokens": tokens, "c_z": c_z,
               "row_block": blk, "blocked": blk < depth}
        try:
            mod = T.PairWeightedAveraging(C_H, N_HEADS, weights(torch, c_z), ckc)
            # Both operands carry a leading 1: `__call__` strips one dim off each before
            # it does anything else, so a 3-D `m` fails a volume check rather than running.
            m_t = torch.randn(1, depth, tokens, C_M, dtype=torch.float32) * 0.3
            z_t = torch.randn(1, tokens, tokens, c_z, dtype=torch.float32) * 0.3

            def run(rows):
                keep, T._PWA_L1_ROWS = T._PWA_L1_ROWS, rows
                T._L1_OUT_RUNG.clear()
                try:
                    m = ttnn.from_torch(m_t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                                        device=dev)
                    z = ttnn.from_torch(z_t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                                        device=dev)
                    o = mod(m, z)
                    r = ttnn.to_torch(o).clone()
                    for t in (o, z, m):
                        ttnn.deallocate(t)
                    return r
                finally:
                    T._PWA_L1_ROWS = keep

            off = run(-1)
            aa = run(-1)
            on = run(0)
            rec["equal_AA"] = bool(torch.equal(off, aa))
            rec["equal_off_vs_on"] = bool(torch.equal(off, on))
            rec["negative_control_must_be_False"] = bool(torch.equal(off, off + 1))
            rec["max_abs_diff"] = float((off.float() - on.float()).abs().max())
            rec["out_shape"] = list(off.shape)
        except Exception as e:                      # a refusal is a result, not a crash
            rec["error"] = f"{type(e).__name__}: {e}"[:400]
        out["cases"].append(rec)
        print(f"  {label:22s} depth {depth:5d} tok {tokens:4d} c_z {c_z:3d} -> block {blk:5d}"
              f"  {'BLOCKED' if rec['blocked'] else 'whole  '}"
              f"  equal {rec.get('equal_off_vs_on')}  maxdiff {rec.get('max_abs_diff')}"
              f"  {rec.get('error','')}", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))

    cases = out["cases"]
    errs = [c["label"] for c in cases if "error" in c]
    # `all(...)` over a list an error filter has emptied is True, which is how a harness that
    # ran nothing reports a clean pass. Every clause below has to be independently true.
    out["summary"] = {
        "cases": len(cases),
        "blocked_cases": len([c for c in cases if c.get("blocked")]),
        "errors": errs,
        "all_bit_exact": (not errs
                          and len(cases) > 0
                          and all(c.get("equal_off_vs_on") for c in cases)
                          and all(c.get("equal_AA") for c in cases)
                          and not any(c.get("negative_control_must_be_False") for c in cases))}
    a.out.write_text(json.dumps(out, indent=1))
    print("SUMMARY", json.dumps(out["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
