#!/usr/bin/env python3
"""Can the head split be folded into the token-DiT qkv matmul at all? The floor screen.

Arm B deletes `nlp_create_qkv_heads` by having the qkv matmul write head-major, which is what
`tt_bio/triatt_qkv.py` already does at the tri-attention sites: the tile ids the writer emits
change, nothing else does. Three things stand between that and the diffusion step, and this screens
the one that decides the rest.

  * the token-DiT qkv weight is [768, 3072], `(kt, nt) = (24, 96)`, and `_MM_BLOCK` has no entry
    there -- the shipped op is `ttnn.linear` with `CORE_GRID_MAIN`, so arm B swaps the matmul
    IMPLEMENTATION as well as its writer;
  * `padded_head_dim` is 64, two tiles, and the head-major macro is written for one;
  * `mm_generic` carries **no bias**, and this site's qkv linear does (`q_bias`, zeros for k/v).

The head-major macro costs nothing at run time (same transactions, different addresses) and the
bias needs one extra broadcast add on q. So the whole arm is bounded by:

    generic_minimal_matmul(no bias) + one add   vs   ttnn.linear(bias) + nlp_create_qkv_heads

If the generic matmul at this shape is slower than the shipped pair minus the add, arm B cannot pay
for itself whatever the writer does, and no kernel work is worth starting.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

S, C, H, HD = 512, 768, 16, 64          # the real token-DiT shape at 512 aa
# (8,8,8,2,2) is `_MM_DEFAULT`, which tenstorrent.py records as being the unconfigured op's OWN
# contraction order -- the only candidate here that can be bit-exact against `ttnn.linear`. The
# rest fold K differently and would cost arm B the one property that made it worth building.
CANDIDATES = [(8, 8, 8, 2, 2), (4, 8, 2, 4, 2), (4, 24, 1, 4, 1), (4, 12, 1, 4, 1),
              (4, 8, 1, 4, 1), (4, 4, 1, 4, 1), (8, 8, 1, 4, 1), (4, 8, 4, 4, 2)]


def timed(ttnn, dev, fn, reps=20, blocks=5):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    walls = []
    for _ in range(blocks):
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        walls.append(1e6 * (time.perf_counter() - t) / reps)
    return st.median(walls), walls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio import mm_generic as G

    dev = T.get_device(trace_region_size=512 << 20)
    out = {"env": {"card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                   "shape": {"S": S, "C": C, "qkv_N": 3 * H * HD},
                   "loadavg": open("/proc/loadavg").read().split()[:3]}, "arms": {}}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        a.out.write_text(json.dumps(out, indent=1))

    tt = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,  # noqa: E731
                                   device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    s = tt(torch.randn(1, S, C))
    w = tt(torch.randn(C, 3 * H * HD))
    bias = tt(torch.zeros(1, 3 * H * HD))
    bias_q = tt(torch.zeros(1, 1, H * HD))       # what the missing q bias would have to be added as
    ckc = T.get_compute_kernel_config() if hasattr(T, "get_compute_kernel_config") else \
        ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2,
                                         math_approx_mode=False, fp32_dest_acc_en=False,
                                         packer_l1_acc=True)

    # --- the shipped pair -----------------------------------------------------------------
    def shipped():
        qkv = ttnn.linear(s, w, bias=bias, compute_kernel_config=ckc,
                          core_grid=T.CORE_GRID_MAIN)
        qkv = ttnn.unsqueeze(qkv, 1)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=False)
        for t in (qkv, q, k, v):
            ttnn.deallocate(t)

    def linear_only():
        qkv = ttnn.linear(s, w, bias=bias, compute_kernel_config=ckc, core_grid=T.CORE_GRID_MAIN)
        ttnn.deallocate(qkv)

    out["arms"]["shipped_linear_plus_heads_us"], _ = timed(ttnn, dev, shipped)
    out["arms"]["shipped_linear_only_us"], _ = timed(ttnn, dev, linear_only)
    out["arms"]["nlp_create_qkv_heads_us"] = round(
        out["arms"]["shipped_linear_plus_heads_us"] - out["arms"]["shipped_linear_only_us"], 2)
    for k_ in ("shipped_linear_plus_heads_us", "shipped_linear_only_us"):
        out["arms"][k_] = round(out["arms"][k_], 2)
    dump()
    print(f"  shipped linear+heads {out['arms']['shipped_linear_plus_heads_us']:8.2f} us   "
          f"linear only {out['arms']['shipped_linear_only_us']:8.2f} us   "
          f"heads {out['arms']['nlp_create_qkv_heads_us']:8.2f} us", flush=True)

    # --- the floor arm B lives under: the generic matmul, no bias, three outputs ------------
    outs = [ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, S, H * HD]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
        ttnn.DRAM_MEMORY_CONFIG) for _ in range(3)]
    grid = tuple(T.COMPUTE_GRID_MAIN)
    out["arms"]["generic"] = {}
    for blk in CANDIDATES:
        tag = "x".join(str(b) for b in blk)
        try:
            def run(blk=blk):
                G.generic_minimal_matmul(dev, s, w, outs, (blk, grid), G.ckc_args(ckc))
            us, walls = timed(ttnn, dev, run)
            out["arms"]["generic"][tag] = {"us": round(us, 2),
                                           "walls": [round(x, 2) for x in walls]}
            print(f"  generic {tag:14s} {us:8.2f} us", flush=True)
        except Exception as e:                                            # noqa: BLE001
            out["arms"]["generic"][tag] = {"error": str(e)[:300]}
            print(f"  generic {tag:14s} FAILED {str(e)[:120]}", flush=True)
        dump()

    ref = ttnn.to_torch(ttnn.linear(s, w, compute_kernel_config=ckc,
                                    core_grid=T.CORE_GRID_MAIN)).clone()
    for blk in CANDIDATES:
        tag = "x".join(str(b) for b in blk)
        if "us" not in out["arms"]["generic"].get(tag, {}):
            continue
        G.generic_minimal_matmul(dev, s, w, outs, (blk, grid), G.ckc_args(ckc))
        got = torch.cat([ttnn.to_torch(o) for o in outs], dim=-1)
        d = (ref.float() - got.float()).abs()
        out["arms"]["generic"][tag]["bit_exact_vs_linear"] = bool(d.max() == 0)
        out["arms"]["generic"][tag]["max_abs_vs_linear"] = float(d.max())
        print(f"  {tag:14s} bit-exact vs ttnn.linear: "
              f"{out['arms']['generic'][tag]['bit_exact_vs_linear']}  "
              f"max {out['arms']['generic'][tag]['max_abs_vs_linear']}", flush=True)
    dump()

    ok = {k: v["us"] for k, v in out["arms"]["generic"].items() if "us" in v}
    if ok:
        best = min(ok, key=ok.get)
        # one broadcast add on q is what the missing bias costs
        add_us, _ = timed(ttnn, dev, lambda: ttnn.deallocate(
            ttnn.add(outs[0], bias_q, memory_config=ttnn.DRAM_MEMORY_CONFIG)))
        out["verdict"] = {
            "best_generic": best, "best_generic_us": ok[best],
            "bias_add_us": round(add_us, 2),
            "arm_b_floor_us": round(ok[best] + add_us, 2),
            "shipped_us": out["arms"]["shipped_linear_plus_heads_us"],
            "headroom_us": round(out["arms"]["shipped_linear_plus_heads_us"]
                                 - ok[best] - add_us, 2),
        }
        out["verdict"]["arm_b_can_pay"] = out["verdict"]["headroom_us"] > 0
        print(f"  ARM B FLOOR {out['verdict']['arm_b_floor_us']:.2f} us vs shipped "
              f"{out['verdict']['shipped_us']:.2f} us -> headroom "
              f"{out['verdict']['headroom_us']:+.2f} us", flush=True)
    dump()
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
