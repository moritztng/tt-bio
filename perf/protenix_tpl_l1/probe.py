"""Does the protenix template embedder's L1-resident normed pair tensor starve its consumer?

`Protenix._ln(..., l1=True)` (tt_bio/protenix.py) admits the normed pair tensor to L1 when 1.5
copies of it fit across the whole grid's banks. That budget is an aggregate of interleaved bytes;
the wall its consumer hits is per core. This replays exactly the residency window `_template`
opens -- the layer_norm, then nt narrow projections of the whole tensor -- at a ladder of token
counts, and reports for each one what the gate decided and what the consumer actually got.

Prints one JSON line per token count.
"""
import json
import os
import sys
import traceback

import torch
import ttnn

from tt_bio import tenstorrent as T
from tt_bio.tenstorrent import get_device, _l1_layer_norm, _narrow_proj_linear, _pair_proj_config

C_Z = int(os.environ.get("PROBE_CZ", "256"))    # Protenix-v2 pair channels (OpenDDE: 384)
C_OUT = int(os.environ.get("PROBE_COUT", "64"))  # template_embedder.linear_no_bias_z: c_z -> 64
NT = 4              # dummy_template_features(max_templates=4): always 4


def l1_free(dev):
    """(free bytes per bank, banks) from the ttnn allocator's own L1 view."""
    try:
        mv = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
        return int(mv.total_bytes_free_per_bank), int(mv.num_banks)
    except Exception:
        return None, None


def ballast(dev, per_core_bytes):
    """An L1-interleaved tensor holding `per_core_bytes` on every bank, to stand in for the
    other buffers a real fold has live when the template projections run. One tile per bank
    per 2048 B, so the per-core figure is exact."""
    if per_core_bytes <= 0:
        return None
    banks = int(ttnn.get_memory_view(dev, ttnn.BufferType.L1).num_banks)
    tiles = banks * (per_core_bytes // 2048)
    return ttnn.from_torch(torch.zeros(1, tiles * 32, 32, dtype=torch.bfloat16),
                           layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=ttnn.L1_MEMORY_CONFIG)


def one(dev, N, ckc, headroom, reserve, ballast_per_core=0):
    w = ttnn.from_torch(torch.randn(C_Z, C_OUT, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    gw = ttnn.from_torch(torch.randn(C_Z, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                         device=dev, dtype=ttnn.bfloat16)
    gb = ttnn.from_torch(torch.zeros(C_Z, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                         device=dev, dtype=ttnn.bfloat16)
    z3 = ttnn.from_torch(torch.randn(1, N, N, C_Z, dtype=torch.bfloat16),
                         layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rec = {"N": N, "c_z": C_Z, "headroom": headroom, "reserve": reserve,
           "ballast": ballast_per_core}
    bal = None
    T._L1_OUT_REFUSED.clear()   # one memo per row, or row N+1 reads row N's refusal
    try:
        bal = ballast(dev, ballast_per_core)
        zn, in_l1 = _l1_layer_norm(z3, headroom, reserve, weight=gw, bias=gb, epsilon=1e-5,
                                   compute_kernel_config=ckc)
        rec["zn_in_l1"] = bool(in_l1)
        rec["free_per_bank_after_zn"], rec["banks"] = l1_free(dev)
        cfg = _pair_proj_config(zn, w, bw_cap=T._NARROW_PROJ_BW, out_l1=in_l1)
        rec["proj_cfg"] = "none" if cfg is None else "tuned"
        before = set(T._L1_OUT_REFUSED)
        outs = []
        for t in range(NT):
            o = T._narrow_proj_linear(zn, w, ckc, ttnn.bfloat16, l1_out=in_l1)
            rec.setdefault("proj_ret", "none" if o is None else "tensor")
            if o is None:                      # would fall back to ttnn.linear(core_grid=)
                o = ttnn.linear(zn, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                core_grid=T.CORE_GRID_MAIN)
                rec["proj_ret"] = "core_grid_fallback"
            rec[f"out_buf{t}"] = str(o.memory_config().buffer_type).split(".")[-1]
            if t == 0:
                rec["free_per_bank_after_proj"] = l1_free(dev)[0]
            outs.append(ttnn.add(ttnn.from_torch(
                torch.zeros(1, N, N, C_OUT, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                o, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            del o
        rec["l1_out_refused"] = sorted(str(k) for k in set(T._L1_OUT_REFUSED) - before)
        rec["ok"] = True
        del outs
        ttnn.deallocate(zn)
    except Exception as e:                                                    # noqa: BLE001
        rec["ok"] = False
        rec["err"] = " ".join(str(e).split())[:400]
        rec["tb"] = traceback.format_exc().splitlines()[-3:]
    ttnn.deallocate(z3)
    if bal is not None:
        ttnn.deallocate(bal)
    return rec


def main():
    ns = [int(x) for x in sys.argv[1].split(",")]
    bals = [int(x) for x in os.environ.get("PROBE_BALLAST", "0").split(",")]
    headroom = float(os.environ.get("PROBE_HEADROOM", "1.5"))
    reserve = int(os.environ.get("PROBE_RESERVE", "0"))
    dev = get_device()
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi3,
                                           fp32_dest_acc_en=False, packer_l1_acc=True)
    print(json.dumps({"grid": list(T.COMPUTE_GRID_MAIN),
                      "per_core_unreserved": int(ttnn.get_max_worker_l1_unreserved_size()),
                      "l1_bank_bytes": int(T._l1_bank_bytes()),
                      "free_per_bank_idle": l1_free(dev)[0],
                      "banks": l1_free(dev)[1]}), flush=True)
    for N in ns:
        for b in bals:
            print(json.dumps(one(dev, N, ckc, headroom, reserve, b)), flush=True)


if __name__ == "__main__":
    main()
