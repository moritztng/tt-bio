#!/usr/bin/env python3
"""Screen: can the fused SDPA be made precise enough to replace openfold3's fp32-softmax attention?

`perf/other512/screen_of3_nofp32_512.json` measured the prize and the price at the fold: routing
openfold3's triangle attention through the fused flash SDPA takes the fold from 107.942 s to
79.646 s (1.3553x) and costs 0.108 plDDT. The 28.3 s is re-materialisation traffic and it is real;
the plDDT is the reason that arm is a screen and not a patch.

The planning doc scoped the fix as "a targeted edit of K2's kernel tree to move the softmax
reduction to fp32 in DST". Read at source, that edit is not needed: `sdpa_generic.plan` already
takes the compute kernel config as `(math_fidelity, math_approx, fp32_dest_acc, dst_full_sync)` and
threads `fp32_dest_acc` into `dst_size` and every subblock, and `build` already takes
`exp_approx_mode`. `tt_bio/triatt_sdpa.py:86` simply hard-codes
`(HiFi2, math_approx=True, fp32_dest_acc=False, dst_full_sync=False)`. So the lever is a config, and
this screen is the whole design space at the production shape before any fold is spent on it.

Reference is a host torch fp32 attention over the SAME device inputs -- the gold, not another device
path. The shipped `_fp32_softmax_attention` is measured against that same gold, so the question the
screen answers is precise: **does any fused config sit at or inside the shipped path's own error?**
If one does, it inherits the shipped path's accuracy and keeps most of the 28.3 s.

Also timed, median of 7 after 2 warm, because a config that recovers the accuracy and gives back the
whole speedup is a NO-GO and has to be visible as one.
"""
from __future__ import annotations

import json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os                                                                     # noqa: E402
import torch, ttnn                                                            # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
import tt_bio.triatt_sdpa as PM                                               # noqa: E402

from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor  # noqa: E402
if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
    mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
    if mgd:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

dev = T.get_device()
g = dev.compute_with_storage_grid_size()

# openfold3's dominant triangle-attention call, read off the fold census:
# 424 of 488 calls at c_z=128 -> B=512 rows, H=4, S=512, head_dim=32; bias [1, H, S, S].
B, H, S, D = 512, 4, 512, 32
torch.manual_seed(0)

CASES = {
    # name: (math_fidelity, math_approx, fp32_dest_acc, dst_full_sync)
    "shipped_hifi2_approx":      (ttnn.MathFidelity.HiFi2, True,  False, False),
    "fp32dst":                   (ttnn.MathFidelity.HiFi2, True,  True,  False),
    "fp32dst_noapprox":          (ttnn.MathFidelity.HiFi2, False, True,  False),
    "fp32dst_noapprox_hifi4":    (ttnn.MathFidelity.HiFi4, False, True,  False),
    "hifi4_approx":              (ttnn.MathFidelity.HiFi4, True,  False, False),
}


def stats(got: torch.Tensor, gold: torch.Tensor) -> dict:
    g32, x32 = gold.float().flatten(), got.float().flatten()
    err = x32 - g32
    denom = g32.std().item()
    pcc = torch.corrcoef(torch.stack([x32, g32]))[0, 1].item()
    return {"rmsd": round(err.pow(2).mean().sqrt().item(), 8),
            "rmsd_over_std": round(err.pow(2).mean().sqrt().item() / denom, 8),
            "max_abs": round(err.abs().max().item(), 8),
            "pcc": round(pcc, 9)}


def bench(fn, n=7, warm=2):
    for _ in range(warm):
        r = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(r)
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(r)
    return st.median(ts) * 1e3


def main():
    scale = float(D) ** -0.5
    qt = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    kt = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    vt = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    bt = torch.randn(1, H, S, S, dtype=torch.bfloat16) * 0.5

    # ---- host fp32 gold, over exactly the tensors the device will see ------------------------
    sc = torch.matmul(qt.float(), kt.float().transpose(-1, -2)) * scale + bt.float()
    gold = torch.matmul(torch.softmax(sc, dim=-1), vt.float())
    del sc

    def dev_t(x):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    q, k, v, bias = dev_t(qt), dev_t(kt), dev_t(vt), dev_t(bt)
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": [g.x, g.y], "shape": {"B": B, "H": H, "S": S, "D": D},
           "note": "gold = host torch fp32 over the same device inputs", "rows": []}

    # ---- the shipped reference path openfold3 runs today --------------------------------------
    ckc_trunk = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    def run_fp32soft():
        return T._fp32_softmax_attention(q, k, v, bias, scale_inv=1.0 / scale,
                                         compute_kernel_config=ckc_trunk,
                                         out_dtype=ttnn.bfloat16, bias_scale_inv=1.0)

    o = run_fp32soft()
    row = {"name": "_fp32_softmax_attention (shipped for openfold3)", "ms": round(bench(run_fp32soft), 4)}
    row.update(stats(ttnn.to_torch(o), gold))
    ttnn.deallocate(o)
    res["rows"].append(row)
    print(json.dumps(row), flush=True)
    ref_err = row["rmsd_over_std"]

    # ---- every fused config -------------------------------------------------------------------
    q_chunk, k_chunk = None, T._sdpa_chunks_shipped(S, S)[1]
    fits = [qc for qc in T._tri_att_q_chunks(S, S) if (S, S, qc) not in T._SDPA_Q_CHUNK_OVER_L1]
    for name, ckc in CASES.items():
        got = None
        for qc in fits:
            got = PM.sdpa(q, k, v, bias, scale, qc, k_chunk, ckc_default=ckc)
            if got is not None:
                q_chunk = qc
                break
        if got is None:
            row = {"name": name, "error": "K2 declined every q_chunk",
                   "rejects": {f"{r}:{sh}": n for (r, sh), n in PM.REJECTS.items()}}
            res["rows"].append(row)
            print(json.dumps(row), flush=True)
            continue
        row = {"name": name, "ckc": [str(ckc[0]).rsplit(".", 1)[-1], *map(bool, ckc[1:])],
               "q_chunk": q_chunk}
        row.update(stats(ttnn.to_torch(got), gold))
        ttnn.deallocate(got)
        qc_fixed = q_chunk
        row["ms"] = round(bench(lambda: PM.sdpa(q, k, v, bias, scale, qc_fixed, k_chunk,
                                                ckc_default=ckc)), 4)
        row["inside_shipped_error"] = row["rmsd_over_std"] <= ref_err
        res["rows"].append(row)
        print(json.dumps(row), flush=True)

    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    print("wrote", sys.argv[1], flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
