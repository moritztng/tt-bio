#!/usr/bin/env python3
"""p123 -- S2 of the RFD3 fusion programme: pin the PV matmul's accumulation contract for L5b.

L5b folds the PV matmul into the landed fused softmax kernel, deleting the softmax's 294.3 MB
bf16 write and the matmul's read of it. The kernel must then reproduce
`attn_value_matmul(attention, vv)` BIT-EXACTLY at [1,4,6051,6080] @ [1,4,6080,32].

This does NOT re-derive whether the K-blocking matters. That is closed:
`tenstorrent.py:_attn_value_program_config` records "measured over 8 shape classes x ~50 arms
on two cards and two ttnn versions: every arm whose `in0_block_w` matched came back
`torch.equal`, every arm that differed did not, no exception." The rule is match-or-differ, and
the shipped PV matmul already runs on an EXPLICIT config, so its blocking is already pinned.

What is open, and what this screens:

  A  the contract   the config's fields at the live atom shape, so the kernel can mirror them
  B  the mirror     attn_value_matmul == plain ttnn.matmul at THIS shape (the one shape L5b
                    depends on; the 8-class study need not have covered it)
  C  the rate       ms/call of the PV matmul, the number the fused kernel has to beat, and
                    the ms/call of the softmax kernel it folds into
  D  the prize      C priced against the bytes L5b deletes, as a COST-MODEL ESTIMATE only

Kill gate: if the config returns None the shape falls to the heuristic and L5b is NO-GO until
the heuristic's own blocking is pinned. If B is False, attn_value_matmul is already not
bit-exact against the naive path here and L5b's contract is the tuned config; record which.
"""
import json
import os
import pathlib
import statistics
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import model as M                                  # noqa: E402
from tt_bio import tenstorrent as T                                 # noqa: E402
from tt_bio import softmax_generic                                  # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p123/pv_korder.json")
NB, NH, Q, K, D = 1, 4, 6051, 6080, 32
REPS = 6
CALLS_PER_STEP = 9          # 6 decoder + 3 atom encoder, confirmed by perf/p49x


def med(fn):
    fn()
    ttnn.synchronize_device(M.get_device())
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(M.get_device())
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


def main():
    mt, kt, nt = -(-Q // 32), K // 32, D // 32
    cfg = T._attn_value_program_config(mt, kt, nt, NB * NH, 2)
    res = {"shape": [NB, NH, Q, K, D], "m_tiles": mt, "k_tiles": kt, "n_tiles": nt,
           "calls_per_step": CALLS_PER_STEP, "reps": REPS}
    if cfg is None:
        res["contract"] = None
        res["verdict"] = "NO-GO: the shape falls outside _attn_value_program_config"
        print("VERDICT: " + res["verdict"])
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=2) + "\n")
        return
    res["contract"] = {"in0_block_w": cfg.in0_block_w, "per_core_M": cfg.per_core_M,
                       "per_core_N": cfg.per_core_N, "out_subblock_h": cfg.out_subblock_h,
                       "out_subblock_w": cfg.out_subblock_w}
    print("A contract: %s" % res["contract"])

    dev = M.get_device()
    ckc = M._default_compute_kernel_config()
    torch.manual_seed(42)
    # A real softmax output: non-negative, rows summing to 1. Cancellation is what makes a
    # regrouping visible, so white noise would be a weaker test than the operand it stands in for.
    a_t = torch.rand(NB, NH, Q, K, dtype=torch.float32)
    a_t = a_t / a_t.sum(-1, keepdim=True)
    v_t = torch.randn(NB, NH, K, D, dtype=torch.float32)
    a = ttnn.from_torch(a_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    v = ttnn.from_torch(v_t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    sc = ttnn.from_torch(a_t, ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    del a_t, v_t

    tuned = ttnn.to_torch(T.attn_value_matmul(a, v, ckc, ttnn.bfloat16))
    naive = ttnn.to_torch(ttnn.matmul(a, v, compute_kernel_config=ckc, dtype=ttnn.bfloat16))
    res["mirror_equal"] = bool(torch.equal(tuned, naive))
    res["mirror_maxabs"] = float((tuned.float() - naive.float()).abs().max())
    print("B mirror: attn_value_matmul == ttnn.matmul  equal=%s maxabs=%.6g"
          % (res["mirror_equal"], res["mirror_maxabs"]))
    del tuned, naive

    t = med(lambda: T.attn_value_matmul(a, v, ckc, ttnn.bfloat16))
    res["pv_ms"] = round(1000 * t[0], 4)
    res["pv_ms_min"], res["pv_ms_max"] = round(1000 * t[1], 4), round(1000 * t[2], 4)
    print("C PV matmul      %8.4f ms/call  (min %.4f max %.4f)"
          % (1000 * t[0], 1000 * t[1], 1000 * t[2]))

    res["softmax_eligible"] = bool(softmax_generic.eligible(sc, ttnn.bfloat16))
    t = med(lambda: softmax_generic.softmax_bf16(sc, ttnn.bfloat16))
    res["softmax_ms"] = round(1000 * t[0], 4)
    print("C softmax kernel %8.4f ms/call  eligible=%s" % (1000 * t[0], res["softmax_eligible"]))

    # D: the prize. Deleted = the kernel's 294.3 MB bf16 pack and the matmul's re-read of it,
    # at the census's measured roofs (163.9 GB/s single-RISC drain, 390 GB/s read). This is a
    # COST-MODEL ESTIMATE and stays one until a fold A/B says otherwise: the lineage's block-
    # sparse lever was projected at 28.506 s/design by per-call-times-census and measured 2.899.
    mb = NH * Q * K * 2 / 1e6
    roof = mb / 163.9 + mb / 390.0
    res["deleted_mb_per_call"] = round(mb, 1)
    res["roof_ms_per_call"] = round(roof, 3)
    res["prize_estimate_s_per_design"] = {
        "floor_110pct": round(-1.10 * roof * CALLS_PER_STEP * 200 / 1000, 3),
        "central_75pct": round(-0.75 * roof * CALLS_PER_STEP * 200 / 1000, 3),
        "pessimistic_41pct": round(-0.41 * roof * CALLS_PER_STEP * 200 / 1000, 3),
    }
    res["measured_chain_ms_per_step"] = round(
        (res["pv_ms"] + res["softmax_ms"]) * CALLS_PER_STEP, 3)
    print("D deleted %.1f MB/call, roof %.3f ms/call, estimate %s s/design"
          % (mb, roof, res["prize_estimate_s_per_design"]))
    print("D shipped softmax+PV chain: %.1f ms/step over %d calls"
          % (res["measured_chain_ms_per_step"], CALLS_PER_STEP))

    res["verdict"] = ("GO to build: contract pinned at in0_block_w=%d, mirror %s"
                      % (cfg.in0_block_w,
                         "holds" if res["mirror_equal"] else
                         "DOES NOT hold -- the contract is the tuned config, not the default"))
    print("\nVERDICT: " + res["verdict"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
