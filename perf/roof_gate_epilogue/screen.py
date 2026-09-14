#!/usr/bin/env python3
"""Price the gate multiply the fusion ranking wants deleted, against the block that would host it.

The ranking's row is `generic_minimal_matmul (fused TriAtt in-projection) -> gate_and_project's
multiply_`, 268.4 MB a Pairformer block. The producer cannot host the epilogue (the multiply's other
operand is downstream of the producer's own output -- see
state/roof-fuse-gate-epilogue.md), so what is measured here is the PRIZE, not a build:

  mul      the op that would be deleted, at the fold's own shape and memory config
  sdpa     the fused SDPA that would host the epilogue instead, same shape
  triatt   the whole TriangleAttention call, as the denominator for an op ratio
  triatt2  the same call again, reported separately, as the A/A floor

Arms are interleaved per rep (`op-ab-must-interleave-arms-compile-warmup-bias`), every timed region
is bracketed by `ttnn.synchronize_device`, and the first reps are discarded as warmup.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import triatt_qkv as TQ
from tt_bio import triatt_sdpa as TS


def weights(c_z, n_heads, head_dim):
    torch.manual_seed(0)
    d = n_heads * head_dim
    return {
        "layer_norm.weight": torch.ones(c_z),
        "layer_norm.bias": torch.zeros(c_z),
        "linear_q.weight": torch.randn(d, c_z) * (c_z ** -0.5),
        "linear_k.weight": torch.randn(d, c_z) * (c_z ** -0.5),
        "linear_v.weight": torch.randn(d, c_z) * (c_z ** -0.5),
        "linear_g.weight": torch.randn(d, c_z) * (c_z ** -0.5),
        "linear_o.weight": torch.randn(c_z, d) * (d ** -0.5),
        "linear.weight": torch.randn(n_heads, c_z) * (c_z ** -0.5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--c-z", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--out", default="perf/roof_gate_epilogue/screen_512_whglx_c2.json")
    a = ap.parse_args()
    S, C, H, D = a.n, a.c_z, a.heads, a.head_dim

    dev = T.get_device()
    kernel_cls = (ttnn.types.WormholeComputeKernelConfig
                  if dev.arch() == ttnn.Arch.WORMHOLE_B0
                  else ttnn.types.BlackholeComputeKernelConfig)
    ckc = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                     fp32_dest_acc_en=True, packer_l1_acc=True)

    res = {"meta": {"n": S, "c_z": C, "heads": H, "head_dim": D,
                    "arch": str(dev.arch()), "grid": list(T.COMPUTE_GRID_MAIN),
                    "card": os.environ.get("TT_VISIBLE_DEVICES"),
                    "host": os.uname().nodename, "loadavg": os.getloadavg(),
                    "reps": a.reps, "warm": a.warm}, "arms": {}}
    print(json.dumps(res["meta"]), flush=True)

    def dram(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    att = T.TriangleAttention(D, H, False, weights(C, H, D), ckc)
    z = dram(torch.randn(1, S, S, C).to(torch.bfloat16) * 0.1)

    # the tail's two operands at the shape the head-major path produces them in
    o = dram(torch.randn(S, H, S, D).to(torch.bfloat16))
    g = dram(torch.randn(S, H, S, D).to(torch.bfloat16))
    q = dram(torch.randn(S, H, S, D).to(torch.bfloat16))
    k = dram(torch.randn(S, H, S, D).to(torch.bfloat16))
    v = dram(torch.randn(S, H, S, D).to(torch.bfloat16))
    bias = dram(torch.randn(1, H, S, S).to(torch.bfloat16) * 0.1)

    hold = {"o": o}

    def f_mul():
        # in place, exactly as gate_and_project issues it; o is refreshed between rep groups
        ttnn.multiply_(hold["o"], g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])

    def f_sdpa():
        r = T._tri_att_sdpa(q, k, v, bias, (D ** 0.5) ** -1)
        ttnn.deallocate(r)

    def f_triatt():
        r = att(z)
        ttnn.deallocate(r)

    arms = {"mul": f_mul, "sdpa": f_sdpa, "triatt": f_triatt, "triatt2": f_triatt}
    ts = {kk: [] for kk in arms}

    for i in range(a.warm + a.reps):
        for name, fn in arms.items():
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            fn()
            ttnn.synchronize_device(dev)
            dt = time.perf_counter() - t0
            if i >= a.warm:
                ts[name].append(dt)
        # keep the in-place multiply's operand from collapsing to zero over reps
        ttnn.deallocate(hold["o"])
        hold["o"] = dram(torch.randn(S, H, S, D).to(torch.bfloat16))

    for name, v_ in ts.items():
        res["arms"][name] = {"ms": st.median(v_) * 1e3,
                             "spread": (max(v_) - min(v_)) / st.median(v_),
                             "all_ms": [x * 1e3 for x in v_]}
        print(f"{name:8s} {st.median(v_)*1e3:9.3f} ms  spread {(max(v_)-min(v_))/st.median(v_)*100:5.2f}%",
              flush=True)

    res["gates"] = {"qkv": list(TQ.STATS), "tail": list(TQ.TAIL_STATS),
                    "qkvg": list(TQ.QKVG_STATS), "sdpa_fused": list(TS.STATS),
                    "tail_rejects": {str(kk): vv for kk, vv in TQ.TAIL_REJECTS.items()},
                    "qkvg_rejects": {str(kk): vv for kk, vv in TQ.QKVG_REJECTS.items()}}
    print(json.dumps(res["gates"]), flush=True)

    aa = res["arms"]
    floor = abs(aa["triatt2"]["ms"] - aa["triatt"]["ms"]) / aa["triatt"]["ms"]
    block_mul = 2 * aa["mul"]["ms"]
    res["derived"] = {
        "aa_floor_frac": floor,
        "mul_ms_per_call": aa["mul"]["ms"],
        "mul_GBps": 3 * S * H * S * D * 2 / (aa["mul"]["ms"] * 1e-3) / 1e9,
        "two_mul_ms_per_block": block_mul,
        "triatt_ms": aa["triatt"]["ms"],
        "ceiling_ratio_on_triatt": aa["triatt"]["ms"] / (aa["triatt"]["ms"] - aa["mul"]["ms"]),
        "realistic_ratio_on_triatt": aa["triatt"]["ms"] / (aa["triatt"]["ms"] - aa["mul"]["ms"] * 2 / 3),
    }
    print(json.dumps(res["derived"], indent=1), flush=True)
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
