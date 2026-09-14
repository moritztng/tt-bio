#!/usr/bin/env python3
"""Is the gate epilogue the RIGHT transform? Both arms against an fp64 reference of the same bf16
operands, so only each arm's own error is left.

arm A  the shipped pair: the fused SDPA, then `gate_and_project`'s `ttnn.multiply_(o, g,
       activations=[SIGMOID])`. sigmoid(g) is packed to bf16 in L1 and the multiply is on the FPU.
arm B  the gated kernel: the same SDPA with `gate=g`. sigmoid(g) stays in DST and the multiply is
       an SFPU binary op, so B rounds once where A rounds twice.

The reference is float64 over the float64 upcast of the SAME bf16 q/k/v/bias/gate tensors. Neither
arm is compared against the other as a reference -- `torch.equal(A, B)` is reported separately, as
the cheap regression signal it is, not as the accuracy bar.
"""
import argparse, json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import triatt_sdpa as TS


def err(x, ref):
    d = (x - ref)
    return {"rel_rms": float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
            "max_abs": float(d.abs().max()),
            "rel_max": float(d.abs().max() / ref.abs().max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--out", default="perf/roof_gate_epilogue/parity_op_512_qb2_c2.json")
    a = ap.parse_args()
    S, H, D = a.n, a.heads, a.head_dim
    # The kernel's own scale, which rides the exp AFTER the bias add. The model passes
    # `self.scale ** -1` and `self.scale` is sqrt(head_dim) there, so this is 1/sqrt(head_dim).
    scale = D ** -0.5

    dev = T.get_device()
    grid = tuple(T.COMPUTE_GRID_MAIN)
    torch.manual_seed(0)
    # bf16 operands, the layout and dtype the fold hands the kernel
    qt = (torch.randn(S, H, S, D) * 0.5).bfloat16()
    kt = (torch.randn(S, H, S, D) * 0.5).bfloat16()
    vt = (torch.randn(S, H, S, D) * 0.5).bfloat16()
    bt = (torch.randn(1, H, S, S) * 2.0).bfloat16()
    gt = (torch.randn(S, H, S, D) * 2.0).bfloat16()

    # fp64 reference. The kernel applies `scale` AFTER the bias add (compute_common: the scale
    # rides the exp, exp((qk + mask - max) * scale)), so the reference must too.
    q64, k64, v64 = qt.double(), kt.double(), vt.double()
    logits = (q64 @ k64.transpose(-1, -2) + bt.double()) * scale
    w = torch.softmax(logits, dim=-1)
    ref_ungated = w @ v64
    ref = ref_ungated * torch.sigmoid(gt.double())
    del logits, w, q64, k64, v64

    f = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cores = grid[0] * grid[1]
    pairs = TS.fused_pairs(S, H, D, cores, ttnn.bfloat16)
    res = {"n": S, "heads": H, "head_dim": D, "arch": str(dev.arch()), "grid": list(grid),
           "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "fused_pairs": [list(p) for p in pairs]}

    # the rung the fold actually takes: `_tri_att_sdpa_at`'s `fits` loop, widest q first
    k_chunk = T._tri_att_k_chunks(S, S)[-1]
    fits = [qc for qc in T._tri_att_q_chunks(S, S)]
    res["k_chunk"] = k_chunk
    res["q_chunks_offered"] = fits

    def arm(gated):
        q, k, v, b, g = f(qt), f(kt), f(vt), f(bt), f(gt)
        o = None
        for qc in fits:
            o = TS.sdpa(q, k, v, b, scale, qc, k_chunk, gate=g if gated else None)
            if o is not None:
                used = qc
                break
        if o is None:
            for t in (q, k, v, b, g):
                ttnn.deallocate(t)
            return None, None
        if not gated:
            o = ttnn.multiply_(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        out = ttnn.to_torch(o).double()
        for t in (q, k, v, b, o):
            ttnn.deallocate(t)
        if not gated:
            ttnn.deallocate(g)
        return out, used

    a_out, a_q = arm(False)
    print("arm A (shipped pair) served at q_chunk", a_q, flush=True)
    b_out, b_q = arm(True)
    print("arm B (gated kernel) served at q_chunk", b_q, flush=True)
    res["served_q_chunk"] = {"A": a_q, "B": b_q}
    res["gate_stats"] = list(TS.GATE_STATS)
    res["gate_rejects"] = {str(kk): vv for kk, vv in TS.GATE_REJECTS.items()}
    res["sdpa_stats"] = list(TS.STATS)
    if b_out is None:
        res["verdict"] = "GATED PATH DECLINED"
        print(json.dumps(res, indent=1))
        (REPO / a.out).write_text(json.dumps(res, indent=1))
        return 1

    # the bf16 storage ceiling: what the answer costs just by being written in bf16
    res["bf16_ceiling"] = err(ref.bfloat16().double(), ref)
    res["A_vs_fp64"] = err(a_out, ref)
    res["B_vs_fp64"] = err(b_out, ref)
    res["bit_exact_A_vs_B"] = bool(torch.equal(a_out, b_out))
    res["B_vs_A"] = err(b_out, a_out)
    print(json.dumps({k2: res[k2] for k2 in
                      ("bf16_ceiling", "A_vs_fp64", "B_vs_fp64", "bit_exact_A_vs_B", "B_vs_A",
                       "gate_stats", "gate_rejects")}, indent=1), flush=True)
    p = REPO / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    print("wrote", p, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
