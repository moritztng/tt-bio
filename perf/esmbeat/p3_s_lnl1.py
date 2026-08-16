#!/usr/bin/env python3
"""S-E: verify the L1 layer_norm operand lever -- parity at the BODY output, gate census, sizes.

p3_s_fork measured `body_l1ln` 2.47 ms/call faster than `body_ln` at 512 aa on qb1 card 0. Three
things have to be true before that is a lever and not an artefact:
  1. the fc1 L1-output gate is still SERVING in the l1ln arm (if it silently fell back to
     `_pair_proj_minimal_matmul` the delta is a different lever wearing this one's name);
  2. the BODY output is `torch.equal`, not just the layer_norm output;
  3. the L1 budget still admits it at 640 and 768 aa, where h1+h2+gated grow with L.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch, ttnn
from tt_bio import tenstorrent as T
from tt_bio import esmc as EC
CALLS = {256: 538, 298: 538, 512: 538, 640: 538, 768: 538}


def timed(fn, dev, reps=4, batches=5, warm=2):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3 / reps)
    return st.median(out), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default="512")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    C_Z, D_FF, ROWS = 256, 1024, EC._PAIR_FFN_ROW_BLOCK
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    ck = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": [g.x, g.y], "rows": ROWS, "sizes": {}}
    torch.manual_seed(0)
    to = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    nw = to(torch.ones(C_Z)); nb = to(torch.zeros(C_Z))
    w1a = to(torch.randn(C_Z, D_FF) * 0.02)
    w1b = to(torch.randn(C_Z, D_FF) * 0.02)
    w2 = to(torch.randn(D_FF, C_Z) * 0.02)
    l1cfg = dict(l1_out=True, l1_bw=T._PAIR_FFN_FC1_BW, l1_block_w=T._PAIR_FFN_FC1_BLOCK_W)

    for L in [int(s) for s in a.sizes.split(",")]:
        x = to(torch.randn(1, L, L, C_Z))
        nblk = -(-L // ROWS)
        pre = ttnn.chunk(x, nblk, dim=1)

        def ln(t, l1):
            kw = {"memory_config": ttnn.L1_MEMORY_CONFIG} if l1 else {}
            return ttnn.layer_norm(t, weight=nw, bias=nb, epsilon=1e-5,
                                   compute_kernel_config=ck, **kw)

        def body(l1, keep=False):
            outs = []
            for p in pre:
                xn = ln(p, l1)
                h1 = T._pair_proj_linear(xn, w1a, ck, ttnn.bfloat16, **l1cfg)
                h2 = T._pair_proj_linear(xn, w1b, ck, ttnn.bfloat16, **l1cfg)
                ttnn.deallocate(xn)
                gated = ttnn.multiply(h1, h2, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                                      memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(h1); ttnn.deallocate(h2)
                o = ttnn.linear(gated, w2, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                                core_grid=T.CORE_GRID_MAIN)
                ttnn.deallocate(gated)
                if keep:
                    outs.append(o)
                else:
                    ttnn.deallocate(o)
            if keep:
                r = ttnn.concat(outs, dim=1)
                for o in outs:
                    ttnn.deallocate(o)
                return r

        row = {}
        for name, l1 in (("dram_ln", False), ("l1_ln", True)):
            EC.L1_FC1_STATS[0] = EC.L1_FC1_STATS[1] = 0
            m, raw = timed(lambda: body(l1), dev)
            row[name] = round(m, 4)
            row[name + "_raw"] = [round(v, 4) for v in raw]
            row[name + "_l1fc1"] = list(EC.L1_FC1_STATS)
        ra = body(False, keep=True); rb = body(True, keep=True)
        ta, tb = ttnn.to_torch(ra), ttnn.to_torch(rb)
        row["body_torch_equal"] = bool(torch.equal(ta, tb))
        row["max_abs_diff"] = float((ta - tb).abs().max())
        ttnn.deallocate(ra); ttnn.deallocate(rb)
        row["delta_ms"] = round(row["l1_ln"] - row["dram_ln"], 4)
        row["delta_s_per_fold"] = round(row["delta_ms"] * CALLS[L] / 1e3, 3)
        res["sizes"][L] = row
        print(L, json.dumps({k: v for k, v in row.items() if not k.endswith("_raw")}), flush=True)
        for p in pre:
            ttnn.deallocate(p)
        ttnn.deallocate(x)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
