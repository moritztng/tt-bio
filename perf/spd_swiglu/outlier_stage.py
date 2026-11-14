"""Which stage of the sharded pair-transition swiglu writes the Wormhole wrong values, and at which K block.

    TT_VISIBLE_DEVICES=N python perf/spd_swiglu/outlier_stage.py --out OUT.json [--S 736] [--seed 0]

outlier_ab.py found transition_shard putting ~4.0 errors into 1-2 output pixels per call at the trunk's HiFi3,
where the interleaved path puts none. This walks every row block of the same input (seed and generator order as
outlier_ab.py), runs each stage on its own and compares it with float64 on the device's own bf16 inputs to that
stage, so a wrong value is pinned to the op and program config that made it. fc1/fc2 sweep in0_block_w (K is
8 tiles); fc3 sweeps its K block. The interleaved arm runs the same stages through `_transition_linear`.
Counts are elements with |err| > --bad; max_err beside them.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--S", type=int, default=736)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--bad", type=float, default=0.1)
ap.add_argument("--ckc", default="trunk", choices=("trunk", "hifi4", "hifi3", "hifi2"))
a = ap.parse_args()
os.environ["TT_BIO_LEVERS"] = "normal"

import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()
dev = T.get_device()
ARCH = "wormhole" if T.is_wormhole() else "blackhole"
CKC_CLS = ttnn.WormholeComputeKernelConfig if ARCH == "wormhole" else ttnn.types.BlackholeComputeKernelConfig
T._LEVERS = T.parse_levers("normal")
FID = {"hifi4": ttnn.MathFidelity.HiFi4, "hifi3": ttnn.MathFidelity.HiFi3, "hifi2": ttnn.MathFidelity.HiFi2}
HIFI4 = CKC_CLS(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True, fp32_dest_acc_en=True,
                packer_l1_acc=True)
CKC = (T.trunk_compute_kernel_config(HIFI4) if a.ckc == "trunk" else
       CKC_CLS(math_fidelity=FID[a.ckc], math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=True))
SILU_CKC = T.silu_ckc(CKC)
SILU = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
C, HID, S = 256, 1024, a.S
F = torch.nn.functional
bf = lambda t: t.to(torch.bfloat16).to(torch.float64)  # noqa: E731
dt = lambda t: ttnn.to_torch(t).to(torch.float64).reshape(-1, t.shape[-1])  # noqa: E731

g = torch.Generator().manual_seed(a.seed)
sd = {"norm.weight": bf(1 + 0.1 * torch.randn(C, generator=g)),
      "norm.bias": bf(0.1 * torch.randn(C, generator=g)),
      "fc1.weight": bf(torch.randn(HID, C, generator=g) / C ** 0.5),
      "fc2.weight": bf(torch.randn(HID, C, generator=g) / C ** 0.5),
      "fc3.weight": bf(torch.randn(C, HID, generator=g) / HID ** 0.5)}
zt = bf(torch.randn(1, S, S, C, generator=g))
tr = T.Transition({k: v.float() for k, v in sd.items()}, CKC)
W1, W2, W3 = (dt(w) for w in (tr.fc1_weight, tr.fc2_weight, tr.fc3_weight))
rows = T._transition_shard_rows(S, C, HID)
wt, nt, ct, kt = -(-S // 32), HID // 32, C // 32, C // 32
gx, gy = T._transition_shard_grid(rows * wt, nt, ct)
pm, pn = rows * wt // gy, nt // gx
cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
MC = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
                       ttnn.ShardSpec(cores, [pm * 32, pn * 32], ttnn.ShardOrientation.ROW_MAJOR))


def cfg(bw, n, act=None):
    sh, sw = max(((h, w) for h in range(1, 5) for w in range(1, 5)
                  if h * w <= 4 and pm % h == 0 and n % w == 0), key=lambda s: (s[0] * s[1], s[1]))
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), in0_block_w=bw, out_subblock_h=sh,
        out_subblock_w=sw, out_block_h=pm, out_block_w=n, per_core_M=pm, per_core_N=n,
        transpose_mcast=False, fused_activation=act, fuse_batch=True)


def tally(acc, key, got, want):
    e = (got - want).abs()
    c = acc.setdefault(key, {"n_bad": 0, "max_err": 0.0, "blocks_bad": []})
    nb = int((e > a.bad).sum())
    c["n_bad"] += nb
    c["max_err"] = max(c["max_err"], float(e.max()))
    return nb


def block(s, h):
    """x_norm of rows s..s+h on device, its float64 copy, and float64 fc1/fc2 pre-activations."""
    blk = ttnn.from_torch(zt[:, s:s + h].float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    xn = ttnn.layer_norm(blk, weight=tr.norm_weight, bias=tr.norm_bias, epsilon=1e-5, compute_kernel_config=CKC,
                         memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(blk)
    X = dt(xn)
    return xn, X @ W1, X @ W2


# The interleaved path at the fold's own row height (the module's chunk), the sharded one at the shard rows.
T.TRANSITION_H_CHUNK_SHAPES.clear()
with T.levers("normal-transition_shard"):
    ttnn.deallocate(tr(ttnn.from_torch(zt.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)))
il_rows = [k[2] for k in T.TRANSITION_H_CHUNK_SHAPES][0]
acc = {}
for s in range(0, S - il_rows + 1, il_rows):
    xn, Z1, Z2 = block(s, il_rows)
    x1 = T._transition_linear("fc1", xn, tr.fc1_weight, silu=True, compute_kernel_config=SILU_CKC,
                              memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    x2 = T._transition_linear("fc2", xn, tr.fc2_weight, compute_kernel_config=CKC,
                              memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    A1, A2 = dt(x1), dt(x2)
    bad = tally(acc, "il.fc1_silu", A1, F.silu(Z1)) + tally(acc, "il.fc2", A2, Z2)
    h = ttnn.multiply_(x1, x2)
    ttnn.deallocate(x2)
    H_ = dt(h)
    bad += tally(acc, "il.mul", H_, A1 * A2)
    o = T._transition_linear("fc3", h, tr.fc3_weight, compute_kernel_config=CKC, dtype=ttnn.bfloat16,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
    bad += tally(acc, "il.fc3", dt(o), H_ @ W3)
    for t in (h, o):
        ttnn.deallocate(t)
    if bad:
        acc.setdefault("il.blocks_bad", []).append(s)
    ttnn.deallocate(xn)
for s in range(0, S - rows + 1, rows):
    xn, Z1, Z2 = block(s, rows)
    # sharded: fc1/fc2 per K block, the product at the shipped K block, fc3 per K block
    for bw in (8, 4, 2, 1):
        x1 = ttnn.linear(xn, tr.fc1_weight, program_config=cfg(bw, pn, SILU), compute_kernel_config=SILU_CKC,
                         memory_config=MC, dtype=ttnn.bfloat16)
        x2 = ttnn.linear(xn, tr.fc2_weight, program_config=cfg(bw, pn), compute_kernel_config=CKC,
                         memory_config=MC, dtype=ttnn.bfloat16)
        A1, A2 = dt(x1), dt(x2)
        bad = tally(acc, f"sh.fc1_silu.bw{bw}", A1, F.silu(Z1)) + tally(acc, f"sh.fc2.bw{bw}", A2, Z2)
        if bad:
            acc[f"sh.fc1_silu.bw{bw}"]["blocks_bad"].append(s)
        if bw == 8:
            h = ttnn.multiply_(x1, x2)
            H_ = dt(h)
            tally(acc, "sh.mul", H_, A1 * A2)
            for bw3 in (pn, 2, 1):
                o = ttnn.linear(h, tr.fc3_weight, program_config=cfg(bw3, ct // gx), compute_kernel_config=CKC,
                                dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if tally(acc, f"sh.fc3.bw{bw3}", dt(o), H_ @ W3):
                    acc[f"sh.fc3.bw{bw3}"]["blocks_bad"].append(s)
                ttnn.deallocate(o)
            ttnn.deallocate(h)
        else:
            ttnn.deallocate(x1)
        ttnn.deallocate(x2)
    ttnn.deallocate(xn)
    print(json.dumps({"block": s, "bad_so_far": {k: v["n_bad"] for k, v in acc.items() if isinstance(v, dict)}}),
          flush=True)

res = {"host": os.uname().nodename, "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arch": ARCH, "S": S,
       "seed": a.seed, "ckc": a.ckc, "fidelity": str(CKC.math_fidelity), "rows": rows, "il_rows": il_rows, "grid": [gx, gy],
       "pm": pm, "pn": pn, "bad": a.bad, "stages": acc}
print(json.dumps(res), flush=True)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=1))
