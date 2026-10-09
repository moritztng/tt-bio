"""The transition's swiglu body (fc1 + silu, fc2, gate multiply, fc3) with its intermediates block-sharded.

    TT_VISIBLE_DEVICES=N python perf/spd_swiglu/shard_bench.py --out OUT.json [--mode normal|fast] [--rows pair:5,9]

The fold keeps fc1's, fc2's and the product's outputs interleaved in L1 (`Transition.swiglu`): every tile a matmul
core writes goes over the NoC to whichever core owns the page, the multiply reads both operands back the same way,
and fc3's in0 reader fetches the product again. Block-sharding all three on the matmul grid makes every one of those
transfers core-local: fc1 and fc2 write their own output block, the multiply is a per-core eltwise, and fc3 takes its
in0 straight from the shard (2D mcast, in0 block-sharded, K split over the grid's x). The arithmetic of fc1 and fc2
is the same program config, so their bytes do not move; fc3's K block becomes the shard width.

Arms per shape and row count:
  cur              what the fold runs today (ttnn.linear core_grid, interleaved L1 intermediates)
  bw               the same with the `transition_bw` K-block table
  shard:gXxY:bwB:bw3K  block-sharded intermediates on an X x Y grid, fc1/fc2 in0_block_w B, fc3 K block K from the
                   shard; K = 0 converts the product back to interleaved L1 and runs fc3 as the fold does
us per call and us per token row (rows x W), error of the body vs float64, digest of fc3's output.
"""
import argparse
import hashlib
import json
import os
import statistics as st
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--mode", default="normal", choices=("normal", "fast"))
ap.add_argument("--batches", type=int, default=7)
ap.add_argument("--W", type=int, default=736)
ap.add_argument("--rows", default="pair:5,9;msa:16,8")
a = ap.parse_args()
os.environ["TT_BIO_LEVERS"] = a.mode

import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()
dev = T.get_device()
ARCH = "wormhole" if T.is_wormhole() else "blackhole"
OPENED = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
                 if os.path.exists(f"/proc/self/fd/{fd}") and
                 os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")})
samples = []


def _sample():
    row = {}
    for n in OPENED:
        try:
            v = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0])
            if 100 <= v <= 3000:
                row[n] = v
        except Exception:
            pass
    samples.append((time.monotonic(), row))


def _sampler():
    while True:
        _sample()
        time.sleep(0.2)


threading.Thread(target=_sampler, daemon=True).start()


def clock(t0, t1):
    _sample()
    t1 = max(t1, samples[-1][0])
    v = sorted(x for ts, r in samples if t0 <= ts <= t1 for x in r.values())
    return dict(median=v[len(v) // 2], min=v[0], max=v[-1], n=len(v)) if v else None


def sync():
    ttnn.synchronize_device(dev)


def timed(fn, batches):
    for _ in range(2):
        ttnn.deallocate(fn())
    sync()
    ts = []
    t0 = time.monotonic()
    for _ in range(batches):
        sync()
        t = time.perf_counter()
        for _ in range(10):
            ttnn.deallocate(fn())
        sync()
        ts.append((time.perf_counter() - t) / 10 * 1e6)
    return dict(us_min=round(min(ts), 2), us_med=round(st.median(ts), 2), clock=clock(t0, time.monotonic()))


CKC_CLS = ttnn.WormholeComputeKernelConfig if ARCH == "wormhole" else ttnn.types.BlackholeComputeKernelConfig
T._LEVERS = T.parse_levers(a.mode)
# Protenix's trunk config as the fold builds it: HiFi4 base, the mode's fidelity lever on top.
CKC = T.trunk_compute_kernel_config(CKC_CLS(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True,
                                            fp32_dest_acc_en=True, packer_l1_acc=True))
SILU_CKC = T.silu_ckc(CKC)
B8 = T.lever("transition_b8")
WDT = ttnn.bfloat8_b if B8 else ttnn.bfloat16
HDT = ttnn.bfloat8_b if B8 else ttnn.bfloat16
GX, GY = T.COMPUTE_GRID_MAIN
CH = {"pair": (256, 1024), "msa": (128, 512)}
SILU = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)


def subblock(h, w):
    return max(((sh, sw) for sh in range(1, 5) for sw in range(1, 5)
                if sh * sw <= 4 and h % sh == 0 and w % sw == 0), key=lambda t: (t[0] * t[1], t[1]))


def block_sharded(gx, gy, h, w):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
                             ttnn.ShardSpec(grid, [h * 32, w * 32], ttnn.ShardOrientation.ROW_MAJOR))


def cfg2d(gx, gy, pm, pn, bw, act=None):
    sh, sw = subblock(pm, pn)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), in0_block_w=bw, out_subblock_h=sh, out_subblock_w=sw,
        out_block_h=pm, out_block_w=pn, per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=act,
        fuse_batch=True)


def body_cur(x, w1, w2, w3, transition_bw):
    names = a.mode + ("+transition_bw" if transition_bw else "")
    with T.levers(names):
        x1 = T._transition_linear("fc1", x, w1, silu=True, compute_kernel_config=SILU_CKC,
                                  memory_config=ttnn.L1_MEMORY_CONFIG, dtype=HDT)
        x2 = T._transition_linear("fc2", x, w2, compute_kernel_config=CKC, memory_config=ttnn.L1_MEMORY_CONFIG,
                                  dtype=HDT)
        h = ttnn.multiply_(x1, x2)
        ttnn.deallocate(x2)
        out = T._transition_linear("fc3", h, w3, compute_kernel_config=CKC, dtype=ttnn.bfloat16,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(h)
    return out


def body_shard(x, w1, w2, w3, gx, gy, bw, bw3):
    nt, ct = w1.shape[-1] // 32, w3.shape[-1] // 32
    mt = x.padded_shape[1] * x.padded_shape[2] // 32
    pm, pn = mt // gy, nt // gx
    mc = block_sharded(gx, gy, pm, pn)
    x1 = ttnn.linear(x, w1, program_config=cfg2d(gx, gy, pm, pn, bw, SILU), compute_kernel_config=SILU_CKC,
                     memory_config=mc, dtype=HDT)
    x2 = ttnn.linear(x, w2, program_config=cfg2d(gx, gy, pm, pn, bw), compute_kernel_config=CKC,
                     memory_config=mc, dtype=HDT)
    h = ttnn.multiply_(x1, x2)
    ttnn.deallocate(x2)
    if bw3 == 0:  # fc3's output does not split over gx: back to interleaved L1, fc3 as the fold runs it
        hi = ttnn.sharded_to_interleaved(h, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(h)
        with T.levers(a.mode + "+transition_bw"):
            out = T._transition_linear("fc3", hi, w3, compute_kernel_config=CKC, dtype=ttnn.bfloat16,
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(hi)
        return out
    out = ttnn.linear(h, w3, program_config=cfg2d(gx, gy, pm, ct // gx, bw3), compute_kernel_config=CKC,
                      dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(h)
    return out


def shard_arms(mt, kt, nt, ct):
    out = []
    for gx in sorted({GX, 8, 4}, reverse=True):
        if gx > GX or nt % gx:
            continue
        for gy in range(GY, 0, -1):
            if mt % gy == 0:
                break
        for bw in [b for b in (8, 4, 2, 1) if kt % b == 0][:2]:
            # bw3 0: product back to interleaved, fc3 on the fold's path (the only option when ct % gx)
            for bw3 in [b for b in (nt // gx, 2, 1) if (nt // gx) % b == 0 and not ct % gx] + [0]:
                out.append((f"shard:g{gx}x{gy}:bw{bw}:bw3{bw3}", (gx, gy, bw, bw3)))
    return list(dict(out).items())


res = {"host": os.uname().nodename, "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arch": ARCH, "nodes": OPENED,
       "grid": [GX, GY], "mode": a.mode, "fidelity": str(CKC.math_fidelity), "b8": B8, "shapes": {}}
g = torch.Generator().manual_seed(0)
for spec in a.rows.split(";"):
    name, rows = spec.split(":")
    C, HID = CH[name]
    w1h, w2h = (torch.randn(C, HID, generator=g) / C ** 0.5 for _ in range(2))
    w3h = torch.randn(HID, C, generator=g) / HID ** 0.5
    w1, w2, w3 = (ttnn.from_torch(t, dtype=WDT, layout=ttnn.TILE_LAYOUT, device=dev) for t in (w1h, w2h, w3h))
    W1, W2, W3 = (ttnn.to_torch(t).double().view(t.shape[-2], t.shape[-1]) for t in (w1, w2, w3))
    for R in (int(r) for r in rows.split(",")):
        xh = torch.randn(1, R, a.W, C, generator=g)
        x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.L1_MEMORY_CONFIG)
        X = ttnn.to_torch(x).double().view(-1, C)
        ref = (torch.nn.functional.silu(X @ W1) * (X @ W2)) @ W3
        mt = R * a.W // 32
        arms = [("cur", lambda: body_cur(x, w1, w2, w3, False)), ("bw", lambda: body_cur(x, w1, w2, w3, True))]
        arms += [(n, (lambda p=p: body_shard(x, w1, w2, w3, *p))) for n, p in shard_arms(mt, C // 32, HID // 32,
                                                                                            C // 32)]
        rows_out = {}
        for arm, fn in arms:
            try:
                o = fn()
                got = ttnn.to_torch(o).double().view(-1, C)
                ttnn.deallocate(o)
                row = dict(rel_rms=float((got - ref).norm() / ref.norm()),
                           digest=hashlib.sha256(got.float().numpy().tobytes()).hexdigest()[:16])
                row.update(timed(fn, a.batches))
                row["us_per_row"] = round(row["us_min"] / R, 3)
            except Exception as e:  # noqa: BLE001  an arm that does not fit L1 or is rejected is a result too
                row = dict(err=str(e).splitlines()[0][:240])
            rows_out[arm] = row
            print(json.dumps({f"{name}.r{R}": {arm: row}}), flush=True)
        res["shapes"][f"{name}.r{R}"] = dict(rows=R, W=a.W, mt=mt, arms=rows_out)
        ttnn.deallocate(x)
    for t in (w1, w2, w3):
        ttnn.deallocate(t)

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=1))
