#!/usr/bin/env python3
"""Essential-FLOP and DRAM-byte census of one real Pairformer block, by shape.

Wraps every ttnn entry point the trunk uses and records, per call, the input and
output shapes / dtypes / buffer types. FLOPs come from the shapes (2 * output
elements * K for a matmul), so the census is arithmetic ground truth, not a
guess at the architecture. Bytes are split into DRAM-resident operands (what an
implementation actually moves today) and the weight/activation footprint an
optimal implementation could not avoid.

    TT_MESH_GRAPH_DESC_PATH=<...>/p150_mesh_graph_descriptor.textproto \
    TT_VISIBLE_DEVICES=3 python3 perf/ceiling/flopcount.py --n 320 --out census_320.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch

import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "stage_split_298"))

from pf_layer import build_layer  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

REC = []
ACTIVE = [False]
DT_BYTES = {"BFLOAT16": 2, "FLOAT32": 4, "BFLOAT8_B": 1, "UINT32": 4, "INT32": 4, "UINT16": 2}


def _dtb(t):
    return DT_BYTES.get(str(t.dtype).split(".")[-1].upper(), 2)


def _info(t):
    try:
        shape = list(t.shape)
        buf = str(t.memory_config().buffer_type).split(".")[-1]
        return {"shape": shape, "dt": str(t.dtype).split(".")[-1],
                "buf": buf, "bytes": math.prod(shape) * _dtb(t)}
    except Exception:
        return None


def _wrap(mod, name, kind):
    fn = getattr(mod, name, None)
    if fn is None:
        return

    def w(*a, **k):
        out = fn(*a, **k)
        if ACTIVE[0]:
            ins = [i for i in (_info(x) for x in a if isinstance(x, ttnn.Tensor)) if i]
            for v in k.values():
                if isinstance(v, ttnn.Tensor):
                    i = _info(v)
                    if i:
                        ins.append(i)
            outs = []
            for o in (out if isinstance(out, (tuple, list)) else [out]):
                if isinstance(o, ttnn.Tensor):
                    i = _info(o)
                    if i:
                        outs.append(i)
            REC.append({"op": name, "kind": kind, "in": ins, "out": outs})
        return out

    setattr(mod, name, w)
    return fn


MM = ["matmul", "linear"]
EW = ["add", "add_", "multiply", "multiply_", "subtract", "sigmoid", "silu", "mul", "gelu"]
NORM = ["layer_norm", "rms_norm", "softmax"]
MOVE = ["permute", "concat", "chunk", "clone", "reshape", "slice", "transpose", "to_layout",
        "unsqueeze", "squeeze", "pad", "embedding", "reallocate"]


def install():
    for n in MM:
        _wrap(ttnn, n, "matmul")
    _wrap(ttnn.experimental, "minimal_matmul", "matmul")
    _wrap(ttnn.transformer, "scaled_dot_product_attention", "sdpa")
    _wrap(ttnn.experimental, "nlp_create_qkv_heads", "move")
    _wrap(ttnn.experimental, "nlp_concat_heads", "move")
    for n in EW:
        _wrap(ttnn, n, "eltwise")
    for n in NORM:
        _wrap(ttnn, n, "norm")
    for n in MOVE:
        _wrap(ttnn, n, "move")


def flops(r):
    """2 * output elements * contraction length. Exact for matmul/linear/SDPA."""
    if r["kind"] == "matmul":
        if not r["out"] or len(r["in"]) < 2:
            return 0
        o = r["out"][0]["shape"]
        k = r["in"][0]["shape"][-1]
        return 2 * math.prod(o) * k
    if r["kind"] == "sdpa":
        q = r["in"][0]["shape"]           # (B,H,S,D)
        kk = r["in"][1]["shape"]
        b, h, s, d = q[0], q[1], q[2], q[3]
        sk = kk[2]
        return 2 * b * h * s * sk * d + 2 * b * h * s * sk * d
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench-iters", type=int, default=7)
    args = ap.parse_args()

    dev = get_device()
    grid = dev.compute_with_storage_grid_size()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    layer, c_z = build_layer(ckc)
    N = args.n
    torch.manual_seed(0)
    s = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    install()
    for _ in range(2):
        s, z = layer(s, z)
    ttnn.synchronize_device(dev)

    ACTIVE[0] = True
    s, z = layer(s, z)
    ACTIVE[0] = False
    ttnn.synchronize_device(dev)

    # per-block wall, synced both sides
    ts = []
    for _ in range(args.bench_iters):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        s, z = layer(s, z)
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    block_ms = sorted(ts)[len(ts) // 2]

    tot_f = 0
    by_kind = OrderedDict()
    dram_r = dram_w = 0
    ops = []
    for r in REC:
        f = flops(r)
        tot_f += f
        k = by_kind.setdefault(r["kind"], {"n": 0, "flops": 0, "dram_in": 0, "dram_out": 0})
        k["n"] += 1
        k["flops"] += f
        di = sum(i["bytes"] for i in r["in"] if i["buf"] == "DRAM")
        do = sum(o["bytes"] for o in r["out"] if o["buf"] == "DRAM")
        k["dram_in"] += di
        k["dram_out"] += do
        dram_r += di
        dram_w += do
        ops.append({"op": r["op"], "kind": r["kind"], "flops": f, "dram_in": di, "dram_out": do,
                    "in": [f"{'x'.join(map(str, i['shape']))}:{i['dt']}:{i['buf']}" for i in r["in"]],
                    "out": [f"{'x'.join(map(str, o['shape']))}:{o['dt']}:{o['buf']}" for o in r["out"]]})

    res = {"n_padded": N, "c_z": c_z, "grid": [grid.x, grid.y], "n_cores": grid.x * grid.y,
           "block_ms_median": round(block_ms, 3), "block_ms_all": [round(t, 3) for t in ts],
           "total_flops": tot_f, "tflops_achieved": round(tot_f / (block_ms / 1e3) / 1e12, 2),
           "dram_read_bytes_today": dram_r, "dram_write_bytes_today": dram_w,
           "by_kind": by_kind, "n_calls": len(REC), "ops": ops}
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"N={N} c_z={c_z} grid={grid.x}x{grid.y} block={block_ms:.3f} ms "
          f"FLOPs={tot_f/1e9:.2f} G -> {res['tflops_achieved']} TFLOP/s", flush=True)
    for k, v in by_kind.items():
        print(f"  {k:8s} n={v['n']:4d} FLOPs={v['flops']/1e9:8.2f} G  "
              f"DRAMin={v['dram_in']/1e6:8.2f} MB  DRAMout={v['dram_out']/1e6:8.2f} MB", flush=True)
    print(f"  today DRAM read {dram_r/1e6:.1f} MB  write {dram_w/1e6:.1f} MB", flush=True)
    print("wrote", args.out, flush=True)


main()
