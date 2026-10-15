#!/usr/bin/env python3
"""Is a weight-resident generic_op `pair_linear` faster than minimal_matmul at BindCraft 2's pair projections?

Scope: state/spd-gap-bind-jobs/pairlin_scope.md. On a Wormhole chip minimal_matmul runs the [50176, 128] x [128, 128]
pair projection at ~0.22 ms (perf/spd/af2_pair_linear_sweep.py, .114 out/plin_d07ffb063/s224.json, best of 50 blocks),
because with M > N it puts N on the 9-core grid axis and pads N's 4 tiles to 9: 5 of 9 core rows compute padding,
and the 32 useful cores each do 196 x 4 HiFi4 tile products (~0.2 ms). The prototype here (kernels in
perf/spd/pair_linear_kernels, writer shared with tt_bio/kernels/pair_swiglu_ws) is pair_swiglu_ws's machinery with
one matmul: every core holds the whole weight in L1 and takes whole 32-row tiles of x, so all cores do useful work
and x is read from DRAM exactly once with no core-to-core forwarding.

Arms per case: `mm` minimal_matmul with pair_mm's shipped block config (pair_mm.config), `linear` ttnn.linear on
CORE_GRID_MAIN (the pre-pair_mm path), `proto` the prototype at HiFi4 (fp32 dest), and at N=128 `proto_hifi2` (is
the prototype compute- or DRAM-bound). Each arm: warm 2, median of REPS reps of INNER back-to-back calls between two
device syncs, per-call ms. rel L2 against a float64 torch reference, and `proto_vs_mm` (rel L2 of proto against mm).

usage (one chip, tt-bio's device path):
    TT_VISIBLE_DEVICES=N timeout 600 python perf/spd/pair_linear_gate.py --out OUT.json
Writes {arch, grid, n, ms: {case: {arm: ms}}, rel_l2: {case: {arm: v}}} after every case, so a hang keeps the rest.
The bar (scope): proto >= 1.5x over mm at n128 with rel L2 within 1.2x of mm's.
"""
try:
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
except Exception:   # an import-time failure here must not hide the device error that follows
    pass

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import ttnn  # noqa: E402

import tt_bio.af2 as A  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio import core_split, pair_mm  # noqa: E402

KDIR = Path(__file__).resolve().parent / "pair_linear_kernels"
WRITER = REPO / "tt_bio" / "kernels" / "pair_swiglu_ws" / "writer.cpp"
X_CB, W_CB, B_CB, OUT_CB = 0, 1, 2, 16
X_DEPTH = 3            # row tiles of x in flight per core
_TILE = 32 * 32 * 2
S = 224
# (case, K, N, bias): the forward projections, the dX of the qkv projection ([.., 384] x [384, 128], the weight's
# cached transpose), and N=512 with bias for a like-for-like check of the bias seed.
CASES = [("n128", 128, 128, False), ("n384", 128, 384, False), ("n512", 128, 512, False),
         ("dx_k384_n128", 384, 128, False), ("n512_bias", 128, 512, True),
         ("n4", 128, 4, False), ("dx_k4_n128", 4, 128, False)]
# n4 / dx_k4_n128: the pair-bias projection [.., 128] x [128, 4] and its dX [.., 4] x [4, 128]. Tile padding makes
# them one-tile N / K; the padded K rows of the weight are zero, so dx_k4 is exact while x's padding is finite.
WARM, REPS, INNER = 2, 5, 10
_PROGRAMS: dict = {}


def _program(x, w, b, out, fidelity):
    """Kernel and CB descriptors for x [.., K] @ w [K, N], cached on everything but the buffer addresses."""
    dev = x.device()
    xs = [int(d) for d in x.padded_shape]
    mt = 1
    for d in xs[:-1]:
        mt *= d
    mt //= 32
    kt, ct = xs[-1] // 32, int(w.padded_shape[-1]) // 32
    key = (tuple(xs), ct, fidelity, b is not None)
    if key in _PROGRAMS:
        return _PROGRAMS[key]
    g = dev.compute_with_storage_grid_size()
    _, grid, cg1, cg2, work1, work2 = core_split.split_work_to_cores(g, mt, 0)

    def cb(idx, tiles):
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=_TILE)
        return ttnn.CBDescriptor(total_size=tiles * _TILE, core_ranges=grid, format_descriptors=[fmt])

    cbs = [cb(X_CB, X_DEPTH * kt), cb(W_CB, kt * ct), cb(OUT_CB, 2 * ct)]
    defines = []
    if b is not None:
        cbs.append(cb(B_CB, ct))
        defines.append(("PL_BIAS", "1"))
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    first = 0
    for group, per in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    rd[cx][cy] = cp[cx][cy] = wr[cx][cy] = [first, per]
                    first += per
    assert first == mt, (first, mt)
    rct = [kt, ct]
    for t in (x, w) + (() if b is None else (b,)):
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = [ct] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    path = ttnn.KernelDescriptor.SourceType.FILE_PATH
    reader = ttnn.KernelDescriptor(kernel_source=str(KDIR / "reader.cpp"), source_type=path, core_ranges=grid,
                                   compile_time_args=rct, runtime_args=rd, common_runtime_args=[0] * 3,
                                   defines=defines, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(kernel_source=str(WRITER), source_type=path, core_ranges=grid,
                                   compile_time_args=wct, runtime_args=wr, common_runtime_args=[0],
                                   defines=[], config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "compute.cpp"), source_type=path, core_ranges=grid, compile_time_args=[kt, ct],
        runtime_args=cp, defines=defines,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True))
    _PROGRAMS[key] = ([reader, writer, compute], cbs)
    return _PROGRAMS[key]


def pair_linear(x, w, b=None, fidelity=ttnn.MathFidelity.HiFi4):
    """x [.., K] @ w [K, N] (+ b, row-replicated [32, N]) as a new bf16 DRAM tensor; all TILE, interleaved."""
    dev = x.device()
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([int(d) for d in x.shape][:-1] + [int(w.shape[-1])]),
                                         ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    kernels, cbs = _program(x, w, b, out, fidelity)
    kernels[0].common_runtime_args = [x.buffer_address(), w.buffer_address(), 0 if b is None else b.buffer_address()]
    kernels[1].common_runtime_args = [out.buffer_address()]
    pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    return ttnn.generic_op([x, w] + ([] if b is None else [b]) + [out], pd)


def timed(dev, fn):
    for _ in range(WARM):
        o = fn()
        ttnn.synchronize_device(dev)
        ttnn.deallocate(o)
    ts = []
    for _ in range(REPS):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(INNER):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) / INNER)
    o = fn()
    ttnn.synchronize_device(dev)
    return statistics.median(ts), o


def rel_l2(got, ref):
    return float((got.double() - ref).norm() / ref.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=",".join(c[0] for c in CASES))
    a = ap.parse_args()
    dev = T.get_device()
    ckc = A.compute_kernel_config()
    DRAM = ttnn.DRAM_MEMORY_CONFIG
    up = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                   device=dev, memory_config=DRAM)
    res = {"arch": "wormhole" if T.is_wormhole() else "blackhole",
           "grid": [int(dev.compute_with_storage_grid_size().x), int(dev.compute_with_storage_grid_size().y)],
           "s": S, "warm": WARM, "reps": REPS, "inner": INNER, "loadavg": open("/proc/loadavg").read().split()[:3],
           "n": [], "ms": {}, "rel_l2": {}, "errors": {}}
    torch.manual_seed(0)
    want = set(a.cases.split(","))
    for name, K, N, bias in CASES:
        if name not in want:
            continue
        xt = torch.randn(1, S, S, K).to(torch.bfloat16)
        wt = (torch.randn(K, N) / K ** 0.5).to(torch.bfloat16)
        bt = (torch.randn(N) * 0.1).to(torch.bfloat16) if bias else None
        ref = xt.double() @ wt.double()
        if bias:
            ref = ref + bt.double()
        x, w = up(xt), up(wt)
        b_mm = up(bt.reshape(1, N)) if bias else None              # minimal_matmul / linear: a [1, N] bias
        b_pl = up(bt.reshape(1, N).expand(32, N)) if bias else None  # the prototype: row-replicated tiles
        cfg = pair_mm.config(x, w)
        arms = {
            "mm": (lambda: ttnn.experimental.minimal_matmul(
                input_tensor=x, weight_tensor=w, bias_tensor=b_mm, compute_kernel_config=ckc,
                dtype=ttnn.bfloat16, config=cfg)) if cfg is not None else None,
            "linear": lambda: ttnn.linear(x, w, bias=b_mm, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                          memory_config=DRAM, core_grid=T.CORE_GRID_MAIN),
            "proto": lambda: pair_linear(x, w, b_pl),
        }
        if name == "n128":
            arms["proto_hifi2"] = lambda: pair_linear(x, w, b_pl, ttnn.MathFidelity.HiFi2)
        ms, rl, err, outs = {}, {}, {}, {}
        for arm, fn in arms.items():
            if fn is None:
                err[arm] = "pair_mm.config declined this shape"
                continue
            try:
                s, o = timed(dev, fn)
                got = ttnn.to_torch(o).reshape(ref.shape)
                ttnn.deallocate(o)
                ms[arm] = round(1e3 * s, 4)
                rl[arm] = rel_l2(got, ref)
                outs[arm] = got
            except Exception as e:  # noqa: BLE001 -- a refused arm is an answer, not a crash
                err[arm] = f"{type(e).__name__}: {str(e)[:200]}"
            print(name, arm, ms.get(arm), rl.get(arm), err.get(arm), flush=True)
        if "proto" in outs and "mm" in outs:
            rl["proto_vs_mm"] = rel_l2(outs["proto"], outs["mm"].double())
        if "proto" in ms and "mm" in ms:
            ms["mm_over_proto"] = round(ms["mm"] / ms["proto"], 3)
        if "proto" in ms and "linear" in ms:
            ms["linear_over_proto"] = round(ms["linear"] / ms["proto"], 3)
        res["n"].append(N)
        res["ms"][name], res["rel_l2"][name] = ms, rl
        if err:
            res["errors"][name] = err
        for t in (x, w, b_mm, b_pl):
            if t is not None:
                ttnn.deallocate(t)
        Path(a.out).write_text(json.dumps(res, indent=1))
    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
