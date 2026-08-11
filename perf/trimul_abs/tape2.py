#!/usr/bin/env python3
"""Close the hole in `perf/trimul_root/tape.py`: it wraps only `ttnn.*` entry points, so the 16
`_channel_move` calls (which reach `tt_bio.reblock_permute.reblock_permute`, a `generic_op`) are
invisible. Its 12 listed classes sum to 28.145 ms against a 28.485 ms wall, leaving 0.34 ms for
268 MB read + 268 MB write, which is 1576 GB/s and impossible. This tapes the move too.

Also measures every alternative for the two layout moves at 512 aa, and whether the channel loop's
tensors can be L1-resident at this size.
"""
import collections, json, statistics as st, time
from pathlib import Path
import sys

import torch
import ttnn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402
import tt_bio.reblock_permute as RB  # noqa: E402
from tt_bio.tenstorrent import COMPUTE_GRID_MAIN, get_device  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
layer, c_z = build_layer(ckc)
tm = layer.triangle_multiplication_start
torch.manual_seed(0)
z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev,
                    dtype=ttnn.bfloat16)

OUT = {"n": N, "c_z": c_z, "hidden": tm._hidden, "grid": list(COMPUTE_GRID_MAIN)}
ROWS, ON = [], [False]


def shp(t):
    try:
        return "x".join(str(d) for d in t.shape)
    except Exception:
        return "?"


def buf(kw):
    mc = kw.get("memory_config")
    if mc is None:
        return "-"
    return "L1" if mc.buffer_type == ttnn.BufferType.L1 else "DRAM"


def wrap(mod, name, tagger):
    orig = getattr(mod, name)

    def f(*a, **kw):
        if not ON[0]:
            return orig(*a, **kw)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = orig(*a, **kw)
        ttnn.synchronize_device(dev)
        ROWS.append((tagger(a, kw), (time.perf_counter() - t0) * 1e3))
        return out
    setattr(mod, name, f)


wrap(ttnn, "matmul", lambda a, kw: f"matmul[{shp(a[0])}@{shp(a[1])}]->{buf(kw)}")
wrap(ttnn, "linear", lambda a, kw: f"linear[{shp(a[0])}@{shp(a[1])}]->{buf(kw)}")
wrap(ttnn.experimental, "minimal_matmul",
     lambda a, kw: f"minimal_matmul[{shp(a[0])}@{shp(a[1])}]->{buf(kw)}")
wrap(ttnn, "permute", lambda a, kw: f"permute{tuple(a[1])}[{shp(a[0])}]->{buf(kw)}")
wrap(ttnn, "transpose", lambda a, kw: f"transpose({a[1]},{a[2]})[{shp(a[0])}]->{buf(kw)}")
wrap(ttnn, "layer_norm", lambda a, kw: f"layer_norm[{shp(a[0])}]")
wrap(ttnn, "multiply_", lambda a, kw: f"multiply_[{shp(a[0])}]")
wrap(ttnn, "chunk", lambda a, kw: f"chunk{kw.get('chunks', '')}[{shp(a[0])}]")
wrap(ttnn, "concat", lambda a, kw: f"concat[{len(a[0])}x{shp(a[0][0])}]")
wrap(ttnn, "clone", lambda a, kw: f"clone[{shp(a[0])}]->{buf(kw)}")
wrap(ttnn, "reallocate", lambda a, kw: f"reallocate[{shp(a[0])}]")
# THE FIX: the channel move goes through tt_bio's own generic_op, not a ttnn entry point.
wrap(RB, "reblock_permute", lambda a, kw: f"REBLOCK_permute(0,3,1,2)[{shp(a[0])}]->{buf(kw)}")
wrap(T, "_channel_move", lambda a, kw: f"_channel_move[{shp(a[0])}]")


def once():
    return tm(z, None)


def module_wall():
    for _ in range(3):
        ttnn.deallocate(once())
    ttnn.synchronize_device(dev)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = [once() for _ in range(6)]
    ttnn.synchronize_device(dev)
    pipe = (time.perf_counter() - t0) * 1e3 / 6
    for o in outs:
        ttnn.deallocate(o)
    return pipe


def tape(label):
    global ROWS
    ROWS = []
    RB.STATS[0] = RB.STATS[1] = 0
    wall = module_wall()
    served_wall, decl_wall = RB.STATS[0], RB.STATS[1]
    ON[0] = True
    ttnn.deallocate(once())
    ON[0] = False
    agg = collections.OrderedDict()
    for tag, ms in ROWS:
        a = agg.setdefault(tag, dict(tag=tag, n=0, ms=0.0))
        a["n"] += 1
        a["ms"] += ms
    # `_channel_move` brackets REBLOCK_permute; report both but only count the outer once.
    inner = sum(a["ms"] for a in agg.values() if a["tag"].startswith("REBLOCK"))
    total = sum(a["ms"] for a in agg.values() if not a["tag"].startswith("REBLOCK"))
    rows = sorted(agg.values(), key=lambda a: -a["ms"])
    print(f"\n=== {label}: module wall {wall:.3f} ms | taped sum {total:.3f} ms "
          f"({total / wall:.3f}x) | reblock served {served_wall // 6} declined {decl_wall // 6} "
          f"per call ===")
    for a in rows:
        a["ms"] = round(a["ms"], 4)
        a["share"] = round(a["ms"] / total, 4)
        mark = "  (inner)" if a["tag"].startswith("REBLOCK") else ""
        print(f"{a['tag']:60s} {a['n']:3d} {a['ms']:8.3f} {100 * a['share']:5.1f}%{mark}")
    if RB.REJECTS:
        print("reblock refusals:", dict(RB.REJECTS))
    return dict(label=label, wall_ms=round(wall, 4), taped_sum_ms=round(total, 4),
                reblock_inner_ms=round(inner, 4), addivity=round(total / wall, 4),
                reblock_served=served_wall // 6, reblock_declined=decl_wall // 6, ops=rows)


tapes = []
for g in (1, 8):
    T._TRIMUL_INPROJ_GROUP = g
    tapes.append(tape(f"G={g}"))
T._TRIMUL_INPROJ_GROUP = 1
OUT["tapes"] = tapes

# ---------------------------------------------------------------- move-op alternatives at 512 aa
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def timeit(fn, warm=3, reps=5):
    for _ in range(warm):
        r = fn()
        if r is not None:
            ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        if r is not None:
            ttnn.deallocate(r)
    return st.median(ts)


def mb(t):
    v = 1
    for d in t.shape:
        v *= int(d)
    return v * 2 / 2 ** 20


moves = []


def rec(name, ms, nbytes_mb, ref=None, extra=""):
    gbs = nbytes_mb / 1024 / (ms / 1e3)
    r = dict(name=name, ms=round(ms, 4), mb_each_way=round(nbytes_mb, 1),
             gbs_each_way=round(gbs, 1), extra=extra)
    if ref:
        r["vs_ref"] = round(ref / ms, 4)
    moves.append(r)
    print(f"{name:56s} {ms:8.4f} ms  {gbs:7.1f} GB/s each way"
          + (f"  {ref / ms:.3f}x" if ref else "") + ("  " + extra if extra else ""))
    return ms


print("\n=== the FORWARD channel move (0,3,1,2), [1,N,N,C] -> [1,C,N,N] ===")
for C in (32, 256):
    src_d = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
    for tag, mc in (("DRAM", DRAM), ("L1", L1)):
        if tag == "L1" and mb(src_d) * 2 > 100:
            continue
        el = RB.eligible(src_d, mc)
        base = rec(f"fwd C={C} ttnn.permute ->{tag}",
                   timeit(lambda: ttnn.permute(src_d, (0, 3, 1, 2), memory_config=mc)), mb(src_d))
        if el:
            rec(f"fwd C={C} REBLOCK ->{tag}",
                timeit(lambda: RB.reblock_permute(src_d, mc)), mb(src_d), base,
                extra="eligible")
        else:
            print(f"    (reblock NOT eligible ->{tag})")
    ttnn.deallocate(src_d)

print("\n=== the BACK channel move (0,2,3,1), [1,C,N,N] -> [1,N,N,C] ===")
for C in (32, 256):
    src_c = ttnn.from_torch(torch.randn(1, C, N, N), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=DRAM)
    ref_t = ttnn.to_torch(src_c).permute(0, 2, 3, 1)
    for tag, mc in (("DRAM", DRAM), ("L1", L1)):
        if tag == "L1" and mb(src_c) * 2 > 100:
            continue

        def two_transpose():
            a = ttnn.transpose(src_c, 1, 2, memory_config=mc)
            b = ttnn.transpose(a, 2, 3, memory_config=mc)
            ttnn.deallocate(a)
            return b

        base = rec(f"back C={C} transpose(1,2)+transpose(2,3) ->{tag}  [TODAY]",
                   timeit(two_transpose), mb(src_c))
        rec(f"back C={C} single ttnn.permute(0,2,3,1) ->{tag}",
            timeit(lambda: ttnn.permute(src_c, (0, 2, 3, 1), memory_config=mc)), mb(src_c), base)
        r = two_transpose()
        ok = torch.equal(ttnn.to_torch(r), ref_t)
        ttnn.deallocate(r)
        print(f"    two-transpose == torch permute: {ok}")
    ttnn.deallocate(src_c)

print("\n=== same-bytes reference: clone of the loop tensors ===")
for shape in ((1, 32, N, N), (1, N, N, 32)):
    s = ttnn.from_torch(torch.randn(*shape), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    rec(f"clone{list(shape)} DRAM->DRAM", timeit(lambda: ttnn.clone(s, memory_config=DRAM)), mb(s))
    rec(f"clone{list(shape)} DRAM->L1", timeit(lambda: ttnn.clone(s, memory_config=L1)), mb(s))
    ttnn.deallocate(s)
OUT["moves"] = moves

# ---------------------------------------------------------------- can the channel loop be L1?
print("\n=== L1 capacity for the channel loop at 512 aa ===")
try:
    v = dev.get_memory_view(ttnn.BufferType.L1)
    print("L1 view:", v)
except Exception as e:
    print("get_memory_view:", e)
cap = {}
for name, fn in (("max_worker_l1_unreserved", lambda: ttnn.get_max_worker_l1_unreserved_size()),):
    try:
        cap[name] = fn()
    except Exception as e:
        cap[name] = str(e)
print(cap)
OUT["l1"] = cap

l1_res = []
for C in (32, 64, 128):
    held = []
    try:
        for _ in range(3):  # a_chunk, b_chunk, x_chunk
            held.append(ttnn.from_torch(torch.zeros(1, C, N, N), layout=ttnn.TILE_LAYOUT,
                                        device=dev, dtype=ttnn.bfloat16, memory_config=L1))
        # and the contraction on top of them
        pc = T._triangle_mul_program_config((N + 31) // 32)
        t = timeit(lambda: ttnn.matmul(held[0], held[1], compute_kernel_config=ckc,
                                       memory_config=L1, program_config=pc,
                                       dtype=ttnn.bfloat16))
        ok = f"3x[1,{C},{N},{N}] L1-resident ({3 * C * N * N * 2 / 2**20:.1f} MB) + L1 matmul {t:.4f} ms"
        l1_res.append(dict(C=C, ok=True, mb=round(3 * C * N * N * 2 / 2 ** 20, 1),
                           l1_matmul_ms=round(t, 4)))
    except Exception as e:
        ok = f"3x[1,{C},{N},{N}] ({3 * C * N * N * 2 / 2**20:.1f} MB): {type(e).__name__} {str(e)[:120]}"
        l1_res.append(dict(C=C, ok=False, mb=round(3 * C * N * N * 2 / 2 ** 20, 1), err=str(e)[:200]))
    print("  " + ok)
    for h in held:
        ttnn.deallocate(h)
OUT["l1_residency"] = l1_res

Path(HERE).mkdir(parents=True, exist_ok=True)
Path(HERE / f"tape2_{N}_qb2c0.json").write_text(json.dumps(OUT, indent=2))
print("\nWROTE " + str(HERE / f"tape2_{N}_qb2c0.json"))
