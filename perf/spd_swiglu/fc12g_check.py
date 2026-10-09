"""fc12g against today's sharded swiglu body: accuracy against float64 and time, one process.

    TT_VISIBLE_DEVICES=N PYTHONPATH=. python perf/spd_swiglu/fc12g_check.py --out OUT.json [--mode fast|normal]

h = silu(x @ w1) * (x @ w2) on the pair Transition shape (x [1, 9, 736, 256], 8x9 grid, per_core_M 23):
  base        today's body: fc1 + fused silu, fc2, multiply_, each block-sharded [23 x 4]
  fc12g_math  one 2D-mcast matmul over [w1_j | w2_j] with tt_bio/kernels/fc12g/compute_fc12g.cpp, silu on MATH
  fc12g_pack  the same with the silu on the PACK thread (FC12G_PACK_SILU)
Accuracy: rel rms and max |err| of h against float64 on the device's own operand values (x, w1, w2 read back), so
only the arithmetic differs. Time: pipelined, reps calls back to back and one sync, best of 3 rounds;
enqueue_us is the host's cost per call.
"""
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()  # noqa: E702,I001

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import statistics as st  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--mode", default="fast", choices=("normal", "fast"))
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--arms", default="base,fc12g_pack,fc12g_pack_alt,fc12g_lever,fc12g_math")
a = ap.parse_args()
os.environ["TT_BIO_LEVERS"] = a.mode

import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio import mm2d_generic as G  # noqa: E402
from tt_bio.mm2d_generic import generic_matmul_2d  # noqa: E402

dev = T.get_device()
T._LEVERS = T.parse_levers(a.mode)
ARCH = "wormhole" if T.is_wormhole() else "blackhole"
NODES = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
                if os.path.exists(f"/proc/self/fd/{fd}") and
                os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")})
clk = []


def _sampler():
    while True:
        for n in NODES:
            try:
                v = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0])
                if 100 <= v <= 3000:
                    clk.append((time.monotonic(), v))
            except Exception:  # noqa: BLE001
                pass
        time.sleep(0.05)


threading.Thread(target=_sampler, daemon=True).start()
CKC_CLS = ttnn.WormholeComputeKernelConfig if ARCH == "wormhole" else ttnn.types.BlackholeComputeKernelConfig
CKC = T.trunk_compute_kernel_config(CKC_CLS(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True,
                                            fp32_dest_acc_en=True, packer_l1_acc=True))
SILU_CKC = T.silu_ckc(CKC)
B8 = T.lever("transition_b8")
WDT = HDT = ttnn.bfloat8_b if B8 else ttnn.bfloat16
C, HID, R, W = 256, 1024, 9, 736
KERNEL = Path(T.__file__).resolve().parent / "kernels" / "fc12g" / "compute_fc12g.cpp"

g = torch.Generator().manual_seed(0)
w1h, w2h = (torch.randn(C, HID, generator=g) / C ** 0.5 for _ in range(2))
xh = torch.randn(1, R, W, C, generator=g)
mt, nt = R * W // 32, HID // 32
gx, gy = T._transition_shard_grid(mt, nt, C // 32)
pm, pn, bw = mt // gy, nt // gx, 8
grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
                       ttnn.ShardSpec(grid, [pm * 32, pn * 32], ttnn.ShardOrientation.ROW_MAJOR))


def cfg(n, sw, act=None):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), in0_block_w=bw, out_subblock_h=1, out_subblock_w=sw,
        out_block_h=pm, out_block_w=n, per_core_M=pm, per_core_N=n, transpose_mcast=False, fused_activation=act,
        fuse_batch=True)


b = HID // gx
w12h = torch.cat([torch.cat([w1h[:, j * b:(j + 1) * b], w2h[:, j * b:(j + 1) * b]], 1) for j in range(gx)], 1)
x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
w1, w2, w12 = (ttnn.from_torch(t, dtype=WDT, layout=ttnn.TILE_LAYOUT, device=dev) for t in (w1h, w2h, w12h))
xd, w1d, w2d = (ttnn.to_torch(t).double() for t in (x, w1, w2))
ref = torch.nn.functional.silu(xd @ w1d) * (xd @ w2d)
assert pn == 4, pn  # fc12g's subblock is one fc half: 1 x pn tiles, which must fit an fp32 dest half

PC1, PC2, PC12 = cfg(pn, pn, ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)), cfg(pn, pn), cfg(2 * pn, pn)


def base():
    x1 = ttnn.linear(x, w1, program_config=PC1, compute_kernel_config=SILU_CKC, memory_config=mc, dtype=HDT)
    x2 = ttnn.linear(x, w2, program_config=PC2, compute_kernel_config=CKC, memory_config=mc, dtype=HDT)
    h = ttnn.multiply_(x1, x2)
    ttnn.deallocate(x2)
    return h


def defines(pack, math_silu=0):
    d = {"FC12G_PACK_SILU": "1"} if pack else {}
    if pack and math_silu:
        d["FC12G_MATH_SILU"] = str(math_silu)
    return d


# fc12g's outputs are allocated on its first call, after base has run: in normal mode two live [23 x 4] shards
# beside x leave no room for stock fc1's fp32 c_5 (L1 clash).
outs = []
flip = [0]


def out_pair():
    if not outs:
        outs.extend(ttnn.allocate_tensor_on_device(ttnn.Shape([1, R, W, HID]), HDT, ttnn.TILE_LAYOUT, dev, mc)
                    for _ in range(2))
    return outs


def fc12g(pack, alternate=False, math_silu=0):
    def run():
        out_g, out_alt = out_pair()
        o = out_g
        if alternate:  # a new output buffer every call, as in the fold: the cached runtime args are rebound
            flip[0] ^= 1
            o = out_alt if flip[0] else out_g
        return generic_matmul_2d(dev, x, w12, o, PC12, SILU_CKC, out_nzsb_w=1, compute_src=str(KERNEL),
                                 compute_defines=defines(pack, math_silu), gate_tiles=2 * pn)
    return run


ARMS = {"base": (base, True), "fc12g_math": (fc12g(False), False), "fc12g_pack": (fc12g(True), False),
        "fc12g_pack_alt": (fc12g(True, True), False),
        **{f"fc12g_split{k}": (fc12g(True, False, k), False) for k in (1, 2, 3)},
        # the lever's own call: a fresh output tensor every call and the caller's short cache key
        "fc12g_lever": (lambda: T._fc12g(x, w12, mc, pm, pn, bw, gx, gy, SILU_CKC, HDT), True)}
arms = [s for s in a.arms.split(",") if s]
res = {"host": os.uname().nodename, "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arch": ARCH, "mode": a.mode,
       "b8": B8, "grid": [gx, gy], "pm": pm, "pn": pn, "fidelity": str(SILU_CKC.math_fidelity),
       "fp32_dest_acc": bool(SILU_CKC.fp32_dest_acc_en), "arms": {}}
for name in arms:
    fn, owns = ARMS[name]
    row = res["arms"][name] = {}
    try:
        o = fn()
        got = ttnn.to_torch(o).double()
        if owns:
            ttnn.deallocate(o)
        e = got - ref
        row.update(rel_rms=float(e.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()), max_err=float(e.abs().max()),
                   n_over_0p1=int((e.abs() > 0.1).sum()), finite=bool(torch.isfinite(got).all()))
        o2 = fn()
        row["repeat_equal"] = bool(torch.equal(ttnn.to_torch(o2).double(), got))
        if owns:
            ttnn.deallocate(o2)
    except Exception as ex:  # noqa: BLE001
        row["err"] = str(ex).splitlines()[0][:300]
    print(json.dumps({name: row}), flush=True)

live = [n for n in arms if "err" not in res["arms"][n]]
t0 = time.monotonic()
for n in live:
    fn, owns = ARMS[n]
    for i in range(3):
        o = fn()
        if owns:
            ttnn.deallocate(o)
    best = []
    for _ in range(3):  # pipelined: reps calls back to back, one sync; best of 3 rounds
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        for _ in range(a.reps):
            o = fn()
            if owns:
                ttnn.deallocate(o)
        te = time.perf_counter()
        ttnn.synchronize_device(dev)
        best.append(((time.perf_counter() - t) * 1e6 / a.reps, (te - t) * 1e6 / a.reps))
    us, enq = min(best)
    res["arms"][n].update(us=round(us, 1), enqueue_us=round(enq, 1))
# Host cost of the generic path's pieces, per call (no device work in any of them).
if "fc12g_pack" in live:
    out_g, out_alt = out_pair()
    args = (x, w12, out_g, PC12, SILU_CKC, 1, str(KERNEL), defines(True), None, 2 * pn)
    entry = G._CACHE[G._key(*args)]
    host = {}
    t = time.perf_counter()
    for _ in range(200):
        G._key(*args)
    host["key_us"] = round((time.perf_counter() - t) * 1e6 / 200, 1)
    t = time.perf_counter()
    for i in range(100):
        G.rebind(entry, x, w12, out_alt if i % 2 == 0 else out_g)
    host["rebind_us"] = round((time.perf_counter() - t) * 1e6 / 100, 1)
    t = time.perf_counter()
    for _ in range(200):
        (x.buffer_address(), w12.buffer_address(), out_g.buffer_address())
    host["addr_us"] = round((time.perf_counter() - t) * 1e6 / 200, 1)
    G.rebind(entry, x, w12, out_g)
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(a.reps):
        ttnn.generic_op([x, w12, out_g], entry["pd"])
    host["generic_op_us"] = round((time.perf_counter() - t) * 1e6 / a.reps, 1)
    ttnn.synchronize_device(dev)
    res["host"] = host
    print(json.dumps({"host": host}), flush=True)
v = sorted(c for t_, c in clk if t_ >= t0)
res["aiclk"] = dict(median=v[len(v) // 2], min=v[0]) if v else None
print(json.dumps({n: {k: res["arms"][n].get(k) for k in ("us", "enqueue_us", "rel_rms")} for n in arms},
                 ), json.dumps(res["aiclk"]), flush=True)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=1))
