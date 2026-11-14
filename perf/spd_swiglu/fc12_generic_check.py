"""S0 kill gate for fc12g: the stock 2D-mcast matmul re-driven through generic_op (tt_bio/mm2d_generic.py).

    TT_VISIBLE_DEVICES=N PYTHONPATH=. python perf/spd_swiglu/fc12_generic_check.py --out OUT.json [--mode fast|normal]

Operands as shard_bench.py's `--stages` fc12 arm: x [1, 9, 736, 256] bf16 in L1 interleaved, the pair Transition's
fc1|fc2 weights column-interleaved per core column into w12 [256, 2048] in DRAM (bfp8_b under transition_b8),
2D mcast on 8x9, per_core_M 23, per_core_N 8, in0_block_w 8, output block-sharded [23x8] tiles in the hidden dtype.
Cases:
  fc12       w12, no activation, the trunk CKC                (the call fc12g replaces)
  fc1_silu   w1 [256, 1024], fused SILU, silu_ckc(CKC), [23x4] (exercises the activation defines)
Each: the stock ttnn.linear and generic_matmul_2d on the same inputs -> torch.equal, max |diff|, differing count;
then 3 warm calls each and 20 paired reps (stock, generic alternating, synchronize_device around each call),
min and median us, AICLK sampled from /sys/class/tenstorrent/tenstorrent!<node>/tt_aiclk during the reps.
Stock times include its output allocation; generic writes into one pre-allocated tensor.

KILL CRITERION (written before the run): generic fc12 within +-3 % of stock fc12 (stock measured 143.1 us at
AICLK 1000, fast, WH) AND bit-identical (torch.equal). The ratio is taken on pipelined time (reps calls, one
sync): synced single calls read the host's dispatch (stock 242 us host-timed for a 143 us op, g12 16:26Z). Otherwise the generic_op route to fc12g is dead.
A case whose stock call raises (e.g. an L1 circular-buffer clash: normal mode's fp32 c_5 is 736 KB) is recorded as
an error and skipped, never retried.
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
ap.add_argument("--warm", type=int, default=3)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--R", type=int, default=9, help="pair rows per block (9 -> M = 207 tiles)")
ap.add_argument("--W", type=int, default=736)
a = ap.parse_args()
os.environ["TT_BIO_LEVERS"] = a.mode

import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.mm2d_generic import generic_matmul_2d, _CACHE  # noqa: E402

dev = T.get_device()
T._LEVERS = T.parse_levers(a.mode)
ARCH = "wormhole" if T.is_wormhole() else "blackhole"
NODES = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
                if os.path.exists(f"/proc/self/fd/{fd}") and
                os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")})
clk = []
_stop = threading.Event()


def _read_clk():
    for n in NODES:
        try:
            v = int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0])
            if 100 <= v <= 3000:
                clk.append((time.monotonic(), v))
        except Exception:  # noqa: BLE001
            pass


def _sampler():
    while not _stop.is_set():
        _read_clk()
        time.sleep(0.05)


threading.Thread(target=_sampler, daemon=True).start()


def aiclk_since(t0):
    _read_clk()
    v = sorted(c for ts, c in clk if ts >= t0)
    return dict(median=v[len(v) // 2], min=v[0], max=v[-1], n=len(v)) if v else None


def sync():
    ttnn.synchronize_device(dev)


CKC_CLS = ttnn.WormholeComputeKernelConfig if ARCH == "wormhole" else ttnn.types.BlackholeComputeKernelConfig
# Protenix's trunk config as the fold builds it (shard_bench.py / b8fid_check.py).
CKC = T.trunk_compute_kernel_config(CKC_CLS(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True,
                                            fp32_dest_acc_en=True, packer_l1_acc=True))
SILU_CKC = T.silu_ckc(CKC)
B8 = T.lever("transition_b8")
WDT = ttnn.bfloat8_b if B8 else ttnn.bfloat16
HDT = ttnn.bfloat8_b if B8 else ttnn.bfloat16
SILU = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
C, HID = 256, 1024


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


def interleave_cols(w1h, w2h, gx):
    """[C, 2*HID] whose j-th of gx column blocks is fc1's j-th block then fc2's (shard_bench.py)."""
    b = w1h.shape[1] // gx
    return torch.cat([torch.cat([w1h[:, j * b:(j + 1) * b], w2h[:, j * b:(j + 1) * b]], 1) for j in range(gx)], 1)


def stats(ts):
    return dict(us_min=round(min(ts), 2), us_med=round(st.median(ts), 2), n=len(ts))


res = {"host": os.uname().nodename, "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arch": ARCH, "nodes": NODES,
       "mode": a.mode, "b8": B8, "fidelity": str(CKC.math_fidelity), "fp32_dest_acc": bool(CKC.fp32_dest_acc_en),
       "packer_l1_acc": bool(CKC.packer_l1_acc), "silu_approx": bool(SILU_CKC.math_approx_mode),
       "overlay": os.environ.get("TT_BIO_METAL_OVERLAY"), "R": a.R, "W": a.W, "cases": {}}

g = torch.Generator().manual_seed(0)
w1h, w2h = (torch.randn(C, HID, generator=g) / C ** 0.5 for _ in range(2))
xh = torch.randn(1, a.R, a.W, C, generator=g)
mt, nt, ct = a.R * a.W // 32, HID // 32, C // 32
grid = T._transition_shard_grid(mt, nt, ct)
assert grid is not None, f"no shard grid for mt={mt} on {T.COMPUTE_GRID_MAIN}"
gx, gy = grid
pm, pn = mt // gy, nt // gx
bw = max(b for b in (8, 4, 2, 1) if (C // 32) % b == 0)
res.update(grid=[gx, gy], pm=pm, pn=pn, bw=bw)

x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
w12 = ttnn.from_torch(interleave_cols(w1h, w2h, gx), dtype=WDT, layout=ttnn.TILE_LAYOUT, device=dev)
w1 = ttnn.from_torch(w1h, dtype=WDT, layout=ttnn.TILE_LAYOUT, device=dev)

CASES = [("fc12", w12, 2 * pn, None, CKC), ("fc1_silu", w1, pn, SILU, SILU_CKC)]
for name, w, n, act, ckc in CASES:
    row = dict(per_core_N=n, subblock=list(subblock(pm, n)), act=None if act is None else "silu")
    out_g = None
    try:
        pc = cfg2d(gx, gy, pm, n, bw, act)
        mc = block_sharded(gx, gy, pm, n)

        def stock():
            return ttnn.linear(x, w, program_config=pc, compute_kernel_config=ckc, memory_config=mc, dtype=HDT)

        try:
            o = stock()
            ref = ttnn.to_torch(o).float()
            ttnn.deallocate(o)
        except Exception as e:  # noqa: BLE001  e.g. the CB clash in normal mode: the case cannot run at all
            row["stock_err"] = str(e).splitlines()[0][:300]
            raise
        out_g = ttnn.allocate_tensor_on_device(ttnn.Shape([1, a.R, a.W, n * 32 * gx]), HDT, ttnn.TILE_LAYOUT, dev, mc)

        def gen():
            return generic_matmul_2d(dev, x, w, out_g, pc, ckc)

        gen()
        got = ttnn.to_torch(out_g).float()
        d = (got - ref).abs()
        row.update(equal=bool(torch.equal(got, ref)), max_abs_diff=float(d.max()), n_diff=int((d > 0).sum()),
                   n_elems=int(d.numel()))
        gen()
        row["generic_repeat_equal"] = bool(torch.equal(ttnn.to_torch(out_g).float(), got))
        info = list(_CACHE.values())[-1]["info"]  # the entry this case just built
        row["generic_info"] = {k: info[k] for k in ("num_blocks", "split_half", "half_core", "packer_l1_acc_en",
                                                    "interm", "interm_separate", "nzsb_w", "defines", "kernel_root")}
        print(json.dumps({name: {k: row[k] for k in ("equal", "max_abs_diff", "n_diff", "generic_repeat_equal")}}),
              flush=True)

        for _ in range(a.warm):
            ttnn.deallocate(stock())
            gen()
        sync()
        ts_s, ts_g = [], []
        t0 = time.monotonic()
        for _ in range(a.reps):
            sync()
            t = time.perf_counter()
            o = stock()
            sync()
            ts_s.append((time.perf_counter() - t) * 1e6)
            ttnn.deallocate(o)
            sync()
            t = time.perf_counter()
            gen()
            sync()
            ts_g.append((time.perf_counter() - t) * 1e6)
        row["aiclk"] = aiclk_since(t0)
        row["stock"], row["generic"] = stats(ts_s), stats(ts_g)
        # Pipelined: reps calls queued back to back, one sync, so the device time shows when the host keeps ahead;
        # enqueue_us is the host's own cost per call (time until the last call returned).
        for nm, fn, owns in (("stock", stock, True), ("generic", gen, False)):
            sync()
            t = time.perf_counter()
            for _ in range(a.reps):
                o = fn()
                if owns:
                    ttnn.deallocate(o)
            te = time.perf_counter()
            sync()
            row[nm].update(pipelined_us=round((time.perf_counter() - t) * 1e6 / a.reps, 2),
                           enqueue_us=round((te - t) * 1e6 / a.reps, 2))
        row["ratio_min"] = round(row["generic"]["pipelined_us"] / row["stock"]["pipelined_us"], 4)
        row["ratio_med"] = round(row["generic"]["us_med"] / row["stock"]["us_med"], 4)
        row["pass"] = bool(row["equal"] and abs(row["ratio_min"] - 1) <= 0.03)
    except Exception as e:  # noqa: BLE001  a failing case is a result; the next case still runs
        row.setdefault("err", str(e).splitlines()[0][:300])
    if out_g is not None:
        ttnn.deallocate(out_g)
    res["cases"][name] = row
    print(json.dumps({name: row}), flush=True)

_stop.set()
for t in (x, w12, w1):
    ttnn.deallocate(t)
res["verdict"] = ("PASS" if res["cases"].get("fc12", {}).get("pass") else
                  "FAIL" if "pass" in res["cases"].get("fc12", {}) else "NO-RESULT")
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=1))
print(json.dumps({"verdict": res["verdict"], "out": str(a.out)}), flush=True)
