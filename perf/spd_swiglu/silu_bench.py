"""Fused silu on one chip: exactness against float64 and us per fc1 call, for one kernel arm per process.

    TT_VISIBLE_DEVICES=N python perf/spd_swiglu/silu_bench.py --arm {base,f32} --out OUT.json

The arm decides which ttnn kernel headers the process compiles against, so it is one per process:
  base  the wheel as shipped (its fp32-dest silu ignores math_approx_mode)
  f32   metal_overlay silu_f32 (calculate_silu_f32 under approx; the silu_f32 lever)
Every matmul here runs Protenix's transition config (HiFi4, fp32 dest, packer L1 acc) with
math_approx_mode=True, the switch the overlay keys on; under base the flag is inert.

exact: x @ I for every bfloat16 x in [-100, 100] (and its negation), fc1's fused silu written fp32.
  The product is exact, so the dest holds x and the output is the SFPU's silu of x, compared with
  float64 silu(x): max/p99.9 relative error and the fraction whose bf16 rounding differs, both where
  |silu(x)| > 1e-30 (below that the device flushes what float64 keeps).
ops: fc1 + fused silu, fc1 without activation (the matmul alone: the roof this kernel can reach) and
  fc1 + standalone ttnn.silu (the held unfused form), at the fold's row-block shapes, L1 in and out,
  bf16 out, CORE_GRID_MAIN. Min and median over batches of 10 device-synchronised calls, AICLK of
  the opened node sampled every 0.2 s during each op's timing window.
transition: the real Transition module on the whole pair tensor (row blocking included), f32 arm
  inside `levers("silu_f32")`, base arm without it.
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
ap.add_argument("--arm", choices=("base", "f32"), required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--batches", type=int, default=7)
ap.add_argument("--skip", default="", help="comma list of sections to skip: exact,ops,transition")
a = ap.parse_args()
if a.arm == "f32":
    os.environ["TT_BIO_LEVERS"] = "silu_f32"

import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()
print(json.dumps({"runtime_root": os.environ.get("TT_METAL_RUNTIME_ROOT"), "jit_cache": os.environ.get("TT_METAL_CACHE")}),
      flush=True)
assert Path(T.__file__).resolve().is_relative_to(ROOT), T.__file__
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
    _sample()  # a window shorter than the sampler's period still gets one reading inside it
    t1 = max(t1, samples[-1][0])
    v = sorted(x for ts, r in samples if t0 <= ts <= t1 for x in r.values())
    return dict(median=v[len(v) // 2], min=v[0], max=v[-1], n=len(v)) if v else None


CKC_CLS = ttnn.WormholeComputeKernelConfig if ARCH == "wormhole" else ttnn.types.BlackholeComputeKernelConfig
CKC = CKC_CLS(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True, fp32_dest_acc_en=True,
              packer_l1_acc=True)
L1 = ttnn.L1_MEMORY_CONFIG
res = {"arm": a.arm, "host": os.uname().nodename, "chip": os.environ.get("TT_VISIBLE_DEVICES"), "arch": ARCH,
       "nodes": OPENED, "grid": list(T.COMPUTE_GRID_MAIN), "runtime_root": os.environ.get("TT_METAL_RUNTIME_ROOT"),
       "jit_cache": os.environ.get("TT_METAL_CACHE"),
       "loadavg0": os.getloadavg()}
skip = set(filter(None, a.skip.split(",")))


def sync():
    ttnn.synchronize_device(dev)


def bf16_round(t):
    return t.to(torch.bfloat16).to(torch.float64)


# ---- exact: the SFPU silu on every bfloat16 in [-100, 100]
if "exact" not in skip:
    pos = torch.arange(0, 0x42C9, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)  # 0 .. 100.5
    xs = torch.cat([pos, -pos]).to(torch.float32)
    n = xs.numel()
    rows = -(-n // 32)
    xin = torch.zeros(rows * 32, dtype=torch.float32)
    xin[:n] = xs
    rows_p = -(-rows // 32) * 32
    X = torch.zeros(rows_p, 32)
    X[:rows] = xin.view(rows, 32)
    xt = ttnn.from_torch(X.view(1, 1, rows_p, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    it = ttnn.from_torch(torch.eye(32).view(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    # core_grid matters: without it ttnn.linear runs the activation as a separate unary op (approx off,
    # so the wheel's silu whatever the overlay), which is what run 1 measured in every arm.
    o = ttnn.linear(xt, it, activation="silu", compute_kernel_config=CKC, dtype=ttnn.float32,
                    core_grid=T.CORE_GRID_MAIN)
    got = ttnn.to_torch(o).view(-1)[:n].to(torch.float64)
    x64 = xs.to(torch.float64)
    ref = x64 / (1 + torch.exp(-x64))
    ok = ref.abs() > 1e-30
    rel = (got[ok] / ref[ok] - 1).abs()
    res["exact"] = dict(n=n, max_rel=float(rel.max()), p999_rel=float(torch.quantile(rel.float(), 0.999)),
                        at=float(x64[ok][int(rel.argmax())]),
                        bf16_differs=float((bf16_round(got[ok]) != bf16_round(ref[ok])).double().mean()),
                        finite=bool(torch.isfinite(got).all()))
    res["exact"]["digest"] = hashlib.sha256(got.to(torch.float32).numpy().tobytes()).hexdigest()[:16]
    # fp32 dest values that are not bf16: fp32 x @ I. The product is not exact, but every arm gets the same
    # dest bits, so equal digests across kernels mean equal silu bits on these inputs too.
    g = torch.Generator().manual_seed(7)
    X = 4 * torch.randn(1, 1, 32768, 32, generator=g)
    xt32 = ttnn.from_torch(X, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    o32 = ttnn.linear(xt32, it, activation="silu", compute_kernel_config=CKC, dtype=ttnn.float32,
                      core_grid=T.CORE_GRID_MAIN)
    res["exact"]["digest_fp32_in"] = hashlib.sha256(ttnn.to_torch(o32).to(torch.float32).numpy().tobytes()).hexdigest()[:16]
    print(json.dumps({"exact": res["exact"]}), flush=True)
    for t in (xt, it, o, xt32, o32):
        ttnn.deallocate(t)


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
            ttnn.deallocate(fn())  # one live output: ten of them do not fit next to the matmul's CBs
        sync()
        ts.append((time.perf_counter() - t) / 10 * 1e6)
    return dict(us_min=round(min(ts), 2), us_med=round(st.median(ts), 2), clock=clock(t0, time.monotonic()))


# Row-block shapes the census measured: rows x c_in x hidden.
SHAPES = {"wormhole": {"pair": (5 * 736, 256, 1024), "msa": (16 * 736, 128, 512)},
          "blackhole": {"pair": (11 * 736, 256, 1024), "msa": (32 * 736, 128, 512)}}[ARCH]
if "ops" not in skip:
    res["ops"] = {}
    g = torch.Generator().manual_seed(0)
    for name, (M, K, N) in SHAPES.items():
        x = ttnn.from_torch(torch.randn(1, 1, M, K, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=L1)
        w = ttnn.from_torch(torch.randn(K, N, generator=g) / K ** 0.5, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev)
        lin = lambda act=None: ttnn.linear(x, w, activation=act, compute_kernel_config=CKC, memory_config=L1,
                                           dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN)

        def unfused():
            y = lin()
            return ttnn.silu(y, memory_config=L1, output_tensor=y)
        row = {"shape": [M, K, N]}
        for arm, fn in (("fused", lambda: lin("silu")), ("matmul", lin), ("unfused", unfused)):
            row[arm] = timed(fn, a.batches)
        print(json.dumps({name: row}), flush=True)
        res["ops"][name] = row
        ttnn.deallocate(x)
        ttnn.deallocate(w)

if "transition" not in skip:
    res["transition"] = {}
    for S, C, HID in ((736, 256, 1024),):
        g = torch.Generator().manual_seed(1)
        sd = {"norm.weight": 1 + 0.1 * torch.randn(C, generator=g), "norm.bias": 0.1 * torch.randn(C, generator=g),
              "fc1.weight": torch.randn(HID, C, generator=g) / C ** 0.5,
              "fc2.weight": torch.randn(HID, C, generator=g) / C ** 0.5,
              "fc3.weight": torch.randn(C, HID, generator=g) / HID ** 0.5}
        sd = {k: bf16_round(v).float() for k, v in sd.items()}
        zt = bf16_round(torch.randn(1, S, S, C, generator=g))
        xn = torch.nn.functional.layer_norm(zt, (C,), sd["norm.weight"].double(), sd["norm.bias"].double(), 1e-5)
        ref = (torch.nn.functional.silu(xn @ sd["fc1.weight"].double().T) * (xn @ sd["fc2.weight"].double().T)) \
            @ sd["fc3.weight"].double().T
        names = ("silu_f32",) if a.arm == "f32" else ()
        with T.levers(names):
            tr = T.Transition(sd, CKC)
            z = ttnn.from_torch(zt.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            out = ttnn.to_torch(tr(z)).to(torch.float64)
            d = out - ref
            row = {"rel_rms": float(d.norm() / ref.norm()), "max_abs": float(d.abs().max())}
            # Whole-pair module call: few calls, so batches of one.
            ts = []
            t0 = time.monotonic()
            for _ in range(5):
                sync()
                t = time.perf_counter()
                o = tr(z)
                sync()
                ts.append((time.perf_counter() - t) * 1e3)
                ttnn.deallocate(o)
            row.update(ms_min=round(min(ts), 3), ms_med=round(st.median(ts), 3), clock=clock(t0, time.monotonic()))
            ttnn.deallocate(z)
        print(json.dumps({f"transition_{S}x{C}x{HID}": row}), flush=True)
        res["transition"][f"{S}x{C}x{HID}"] = row

res["loadavg1"] = os.getloadavg()
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=1))
