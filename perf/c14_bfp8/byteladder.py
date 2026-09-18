#!/usr/bin/env python3
"""Storage-width ladder at the pair class largest executed key, on Blackhole.

Every bfp8 realization figure this campaign owns is Wormhole (b2z-bfp8-narrow: 0.26-0.30 at
c_z = 128). The one Blackhole byte number, c14-byte-deletion 1.9 % of the op for 50 % of its DRAM
bytes, is a destination RETARGET over the same NoC, not a width cut. So the question "does a
narrower storage dtype convert to time here" has never been measured on this part, and every
region this row could build rests on the answer.

Arms are interleaved rep by rep with the interior order reversed on odd reps, and the bf16 clone
runs twice per rep as its own A/A twin. Refuses to start without prediction.json.
"""
from __future__ import annotations

import argparse, json, statistics, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRED = HERE / "prediction.json"
if not PRED.is_file():
    sys.exit("refusing to run: no pre-registered prediction at %s" % PRED)

sys.path.insert(0, str(HERE.parent / "c12_orchestrator" / "relayed" / "c12_kblock"))
import clk  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE / "byteladder.json"))
ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--mhz", type=int, default=1350)
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--c", type=int, default=128)
A = ap.parse_args()

Z = 67.108864e6          # one bf16 [1,512,512,128] tensor, the campaign unit
ELTS = 1 * A.n * A.n * A.c
B16, F32, B8 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b
WIDTH = {B16: 2.0, F32: 4.0, B8: 1.0625}   # bytes an element; bfp8_b tile is 1088 B / 1024 elts
DRAM = ttnn.DRAM_MEMORY_CONFIG

dev = ttnn.open_device(device_id=0)
nodes = clk.nodes_open_by_this_process()
grid = dev.compute_with_storage_grid_size()
CORE_GRID = ttnn.CoreGrid(y=grid.y, x=grid.x)
ckc = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)


def t(shape, dt, mc=DRAM):
    return ttnn.from_torch(torch.randn(*shape, dtype=torch.float32), layout=ttnn.TILE_LAYOUT,
                           device=dev, dtype=dt, memory_config=mc)


def timeit(fn, reps=3):
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


# ---------------- operands, allocated once so no arm pays an allocation ----------------
PAIR = (1, A.n, A.n, A.c)
src = {dt: t(PAIR, dt) for dt in (B16, F32, B8)}
dst = {dt: t(PAIR, dt) for dt in (B16, F32, B8)}
rhs = {dt: t(PAIR, dt) for dt in (B16, F32, B8)}
gamma = {dt: t((1, 1, 32, A.c), dt) for dt in (B16, B8)}
beta = {dt: t((1, 1, 32, A.c), dt) for dt in (B16, B8)}
w = {dt: t((A.c, A.c), dt) for dt in (B16, B8)}


def clone(dt):
    def f():
        y = ttnn.clone(src[dt], memory_config=DRAM, dtype=dt)
        ttnn.deallocate(y)
    return f, 2 * WIDTH[dt] * ELTS


def cast(a, b):
    def f():
        y = ttnn.typecast(src[a], b, memory_config=DRAM)
        ttnn.deallocate(y)
    return f, (WIDTH[a] + WIDTH[b]) * ELTS


def ln(dt, affine=True):
    def f():
        y = ttnn.layer_norm(src[dt], weight=gamma[dt] if affine else None,
                            bias=beta[dt] if affine else None, memory_config=DRAM,
                            compute_kernel_config=ckc)
        ttnn.deallocate(y)
    return f, 2 * WIDTH[dt] * ELTS


def binop(op, dt):
    # in place into a scratch copy so the operand never degrades across reps
    def f():
        a = ttnn.clone(dst[dt], memory_config=DRAM, dtype=dt)
        op(a, rhs[dt])
        ttnn.deallocate(a)
    return f, 3 * WIDTH[dt] * ELTS       # the clone is charged separately and subtracted below


def binop_net(op, dt):
    f, _ = binop(op, dt)
    return f, 3 * WIDTH[dt] * ELTS


def lin(act_dt, w_dt, out_dt):
    def f():
        y = ttnn.linear(src[act_dt], w[w_dt], compute_kernel_config=ckc, core_grid=CORE_GRID,
                        dtype=out_dt, memory_config=DRAM)
        ttnn.deallocate(y)
    return f, (WIDTH[act_dt] + WIDTH[out_dt]) * ELTS


ARMS = [
    ("clone_b16", clone(B16)),
    ("clone_b16_aa", clone(B16)),
    ("clone_f32", clone(F32)),
    ("clone_b8", clone(B8)),
    ("cast_b16_to_b8", cast(B16, B8)),
    ("cast_b8_to_b16", cast(B8, B16)),
    ("ln_b16", ln(B16)),
    ("ln_b8", ln(B8)),
    ("mul_b16", binop_net(ttnn.multiply_, B16)),
    ("mul_b8", binop_net(ttnn.multiply_, B8)),
    ("add_b16", binop_net(ttnn.add_, B16)),
    ("add_b8", binop_net(ttnn.add_, B8)),
    ("lin_b16", lin(B16, B16, B16)),
    ("lin_b8act", lin(B8, B16, B16)),
    ("lin_b8all", lin(B8, B8, B8)),
]

res = {"key": list(PAIR), "grid": [grid.y, grid.x], "cores": grid.x * grid.y, "nodes": nodes,
       "reps": A.reps, "mhz_requested": A.mhz, "prediction": json.loads(PRED.read_text()),
       "arms": {}, "errors": {}}

held = clk.force(A.mhz, nodes)
res["nodes_forced"] = held
for _ in range(200):
    if all(clk.aiclk(n) >= A.mhz - 5 for n in held):
        break
    time.sleep(0.05)
sampler = clk.Sampler(held[0])
time.sleep(0.3)

# warm/compile pass, discarded, and it tells us which arms are not bfp8-capable at all
live = []
for name, (fn, nbytes) in ARMS:
    try:
        fn()
        ttnn.synchronize_device(dev)
        live.append((name, fn, nbytes))
    except Exception as e:                                    # noqa: BLE001
        res["errors"][name] = "%s: %s" % (type(e).__name__, str(e)[:400])
        print("REFUSED %-16s %s" % (name, str(e)[:160]), flush=True)
res["capable"] = [n for n, _, _ in live]

samples = {n: [] for n, _, _ in live}
for rep in range(A.reps):
    order = live if rep % 2 == 0 else list(reversed(live))
    for name, fn, _ in order:
        samples[name].extend(timeit(fn, reps=1))
    print("rep %d done" % rep, flush=True)

res["clock"] = sampler.stop()
res["aiclk_after"] = {n: clk.aiclk(n) for n in held}
for name, _, nbytes in live:
    v = samples[name]
    med = statistics.median(v)
    res["arms"][name] = {"ms_med": med, "ms_min": min(v), "ms": v, "bytes": nbytes,
                         "gbs": nbytes / (med * 1e-3) / 1e9, "Z": nbytes / Z}
base = res["arms"].get("clone_b16", {}).get("ms_med")
if base:
    aa = res["arms"].get("clone_b16_aa", {}).get("ms_med")
    res["aa_floor_pct"] = abs(aa - base) / base * 100 if aa else None
    for name, a in res["arms"].items():
        a["vs_clone_b16"] = a["ms_med"] / base
for pair in (("clone_b8", "clone_b16"), ("ln_b8", "ln_b16"), ("mul_b8", "mul_b16"),
             ("add_b8", "add_b16"), ("lin_b8act", "lin_b16"), ("lin_b8all", "lin_b16"),
             ("clone_f32", "clone_b16")):
    a, b = pair
    if a in res["arms"] and b in res["arms"]:
        r = res["arms"][a]["ms_med"] / res["arms"][b]["ms_med"]
        byte_r = res["arms"][a]["bytes"] / res["arms"][b]["bytes"]
        res["arms"][a]["ratio_vs_own_b16"] = r
        res["arms"][a]["byte_ratio"] = byte_r
        # realization: share of the byte change that showed up as a time change
        res["arms"][a]["realization"] = (1 - r) / (1 - byte_r) if byte_r != 1 else None

Path(A.out).write_text(json.dumps(res, indent=2))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "ms"}
                  for k, v in res["arms"].items()}, indent=2))
print("A/A floor %.3f %%   clock %s" % (res.get("aa_floor_pct") or -1, res["clock"]))
ttnn.close_device(dev)
