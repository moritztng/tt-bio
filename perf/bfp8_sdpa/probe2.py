#!/usr/bin/env python3
"""probe.py with its two instrument defects fixed, so the numbers can be quoted.

probe.py answered the row question -- both dtype refusals lift and the kernel serves bfp8 on both
paths, six of six arms served with zero declines -- but neither number it printed was usable:

  * the arms ran in sequence, one dtype at a time, so the bf16 A/A pair landed 19.63 % apart and
    the whole ratio spread was compile/warmup drift. Here every arm is timed once per round and
    the rounds are interleaved, which is the only form an A/B on this box is allowed to take.
  * the fp64 reference used `softmax(qk * scale + bias)`, and this kernel adds the bias BEFORE
    applying the scale. So the bf16 control read 0.689 rel_rms, which is not a bf16 floor but a
    wrong reference. Rather than assume the fixed convention, all four are evaluated and the
    bf16 arm picks the winner -- a known-answer instrument control, not an assertion.
"""
from __future__ import annotations
import argparse, json, statistics, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "c12_orchestrator" / "relayed" / "c12_kblock"))
import clk  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402
from tt_bio import triatt_sdpa as TS  # noqa: E402
from tt_bio import mm_generic as MG  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(HERE / "probe2.json"))
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--head-dim", type=int, default=32)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--acc-batch", type=int, default=4)
ap.add_argument("--rounds", type=int, default=9)
ap.add_argument("--mhz", type=int, default=1350)
A = ap.parse_args()

dev = ttnn.open_device(device_id=0)
nodes = clk.nodes_open_by_this_process()
grid = dev.compute_with_storage_grid_size()
cores = grid.x * grid.y
DRAM = ttnn.DRAM_MEMORY_CONFIG
B8, B16 = ttnn.bfloat8_b, ttnn.bfloat16
S, H, D, Bt = A.seq, A.heads, A.head_dim, A.batch
SCALE = D ** -0.5
res = {"host": "qb1", "card": 1, "seq": S, "heads": H, "head_dim": D, "batch": Bt,
       "grid": [grid.y, grid.x], "cores": cores, "rounds": A.rounds, "mhz_requested": A.mhz,
       "interleaved": True, "benchlocked": False}

held = clk.force(A.mhz, nodes)
for _ in range(200):
    if all(clk.aiclk(n) >= A.mhz - 5 for n in held):
        break
    time.sleep(0.05)
sampler = clk.Sampler(held[0])

_orig_ok = TS.fast_dtypes_ok


def mixed_ok(*dts):
    return set(dts) <= MG.FAST_DTYPES


def mk(shape, dt):
    h = torch.randn(*shape, dtype=torch.float32)
    return h, ttnn.from_torch(h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt,
                              memory_config=DRAM)


def refs(hq, hk, hv, hb):
    """Every bias/scale convention the call site could mean, in fp64. The bf16 arm picks one."""
    q, k, v, b = (x.to(torch.float64) for x in (hq, hk, hv, hb))
    qk = q @ k.transpose(-1, -2)
    out = {}
    for name, sc in (("qk_s_plus_b", qk * SCALE + b), ("qk_plus_b_times_s", (qk + b) * SCALE),
                     ("qk_plus_b_div_s", (qk + b) / SCALE), ("qk_div_s_plus_b", qk / SCALE + b)):
        out[name] = torch.softmax(sc, dim=-1) @ v
    return out


def rel_rms(got, ref):
    got = got.to(torch.float64)
    return float(torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref))


pairs = TS.fused_pairs(S, H, D, cores, B16)
QC, KC = pairs[0]
res["pair_used"] = [QC, KC]

C = H * D
# operands built once, outside the timing loop, so a round times the op and not from_torch
OPS = {}
for name, qdt, bdt in (("bf16", B16, B16), ("bfp8", B8, B8), ("bfp8_qkv_bf16_mask", B8, B16)):
    hq, q = mk([Bt, H, S, D], qdt)
    hk, k = mk([Bt, H, S, D], qdt)
    hv, v = mk([Bt, H, S, D], qdt)
    hb, bias = mk([1, H, S, S], bdt)
    OPS[name] = {"kind": "sdpa", "t": (q, k, v, bias), "h": (hq, hk, hv, hb),
                 "ok": mixed_ok if qdt != bdt else _orig_ok}
for name, dt in (("fuseqkv_bf16", B16), ("fuseqkv_bfp8", B8)):
    hx, x = mk([Bt, S, C], dt)
    hw, w = mk([C, 3 * C], dt)
    hb, bias = mk([1, H, S, S], dt)
    OPS[name] = {"kind": "fuseqkv", "t": (x, w, bias), "h": (hx, hw, hb), "ok": _orig_ok}
# bf16 twice under different names is the A/A control; it must sit in the same interleave
OPS["bf16_aa"] = dict(OPS["bf16"])
ORDER = ["bf16", "bfp8", "bfp8_qkv_bf16_mask", "fuseqkv_bf16", "fuseqkv_bfp8", "bf16_aa"]


def call(name):
    o = OPS[name]
    TS.fast_dtypes_ok = o["ok"]
    try:
        if o["kind"] == "sdpa":
            q, k, v, bias = o["t"]
            return TS.sdpa(q, k, v, bias, SCALE, QC, KC)
        x, w, bias = o["t"]
        return TS.sdpa_fused_qkv(x, w, bias, SCALE, H, D, S, KC, force=True)
    finally:
        TS.fast_dtypes_ok = _orig_ok


arms = {n: {"ms": [], "served": 0, "declined": 0} for n in ORDER}
for n in ORDER:                                        # warm every program before any is timed
    b0 = (TS.STATS[0], TS.STATS[1])
    o = call(n)
    arms[n]["kernel_ran"] = o is not None
    arms[n]["served"] = TS.STATS[0] - b0[0]
    arms[n]["declined"] = TS.STATS[1] - b0[1]
    if o is not None:
        arms[n]["out_dtype"] = str(o.dtype)
        ttnn.deallocate(o)
ttnn.synchronize_device(dev)
for _ in range(A.rounds):
    for n in ORDER:
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = call(n)
        ttnn.synchronize_device(dev)
        arms[n]["ms"].append((time.perf_counter() - t0) * 1e3)
        if o is not None:
            ttnn.deallocate(o)

# ---------------------------------------------------------------- accuracy, all four conventions
n_acc = min(A.acc_batch, Bt)
for name in ORDER:
    o = OPS[name]
    out = call(name)
    got = ttnn.to_torch(out)[:n_acc]
    if o["kind"] == "sdpa":
        hq, hk, hv, hb = o["h"]
        r = refs(hq[:n_acc], hk[:n_acc], hv[:n_acc], hb)
    else:
        hx, hw, hb = o["h"]
        qkv = hx[:n_acc].to(torch.float64) @ hw.to(torch.float64)
        hq, hk, hv = (qkv[..., i * C:(i + 1) * C].reshape(n_acc, S, H, D).permute(0, 2, 1, 3)
                      for i in range(3))
        r = refs(hq, hk, hv, hb)
    arms[name]["rel_rms"] = {k: rel_rms(got, vv) for k, vv in r.items()}
    ttnn.deallocate(out)

for n in ORDER:
    a = arms[n]
    a["ms_med"], a["ms_min"] = statistics.median(a["ms"]), min(a["ms"])
res["clock"] = sampler.stop()
res["arms"] = arms
# the bf16 arm picks the reference convention, then every arm is scored on that one
best = min(arms["bf16"]["rel_rms"], key=arms["bf16"]["rel_rms"].get)
res["convention"] = best
res["bf16_control_rel_rms"] = arms["bf16"]["rel_rms"][best]
res["accuracy"] = {n: arms[n]["rel_rms"][best] for n in ORDER}
med = {n: arms[n]["ms_med"] for n in ORDER}
res["aa_floor_pct"] = abs(med["bf16_aa"] - med["bf16"]) / med["bf16"] * 100
res["ratios"] = {"bfp8": med["bf16"] / med["bfp8"],
                 "bfp8_qkv_bf16_mask": med["bf16"] / med["bfp8_qkv_bf16_mask"],
                 "fuseqkv_bfp8": med["fuseqkv_bf16"] / med["fuseqkv_bfp8"]}
Path(A.out).write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != "arms"}, indent=2))
print(json.dumps({n: {"ms_med": round(arms[n]["ms_med"], 4), "served": arms[n]["served"],
                      "declined": arms[n]["declined"], "out": arms[n].get("out_dtype")}
                  for n in ORDER}, indent=2))
ttnn.close_device(dev)
