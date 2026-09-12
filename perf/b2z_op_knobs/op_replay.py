#!/usr/bin/env python3
"""Replay one Boltz-2 op instance standalone and time it, under an arbitrary knob config.

The manifest (bench_manifest.json) says what the fold actually dispatches. This file rebuilds
each instance from that description, runs it back-to-back with the program cache hot, and
reports median device us/call plus its own A/A floor.

Timing method: R calls enqueued back-to-back with a single synchronize at the end, which is how
the fold issues them, then wall/R. The queue is deliberately kept full -- `time.thread_time()`
does not separate host from device on this stack (`b2x-op-cost-curve`), so only the drained wall
is trustworthy.

Knob overrides (--knob k=v,...) replace the captured config so the same instance can be swept:
  fidelity=LoFi|HiFi2|HiFi3|HiFi4   fp32acc=0|1   packerl1=0|1   dstfull=0|1  approx=0|1
  outbuf=DRAM|L1                     inbuf=DRAM|L1
  outdtype=bfloat16|bfloat8_b|float32
  grid=XxY                           (matmul core grid)
  shard=none|width|height|block      (output sharding for matmul/binary/layernorm)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics as st
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent

FIDELITY = {}
DTYPES = {}


def _init_tables(ttnn):
    global FIDELITY, DTYPES
    FIDELITY = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2,
                "HiFi3": ttnn.MathFidelity.HiFi3, "HiFi4": ttnn.MathFidelity.HiFi4}
    DTYPES = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32, "bfloat8_b": ttnn.bfloat8_b,
              "bfloat4_b": ttnn.bfloat4_b, "uint32": ttnn.uint32, "int32": ttnn.int32,
              "uint16": ttnn.uint16}


def parse_knobs(s: str) -> dict:
    out = {}
    for kv in (s or "").split(","):
        if not kv.strip():
            continue
        k, _, v = kv.partition("=")
        out[k.strip()] = v.strip()
    return out


def ckc_from(ttnn, captured: dict | None, knobs: dict):
    c = dict(captured or {"math_fidelity": "HiFi4", "math_approx_mode": 0,
                          "fp32_dest_acc_en": 1, "packer_l1_acc": 1, "dst_full_sync_en": 0})
    if "fidelity" in knobs:
        c["math_fidelity"] = knobs["fidelity"]
    for kn, fld in (("fp32acc", "fp32_dest_acc_en"), ("packerl1", "packer_l1_acc"),
                    ("dstfull", "dst_full_sync_en"), ("approx", "math_approx_mode")):
        if kn in knobs:
            c[fld] = int(knobs[kn])
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=FIDELITY[c["math_fidelity"]],
        math_approx_mode=bool(c["math_approx_mode"]),
        fp32_dest_acc_en=bool(c["fp32_dest_acc_en"]),
        packer_l1_acc=bool(c["packer_l1_acc"]),
        dst_full_sync_en=bool(c["dst_full_sync_en"]),
    )


def memcfg(ttnn, buf: str, knobs: dict, which: str, shape=None, device=None):
    """Interleaved DRAM/L1, or a sharded config when --knob shard= asks for one."""
    shard = knobs.get("shard", "none") if which == "out" else knobs.get("inshard", "none")
    buf = knobs.get(f"{which}buf", buf)
    if shard in ("none", "", None):
        return ttnn.DRAM_MEMORY_CONFIG if buf == "DRAM" else ttnn.L1_MEMORY_CONFIG
    gx, gy = (int(v) for v in knobs.get("grid", "8x8").split("x"))
    n = 1
    for d in shape[:-1]:
        n *= int(d)
    w = int(shape[-1])
    cores = gx * gy
    if shard == "width":
        per = math.ceil(w / cores / 32) * 32
        spec = ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(
            ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))}), [n, per],
            ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)
    if shard == "height":
        per = math.ceil(n / cores / 32) * 32
        spec = ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(
            ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))}), [per, w],
            ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    if shard == "block":
        ph = math.ceil(n / gy / 32) * 32
        pw = math.ceil(w / gx / 32) * 32
        spec = ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(
            ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))}), [ph, pw],
            ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)
    raise ValueError(f"shard={shard}")


def make_tensor(ttnn, torch, spec, device, knobs):
    dt = DTYPES[spec["dtype"]]
    layout = ttnn.TILE_LAYOUT if spec["layout"] == "TILE" else ttnn.ROW_MAJOR_LAYOUT
    shape = [int(x) for x in spec["shape"]]
    if spec["dtype"] in ("uint32", "int32", "uint16"):
        t = torch.zeros(shape, dtype=torch.int32)
    else:
        t = torch.randn(shape, dtype=torch.float32) * 0.1
    mc = ttnn.DRAM_MEMORY_CONFIG if knobs.get("inbuf", spec["buffer"]) == "DRAM" \
        else ttnn.L1_MEMORY_CONFIG
    return ttnn.from_torch(t, dtype=dt, layout=layout, device=device, memory_config=mc)


def build_call(ttnn, torch, rec, device, knobs):
    """Return (fn, tensors) where fn() issues exactly the captured op once."""
    ins = [make_tensor(ttnn, torch, s, device, knobs) for s in rec["inputs"]]
    kind = rec["kind"]
    obuf = (rec.get("out_mem") or {}).get("buffer", "DRAM")
    oshape = None

    if kind == "matmul":
        a, b = ins[0], ins[1]
        oshape = list(a.shape)[:-1] + [int(b.shape[-1])]
        mc = memcfg(ttnn, obuf, knobs, "out", oshape, device)
        ckc = ckc_from(ttnn, rec.get("ckc"), knobs)
        od = DTYPES[knobs.get("outdtype", rec.get("out_dtype", "bfloat16"))]
        kw = dict(memory_config=mc, dtype=od, compute_kernel_config=ckc)
        if "grid" in knobs and knobs.get("shard", "none") == "none":
            gx, gy = (int(v) for v in knobs["grid"].split("x"))
            kw["core_grid"] = ttnn.CoreGrid(x=gx, y=gy)
        if rec.get("has_bias"):
            bias = ins[2]
            fn = lambda: ttnn.linear(a, b, bias=bias, **kw)          # noqa: E731
        elif rec["api"] == "ttnn.linear":
            fn = lambda: ttnn.linear(a, b, **kw)                      # noqa: E731
        else:
            fn = lambda: ttnn.matmul(a, b, **kw)                      # noqa: E731
    elif kind == "binary":
        op = getattr(ttnn, rec["binop"].lower(), None) or ttnn.add
        a, b = ins[0], ins[1]
        oshape = list(a.shape)
        mc = memcfg(ttnn, obuf, knobs, "out", oshape, device)
        od = DTYPES[knobs.get("outdtype", rec.get("out_dtype", "bfloat16"))]
        kw = dict(memory_config=mc, dtype=od)
        if rec.get("post_act"):
            try:
                kw["activations"] = [rec["post_act"]]
            except Exception:                                          # noqa: BLE001
                pass
        if len(ins) >= 3:
            kw["output_tensor"] = ins[2]
            kw.pop("dtype", None)
        kw = _accepted(op, (a, b), kw)
        fn = lambda: op(a, b, **kw)                                    # noqa: E731
    elif kind == "layernorm":
        x = ins[0]
        w = ins[1] if len(ins) > 1 else None
        bi = ins[2] if len(ins) > 2 else None
        oshape = list(x.shape)
        mc = memcfg(ttnn, obuf, knobs, "out", oshape, device)
        ckc = ckc_from(ttnn, rec.get("ckc"), knobs)
        f = ttnn.rms_norm if rec.get("norm") == "rmsnorm" else ttnn.layer_norm
        fn = lambda: f(x, weight=w, bias=bi, epsilon=rec.get("eps", 1e-5),  # noqa: E731
                       memory_config=mc, compute_kernel_config=ckc)
    elif kind == "softmax":
        x = ins[0]
        mc = memcfg(ttnn, obuf, knobs, "out", list(x.shape), device)
        ckc = ckc_from(ttnn, rec.get("ckc"), knobs)
        fn = lambda: ttnn.softmax(x, dim=rec.get("dim", -1), memory_config=mc,  # noqa: E731
                                  compute_kernel_config=ckc)
    elif kind == "transpose":
        x = ins[0]
        d = rec.get("dim", "WH")
        pair = {"WH": (-2, -1), "HC": (-3, -2), "CN": (-4, -3), "NH": (-4, -2),
                "NW": (-4, -1), "CW": (-3, -1)}.get(d, (-2, -1))
        mc = ttnn.DRAM_MEMORY_CONFIG if knobs.get("outbuf", obuf) == "DRAM" else ttnn.L1_MEMORY_CONFIG
        fn = lambda: ttnn.transpose(x, pair[0], pair[1], memory_config=mc)  # noqa: E731
    elif kind == "permute":
        x = ins[0]
        perm = rec.get("perm") or list(range(len(x.shape)))
        mc = ttnn.DRAM_MEMORY_CONFIG if knobs.get("outbuf", obuf) == "DRAM" else ttnn.L1_MEMORY_CONFIG
        fn = lambda: ttnn.permute(x, perm, memory_config=mc)            # noqa: E731
    elif kind == "slice":
        x = ins[0]
        b, e, s = rec["begins"], rec["ends"], rec["steps"]
        mc = ttnn.DRAM_MEMORY_CONFIG if knobs.get("outbuf", obuf) == "DRAM" else ttnn.L1_MEMORY_CONFIG
        fn = lambda: ttnn.slice(x, b, e, s, memory_config=mc)           # noqa: E731
    elif kind == "concat":
        mc = ttnn.DRAM_MEMORY_CONFIG if knobs.get("outbuf", obuf) == "DRAM" else ttnn.L1_MEMORY_CONFIG
        d = rec.get("dim", 0)
        fn = lambda: ttnn.concat(list(ins), dim=d, memory_config=mc)    # noqa: E731
    elif kind == "reshape":
        x = ins[0]
        sh = rec["out_shape"]
        fn = lambda: ttnn.reshape(x, sh)                                # noqa: E731
    elif kind == "nlp_create_qkv_heads":
        x = ins[0]
        mc = ttnn.DRAM_MEMORY_CONFIG if knobs.get("outbuf", obuf) == "DRAM" else ttnn.L1_MEMORY_CONFIG
        fn = lambda: ttnn.experimental.nlp_create_qkv_heads(            # noqa: E731
            x, num_heads=rec["num_heads"], num_kv_heads=rec["num_kv_heads"],
            transpose_k_heads=rec.get("transpose_k", False), memory_config=mc)
    elif kind == "nlp_concat_heads":
        x = ins[0]
        mc = ttnn.DRAM_MEMORY_CONFIG if knobs.get("outbuf", obuf) == "DRAM" else ttnn.L1_MEMORY_CONFIG
        fn = lambda: ttnn.experimental.nlp_concat_heads(x, memory_config=mc)  # noqa: E731
    elif kind == "sdpa":
        q, k, v = ins[0], ins[1], ins[2]
        rest = ins[3] if len(ins) > 3 else None
        ckc = ckc_from(ttnn, rec.get("ckc"), knobs)
        kw = dict(is_causal=False, scale=rec.get("scale"), compute_kernel_config=ckc)
        if rest is not None:
            kw["attn_mask"] = rest
        fn = lambda: ttnn.transformer.scaled_dot_product_attention(q, k, v, **kw)  # noqa: E731
    elif kind == "unary":
        x = ins[0]
        f = getattr(ttnn, rec.get("unop") or "silu", ttnn.silu)
        mc = ttnn.DRAM_MEMORY_CONFIG if knobs.get("outbuf", obuf) == "DRAM" else ttnn.L1_MEMORY_CONFIG
        fn = lambda: f(x, memory_config=mc)                             # noqa: E731
    elif kind == "copy":
        x = ins[0]
        mc = ttnn.DRAM_MEMORY_CONFIG if knobs.get("outbuf", obuf) == "DRAM" else ttnn.L1_MEMORY_CONFIG
        od = DTYPES[knobs.get("outdtype", "bfloat16")]
        fn = lambda: ttnn.clone(x, memory_config=mc, dtype=od)          # noqa: E731
    else:
        return None, ins
    return fn, ins


def _addr(t):
    try:
        return t.buffer_address()
    except Exception:                                                    # noqa: BLE001
        return None


def _accepted(fn, args, kw):
    """Drop kwargs this ttnn overload does not take, cheapest-first."""
    for drop in ([], ["activations"], ["dtype"], ["activations", "dtype"],
                 ["activations", "dtype", "memory_config"]):
        trial = {k: v for k, v in kw.items() if k not in drop}
        try:
            o = fn(*args, **trial)
            if _addr(o) not in {_addr(x) for x in args}:
                try:
                    o.deallocate()
                except Exception:                                        # noqa: BLE001
                    pass
            return trial
        except Exception:                                                # noqa: BLE001
            continue
    return kw


def time_call(ttnn, device, fn, ins, reps=24, bursts=5):
    """Median us/call over `bursts` drained bursts of `reps` back-to-back issues.

    An op whose output aliases an operand (in-place binaries, whole-tensor slices) must not have
    that output deallocated, or the second call runs on a freed buffer.
    """
    live = {_addr(t) for t in ins}

    def once():
        o = fn()
        if _addr(o) not in live:
            try:
                ttnn.deallocate(o)
            except Exception:                                            # noqa: BLE001
                pass

    for _ in range(3):
        once()
    ttnn.synchronize_device(device)
    meds = []
    for _ in range(bursts):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            once()
        ttnn.synchronize_device(device)
        meds.append((time.perf_counter() - t0) / reps)
    return 1e6 * st.median(meds), [round(1e6 * m, 2) for m in meds]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "bench_manifest.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", default="PairformerLayer,DiffusionStep,MSALayer")
    ap.add_argument("--ids", default="", help="comma list of instance ids to run (default all)")
    ap.add_argument("--knob", default="", help="config overrides, see module docstring")
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--bursts", type=int, default=5)
    ap.add_argument("--aa", action="store_true", help="run every instance twice for an A/A floor")
    ap.add_argument("--max-bytes", type=int, default=600_000_000,
                    help="skip instances whose operands exceed this, to stay inside DRAM")
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    _init_tables(ttnn)
    knobs = parse_knobs(a.knob)

    recs = json.load(open(a.manifest))
    units = set(a.units.split(","))
    want = set(x for x in a.ids.split(",") if x)
    recs = [r for r in recs if r["unit"] in units and (not want or r["id"] in want)]
    recs = [r for r in recs if r["kind"] not in ("unsupported", "generic")]

    device = ttnn.open_device(device_id=0)
    ttnn.enable_program_cache(device) if hasattr(ttnn, "enable_program_cache") else None
    out = {"env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "knob": a.knob, "reps": a.reps, "bursts": a.bursts,
                   "arch": str(device.arch()) if hasattr(device, "arch") else "?",
                   "grid": str(device.compute_with_storage_grid_size())},
           "rows": []}
    print(f"# {len(recs)} instances  knob='{a.knob}'  grid={out['env']['grid']}", flush=True)

    for r in recs:
        tot = sum(i.get("bytes", 0) for i in r["inputs"])
        row = {"id": r["id"], "unit": r["unit"], "kind": r["kind"], "api": r["api"],
               "op": r["op"], "per_unit": r["per_unit"], "calls_per_fold": r["calls_per_fold"],
               "in_bytes": tot,
               "shapes": ["x".join(map(str, i["shape"])) for i in r["inputs"]]}
        if tot > a.max_bytes:
            row["error"] = f"operands {tot/1e6:.0f} MB over cap"
            out["rows"].append(row)
            continue
        fn = ins = None
        try:
            fn, ins = build_call(ttnn, torch, r, device, knobs)
            if fn is None:
                row["error"] = "no builder"
            else:
                us, all_us = time_call(ttnn, device, fn, ins, a.reps, a.bursts)
                row["us_per_call"] = round(us, 2)
                row["bursts_us"] = all_us
                row["ms_per_fold"] = round(us * r["calls_per_fold"] / 1000.0, 3)
                if a.aa:
                    us2, _ = time_call(ttnn, device, fn, ins, a.reps, a.bursts)
                    row["aa_ratio"] = round(us2 / us, 4)
        except Exception as e:                                           # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:220]}"
        finally:
            for t in (ins or []):
                try:
                    ttnn.deallocate(t)
                except Exception:                                        # noqa: BLE001
                    pass
        out["rows"].append(row)
        msg = row.get("error") or (f"{row['us_per_call']:9.2f} us x{row['calls_per_fold']:6d} "
                                   f"= {row['ms_per_fold']:8.2f} ms/fold")
        print(f"{row['id']:22s} {row['kind']:12s} {'|'.join(row['shapes'])[:52]:52s} {msg}",
              flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    ok = [r for r in out["rows"] if "ms_per_fold" in r]
    for u in sorted(units):
        s = sum(r["ms_per_fold"] for r in ok if r["unit"] == u)
        print(f"SUM {u}: {s/1000:.3f} s/fold over {len([r for r in ok if r['unit']==u])} instances",
              flush=True)
    out["totals"] = {u: round(sum(r["ms_per_fold"] for r in ok if r["unit"] == u) / 1000, 4)
                     for u in sorted(units)}
    json.dump(out, open(a.out, "w"), indent=1)
    ttnn.close_device(device)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
