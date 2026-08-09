#!/usr/bin/env python3
"""Working-set ledger for ONE Protenix-v2 Pairformer block at the 298-aa shape.

W7 of the PERF WAR asks a question one level above per-op tuning: at 298 aa, what does the
whole block's working set look like against the 195 MB of aggregate L1, and what has to
stream from DRAM? Nothing measures that today. This does.

Every ttnn call the block makes is wrapped, and each call records its input tensors
(shape, dtype, buffer type), its output tensor, and the device's allocated DRAM and L1
high-water marks right after it. From that we get:

  * the working-set table: every distinct tensor the block holds, its size in MB, and the
    op index range over which it is live (its lifetime);
  * a first-order DRAM traffic model: bytes read = DRAM-resident inputs, bytes written =
    DRAM-resident outputs, per op and per phase;
  * the predicted floor from those bytes at THIS card's measured read/write roofs, which
    the measured block wall then confirms or refutes.

Three modes:

  ledger  wrap + record, one block call, no per-op sync. Ordering and shapes are exact;
          durations here are host-side and instrumented, never quote them.
  roofs   measure DRAM read and write roofs on this card (WARROOM rule: roofs are per-card).
  bench   plain warm block timing, no wrapping, sync on both sides. The ground truth ms.

    TT_VISIBLE_DEVICES=2 TT_MESH_GRAPH_DESC_PATH=... python3 perf/sram_strategy/block_working_set.py \
        --mode ledger --n 320 --out /tmp/ledger_n320.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

from tt_bio.tenstorrent import get_device  # noqa: E402

# ttnn ops the block actually calls (from pf_layer.py --mode count at N=320).
OPS = [
    ("ttnn", "matmul"), ("ttnn", "linear"), ("ttnn", "layer_norm"),
    ("ttnn", "multiply"), ("ttnn", "multiply_"), ("ttnn", "add"), ("ttnn", "add_"),
    ("ttnn", "permute"), ("ttnn", "transpose"), ("ttnn", "concat"), ("ttnn", "clone"),
    ("ttnn", "chunk"), ("ttnn", "reallocate"), ("ttnn", "typecast"), ("ttnn", "reshape"),
    ("ttnn", "unsqueeze"), ("ttnn", "squeeze"), ("ttnn", "softmax"), ("ttnn", "slice"),
    ("ttnn.experimental", "minimal_matmul"),
    ("ttnn.experimental", "nlp_create_qkv_heads"),
    ("ttnn.experimental", "nlp_concat_heads"),
    ("ttnn.transformer", "scaled_dot_product_attention"),
]

ELEM_BYTES = {
    "BFLOAT16": 2.0, "FLOAT32": 4.0, "BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625,
    "UINT32": 4.0, "INT32": 4.0, "UINT16": 2.0, "UINT8": 1.0,
}


def _resolve(path):
    obj = ttnn
    for part in path.split(".")[1:]:
        obj = getattr(obj, part)
    return obj


def tensor_info(t):
    """(shape, dtype, buffer_type, bytes) for a ttnn tensor on device, or None."""
    try:
        shape = [int(d) for d in t.shape]
    except Exception:
        return None
    try:
        dt = str(t.dtype).split(".")[-1]
        eb = ELEM_BYTES.get(dt, 2.0)
    except Exception:
        dt, eb = "?", 2.0
    n = 1
    for d in shape:
        n *= d
    try:
        mc = t.memory_config()
        bt = "DRAM" if mc.buffer_type == ttnn.BufferType.DRAM else "L1"
        layout = str(mc.memory_layout).split(".")[-1]
    except Exception:
        bt, layout = "?", "?"
    # tile-padded byte count: the device stores 32x32 tiles
    padded = n
    if len(shape) >= 2:
        padded = n // (shape[-1] * shape[-2]) * (-(-shape[-1] // 32) * 32) * (-(-shape[-2] // 32) * 32)
    try:
        addr = int(t.buffer_address())
    except Exception:
        addr = -1
    return {"shape": shape, "dtype": dt, "buffer": bt, "layout": layout, "addr": addr,
            "bytes": padded * eb, "logical_bytes": n * eb}


def is_tensor(x):
    return isinstance(x, ttnn.Tensor)


def tensors_in(x, out):
    """Collect tensors from an argument, including a list/tuple (ttnn.concat takes one)."""
    if is_tensor(x):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for e in x:
            tensors_in(e, out)


class Ledger:
    def __init__(self, dev, mem_probe, sync_per_op=False):
        self.dev = dev
        self.mem_probe = mem_probe
        self.sync_per_op = sync_per_op
        self.rows = []
        self.on = False
        self.phase = "residual"

    def mem(self):
        if not self.mem_probe:
            return None, None
        out = []
        for bt in (ttnn.BufferType.DRAM, ttnn.BufferType.L1):
            mv = ttnn.get_memory_view(self.dev, bt)
            out.append((mv.total_bytes_per_bank - mv.total_bytes_free_per_bank) * mv.num_banks)
        return out[0], out[1]

    def wrap(self, modpath, name):
        target = _resolve(modpath)
        orig = getattr(target, name)
        label = f"{modpath}.{name}"

        def wrapper(*args, **kwargs):
            if not self.on:
                return orig(*args, **kwargs)
            raw = []
            for a in args:
                tensors_in(a, raw)
            for v in kwargs.values():
                tensors_in(v, raw)
            ins = [i for i in (tensor_info(t) for t in raw) if i]
            if self.sync_per_op:
                ttnn.synchronize_device(self.dev)
            t0 = time.perf_counter()
            res = orig(*args, **kwargs)
            if self.sync_per_op:
                ttnn.synchronize_device(self.dev)
            dt = time.perf_counter() - t0
            outs = []
            if is_tensor(res):
                outs = [tensor_info(res)]
            elif isinstance(res, (list, tuple)):
                outs = [tensor_info(r) for r in res if is_tensor(r)]
            outs = [o for o in outs if o]
            dram, l1 = self.mem()
            self.rows.append({
                "idx": len(self.rows), "op": label, "phase": self.phase, "ins": ins,
                "outs": outs, "host_ms": dt * 1e3, "dram_alloc": dram, "l1_alloc": l1,
            })
            return res

        setattr(target, name, wrapper)
        return target, name, orig


class PhaseProxy:
    """Tags every ttnn op a sub-module issues with that sub-module's name.

    PairformerLayer.__call__ reaches its parts through instance attributes, so replacing the
    attribute with this proxy is enough; no class patching, no edit to tt_bio.
    """

    def __init__(self, name, inner, led):
        self.name, self.inner, self.led = name, inner, led

    def __call__(self, *a, **k):
        prev, self.led.phase = self.led.phase, self.name
        try:
            return self.inner(*a, **k)
        finally:
            self.led.phase = prev


PHASE_ATTRS = [
    "triangle_multiplication_start", "triangle_multiplication_end",
    "triangle_attention_start", "triangle_attention_end",
    "transition_z", "attention_pair_bias", "transition_s",
]


def run_ledger(dev, layer, c_z, N, warm, mem_probe, sync_per_op=False):
    torch.manual_seed(0)
    s = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    for _ in range(warm):
        s, z = layer(s, z)
    ttnn.synchronize_device(dev)

    led = Ledger(dev, mem_probe, sync_per_op)
    restore = [led.wrap(m, n) for m, n in OPS]
    inners = {}
    for attr in PHASE_ATTRS:
        inner = getattr(layer, attr, None)
        if inner is not None:
            inners[attr] = inner
            setattr(layer, attr, PhaseProxy(attr, inner, led))
    led.on = True
    s, z = layer(s, z)
    led.on = False
    ttnn.synchronize_device(dev)
    for target, name, orig in restore:
        setattr(target, name, orig)
    for attr, inner in inners.items():
        setattr(layer, attr, inner)
    return led.rows


# Metadata-only when the output shares the input's buffer: no bytes move.
VIEW_OPS = {"ttnn.unsqueeze", "ttnn.squeeze", "ttnn.reshape", "ttnn.chunk"}


def is_view(r):
    if r["op"] not in VIEW_OPS:
        return False
    in_addrs = {i["addr"] for i in r["ins"] if i["addr"] >= 0}
    return bool(in_addrs) and all(o["addr"] in in_addrs for o in r["outs"] if o["addr"] >= 0)


def summarize(rows, key):
    """First-order DRAM traffic: DRAM inputs are read, DRAM outputs are written.

    Views (an unsqueeze/reshape/chunk whose output shares the input's buffer) move nothing
    and are excluded, so the totals are traffic the DRAM controller actually sees.
    """
    agg = defaultdict(lambda: {"n": 0, "read": 0.0, "write": 0.0, "l1_out": 0.0, "host_ms": 0.0})
    tot_r = tot_w = tot_l1 = 0.0
    for r in rows:
        a = agg[key(r)]
        a["n"] += 1
        a["host_ms"] += r["host_ms"]
        if is_view(r):
            continue
        for i in r["ins"]:
            if i["buffer"] == "DRAM":
                a["read"] += i["bytes"]
                tot_r += i["bytes"]
        for o in r["outs"]:
            if o["buffer"] == "DRAM":
                a["write"] += o["bytes"]
                tot_w += o["bytes"]
            else:
                a["l1_out"] += o["bytes"]
                tot_l1 += o["bytes"]
    return agg, tot_r, tot_w, tot_l1


def working_set(rows):
    """Distinct device buffers the block touches: size, placement, first/last op index."""
    ts = {}
    for r in rows:
        for t in r["ins"] + r["outs"]:
            if t["addr"] < 0:
                continue
            k = (t["buffer"], t["addr"], round(t["bytes"]))
            e = ts.setdefault(k, {"buffer": t["buffer"], "bytes": t["bytes"],
                                  "shape": t["shape"], "dtype": t["dtype"],
                                  "first": r["idx"], "last": r["idx"], "touches": 0,
                                  "phases": set()})
            e["last"] = r["idx"]
            e["touches"] += 1
            e["phases"].add(r["phase"])
    for e in ts.values():
        e["phases"] = sorted(e["phases"])
    return sorted(ts.values(), key=lambda e: -e["bytes"])


def roofs(dev, iters, warm):
    """DRAM read and write roofs on THIS card, separated by one-sided clones.

    A DRAM->DRAM clone mixes a read and a write and cannot separate them; solving two mixed
    equations assumes read and write time add, which they do not. Instead:
      read  roof: DRAM -> L1 clone (the DRAM side is a pure read; the L1 write is ~free)
      write roof: L1 -> DRAM clone (the DRAM side is a pure write)
    The DRAM->DRAM clone and add are kept as a sanity check on the mixed rate.
    """
    out = {}
    r, c = 4096, 8192  # 67.1 MB: fits in L1 (110 cores x 1.53 MB) with margin
    nb = r * c * 2

    def timed(fn):
        for _ in range(warm):
            fn()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t0) / iters

    a = ttnn.ones((1, 1, r, c), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
    t_rd = timed(lambda: ttnn.clone(a, memory_config=ttnn.L1_MEMORY_CONFIG))
    b = ttnn.ones((1, 1, r, c), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
    t_dd = timed(lambda: ttnn.clone(a, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    t_add = timed(lambda: ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    ttnn.deallocate(b)
    al1 = ttnn.clone(a, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(a)
    t_wr = timed(lambda: ttnn.clone(al1, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    ttnn.deallocate(al1)

    out["bytes_MB"] = nb / 1e6
    out["read_GBs"] = nb / t_rd / 1e9
    out["write_GBs"] = nb / t_wr / 1e9
    out["dram_clone_GBs"] = 2 * nb / t_dd / 1e9
    out["dram_add_GBs"] = 3 * nb / t_add / 1e9
    out["t_read_ms"] = t_rd * 1e3
    out["t_write_ms"] = t_wr * 1e3
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["ledger", "roofs", "bench"], required=True)
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--read-roof", type=float, default=403.2e9, help="measured, this card")
    ap.add_argument("--write-roof", type=float, default=268.3e9, help="measured, this card")
    ap.add_argument("--mix-roof", type=float, default=443.7e9,
                    help="measured aggregate rate for a read+write mix, this card")
    ap.add_argument("--sync-per-op", action="store_true",
                    help="synchronize_device around every op: per-op device time, fully "
                         "serialized (so the sum exceeds the block wall by the overlap)")
    ap.add_argument("--mem-probe", action="store_true",
                    help="sample device DRAM/L1 allocation after every op (slow, ledger only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = get_device()
    import tt_bio.tenstorrent as T
    print(f"grid={T.COMPUTE_GRID_MAIN} cores={T.COMPUTE_GRID_MAIN[0]*T.COMPUTE_GRID_MAIN[1]} "
          f"l1_per_core={ttnn.get_max_worker_l1_unreserved_size()}", flush=True)

    if args.mode == "roofs":
        res = roofs(dev, args.iters, args.warm)
        for k, v in res.items():
            print(f"  {k} = {v:.3f}" if isinstance(v, float) else f"  {k} = {v}")
        if args.out:
            Path(args.out).write_text(json.dumps(res, indent=2))
        return

    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    layer, c_z = build_layer(ckc)
    N = args.n

    if args.mode == "bench":
        torch.manual_seed(0)
        s = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        for _ in range(args.warm):
            s, z = layer(s, z)
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(args.iters):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            s, z = layer(s, z)
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
        med = sorted(ts)[len(ts) // 2]
        print(f"BLOCK_MS n={N} c_z={c_z} median={med:.2f} min={min(ts):.2f} "
              f"series={[round(t,2) for t in ts]}")
        if args.out:
            Path(args.out).write_text(json.dumps({"n": N, "c_z": c_z, "median_ms": med, "ms": ts}, indent=2))
        return

    rows = run_ledger(dev, layer, c_z, N, args.warm, args.mem_probe, args.sync_per_op)
    R, W, MIX = args.read_roof, args.write_roof, args.mix_roof

    def floor_ms(rd, wr):
        """Bytes -> ms at this card's measured roofs. Reads and writes partly overlap, so
        the binding constraint is whichever of read-only, write-only or the mixed aggregate
        rate is largest. Checked against the roof probes themselves: predicts the DRAM->DRAM
        add within 0.2% and the clone within 11%."""
        return max(rd / R, wr / W, (rd + wr) / MIX) * 1e3

    print(f"LEDGER n={N} c_z={c_z} ops={len(rows)}")
    for title, key in [("BY PHASE", lambda r: r["phase"]), ("BY OP", lambda r: r["op"])]:
        agg, tot_r, tot_w, tot_l1 = summarize(rows, key)
        print(f"\n=== {title} ===")
        sort_key = (lambda kv: -kv[1]["host_ms"]) if args.sync_per_op else \
            (lambda kv: -(kv[1]["read"] + kv[1]["write"]))
        print(f"{'name':<44} {'n':>4} {'read_MB':>10} {'write_MB':>10} {'l1_out_MB':>10} "
              f"{'floor_ms':>9} {'meas_ms':>9} {'eff%':>6}")
        tot_ms = sum(a["host_ms"] for a in agg.values())
        for name, a in sorted(agg.items(), key=sort_key):
            fl = floor_ms(a["read"], a["write"])
            eff = 100 * fl / a["host_ms"] if a["host_ms"] > 0 else 0
            print(f"{name:<44} {a['n']:>4} {a['read']/1e6:>10.1f} {a['write']/1e6:>10.1f} "
                  f"{a['l1_out']/1e6:>10.1f} {fl:>9.2f} {a['host_ms']:>9.2f} {eff:>6.1f}")
        fl = floor_ms(tot_r, tot_w)
        print(f"{'TOTAL':<44} {len(rows):>4} {tot_r/1e6:>10.1f} {tot_w/1e6:>10.1f} "
              f"{tot_l1/1e6:>10.1f} {fl:>9.2f} {tot_ms:>9.2f} "
              f"{100*fl/tot_ms if tot_ms else 0:>6.1f}")
    print(f"\nroofs used: read {R/1e9:.1f} write {W/1e9:.1f} mixed {MIX/1e9:.1f} GB/s")
    if args.mem_probe:
        print(f"peak DRAM alloc = {max(r['dram_alloc'] for r in rows)/2**20:.1f} MiB")
        print(f"peak L1   alloc = {max(r['l1_alloc'] for r in rows)/2**20:.1f} MiB")
    ws = working_set(rows)
    print("\n=== WORKING SET (distinct device buffers, largest first) ===")
    print(f"{'MB':>8} {'buf':>5} {'dtype':<10} {'shape':<22} {'first':>6} {'last':>6} "
          f"{'touch':>6}  phases")
    for e in ws[:30]:
        print(f"{e['bytes']/1e6:>8.2f} {e['buffer']:>5} {e['dtype']:<10} "
              f"{'x'.join(str(d) for d in e['shape']):<22} {e['first']:>6} {e['last']:>6} "
              f"{e['touches']:>6}  {','.join(e['phases'])}")
    print(f"distinct buffers = {len(ws)}, total distinct bytes = "
          f"{sum(e['bytes'] for e in ws)/1e6:.1f} MB")
    if args.out:
        agg, tot_r, tot_w, tot_l1 = summarize(rows, lambda r: r["op"])
        Path(args.out).write_text(json.dumps(
            {"n": N, "c_z": c_z, "rows": rows, "working_set": ws,
             "total_read_B": tot_r, "total_write_B": tot_w, "total_l1_out_B": tot_l1},
            indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
