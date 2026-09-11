#!/usr/bin/env python3
"""Per-op DRAM bytes AND a runtime dtype census of one Boltz-2 fold, on device.

Two questions the roofline capture could not answer, from the same 4-minute run:

  1. Which ops inside our 12.17 GB pairformer block move the bytes? The roofline pass summed
     the block; the transfer plan has to rank three candidate fusions against each other and
     is currently scaling BioIR's own 83 Z breakdown by our 181.3/83 ratio to do it.
  2. What dtype were the tensors, actually? `tt_bio/boltz2.py` has no `ttnn.float32` and the
     two device fp32 gates default off, but that is a reachability argument over gate values,
     and per `rf3-runs-bf16-on-gpu-kernel-counter-is-not-a-dtype` a static read of construction
     sites is not proof of what the tensors were.

Method is `perf/bioir_roofline/fold_bytes_512.py`'s, with two changes: the graph walk keeps the
op name that allocated each buffer, and it recovers each tensor's bytes-per-element from
size/numel. 2 B is bf16, 4 B is fp32, ~1.06 B is bfloat8_b — so the dtype histogram is measured
off the captured tensors rather than read off the source.

Run with `--affinity` to census the affinity path, whose 64-block trunk runs fp32 on device by
default (`Fp32PairformerModule`).
"""
import argparse
import json
import math
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402

WALL = defaultdict(list)
CAP = {}
BUSY = {"on": False}
DEV = {"d": None}

# Bytes per element -> the dtype that is the only thing it can be on this stack.
_ELEM = {1: "bfloat8_b/uint8", 2: "bfloat16", 4: "float32/uint32"}


def _numel(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def sig(prefix, args):
    parts = []
    for x in args:
        sh = getattr(x, "shape", None)
        if sh is not None:
            parts.append("x".join(str(int(d)) for d in sh))
    return prefix + "|" + ",".join(parts[:2])


def walk(g):
    """Per-op DRAM bytes and a bytes-per-element histogram over the captured tensors."""
    per_op = defaultdict(lambda: {"calls": 0, "dram_write": 0})
    dtype_hist = defaultdict(lambda: {"tensors": 0, "bytes": 0})
    out = {"dram_read": 0, "dram_write": 0, "l1_write": 0, "n_ops": 0, "n_tensors": 0}
    seen = set()
    stack = []
    for n in g:
        t = n.get("node_type")
        p = n.get("params") or {}
        if t == "function_start":
            name = str(p.get("name", ""))
            stack.append(name)
            if name.startswith("ttnn."):
                out["n_ops"] += 1
                per_op[name]["calls"] += 1
        elif t == "function_end":
            if stack:
                stack.pop()
        elif t == "tensor":
            tid = p.get("tensor_id")
            if tid in seen:
                continue
            seen.add(tid)
            out["n_tensors"] += 1
            size = int(p.get("size", 0) or 0)
            shape = p.get("shape")
            if size and shape:
                try:
                    ne = _numel(eval(str(shape)) if isinstance(shape, str) else shape)
                except Exception:
                    ne = 0
                if ne:
                    bpe = _ELEM.get(int(round(size / ne)), f"{size / ne:.2f}B/elem")
                    dtype_hist[bpe]["tensors"] += 1
                    dtype_hist[bpe]["bytes"] += size
            if "DRAM" in str(p.get("buffer_type", "")):
                out["dram_read"] += size
        elif t == "buffer_allocate":
            size = int(p.get("size", 0) or 0)
            if str(p.get("type")) == "DRAM":
                out["dram_write"] += size
                owner = next((s for s in reversed(stack) if s.startswith("ttnn.")), "?")
                per_op[owner]["dram_write"] += size
            else:
                out["l1_write"] += size
    out["dram_total"] = out["dram_read"] + out["dram_write"]
    out["per_op"] = dict(sorted(per_op.items(), key=lambda kv: -kv[1]["dram_write"])[:40])
    out["bytes_per_elem"] = dict(dtype_hist)
    return out


def wrap(cls, prefix, capture_on_call):
    orig = cls.__call__

    def call(self, *a, **k):
        s = sig(prefix, a)
        n = len(WALL[s])
        want = (n == capture_on_call) and s not in CAP and not BUSY["on"]
        if want:
            BUSY["on"] = True
            ttnn.synchronize_device(DEV["d"])
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        ttnn.synchronize_device(DEV["d"])
        t0 = time.perf_counter()
        r = orig(self, *a, **k)
        ttnn.synchronize_device(DEV["d"])
        dt = time.perf_counter() - t0
        if want:
            CAP[s] = walk(ttnn.graph.end_graph_capture())
            BUSY["on"] = False
        WALL[s].append(dt)
        return r

    cls.__call__ = call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import tt_baseline as B
    B.SAMPLING_STEPS = a.steps
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    import fold_ab_multi as FAM
    from tt_bio.main import _resolve_recycling_steps
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    FAM.patch_boltz2_cfg()

    fix = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, state = B.build_fold(
        "boltz2", ROOT / f".msa_bytes_{a.size}",
        fix / f"cdk2x2_{a.size}.yaml", fix / f"cdk2x2_{a.size}.a3m")
    DEV["d"] = T.get_device()

    wrap(T.PairformerLayer, "pairformer", 3)
    wrap(T.DiffusionTransformerLayer, "difftx", 3)

    fold_s, m = one_fold()
    print("FOLD %.2f s n_tokens=%s plddt=%s" % (fold_s, m.get("n_tokens"), m.get("plddt")),
          flush=True)

    rows = []
    for s in sorted(WALL):
        ts = WALL[s]
        row = {"sig": s, "calls_in_capture_fold": len(ts),
               "median_ms": round(1e3 * st.median(ts), 4)}
        row.update(CAP.get(s, {}))
        rows.append(row)
        hist = row.get("bytes_per_elem", {})
        print("%-44s %6d calls %8.3f ms  dtype=%s" % (
            s[:44], len(ts), row["median_ms"],
            {k: v["tensors"] for k, v in hist.items()}), flush=True)
        for op, v in list(row.get("per_op", {}).items())[:12]:
            print("      %-46s %5d  %8.1f MB" % (op, v["calls"], v["dram_write"] / 1e6),
                  flush=True)

    Path(a.out).write_text(json.dumps(
        {"size": a.size, "capture_steps": a.steps, "fold_s": round(fold_s, 3),
         "n_tokens": m.get("n_tokens"), "plddt": m.get("plddt"), "rows": rows}, indent=1))
    print("WROTE " + a.out)
    T.cleanup()


if __name__ == "__main__":
    main()
