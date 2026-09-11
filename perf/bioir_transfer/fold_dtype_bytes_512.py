#!/usr/bin/env python3
"""Per-op DRAM bytes AND a runtime dtype census of one Boltz-2 fold, on device.

Two questions the roofline capture could not answer, from the same run:

  1. Which ops inside our 12.17 GB pairformer block move the bytes? The roofline pass summed
     the block; the transfer plan has to rank three candidate fusions against each other and
     is currently scaling BioIR's own 83 Z breakdown by our 181.3/83 ratio to do it.
  2. What dtype were the tensors, actually? `tt_bio/boltz2.py` has no `ttnn.float32` and the
     two device fp32 gates default off, but that is a reachability argument over gate values,
     and per `rf3-runs-bf16-on-gpu-kernel-counter-is-not-a-dtype` a static read of construction
     sites is not proof of what the tensors were.

Byte attribution is `perf/bioir_roofline/fold_bytes_512.py`'s graph capture, with the op that
allocated (write) or read (input tensor resident in DRAM) each buffer kept instead of discarded.
Reads are ~73 % of a pairformer block's traffic, so a write-only ranking is not a ranking.

The dtype census does NOT come from the graph. Under `enable_fast_runtime_mode=true` a captured
tensor node carries `size` but no `shape`, so bytes-per-element is not recoverable, and
`FastOperation.__call__` never runs `POST_OPERATION_HOOKS`. Both facts were found by running the
first version of this script. Instead every ttnn op is wrapped at `FastOperation.__call__` and
the real `ttnn.Tensor.dtype` of its inputs and outputs is recorded, for the whole fold rather
than for two layer types. Any float32 output also records the tt_bio line that produced it.
"""
import argparse
import json
import statistics as st
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import ttnn                                                                   # noqa: E402
from ttnn import decorators as _dec                                           # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402

WALL = defaultdict(list)
CAP = {}
BUSY = {"on": False}
DEV = {"d": None}

# Runtime dtype census, filled by the FastOperation wrapper.
OP_DT = defaultdict(lambda: {"calls": 0, "out": Counter(), "in": Counter()})
DT_OUT = Counter()          # dtype -> output tensors produced
DT_ELEM = Counter()         # dtype -> logical elements produced
FP32_SITES = Counter()      # tt_bio call site -> float32 outputs produced
SITES = defaultdict(Counter)  # op name -> tt_bio call site -> calls
SITE_OPS = set()            # ops whose call site is worth the traceback
TOTAL = {"calls": 0}

_BITS = {"bfloat16": 16, "float32": 32, "bfloat8_b": 8.0625, "bfloat4_b": 4.0625,
         "uint32": 32, "int32": 32, "uint16": 16, "uint8": 8}


def _numel(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _tensors(obj, acc, depth=0):
    if isinstance(obj, ttnn.Tensor):
        acc.append(obj)
    elif depth < 2:
        if isinstance(obj, (list, tuple)):
            for o in obj:
                _tensors(o, acc, depth + 1)
        elif isinstance(obj, dict):
            for o in obj.values():
                _tensors(o, acc, depth + 1)


def _site():
    """Innermost tt_bio frame, so a float32 tensor names the line that made it."""
    for fr in reversed(traceback.extract_stack()[:-2]):
        if "/tt_bio/" in fr.filename:
            return "%s:%d" % (fr.filename.split("/tt_bio/")[-1], fr.lineno)
    return "?"


def census_ops():
    """Wrap every ttnn op at the one place they all go through.

    POST_OPERATION_HOOKS are only run by `Operation`; the fold runs `FastOperation`
    (enable_fast_runtime_mode=true), which skips them. Patching __call__ catches every op
    including ttnn.experimental.* and ttnn.generic_op.
    """
    for cls in (_dec.FastOperation, getattr(_dec, "Operation", None)):
        if cls is None:
            continue
        orig = cls.__call__

        def call(self, *a, _orig=orig, **k):
            r = _orig(self, *a, **k)
            name = self.python_fully_qualified_name
            rec = OP_DT[name]
            rec["calls"] += 1
            if name in SITE_OPS:
                SITES[name][_site()] += 1
            TOTAL["calls"] += 1
            ins = []
            _tensors(a, ins)
            _tensors(k, ins)
            for t in ins:
                rec["in"][t.dtype.name.lower()] += 1
            outs = []
            _tensors(r, outs)
            for t in outs:
                dt = t.dtype.name.lower()
                rec["out"][dt] += 1
                DT_OUT[dt] += 1
                try:
                    DT_ELEM[dt] += _numel(t.shape)
                except Exception:
                    pass
                if dt == "float32" and len(FP32_SITES) < 200:
                    FP32_SITES[name + " @ " + _site()] += 1
            return r

        cls.__call__ = call


def sig(prefix, args):
    parts = []
    for x in args:
        sh = getattr(x, "shape", None)
        if sh is not None:
            parts.append("x".join(str(int(d)) for d in sh))
    return prefix + "|" + ",".join(parts[:2])


def walk(g):
    """Per-op DRAM bytes, reads and writes both attributed to the op that moved them."""
    per_op = defaultdict(lambda: {"calls": 0, "dram_read": 0, "dram_write": 0})
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
            if "DRAM" in str(p.get("buffer_type", "")):
                out["dram_read"] += size
                owner = next((s for s in reversed(stack) if s.startswith("ttnn.")), "?")
                per_op[owner]["dram_read"] += size
        elif t == "buffer_allocate":
            size = int(p.get("size", 0) or 0)
            if str(p.get("type")) == "DRAM":
                out["dram_write"] += size
                owner = next((s for s in reversed(stack) if s.startswith("ttnn.")), "?")
                per_op[owner]["dram_write"] += size
            else:
                out["l1_write"] += size
    out["dram_total"] = out["dram_read"] + out["dram_write"]
    for v in per_op.values():
        v["dram_total"] = v["dram_read"] + v["dram_write"]
    out["per_op"] = dict(sorted(per_op.items(), key=lambda kv: -kv[1]["dram_total"])[:40])
    return out


def wrap(cls, prefix, capture_on_call=3, attrs=("ending",)):
    orig = cls.__call__

    def call(self, *a, **k):
        tag = "".join("/" + str(getattr(self, at)) for at in attrs if hasattr(self, at))
        s = sig(prefix + tag, a)
        n = len(WALL[s])
        # `>=`, not `==`: an outer module's capture is live on some inner module's Nth call,
        # and a `==` would drop that inner capture for the whole fold.
        want = (n >= capture_on_call) and s not in CAP and not BUSY["on"]
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
    ap.add_argument("--steps", type=int, default=None,
                    help="sampling steps; default is the shipped one (the published cell)")
    ap.add_argument("--sites", default="ttnn.generic_op,ttnn.allocate_tensor_on_device,"
                    "ttnn.chunk,ttnn.layer_norm,ttnn.matmul,ttnn.transpose,ttnn.concat",
                    help="ops whose tt_bio call site is recorded (costs a traceback per call)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    SITE_OPS.update(x for x in a.sites.split(",") if x)

    import tt_baseline as B
    if a.steps:
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

    census_ops()
    wrap(T.PairformerLayer, "pairformer")
    wrap(T.DiffusionTransformerLayer, "difftx")
    # Submodule captures rank the fusion boundaries inside the block; the op-name table alone
    # cannot, because one op name (layer_norm, chunk, generic_op) serves several of them.
    wrap(T.TriangleMultiplication, "trimul")
    wrap(T.TriangleAttention, "triatt")
    wrap(T.Transition, "transition")
    wrap(T.AttentionPairBias, "attnpairbias")

    fold_s, m = one_fold()
    print("FOLD %.2f s steps=%s meta=%s" % (fold_s, B.SAMPLING_STEPS, m), flush=True)

    print("\n== runtime dtype census, whole fold: %d ttnn op calls ==" % TOTAL["calls"])
    for dt, n in DT_OUT.most_common():
        print("  %-12s %9d output tensors  %14d elements  %8.2f GB" % (
            dt, n, DT_ELEM[dt], DT_ELEM[dt] * _BITS.get(dt, 16) / 8 / 1e9), flush=True)
    for op in sorted(SITES):
        print("  %s call sites:" % op)
        for site, n in SITES[op].most_common(8):
            print("    %-64s %7d" % (site, n), flush=True)
    if FP32_SITES:
        print("  float32 sites:")
        for s, n in FP32_SITES.most_common(30):
            print("    %-70s %7d" % (s, n), flush=True)

    rows = []
    for s in sorted(WALL):
        ts = WALL[s]
        row = {"sig": s, "calls_in_capture_fold": len(ts),
               "median_ms": round(1e3 * st.median(ts), 4)}
        row.update(CAP.get(s, {}))
        rows.append(row)
        print("\n%-44s %6d calls %8.3f ms  %6.2f GB/call" % (
            s[:44], len(ts), row["median_ms"], row.get("dram_total", 0) / 1e9), flush=True)
        for op, v in list(row.get("per_op", {}).items())[:14]:
            print("      %-46s %5d  r %8.1f  w %8.1f  = %8.1f MB" % (
                op, v["calls"], v["dram_read"] / 1e6, v["dram_write"] / 1e6,
                v["dram_total"] / 1e6), flush=True)

    Path(a.out).write_text(json.dumps(
        {"size": a.size, "sampling_steps": B.SAMPLING_STEPS, "fold_s": round(fold_s, 3),
         "meta": {k: v for k, v in (m or {}).items() if isinstance(v, (int, float, str))},
         "ttnn_calls": TOTAL["calls"],
         "dtype_out_tensors": dict(DT_OUT), "dtype_out_elements": dict(DT_ELEM),
         "fp32_sites": dict(FP32_SITES),
         "call_sites": {k: dict(v) for k, v in sorted(SITES.items())},
         "per_op_dtype": {k: {"calls": v["calls"], "out": dict(v["out"]), "in": dict(v["in"])}
                          for k, v in sorted(OP_DT.items())},
         "rows": rows}, indent=1))
    print("WROTE " + a.out)
    T.cleanup()


if __name__ == "__main__":
    main()
