"""Record the fold's executed ttnn graph with buffer addresses, so adjacency is measured.

The question this answers: which hand-written multiply/add and add/layer_norm chains are
actually adjacent in the graph the device runs, i.e. the producer's result has exactly one
consumer and that consumer is the op we would fuse into. A grep over the source cannot tell
you that; a producer whose result is also read by a later residual add is not fusable at all.

Dataflow is tracked on (buffer address, version), never on tensor id. ttnn reuses freed
addresses, so an address alone aliases unrelated tensors; a version counter incremented on
every write to that address gives clean SSA numbering. Every op in the fold's own census
(perf/roof_launch/op_census_512.json, 30 ops) is wrapped, because one unwrapped op that reads
a buffer would make its producer look single-consumer when it is not, an error in the
direction that invents a fusion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

# The 30 ops the 512 aa census records, plus a few eltwise ops it does not reach but our
# source calls, so an unwrapped reader cannot invent a single-consumer producer.
WRAP = [
    ("ttnn", "linear"), ("ttnn", "matmul"), ("ttnn", "multiply_"), ("ttnn", "multiply"),
    ("ttnn", "add_"), ("ttnn", "add"), ("ttnn", "subtract"), ("ttnn", "subtract_"),
    ("ttnn", "layer_norm"), ("ttnn", "rms_norm"), ("ttnn", "reshape"),
    ("ttnn", "to_memory_config"), ("ttnn", "slice"), ("ttnn", "permute"), ("ttnn", "unsqueeze"),
    ("ttnn", "squeeze"), ("ttnn", "pad"), ("ttnn", "to_layout"), ("ttnn", "concat"),
    ("ttnn", "chunk"), ("ttnn", "softmax"), ("ttnn", "transpose"), ("ttnn", "cos"),
    ("ttnn", "sin"), ("ttnn", "exp"), ("ttnn", "sqrt"), ("ttnn", "rsqrt"), ("ttnn", "mean"),
    ("ttnn", "sum"), ("ttnn", "to_torch"), ("ttnn", "from_device"), ("ttnn", "from_torch"),
    ("ttnn", "generic_op"), ("ttnn", "deallocate"), ("ttnn", "typecast"), ("ttnn", "clone"),
    ("ttnn", "sigmoid"), ("ttnn", "silu"), ("ttnn", "gelu"), ("ttnn", "where"),
    ("ttnn", "addcmul"), ("ttnn", "addalpha"), ("ttnn", "div"), ("ttnn", "divide"),
    ("ttnn.transformer", "scaled_dot_product_attention"),
    ("ttnn.experimental", "nlp_create_qkv_heads"), ("ttnn.experimental", "nlp_concat_heads"),
]

REC = []
SITES: dict = {}
SHAPES: dict = {}
VER: dict = defaultdict(int)
_TRACING = [False]
_SEQ = [0]
_TRACE_FILE = os.path.abspath(__file__)


def _site() -> int:
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if fn != _TRACE_FILE and ("tt_bio" in fn or "tt_baseline" in fn):
            s = "%s:%d:%s" % (fn.split("tt_bio/")[-1], f.f_lineno, f.f_code.co_name)
            return SITES.setdefault(s, len(SITES))
        f = f.f_back
    return SITES.setdefault("<unknown>", len(SITES))


def _shape_id(t) -> int:
    try:
        k = "%s|%s" % ("x".join(str(int(d)) for d in t.shape), str(t.dtype).split(".")[-1])
    except Exception:
        k = "?"
    return SHAPES.setdefault(k, len(SHAPES))


def _nbytes(t) -> int:
    try:
        n = 1
        for d in t.shape:
            n *= int(d)
        s = str(t.dtype)
        w = 4 if "float32" in s or "uint32" in s or "int32" in s else (1 if "bfloat8" in s or "bfloat4" in s else 2)
        return n * w
    except Exception:
        return 0


def _addr(t):
    try:
        return int(t.buffer_address())
    except Exception:
        return None


_DRAM = [None]


def _space(t):
    """1 if the tensor lives in DRAM, 0 if in L1, -1 unknown.

    A fusion deletes the producer's output write and the consumer's read of it. If both live in
    L1 (which the shipped _PAIR_PROJ_L1_OUT lever arranges for exactly the trimul's multiply_
    and the Pairformer's residual add_) then no DRAM byte is deleted and the fusion is worth
    nothing at the DRAM roof. Pricing an L1 operand at a DRAM rate is the error that invents a
    lever, so residency is recorded per operand rather than assumed.
    """
    try:
        import ttnn
        if _DRAM[0] is None:
            _DRAM[0] = ttnn.BufferType.DRAM
        return 1 if t.memory_config().buffer_type == _DRAM[0] else 0
    except Exception:
        return -1


def _tensors(obj, out, depth=0):
    if depth > 2 or obj is None:
        return
    if hasattr(obj, "buffer_address") and hasattr(obj, "shape"):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            _tensors(o, out, depth + 1)
    elif isinstance(obj, dict):
        for o in obj.values():
            _tensors(o, out, depth + 1)


def _wrap(name, fn):
    def w(*a, **k):
        if not _TRACING[0]:
            return fn(*a, **k)
        ins = []
        _tensors(a, ins)
        _tensors(k, ins)
        reads = []
        for t in ins:
            ad = _addr(t)
            if ad is not None:
                reads.append((ad, VER[ad], _shape_id(t), _nbytes(t), _space(t)))
        site = _site()
        r = fn(*a, **k)
        outs = []
        _tensors(r, outs)
        writes = []
        for t in outs:
            ad = _addr(t)
            if ad is None:
                continue
            VER[ad] += 1
            writes.append((ad, VER[ad], _shape_id(t), _nbytes(t), _space(t)))
        REC.append((_SEQ[0], name, site, reads, writes))
        _SEQ[0] += 1
        return r

    w.__name__ = name
    return w


def install():
    import ttnn
    n = 0
    for path, name in WRAP:
        mod = ttnn
        ok = True
        for p in path.split(".")[1:]:
            mod = getattr(mod, p, None)
            if mod is None:
                ok = False
                break
        if not ok:
            continue
        fn = getattr(mod, name, None)
        if fn is None or not callable(fn):
            continue
        try:
            setattr(mod, name, _wrap(name, fn))
            n += 1
        except Exception as e:
            print("cannot wrap %s.%s: %s" % (path, name, e))
    print("wrapped %d ops" % n, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="boltz2")
    ap.add_argument("--size", default="512")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn  # noqa: F401
    import tt_baseline as B
    from tt_bio.main import (_detect_p300_devices, _find_ttnn_mesh_graph_descriptor,
                             _resolve_recycling_steps, _resolve_sampling_steps)
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    if a.model == "boltz2":
        sys.path.insert(0, str(ROOT / "perf" / "other512"))
        import fold_ab_multi as _FAM
        _FAM.patch_boltz2_cfg()

    fixdir = ROOT / "perf" / "size512" / "fixtures"
    tgt = fixdir / ("cdk2x2_%s.yaml" % a.size)
    a3m = fixdir / ("cdk2x2_%s.a3m" % a.size)
    t0 = time.time()
    one_fold, meta, state = B.build_fold(
        a.model, ROOT / (".msa_s512_%s_%s" % (a.model, a.size)), tgt, a3m)
    print("build_fold %.1fs" % (time.time() - t0), flush=True)
    install()
    _TRACING[0] = True
    t1 = time.time()
    try:
        one_fold()
    finally:
        _TRACING[0] = False
    print("traced fold %.1fs, %d ops" % (time.time() - t1, len(REC)), flush=True)
    import tt_bio.tenstorrent as _TT
    latch = {k: {kk: vv for kk, vv in v.items() if kk != "why"}
             for k, v in getattr(_TT, "LATCH_STATS", {}).items()
             if any(v.get(f) for f in ("served", "refused", "blocked", "declined"))}
    print("latches: %s" % json.dumps(latch), flush=True)
    print("trimul g_out fused: %s" % getattr(_TT, "TRIMUL_GOUT_STATS", None), flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"model": a.model, "size": a.size, "n_ops": len(REC),
                   "sites": {v: k for k, v in SITES.items()},
                   "shapes": {v: k for k, v in SHAPES.items()},
                   "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
                   "latches": latch,
                   "fields": ["addr", "version", "shape_id", "bytes", "dram"],
                   "records": REC}, f)
    print("wrote %s (%.1f MB)" % (a.out, a.out.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
