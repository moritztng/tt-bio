#!/usr/bin/env python3
"""Op-level census of a boltz-2 fold: how many ttnn ops each region issues, and at which shapes.

Why this and not a profiler. The question the 512 aa fold now turns on is whether the 8.5 s that has
never been decomposed is bound by bytes, by arithmetic, or by per-op fixed cost. Module-level walls
cannot answer it and the tt-metal device profiler needs a profiler-enabled build that the pip wheel
does not carry. What settles it without either is the pair (region wall, ops issued in that region):
their ratio is the region's MEAN COST PER OP, and comparing that against the region's byte and
arithmetic floors says which of the three binds. A region running 20 us/op on tensors whose read+write
is 3.9 us is not short of bandwidth and is not short of FLOPs.

The instrument tax is deliberately asymmetric. Region walls are timed exactly as `decompose.py` times
them -- `ttnn.synchronize_device` on both sides -- so a region wall here is comparable with the walls
already in the state doc. Op counting adds only a dict update per op, ~1 us against ~300 k ops, i.e.
~0.3 s on a 25 s fold. The counter is NOT a timer: it never synchronises, so it cannot serialise the
async queue and cannot move the region walls it is being divided into.

What the counter keys on: the op name plus the shape of every tensor argument. That is enough to
separate the token DiT's [1,512,768] linears from the atom DiT's [1,224,32,128] ones without
reconstructing program configs, and it is what a follow-up off-fold micro-benchmark needs to build
the op at the production shape (the route `s2_atom_pad.py` used to predict S2's landing to 91 %).

Output: perf/b2deep/opcensus_512.json
    {"regions": {<region>: {"wall_ms":, "calls":, "ops":, "us_per_op":,
                            "by_op": {<opkey>: count}}}, ...}
Regions nest; an op is charged to the INNERMOST region live when it is issued, so the per-region
`ops` are disjoint and sum to the total.
"""
import argparse, hashlib, json, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(HERE))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
OPS = defaultdict(Counter)
STACK = ["root"]
STATE = {"dev": None}

# Every ttnn entry point the model reaches. Wrapped by name off the module object, so a name that
# does not exist on this wheel is skipped rather than raising -- the run records which were wrapped.
TTNN_OPS = [
    "linear", "matmul", "layer_norm", "rms_norm", "softmax", "add", "add_", "subtract",
    "multiply", "multiply_", "div", "sigmoid", "silu", "gelu", "exp", "typecast", "clone",
    "permute", "transpose", "reshape", "slice", "pad", "concat", "chunk", "split",
    "to_layout", "to_memory_config", "reallocate", "unsqueeze", "squeeze", "repeat",
    "repeat_interleave", "sum", "mean", "max", "generic_op", "embedding", "where", "sqrt",
    "rsqrt", "relu", "tanh", "log", "abs", "arange", "zeros_like", "full_like",
]
TTNN_SUB = {
    "experimental": ["nlp_create_qkv_heads", "nlp_concat_heads", "minimal_matmul", "rotary_embedding"],
    "transformer": ["scaled_dot_product_attention", "concatenate_heads"],
}


def _shape(x):
    s = getattr(x, "shape", None)
    if s is None:
        return None
    try:
        return "x".join(str(int(d)) for d in s)
    except Exception:                                                            # noqa: BLE001
        return None


def _key(name, args, kw):
    parts = [p for p in (_shape(x) for x in args) if p]
    for k in ("input_tensor_b", "weight", "bias", "keys_indexing"):
        p = _shape(kw.get(k))
        if p:
            parts.append(f"{k}={p}")
    return f"{name}({'|'.join(parts)})" if parts else name


def _count(name, fn):
    def wrapper(*a, **kw):
        OPS[STACK[-1]][_key(name, a, kw)] += 1
        return fn(*a, **kw)
    return wrapper


def install_counters(ttnn):
    wrapped = []
    for n in TTNN_OPS:
        f = getattr(ttnn, n, None)
        if callable(f):
            setattr(ttnn, n, _count(n, f))
            wrapped.append(n)
    for sub, names in TTNN_SUB.items():
        mod = getattr(ttnn, sub, None)
        if mod is None:
            continue
        for n in names:
            f = getattr(mod, n, None)
            if callable(f):
                setattr(mod, n, _count(f"{sub}.{n}", f))
                wrapped.append(f"{sub}.{n}")
    return wrapped


def timed_region(key, fn, *a, **kw):
    """Region wall with a sync on both sides, and the region pushed for op attribution.

    The push happens INSIDE the timed span on purpose: an op issued by this region's own body is
    charged here, and an op issued by a nested region is charged there, so `ops` are disjoint while
    `wall_ms` stays inclusive. That asymmetry is intended -- the ratio wall/ops is only meaningful
    for a LEAF region, and the leaf check is `ops(region) == sum of its own by_op`, which holds by
    construction. For a parent, subtract the children before dividing.
    """
    import ttnn
    ttnn.synchronize_device(STATE["dev"])
    t0 = time.perf_counter()
    STACK.append(key)
    try:
        out = fn(*a, **kw)
    finally:
        STACK.pop()
    ttnn.synchronize_device(STATE["dev"])
    w = WALL[key]
    w["n"] += 1
    w["s"] += time.perf_counter() - t0
    return out


def patch(mod, cls_name, key, split=None):
    """Wrap `cls.__call__`. `split` maps (self, args) -> a key suffix, for level/shape keying."""
    cls = getattr(mod, cls_name, None)
    if cls is None:
        return None
    orig = cls.__call__

    def call(self, *a, **kw):
        k = key if split is None else f"{key}|{split(self, a)}"
        return timed_region(k, orig, self, *a, **kw)

    cls.__call__ = call
    return key


def sha_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(Path(d).glob("*")) if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, default=HERE / "opcensus_512.json")
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import (_resolve_recycling_steps, _resolve_sampling_steps,
                             _detect_p300_devices, _find_ttnn_mesh_graph_descriptor)
    from decompose import patch_boltz2_cfg

    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}, set PYTHONPATH"

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, "boltz2")
    patch_boltz2_cfg()

    lvl = lambda self, args: ("atom" if getattr(self, "atom_level", False) else "token")
    shp = lambda self, args: f"{lvl(self, args)}|{_shape(args[0]) or '?'}"

    installed = [k for k in [
        patch(T, "TriangleMultiplication", "body:TriangleMultiplication"),
        patch(T, "TriangleAttention", "body:TriangleAttention"),
        patch(T, "Transition", "body:Transition", split=lambda s, a: _shape(a[0]) or "?"),
        patch(T, "AttentionPairBias", "body:AttentionPairBias", split=shp),
        patch(T, "AdaLN", "body:AdaLN", split=shp),
        patch(T, "ConditionedTransitionBlock", "body:ConditionedTransitionBlock", split=shp),
        patch(T, "DiffusionTransformerLayer", "block:DiffusionTransformerLayer", split=shp),
        patch(T, "PairWeightedAveraging", "body:PairWeightedAveraging"),
        patch(T, "OuterProductMean", "body:OuterProductMean"),
        patch(T, "MSALayer", "body:MSALayer"),
        patch(T, "PairformerLayer", "block:PairformerLayer"),
    ] if k]

    wrapped = install_counters(ttnn)

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold("boltz2", ROOT / f".msa_om512_{a.size}", tgt, a3m)
    STATE["dev"] = T.get_device()
    struct_dir = Path(meta["struct_dir"])

    print("=== cold ===", flush=True)
    cold_s, _ = one_fold()
    print(f"  cold {cold_s:.2f}s", flush=True)

    WALL.clear(); OPS.clear(); STACK[:] = ["root"]
    print("=== warm (census) ===", flush=True)
    fold_s, m = one_fold()

    regions = {}
    for k, w in sorted(WALL.items(), key=lambda kv: -kv[1]["s"]):
        n_ops = sum(OPS[k].values())
        regions[k] = {
            "wall_ms": round(w["s"] * 1e3, 2),
            "calls": w["n"],
            "ops": n_ops,
            "ops_per_call": round(n_ops / w["n"], 2) if w["n"] else None,
            # own-region time is unknown for a parent; us_per_op is only read for leaves
            "us_per_op": round(w["s"] * 1e6 / n_ops, 2) if n_ops else None,
            "by_op": dict(OPS[k].most_common()),
        }
    res = {
        "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
        "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "model": "boltz2", "size": a.size,
        "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
        "fold_s": round(fold_s, 3), "cold_s": round(cold_s, 3),
        "cif_sha256": sha_dir(struct_dir),
        "loadavg": [f"{x:.2f}" for x in os.getloadavg()],
        "regions_installed": installed, "ttnn_ops_wrapped": wrapped,
        "root_ops": sum(OPS["root"].values()),
        "total_ops": sum(sum(c.values()) for c in OPS.values()),
        "regions": regions,
        "root_by_op": dict(OPS["root"].most_common(60)),
    }
    a.out.write_text(json.dumps(res, indent=1))
    print(f"fold {fold_s:.3f}s  total ops {res['total_ops']}  -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
