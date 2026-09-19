"""One instrumented fold, two counts.

Q1 needs every read and write of a z-sized buffer inside one `PairformerLayer`, keyed by device
ALLOCATION rather than by tensor id (`ttnn-graph-byte-count-must-dedupe-buffer-not-tensor-id`: a
tensor-id key inflated this same block by 1.83x once already). Q2 needs every transpose in the
whole fold with the axes it swaps, because the 0.5175 s `TransposeDeviceOperation` class is a
whole-fold number and a block capture cannot partition it.

Both come out of one fold, so the card is opened once:

  * the buffer-keyed recorder from `perf/b2z2_byte_floor/trace_block.py` is armed on ONE warm
    `PairformerLayer` call inside the real fold, not on a standalone layer. A standalone layer is
    built with a compute-kernel config this file chooses, and the shipped fold's config is the one
    the seconds were measured at.
  * a second, much cheaper recorder stays on for the whole fold and records only transposes and
    permutes: op, unit stack, call site, rank, the axis permutation, shape, dtype and bytes.

The unit stack is this file's own, because `trace_block`'s tag chain records `file:function` and
every sub-unit of a PairformerLayer is `tenstorrent.py:__call__`. Each instrumented class pushes
`<Class>#<n>`, numbered within its parent's call, so the two `TriangleMultiplication` calls come
out as `#0` (outgoing/start) and `#1` (incoming/end) in source order.

Usage:
  TT_VISIBLE_DEVICES=N TT_BIO_LEASE_CARDS=N TT_BIO_LEASE_HOLDER=worker:anthro-zpass-census \
    python3 perf/anthro_zpass/capture.py --size 512 --block-call 8 --out-dir perf/anthro_zpass/out/x
"""
import argparse
import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(ROOT, "perf", "b2z2_byte_floor"), ROOT,
          os.path.join(ROOT, "scripts", "gpu_vs_tt"), os.path.join(ROOT, "perf", "other512")):
    sys.path.insert(0, p)

P = argparse.ArgumentParser(description=__doc__.splitlines()[0])
P.add_argument("--size", type=int, default=512)
P.add_argument("--block-call", type=int, default=8,
               help="which PairformerLayer call of the fold carries the full ledger, 0-based")
P.add_argument("--parent", default="Pairformer",
               help="whose PairformerLayer to arm: Pairformer (the trunk) or MSALayer")
P.add_argument("--out-dir", default=os.path.join(HERE, "out", "dev"))
ARGS = P.parse_args()

import torch  # noqa: E402
import ttnn  # noqa: E402
import trace_block as TB  # noqa: E402

# Classes that get a unit-stack frame. Everything a pair tensor can be touched inside during a
# Boltz-2 fold, plus the top-level units `state/c12-profiled-fold.md` quotes seconds for, so the
# transpose census can be split the same way that doc splits device time.
UNIT_CLASSES = (
    "TrunkModule", "PairformerModule", "Pairformer", "PairformerLayer",
    "TriangleMultiplication", "TriangleAttention", "Transition", "AttentionPairBias",
    "MSAModule", "MSA", "MSALayer", "PairWeightedAveraging", "OuterProductMean",
    "DiffusionModule", "Diffusion", "DiffusionTransformer", "DiffusionTransformerLayer",
    "AdaLN", "ConditionedTransitionBlock",
    "RelPosGather", "PairConditioningDevice", "PairAssemblyDevice", "ConfidenceHeadsDevice",
)

STACK = []              # [[label, Counter]] -- Counter numbers this frame's children
TOP = Counter()
PFL = Counter()         # PairformerLayer calls, keyed by the parent that issued them
TCENSUS = []            # every transpose/permute in the fold
BLOCK = {}              # the armed PairformerLayer call's ledger


def _labels():
    return [f[0] for f in STACK]


# `trace_block`'s tag chain is file:function, and every sub-unit of a PairformerLayer resolves to
# `tenstorrent.py:__call__`, so the ledger cannot say which of the five sub-units owns a buffer.
# Prepending this file's unit stack fixes that without changing the schema: the recorder reads
# `_stack_tags` as a module global, so replacing it is enough.
_TB_TAGS = TB._stack_tags
TB._stack_tags = lambda: [f"u:{x}" for x in _labels()] + _TB_TAGS()


def _unit_wrap(cls, fn):
    def w(self, *a, **kw):
        c = STACK[-1][1] if STACK else TOP
        n = c[cls]
        c[cls] = n + 1
        STACK.append([f"{cls}#{n}", Counter()])
        try:
            return fn(self, *a, **kw)
        finally:
            STACK.pop()
    w.__name__ = f"unit_{cls}"
    return w


TILED_HINT = {"UNTILED": "both moved axes sit outside the tiled pair: tile blocks are relabelled, "
                         "no element leaves its tile",
              "TILED_INNER": "the swap is the tiled (-2,-1) pair: a real element transpose the "
                             "hardware does tile-locally",
              "MIXED": "an untiled axis is exchanged with a tiled one: rows move between tiles, "
                       "which is the row-granular scatter tt_bio measured at 19% of the copy roof",
              "NOOP": "identity permutation"}


def classify(rank, perm):
    """Which of the three shapes a permutation of a TILE-layout tensor is.

    TTNN tiles the last two axes. Whether a transpose is addressing or data movement is therefore
    a property of WHICH axes move, not of the op name.
    """
    moved = {i for i in range(rank) if perm[i] != i}
    inner = {rank - 2, rank - 1} if rank >= 2 else set()
    if not moved:
        return "NOOP"
    if not (moved & inner):
        return "UNTILED"
    if moved <= inner:
        return "TILED_INNER"
    return "MIXED"


def _perm_of(op, rank, a, kw):
    """The permutation this call applies, as a tuple, or None if it cannot be read off."""
    if op.endswith(".transpose"):
        d = [x for x in list(a[1:3]) + [kw.get("dim1"), kw.get("dim2")] if isinstance(x, int)]
        if len(d) < 2:
            return None
        i, j = d[0] % rank, d[1] % rank
        p = list(range(rank))
        p[i], p[j] = p[j], p[i]
        return tuple(p)
    dims = kw.get("dims")
    if dims is None:
        dims = next((x for x in a[1:] if isinstance(x, (list, tuple))), None)
    if dims is None:
        return None
    return tuple(int(x) % rank for x in dims)


def _twrap(name, fn):
    def w(*a, **kw):
        t = a[0] if a else None
        info = TB._tinfo(t)
        r = fn(*a, **kw)
        if info is not None:
            rank = len(list(t.padded_shape))
            perm = _perm_of(name, rank, a, kw)
            TCENSUS.append({
                "op": name, "unit": _labels(), "owner": TB._owner(), "rank": rank,
                "perm": list(perm) if perm else None,
                "kind": classify(rank, perm) if perm else "UNREADABLE",
                "shape": info["shape"], "dtype": info["dtype"], "bytes": info["bytes"],
                "where_in": info["where"],
                "where_out": (TB._tinfo(r) or {}).get("where"),
            })
        return r
    w.__name__ = getattr(fn, "__name__", name)
    return w


def install_transpose():
    """Wrap every transpose/permute in the ttnn tree, on top of whatever already wraps it."""
    n, seen = 0, set()

    def walk(prefix, m, depth):
        nonlocal n
        if id(m) in seen or depth > 3:
            return
        seen.add(id(m))
        for attr in dir(m):
            if attr.startswith("_"):
                continue
            try:
                o = getattr(m, attr)
            except Exception:
                continue
            if type(o).__name__ == "module" and getattr(o, "__name__", "").startswith("ttnn"):
                walk(f"{prefix}.{attr}", o, depth + 1)
            elif attr in ("transpose", "permute") and callable(o):
                try:
                    setattr(m, attr, _twrap(f"{prefix}.{attr}", o))
                    n += 1
                except Exception:
                    pass

    walk("ttnn", ttnn, 0)
    return n


def main():
    os.makedirs(ARGS.out_dir, exist_ok=True)
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import fold_ab_multi as FAM
    from tt_bio.main import _resolve_recycling_steps

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    FAM.patch_boltz2_cfg()
    fix = Path(ROOT) / "perf" / "size512" / "fixtures"
    res = B.build_fold("boltz2", Path(ROOT) / f".msa_bytes_{ARGS.size}",
                       fix / f"cdk2x2_{ARGS.size}.yaml", fix / f"cdk2x2_{ARGS.size}.a3m")
    one_fold, meta = res[0], res[1]
    dev = T.get_device()

    for cls in UNIT_CLASSES:
        c = getattr(T, cls, None)
        if c is not None and "__call__" in c.__dict__:
            c.__call__ = _unit_wrap(cls, c.__dict__["__call__"])

    # trace_block's recorder first, so the transpose recorder wraps it and both fire.
    nall = TB.install()
    ntr = install_transpose()
    print(f"wrapped {nall} ttnn operations, {ntr} transpose/permute entry points", flush=True)

    # The full ledger is armed on one PairformerLayer call. `_unit_wrap` already replaced
    # __call__, so arm around the wrapped one -- the unit stack has to be pushed before the
    # recorder starts or every op in the block records an empty stack.
    inner = T.PairformerLayer.__call__
    armed = {"done": False}

    def call(self, *a, **kw):
        parent = STACK[-1][0].split("#")[0] if STACK else "-"
        i = PFL[parent]
        PFL[parent] = i + 1
        if armed["done"] or parent != ARGS.parent or i != ARGS.block_call:
            return inner(self, *a, **kw)
        TB.REC.clear()
        TB.ACTIVE[0] = True
        try:
            r = inner(self, *a, **kw)
        finally:
            TB.ACTIVE[0] = False
        ttnn.synchronize_device(dev)
        armed["done"] = True
        BLOCK.update(TB.dump(os.path.join(ARGS.out_dir, f"block_{ARGS.size}.json.gz"),
                             TB._arch(), ARGS.size, TB.REC,
                             {"block_call": i, "block_parent": parent}))
        TB.REC.clear()
        return r

    T.PairformerLayer.__call__ = call
    # NOT a timing: every ttnn call in this process carries a wrapper and one call carries a
    # stack walk per operand. Printed only so a run that folded nothing is obvious.
    fold_s, m = one_fold()
    print(f"folded (instrumented, not a timing) {fold_s:.1f} s, plddt {m.get('plddt')}", flush=True)
    if not armed["done"]:
        print(f"WARNING: {ARGS.parent} PairformerLayer call {ARGS.block_call} never ran "
              f"(calls seen: {dict(PFL)})", flush=True)

    tp = os.path.join(ARGS.out_dir, f"transposes_{ARGS.size}.json.gz")
    with gzip.open(tp, "wt") as fh:
        json.dump({"arch": TB._arch(), "size": ARGS.size, "calls": TCENSUS}, fh)
    kinds = Counter(x["kind"] for x in TCENSUS)
    print(f"{len(TCENSUS)} transpose/permute calls -> {tp}", flush=True)
    print("  by kind: " + json.dumps(dict(kinds)), flush=True)
    json.dump({"size": ARGS.size, "arch": TB._arch(), "block_call": ARGS.block_call,
               "block": BLOCK, "n_transpose_calls": len(TCENSUS), "kinds": dict(kinds),
               "unit_calls": dict(TOP), "pairformer_layer_calls": dict(PFL),
               "grid": meta.get("grid"),
               "recycling_steps": meta.get("recycling_steps"), "plddt": m.get("plddt")},
              open(os.path.join(ARGS.out_dir, "manifest.json"), "w"), indent=1)
    T.cleanup()


main()
