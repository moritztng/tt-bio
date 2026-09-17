#!/usr/bin/env python3
"""In-fold per-op device time for the Boltz-2 512 aa fold, one profilable unit at a time.

Why not just profile the fold. The fold dispatches ~465 k ttnn calls. tt-metal's device profiler
holds 48 bytes of DRAM per dispatched program per RISC per core and writes ~350 kB of host CSV per
program, so a whole-fold capture is ~10^5 GB of CSV and several GB of on-chip DRAM markers. It is
not a tuning problem, it is out of reach (measured ladder: knowledgebase skill `ttnn-perf-profiling`
section 7). What IS in reach is one instance of each repeating unit, profiled with the fold's own
operands, weighted by the fold's own call census.

That weighting is the whole point of this row. The `c10-fold-census` replay priced each launch key
standalone with `ttnn.DRAM_MEMORY_CONFIG` on every operand AND every output, while the fold runs
many of them L1-resident; and it published the `@grid110` arm for `linear`/`matmul` while the fold
passes no `core_grid`. Two biases, opposite signs. Grabbing a whole unit out of a live fold and
replaying *that* removes both: the unit's own code allocates its own intermediates, picks its own
memory configs and program configs, and runs its ops back to back the way the fold does.

Phases, one per process:

  counts  no profiler, no truncation. A full fold with the unit classes wrapped in a counting +
          unsynced-wall bracket, interleaved against plain folds (A,B,A,B,A). Gives the fold wall
          of record for this session, every unit's calls-per-fold, and the inclusive/exclusive host
          wall tree. The bracket is unsynced on purpose: a sync at a unit boundary changes the
          thing being measured, and the A/B against the plain arm prices what the bracket costs.

  probe   instrument proof. One 2048^3 bf16 matmul, solo-synced and back-to-back, so the profiler's
          reported kernel time can be checked against a wall this process measured itself.

  unit    grab one settled instance of `--unit <Class>` out of a real fold and run `--reps` of it
          back to back, fenced. Run bare for the wall, and again under `python -m tracy` for the
          per-op device time. `--ops` additionally records the python-level ttnn call sequence with
          operand shapes, which is what lets the ops report's device op codes be split back into
          the census's classes (`linear` and `matmul` are both MatmulDeviceOperation; `multiply_`,
          `add_`, `multiply` and `add` are all BinaryNgDeviceOperation).

The precursor problem is the b2z-kernel-cycle-census solution, reused: a settled `Diffusion`-side
call needs the trunk to have run, so for those units recycling drops to 1 and the pairformer/MSA
stacks are truncated, and the fold is unwound with a sentinel the moment the wanted call is in
hand. Shapes and memory configs are untouched; only the values flowing in differ, and a bf16
matmul's cycle count does not depend on its values.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

OUT: dict = {}
OUT_PATH: Path | None = None

FENCE_N = 3
FENCE_DIM = 32

# Every class on the fold's device path, innermost last. `counts` brackets all of them and reports
# an inclusive/exclusive tree, so which ones form a partition is a decision for the reducer and not
# a guess baked in here.
CLASSES = (
    "PairformerModule", "MSAModule", "DiffusionModule",
    "PairConditioningDevice", "PairAssemblyDevice", "ConfidenceHeadsDevice", "RelPosGather",
    "Pairformer", "MSA", "Diffusion",
    "PairformerLayer", "MSALayer", "DiffusionTransformer", "DiffusionTransformerLayer",
    "TriangleMultiplication", "TriangleAttention", "AttentionPairBias", "Transition",
    "OuterProductMean", "PairWeightedAveraging", "AdaLN", "ConditionedTransitionBlock",
)

# ttnn ops patched by `--ops`. Curated rather than reflective: patching every ttnn attribute costs
# host time in the very region being profiled. The reducer reports how many ops-report rows stayed
# unmatched, which is the check that this list is complete enough for the unit at hand.
TTNN_OPS = (
    "linear", "matmul", "layer_norm", "rms_norm", "softmax", "silu", "sigmoid", "gelu", "relu",
    "add", "add_", "subtract", "subtract_", "multiply", "multiply_", "div", "div_", "mul", "exp",
    "tanh", "sqrt", "rsqrt", "reciprocal", "neg", "clamp", "where", "generic_op",
    "transpose", "permute", "concat", "slice", "reshape", "unsqueeze", "squeeze", "repeat",
    "typecast", "clone", "copy", "to_layout", "pad", "sum", "mean", "max", "min", "embedding",
    "reallocate", "deallocate", "allocate_tensor_on_device", "chunk", "split", "tilize",
    "untilize", "interleaved_to_sharded", "sharded_to_interleaved", "reduce_scatter", "empty",
    "zeros", "ones", "full", "arange", "argmax", "sort", "topk", "nlp_create_qkv_heads",
)


class Grabbed(Exception):
    """Unwind out of the precursor fold as soon as the wanted call has been captured."""


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1, default=str))


def loadavg():
    return open("/proc/loadavg").read().split()[:3]


def shp(x):
    s = getattr(x, "shape", None)
    return list(s) if s is not None else None


class Tree:
    """Calls and unsynced inclusive/exclusive wall per class path, nested by the live call stack.

    Unsynced: a ttnn call is an async enqueue, so these seconds are host-side inclusive wall, which
    on this stack is the fold's own critical path (the calling thread spins for dispatch-queue
    room). They are used to WEIGHT device time by calls-per-fold and to bound coverage, never
    quoted as device time. The device seconds come from the profiler, in the `unit` phase.
    """

    def __init__(self):
        self.incl: dict[str, float] = defaultdict(float)
        self.excl: dict[str, float] = defaultdict(float)
        self.calls: Counter = Counter()
        self.ms: dict[str, list] = defaultdict(list)
        self.child: dict[str, float] = defaultdict(float)
        self.notes: list[str] = []

    def wrap(self, cls, name):
        orig = cls.__dict__.get("__call__") or cls.__dict__.get("forward")
        attr = "__call__" if "__call__" in cls.__dict__ else "forward"
        if orig is None:
            self.notes.append("no __call__/forward on %s" % name)
            return None

        def w(self_obj, *args, **kw):
            p = "/".join(T_STACK + [name])
            T_STACK.append(name)
            kids0 = self.child[p]
            t0 = time.perf_counter()
            try:
                return orig(self_obj, *args, **kw)
            finally:
                dt = time.perf_counter() - t0
                T_STACK.pop()
                self.calls[p] += 1
                self.incl[p] += dt
                self.excl[p] += dt - (self.child[p] - kids0)
                self.ms[p].append(dt)
                if T_STACK:
                    self.child["/".join(T_STACK)] += dt
        setattr(cls, attr, w)
        return (cls, attr, orig)

    def report(self):
        rows = []
        for p, n in self.calls.most_common():
            v = sorted(self.ms[p])
            rows.append({"path": p, "cls": p.split("/")[-1], "calls": n,
                         "incl_s": round(self.incl[p], 5),
                         "excl_s": round(self.excl[p], 5),
                         "median_ms": round(1e3 * v[len(v) // 2], 5),
                         "min_ms": round(1e3 * v[0], 5), "max_ms": round(1e3 * v[-1], 5)})
        rows.sort(key=lambda r: -r["incl_s"])
        return {"rows": rows, "notes": self.notes}


T_STACK: list[str] = []


def make_fence(ttnn, dev):
    import torch
    t = ttnn.from_torch(torch.ones(1, 1, FENCE_DIM, FENCE_DIM), layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16, device=dev)

    def fence():
        for _ in range(FENCE_N):
            ttnn.exp(t)
        ttnn.synchronize_device(dev)
    return fence


def timed_reps(ttnn, dev, fn, args, kwargs, reps, fence):
    for _ in range(3):
        fn(*args, **kwargs)
    ttnn.synchronize_device(dev)
    fence()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*args, **kwargs)
    ttnn.synchronize_device(dev)
    wall = (time.perf_counter() - t0) / reps
    fence()
    return round(1e3 * wall, 4)


def phase_probe(ttnn, dev, reps=20):
    import torch
    n = 2048
    a = ttnn.from_torch(torch.randn(n, n), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                        device=dev)
    b = ttnn.from_torch(torch.randn(n, n), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                        device=dev)
    fence = make_fence(ttnn, dev)
    for _ in range(5):
        ttnn.matmul(a, b)
    ttnn.synchronize_device(dev)
    solo = []
    for _ in range(reps):
        t0 = time.perf_counter()
        ttnn.matmul(a, b)
        ttnn.synchronize_device(dev)
        solo.append(time.perf_counter() - t0)
    ttnn.synchronize_device(dev)
    fence()
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.matmul(a, b)
    ttnn.synchronize_device(dev)
    btb = (time.perf_counter() - t0) / reps
    fence()
    return {"n": n, "reps": reps, "solo_synced_ms": round(1e3 * st.median(solo), 4),
            "back_to_back_ms": round(1e3 * btb, 4), "flops": 2 * n ** 3,
            "bytes": 3 * n * n * 2}


def all_shapes(args, kw):
    """Every tensor shape reachable from a call's arguments, flattened.

    Recording only `[shp(x) for x in args]` is what held `align_quality` at 0.32: `generic_op`
    passes its tensors in a LIST (`ttnn.generic_op([in0, in1, out], pd)`), so the only positional
    argument has no `.shape` at all, and most tt-bio call sites pass operands as keyword arguments.
    Both were invisible. This walks one level of list/tuple nesting and both arg kinds.
    """
    out = []
    for x in list(args) + list(kw.values()):
        s = shp(x)
        if s is not None:
            out.append(s)
        elif isinstance(x, (list, tuple)):
            out.extend(s2 for s2 in (shp(y) for y in x) if s2 is not None)
    return out


def install_ops(ttnn, rec):
    """Record the python-level ttnn call sequence, in order, with every operand shape."""
    undo = []
    for nm in TTNN_OPS:
        f = getattr(ttnn, nm, None)
        if f is None or not callable(f):
            continue

        def mk(nm, f):
            def w(*args, **kw):
                rec.append((nm, all_shapes(args, kw), []))
                return f(*args, **kw)
            return w
        setattr(ttnn, nm, mk(nm, f))
        undo.append((nm, f))

    def restore():
        for nm, f in undo:
            setattr(ttnn, nm, f)
    return restore, len(undo)



def parse_units(spec: str):
    """`Name[:want_attr][:reps]` items, comma separated.

    `PairformerLayer:transform_s:3,MSALayer::5,PairConditioningDevice`
    An empty middle field means no attribute filter. Missing reps falls back to --reps.
    """
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        name = parts[0].strip()
        attr = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
        reps = int(parts[2]) if len(parts) > 2 and parts[2].strip() else None
        out.append((name, attr, reps))
    return out


def truncate_stacks(T, keep):
    import gc
    saved = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, (T.Pairformer, T.MSA)) and isinstance(
                    getattr(obj, "blocks", None), list) and len(obj.blocks) > keep:
                saved.append((obj, obj.blocks))
                obj.blocks = obj.blocks[:keep]
        except ReferenceError:
            continue

    def restore():
        for obj, blocks in saved:
            obj.blocks = blocks
    return restore, len(saved)


def build(a, T):
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps
    B.RECYCLING_STEPS = 1 if a.short_precursor else _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()
    T.get_device(trace_region_size=1 << 30)
    fix = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, state = B.build_fold("boltz2", a.msa_dir,
                                         fix / ("cdk2x2_%d.yaml" % a.size),
                                         fix / ("cdk2x2_%d.a3m" % a.size))
    return one_fold, meta, state


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phase", required=True, choices=("counts", "probe", "unit"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--unit", default=None, help="single unit; --units supersedes it")
    ap.add_argument("--units", default=None,
                    help="comma separated Name[:want_attr][:reps] -- one precursor, several units")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip", type=int, default=2,
                    help="grab the Nth call of the unit, so the grabbed one is settled")
    ap.add_argument("--plain-n", type=int, default=2)
    ap.add_argument("--keep-blocks", type=int, default=2)
    ap.add_argument("--short-precursor", action="store_true")
    ap.add_argument("--ops", action="store_true")
    ap.add_argument("--msa-dir", type=Path,
                    default=ROOT / "perf" / "roof_msa_ladder" / ".msa_512")
    a = ap.parse_args()
    OUT_PATH = a.out
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T

    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "t_start": time.time(),
                  "phase": a.phase, "unit": a.unit, "reps": a.reps, "size": a.size,
                  "visible": os.environ.get("TT_VISIBLE_DEVICES"),
                  "lease": os.environ.get("TT_BIO_LEASE_CARDS"),
                  "commit": os.popen("git -C %s rev-parse --short HEAD" % ROOT).read().strip(),
                  "profiler": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
                  "ttnn": getattr(ttnn, "__file__", "?"),
                  "metal_home": os.environ.get("TT_METAL_HOME"),
                  "loadavg": loadavg()}
    dump()

    if a.phase == "probe":
        dev = T.get_device()
        OUT["probe"] = phase_probe(ttnn, dev)
        OUT["env"]["t_end"] = time.time()
        print("  probe " + json.dumps(OUT["probe"]), flush=True)
        dump()
        print("DONE", a.out, flush=True)
        return 0

    one_fold, meta, state = build(a, T)
    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "card_type", "load_s")
                       if k in meta})
    dev = T.get_device()
    dump()

    if a.phase == "counts":
        print("=== warm fold ===", flush=True)
        t0 = time.time()
        one_fold()
        OUT["warm_fold_s"] = round(time.time() - t0, 4)
        dump()
        legs = []
        # A,B,A,B,A: the bracket arm interleaves with the plain arm inside one session, so a
        # constant co-tenant cancels and the two plain-arm gaps are this session's own A/A floor.
        for i in range(2 * a.plain_n + 1):
            arm = "plain" if i % 2 == 0 else "bracket"
            tree = Tree() if arm == "bracket" else None
            undo = []
            if tree is not None:
                T_STACK.clear()
                for nm in CLASSES:
                    cls = getattr(T, nm, None)
                    if cls is None:
                        tree.notes.append("missing class %s" % nm)
                        continue
                    u = tree.wrap(cls, nm)
                    if u:
                        undo.append(u)
            t0 = time.time()
            try:
                one_fold()
            finally:
                for cls, attr, orig in undo:
                    setattr(cls, attr, orig)
            leg = {"arm": arm, "i": i, "t0": t0, "t1": time.time(),
                   "fold_s": round(time.time() - t0, 4), "loadavg": loadavg()}
            if tree is not None:
                leg["tree"] = tree.report()
            legs.append(leg)
            OUT["legs"] = legs
            dump()
            print("  leg %d %-7s %.4f s" % (i, arm, leg["fold_s"]), flush=True)
        plain = [l["fold_s"] for l in legs if l["arm"] == "plain"]
        brack = [l["fold_s"] for l in legs if l["arm"] == "bracket"]
        OUT["fold_s_plain_median"] = round(st.median(plain), 4)
        OUT["fold_s_bracket_median"] = round(st.median(brack), 4)
        OUT["aa_floor_s"] = round(max(plain) - min(plain), 4)
        OUT["bracket_cost_ratio"] = round(st.median(brack) / st.median(plain), 5)
        OUT["env"]["t_end"] = time.time()
        dump()
        print("DONE", a.out, flush=True)
        return 0

    # --- phase unit -------------------------------------------------------------------------
    # One precursor, several units. A model load plus a precursor fold costs minutes and the card is
    # shared, so collecting every unit that is reachable before the same abort point in ONE process
    # is the difference between one leg and seven. Each unit still gets its own fenced region, so the
    # reducer windows them independently.
    specs = [(n, at, r if r else a.reps)
             for n, at, r in parse_units(a.units or a.unit or "")]
    if not specs:
        OUT["error"] = "--units is required for --phase unit"
        dump()
        return 2
    OUT["specs"] = [{"unit": u, "want_attr": at, "reps": r} for u, at, r in specs]
    wrapped: list = []
    grabs: dict = {}
    counts: Counter = Counter()
    matched: Counter = Counter()

    def clone(x):
        return ttnn.clone(x) if isinstance(x, ttnn.Tensor) else x

    def install(name, attr_req):
        cls = getattr(T, name, None)
        if cls is None:
            OUT.setdefault("missing_classes", []).append(name)
            return
        key = "__call__" if "__call__" in cls.__dict__ else "forward"
        orig = cls.__dict__.get(key)
        if orig is None:
            OUT.setdefault("missing_classes", []).append("%s (no __call__/forward)" % name)
            return

        def w(self_obj, *args, **kw):
            counts[name] += 1
            out = orig(self_obj, *args, **kw)
            if name in grabs:
                return out
            if attr_req is not None and not getattr(self_obj, attr_req, False):
                return out
            matched[name] += 1
            if matched[name] >= a.skip:
                grabs[name] = {"obj": self_obj,
                               "args": tuple(clone(x) for x in args),
                               "kwargs": {k: clone(v) for k, v in kw.items()}}
                print("  grabbed %s on call %d (match %d)"
                      % (name, counts[name], matched[name]), flush=True)
                if len(grabs) == len([1 for n, _at, _r in specs
                                      if n not in OUT.get("missing_classes", [])]):
                    raise Grabbed
            return out
        setattr(cls, key, w)
        wrapped.append((cls, key, orig))

    for name, attr_req, _reps in specs:
        install(name, attr_req)

    restore = None
    if a.short_precursor:
        restore, n_trunc = truncate_stacks(T, a.keep_blocks)
        OUT["env"].update({"stacks_truncated": n_trunc, "keep_blocks": a.keep_blocks})
        print("  truncated %d block stacks to %d" % (n_trunc, a.keep_blocks), flush=True)

    print("=== precursor fold (unwound once every unit is in hand) ===", flush=True)
    t0 = time.perf_counter()
    try:
        one_fold()
    except Grabbed:
        pass
    finally:
        for cls, key, orig in wrapped:
            setattr(cls, key, orig)
        if restore:
            restore()
    OUT["precursor_s"] = round(time.perf_counter() - t0, 3)
    OUT["calls_in_precursor"] = dict(counts)
    OUT["signature_matches_in_precursor"] = dict(matched)
    OUT["grabbed"] = sorted(grabs)
    OUT["not_grabbed"] = [n for n, _at, _r in specs if n not in grabs]
    dump()
    if not grabs:
        OUT["error"] = "nothing was grabbed"
        dump()
        print("FAILED " + OUT["error"], flush=True)
        return 1

    fence = make_fence(ttnn, dev)
    rec: list = []
    undo_ops = None
    if a.ops:
        undo_ops, n_patched = install_ops(ttnn, rec)
        OUT["ops_patched"] = n_patched
    # order matters: the reducer pairs the k-th fenced window with the k-th entry of this list
    OUT["unit_order"] = [n for n, _at, _r in specs if n in grabs]
    OUT["units"] = {}
    try:
        for name, _at, reps in specs:
            if name not in grabs:
                continue
            g = grabs[name]
            print("=== profiled region %d/%d: %d x %s ==="
                  % (len(OUT["units"]) + 1, len(OUT["unit_order"]), reps, name), flush=True)
            t_a = time.time()
            ms = timed_reps(ttnn, dev, g["obj"], g["args"], g["kwargs"], reps, fence)
            OUT["units"][name] = {
                "reps": reps, "synced_wall_ms_per_call": ms,
                "t_region0": t_a, "t_region1": time.time(),
                "arg_shapes": [shp(x) or type(x).__name__ for x in g["args"]],
                "calls_in_precursor": counts[name],
            }
            print("  %s synced wall %.4f ms/call (profiler=%s)"
                  % (name, ms, OUT["env"]["profiler"]), flush=True)
            dump()
    finally:
        if undo_ops:
            undo_ops()
    if a.ops:
        OUT["op_seq_len"] = len(rec)
        (a.out.parent / (a.out.stem + ".opseq.json")).write_text(json.dumps(rec))
        OUT["op_seq_path"] = str(a.out.parent / (a.out.stem + ".opseq.json"))
        OUT["op_seq_counts"] = dict(Counter(r[0] for r in rec).most_common())
    OUT["fence"] = {"op": "ttnn.exp", "n": FENCE_N, "dim": FENCE_DIM}
    OUT["env"]["t_end"] = time.time()
    dump()
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
