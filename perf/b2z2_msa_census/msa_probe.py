#!/usr/bin/env python3
"""One settled `MSALayer.__call__`, grabbed out of a real 512 aa fold and replayed.

The MSA track is the only block of the fold nobody has taken apart per program. Its cost is
`139.47 ms + 0.0995 ms/padded row` (CONTEXT §2-CORRECTION, CONTESTED), and the row-independent
139.47 ms is the same SHAPE as the diffusion step's per-program constant. Whether it is the same
MECHANISM is what this harness exists to answer, with the same instruments the step was answered
with: `b2z2_sampler_stall/stall_split.py` for the CB split and the three-model fit, and
`b2z2_step_fusion/site_cost.py` to put a microsecond on each line of `tenstorrent.py`.

Modes:
  ops    graph-capture one call and print the ordered top-level ttnn op list, so a lever can be
         picked off the real sequence instead of an op-code histogram that has lost the order.
  time   replay the call `--reps` times with the profiler OFF, report the synced wall. A span may
         only be taken here: the profiler inflates gaps (3.81x on the 1066-program diffusion step)
         and leaves kernels alone.
  prof   the same replay, laid out for a profiler-armed capture: warm, sync, fence, exactly
         `--reps` calls, sync, fence. `stall_split.py` windows on the last two fence runs and
         divides by `--reps`, so this mode must not run any other device work in between.
  ab     paired interleaved arms in ONE process, each arm an env setting, base re-timed inside
         every block so the ratio never spans a drift and the A/A floor is measured, not assumed.
         `_L1_OUT_RUNG` is cleared between arms: it is a module-level dict that only grows
         (`tenstorrent.py:3690`) and would otherwise carry one arm's demotions into the next.
  fold   a real fold at the full protocol (200 sampling steps, 3 recycles, full MSA depth), with
         every `MSALayer.__call__` and every `MSA.__call__` bracketed by a device sync. This is
         what settles the CONTESTED fit: the padded row count the fixture actually carries and the
         track's measured share of the fold.

The grab follows `perf/b2z2_step_fusion/step_probe.py`: patch the class, take the arguments of a
settled call, abort the precursor fold with a sentinel. The MSA module runs first in the trunk, so
the precursor is short. Replay is safe without re-cloning because at 512 tokens on Wormhole
`S <= SEQ_LEN_MORE_CHUNKING` and the whole path is in-place: `add_` returns its own operand and
`PairformerLayer` only ever adds into `z`. The returned pair is fed back in so the replay is a
steady state rather than 10 calls on the same values; a bf16 program's cycle count does not depend
on its values.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))

OUT: dict = {}
OUT_PATH: Path | None = None

FENCE_N, FENCE_DIM = 3, 32
GRAB_CALL = 2               # second MSALayer of the first trunk pass: block 0 warmed the kernels
SUBUNITS = ("pair_weighted_averaging", "msa_transition", "outer_product_mean", "pairformer_layer")


class Grabbed(Exception):
    """Unwind out of the precursor fold as soon as the call has been captured."""


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def make_fence(ttnn, dev):
    import torch
    t = ttnn.from_torch(torch.ones(1, 1, FENCE_DIM, FENCE_DIM), layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16, device=dev)

    def fence():
        for _ in range(FENCE_N):
            ttnn.exp(t)
        ttnn.synchronize_device(dev)
    return fence


def set_arm(T, over):
    """Apply one arm's overrides and return the undo. `@name` sets a module attribute."""
    keep = {}
    for k, v in over.items():
        if k.startswith("@"):
            n = k[1:]
            keep[k] = getattr(T, n)
            setattr(T, n, v)
        else:
            keep[k] = os.environ.get(k)
            os.environ[k] = v
    T._L1_OUT_RUNG.clear()
    return keep


def clear_arm(T, keep):
    for k, v in keep.items():
        if k.startswith("@"):
            setattr(T, k[1:], v)
        elif v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class _Marked:
    """Wrap a sub-unit so the program stream carries a labelled boundary.

    One `ttnn.exp` on a 32x32 tile in front of each of the layer's four sub-units. It is a
    single program, so `stall_split.py`'s fence detector (a run of >=3) does not confuse it
    with a window fence, and it costs a few microseconds against a 238 ms call. The marker
    goes into BOTH the graph capture and the armed capture, which is the point: it is the only
    thing that makes the aligned per-program table segmentable by sub-unit without a device
    sync per sub-unit, and a sync per sub-unit is what the earlier census had to pay.
    """

    def __init__(self, inner, ttnn, tile):
        self._inner, self._ttnn, self._tile = inner, ttnn, tile

    def __call__(self, *a, **k):
        self._ttnn.exp(self._tile)
        return self._inner(*a, **k)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def mark_subunits(ttnn, dev, g):
    """Replace the grabbed layer's four sub-units with marked proxies. Returns the undo."""
    import torch
    tile = ttnn.from_torch(torch.ones(1, 1, FENCE_DIM, FENCE_DIM), layout=ttnn.TILE_LAYOUT,
                           dtype=ttnn.bfloat16, device=dev)
    obj = g["obj"]
    saved = {n: getattr(obj, n) for n in SUBUNITS}
    for n, inner in saved.items():
        setattr(obj, n, _Marked(inner, ttnn, tile))

    def undo():
        for n, inner in saved.items():
            setattr(obj, n, inner)
    return undo


def patch_cfg():
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()


def build(ttnn, T, B, size, recycles):
    B.RECYCLING_STEPS = recycles
    B.SAMPLING_STEPS = 200
    patch_cfg()
    T.get_device(trace_region_size=512 << 20)
    fix = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, _state = B.build_fold(
        "boltz2", HERE / f".msa_{size}", fix / f"cdk2x2_{size}.yaml", fix / f"cdk2x2_{size}.a3m")
    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "card_type") if k in meta})
    dump()
    return one_fold, T.get_device()


def grab(ttnn, T, B, size):
    """Run the precursor fold and return the settled `MSALayer.__call__` and its operands."""
    one_fold, dev = build(ttnn, T, B, size, recycles=1)
    grabs, counts = {}, {"n": 0}
    cls = T.MSALayer
    orig = cls.__dict__["__call__"]

    def clone(x):
        return ttnn.clone(x) if isinstance(x, ttnn.Tensor) else x

    def wrapper(self_obj, *args, **kw):
        counts["n"] += 1
        if counts["n"] >= GRAB_CALL and not grabs:
            grabs["g"] = {"obj": self_obj,
                          "args": tuple(clone(x) for x in args),
                          "kwargs": {k: clone(v) for k, v in kw.items()}}
            raise Grabbed
        return orig(self_obj, *args, **kw)
    cls.__call__ = wrapper

    t0 = time.perf_counter()
    try:
        one_fold()
    except Grabbed:
        pass
    finally:
        cls.__call__ = orig
    OUT["precursor_s"] = round(time.perf_counter() - t0, 3)
    OUT["msalayer_calls_in_precursor"] = counts["n"]
    dump()
    if not grabs:
        raise SystemExit("MSALayer was never grabbed")
    g = grabs["g"]
    OUT["arg_shapes"] = [list(x.shape) if hasattr(x, "shape") else repr(x) for x in g["args"]]
    m = g["args"][1]
    OUT["padded_rows"] = int(m.shape[1])
    OUT["tokens"] = int(m.shape[2])
    dump()
    return dev, g


def replay(g, reps):
    """`reps` back-to-back calls, feeding the returned pair forward. Returns nothing."""
    obj, args, kw = g["obj"], list(g["args"]), g["kwargs"]
    for _ in range(reps):
        z, m = obj(*args, **kw)
        args[0], args[1] = z, m
    g["args"] = tuple(args)


def mode_time(ttnn, dev, g, reps, blocks):
    fence = make_fence(ttnn, dev)
    for _ in range(3):
        replay(g, 1)
    ttnn.synchronize_device(dev)
    fence()
    walls = []
    for _ in range(blocks):
        t0 = time.perf_counter()
        replay(g, reps)
        ttnn.synchronize_device(dev)
        walls.append((time.perf_counter() - t0) / reps)
    fence()
    OUT["layer"] = {"ms_per_call": round(1e3 * st.median(walls), 4),
                    "ms_all": [round(1e3 * w, 4) for w in walls],
                    "reps": reps, "blocks": blocks}
    print(f"  MSALayer {OUT['layer']['ms_per_call']:.4f} ms/call  {OUT['layer']['ms_all']}",
          flush=True)


def mode_prof(ttnn, dev, g, reps):
    """One fenced window of exactly `reps` calls, for `stall_split.py` to read."""
    fence = make_fence(ttnn, dev)
    for _ in range(3):
        replay(g, 1)
    ttnn.synchronize_device(dev)
    fence()
    t0 = time.perf_counter()
    replay(g, reps)
    ttnn.synchronize_device(dev)
    armed = (time.perf_counter() - t0) / reps
    fence()
    OUT["env"]["reps"] = reps
    OUT["armed_ms_per_call"] = round(1e3 * armed, 4)
    print(f"  armed window: {reps} calls, {1e3*armed:.4f} ms/call (span, NOT a wall)", flush=True)


def mode_ops(ttnn, dev, g):
    from itemize import top_level_spans
    replay(g, 1)
    ttnn.synchronize_device(dev)
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    replay(g, 1)
    ttnn.synchronize_device(dev)
    nodes = ttnn.graph.end_graph_capture()
    ops, _owner = top_level_spans(nodes)
    names = [o["name"] for o in ops]
    by = defaultdict(int)
    for n in names:
        by[n] += 1
    import gzip
    raw = OUT_PATH.with_suffix(".graph.json.gz")
    with gzip.open(raw, "wt") as fh:
        json.dump(nodes, fh)
    OUT["graph"] = raw.name
    OUT["ops"] = {"n_top_level": len(names),
                  "by_name": dict(sorted(by.items(), key=lambda kv: -kv[1])),
                  "sequence": names}
    print(f"  {len(names)} top-level ttnn ops", flush=True)
    for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"    {v:5d}  {k}", flush=True)


# name -> overrides. A plain key is an environment variable; a key starting with `@` is a
# module-level attribute of `tt_bio.tenstorrent`, because a flag read once at import
# (`env_flag` at module scope) does NOT see an environment variable set after the import --
# which is how an arm can silently measure the base against itself.
ARMS = {
    "base": {},
    "h24": {"TT_BIO_TRANSITION_H_CHUNK": "24"},
    "h32": {"TT_BIO_TRANSITION_H_CHUNK": "32"},
    "h64": {"TT_BIO_TRANSITION_H_CHUNK": "64"},
    "batchw": {"@_PWA_BATCH_HEAD_WEIGHTS": True},
    "perhead": {"@_PWA_BATCH_HEAD_WEIGHTS": False},   # the pre-lever base, now that it ships on
    # The PWA residency sweep. `l1rN` blocks the head loop at N rows AND leaves the normed block
    # L1-resident; `blkN` blocks at the same N with the norm in DRAM, so the pair isolates the
    # residency from the blocking overhead it is paid for with.
    "l1r512": {"@_PWA_L1_ROWS": 512},
    "l1r256": {"@_PWA_L1_ROWS": 256},
    "l1r128": {"@_PWA_L1_ROWS": 128},
    "l1r64": {"@_PWA_L1_ROWS": 64},
    "blk512": {"@_PWA_L1_ROWS": 512, "@_PWA_L1_NORM_M": False},
    "blk256": {"@_PWA_L1_ROWS": 256, "@_PWA_L1_NORM_M": False},
    "blk128": {"@_PWA_L1_ROWS": 128, "@_PWA_L1_NORM_M": False},
    "l1r384": {"@_PWA_L1_ROWS": 384},
    "l1r640": {"@_PWA_L1_ROWS": 640},
    "blk384": {"@_PWA_L1_ROWS": 384, "@_PWA_L1_NORM_M": False},
    "blk640": {"@_PWA_L1_ROWS": 640, "@_PWA_L1_NORM_M": False},
    # The shipped default, and the pre-lever path it has to beat now that it ships on.
    "derived": {"@_PWA_L1_ROWS": 0},
    "l1off": {"@_PWA_L1_ROWS": -1},
}


def mode_ab(ttnn, T, dev, g, arms, reps, blocks):
    """Interleave the arms inside every block, so an arm's ratio never spans a drift."""
    fence = make_fence(ttnn, dev)

    def arm(name):
        return set_arm(T, ARMS[name])

    def unarm(keep):
        clear_arm(T, keep)

    walls = {a: [] for a in arms}
    for a in arms:                              # warm every arm's programs before timing any
        keep = arm(a)
        replay(g, 2)
        unarm(keep)
    ttnn.synchronize_device(dev)
    fence()
    for _ in range(blocks):
        for a in arms:
            keep = arm(a)
            t0 = time.perf_counter()
            replay(g, reps)
            ttnn.synchronize_device(dev)
            walls[a].append((time.perf_counter() - t0) / reps)
            unarm(keep)
    fence()
    med = {a: 1e3 * st.median(v) for a, v in walls.items()}
    base = med[arms[0]]
    OUT["ab"] = {"reps": reps, "blocks": blocks,
                 "ms": {a: round(v, 4) for a, v in med.items()},
                 "all_ms": {a: [round(1e3 * x, 4) for x in v] for a, v in walls.items()},
                 "ratio_vs_base": {a: round(base / v, 5) for a, v in med.items()}}
    for a in arms:
        print(f"  {a:8s} {med[a]:9.4f} ms/call  {base/med[a]:.5f}x  "
              f"{[round(1e3*x,2) for x in walls[a]]}", flush=True)


def mode_parity(ttnn, T, dev, g, arm):
    """Run the grabbed call twice from IDENTICAL inputs, base vs `arm`, and compare exactly."""
    import torch
    obj, args, kw = g["obj"], list(g["args"]), g["kwargs"]

    def once(over):
        keep = set_arm(T, over)
        a = [ttnn.clone(x) if isinstance(x, ttnn.Tensor) else x for x in args]
        z, m = obj(*a, **kw)
        out = (ttnn.to_torch(z).clone(), ttnn.to_torch(m).clone())
        clear_arm(T, keep)
        return out

    if not ARMS[arm]:
        raise SystemExit(f"arm {arm!r} sets nothing; that would compare the base with itself")
    for k in ARMS[arm]:
        if k.startswith("@") and not hasattr(T, k[1:]):
            raise SystemExit(f"tt_bio.tenstorrent has no attribute {k[1:]}")
    base_z, base_m = once({})
    aa_z, aa_m = once({})
    arm_z, arm_m = once(ARMS[arm])
    eq = {"A/A z": torch.equal(base_z, aa_z), "A/A m": torch.equal(base_m, aa_m),
          "arm z": torch.equal(base_z, arm_z), "arm m": torch.equal(base_m, arm_m)}
    # negative control: a check that cannot pass must not pass
    eq["negative control (must be False)"] = torch.equal(base_z, base_z + 1)
    d_z = float((base_z.float() - arm_z.float()).abs().max())
    d_m = float((base_m.float() - arm_m.float()).abs().max())
    OUT["parity"] = {"arm": arm, "equal": eq, "max_abs_diff_z": d_z, "max_abs_diff_m": d_m,
                     "sha_base_z": None}
    for k, v in eq.items():
        print(f"  {k:34s} {v}", flush=True)
    print(f"  max |delta| z {d_z:g}   m {d_m:g}", flush=True)


def mode_fold(ttnn, T, B, size, folds, recycles):
    """A real fold at the full protocol, with the MSA track bracketed by device syncs."""
    dev_box = {}
    layer_ms, module_ms, depths = [], [], []
    cls, mod = T.MSALayer, T.MSA
    o_layer, o_mod = cls.__dict__["__call__"], mod.__dict__["__call__"]

    def sync():
        d = dev_box.get("d")
        if d is not None:
            ttnn.synchronize_device(d)

    def w_layer(self_obj, *a, **k):
        sync()
        t0 = time.perf_counter()
        out = o_layer(self_obj, *a, **k)
        sync()
        layer_ms.append(1e3 * (time.perf_counter() - t0))
        return out

    def w_mod(self_obj, *a, **k):
        depths.append(int(a[1].shape[1]))
        sync()
        t0 = time.perf_counter()
        out = o_mod(self_obj, *a, **k)
        sync()
        module_ms.append(1e3 * (time.perf_counter() - t0))
        return out

    one_fold, dev = build(ttnn, T, B, size, recycles)
    dev_box["d"] = dev
    t0 = time.perf_counter()
    one_fold()                                   # cold fold, discarded
    OUT["cold_fold_s"] = round(time.perf_counter() - t0, 4)
    dump()

    cls.__call__, mod.__call__ = w_layer, w_mod
    walls = []
    try:
        for i in range(folds):
            layer_ms.clear(); module_ms.clear(); depths.clear()
            t0 = time.perf_counter()
            one_fold()
            wall = time.perf_counter() - t0
            walls.append(wall)
            OUT.setdefault("folds", []).append({
                "fold_s": round(wall, 4),
                "msalayer_calls": len(layer_ms),
                "msalayer_ms": [round(x, 3) for x in layer_ms],
                "msalayer_total_ms": round(sum(layer_ms), 3),
                "msa_module_calls": len(module_ms),
                "msa_module_total_ms": round(sum(module_ms), 3),
                "padded_rows": sorted(set(depths)),
                "track_pct_of_fold": round(100 * sum(module_ms) / 1e3 / wall, 3)})
            f = OUT["folds"][-1]
            print(f"  fold {i}: {wall:.4f} s, MSA track {f['msa_module_total_ms']/1e3:.4f} s "
                  f"({f['track_pct_of_fold']:.2f} %), {f['msalayer_calls']} MSALayer calls, "
                  f"depth {f['padded_rows']}", flush=True)
            dump()
    finally:
        cls.__call__, mod.__call__ = o_layer, o_mod
    OUT["fold_median_s"] = round(st.median(walls), 4)
    OUT["track_median_s"] = round(st.median([f["msa_module_total_ms"] for f in OUT["folds"]]) / 1e3, 4)
    OUT["track_pct_median"] = round(st.median([f["track_pct_of_fold"] for f in OUT["folds"]]), 3)
    OUT["layer_ms_median"] = round(st.median(
        [x for f in OUT["folds"] for x in f["msalayer_ms"]]), 4)
    print(f"  MEDIAN fold {OUT['fold_median_s']:.4f} s, track {OUT['track_median_s']:.4f} s "
          f"= {OUT['track_pct_median']:.2f} %, MSALayer {OUT['layer_ms_median']:.4f} ms",
          flush=True)


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", required=True, choices=("ops", "time", "prof", "fold", "ab", "parity"))
    ap.add_argument("--arms", default="base,base,h24,h32",
                    help="comma-separated arm names; the FIRST is the ratio denominator and "
                         "repeating it gives the A/A floor")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--mark", action="store_true",
                    help="one 32x32 ttnn.exp in front of each sub-unit, so the aligned "
                         "per-program table can be cut by sub-unit")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    OUT_PATH = a.out
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B

    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "card": os.environ.get("TT_VISIBLE_DEVICES"),
                  "mode": a.mode, "size": a.size, "label": a.label,
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "profiler": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
                  "ttnn": getattr(ttnn, "__file__", "?"),
                  "flags": {k: v for k, v in sorted(os.environ.items())
                            if k.startswith("TT_BIO_") or k.startswith("B2_")},
                  "loadavg": open("/proc/loadavg").read().split()[:3]}
    dump()

    if a.mode == "fold":
        mode_fold(ttnn, T, B, a.size, a.folds, a.recycles)
    else:
        dev, g = grab(ttnn, T, B, a.size)
        if a.mark:
            mark_subunits(ttnn, dev, g)
            OUT["marked"] = list(SUBUNITS)
        if a.mode == "ops":
            mode_ops(ttnn, dev, g)
        elif a.mode == "time":
            mode_time(ttnn, dev, g, a.reps, a.blocks)
        elif a.mode == "ab":
            names = a.arms.split(",")
            seen, arms = set(), []
            for i, n in enumerate(names):            # a repeated arm gets its own slot
                nm = n if n not in seen else f"{n}#{i}"
                seen.add(n)
                arms.append(nm)
                ARMS.setdefault(nm, ARMS[n])
            mode_ab(ttnn, T, dev, g, arms, a.reps, a.blocks)
        elif a.mode == "parity":
            mode_parity(ttnn, T, dev, g, a.arms.split(",")[-1])
        else:
            mode_prof(ttnn, dev, g, a.reps)
    dump()
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
