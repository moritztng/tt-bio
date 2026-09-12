#!/usr/bin/env python3
"""The MSA ROW axis cost curve: `t = a + b * n_rows` on a real, settled `MSALayer.__call__`.

One number decides this row. Both shard verdicts in this wave were decided by the constant
fraction of a traced cost curve -- the token DiT's 56.3 % killed it, the atom track's 9.893 %
made it -- so this harness fits the same curve on the MSA row axis and publishes `a` as a
fraction at the shipped 1024-row bucket.

The call is grabbed out of a real 512 aa fold the way `perf/b2z2_msa_census/msa_probe.py` grabs
it: patch the class, clone the arguments of a settled call, abort the precursor with a sentinel.
The MSA row axis is then swept by slicing `m` (and its mask) on dim 1 and replaying.

Two curves come out of one device session:

  * the LAYER curve, unsynced inside the call, which is the honest wall; and
  * the SUB-UNIT curve, one device sync around each of the four sub-units, which says WHERE the
    constant lives. `pairformer_layer` carries no MSA row index at all, so the prediction is that
    it is flat in the row count and is most of the constant. A curve that only reports `a` cannot
    tell a replicated sub-unit from a per-call fixed cost, and the two imply different levers.

Sliced operands are CLONED, never handed in as views: `ttnn.slice` copies, the layer path is
in-place (`add_` returns its own operand), and a view would mutate the grabbed source.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

OUT: dict = {"env": {}}
OUT_PATH: Path | None = None
FENCE_N, FENCE_DIM = 3, 32
GRAB_CALL = 2
SUBUNITS = ("pair_weighted_averaging", "msa_transition", "outer_product_mean", "pairformer_layer")


class Grabbed(Exception):
    pass


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def patch_cfg():
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()


def grab(ttnn, T, B, size):
    B.RECYCLING_STEPS = 1
    B.SAMPLING_STEPS = 200
    patch_cfg()
    T.get_device(trace_region_size=512 << 20)
    fix = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, _state = B.build_fold(
        "boltz2", HERE / f".msa_{size}", fix / f"cdk2x2_{size}.yaml", fix / f"cdk2x2_{size}.a3m")
    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "card_type") if k in meta})
    dump()
    dev = T.get_device()

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
    try:
        one_fold()
    except Grabbed:
        pass
    finally:
        cls.__call__ = orig
    if not grabs:
        raise SystemExit("MSALayer was never grabbed")
    g = grabs["g"]
    OUT["arg_shapes"] = [list(x.shape) if hasattr(x, "shape") else repr(x) for x in g["args"]]
    OUT["kwarg_shapes"] = {k: (list(v.shape) if hasattr(v, "shape") else repr(v))
                           for k, v in g["kwargs"].items()}
    dump()
    return dev, g


def make_fence(ttnn, dev):
    import torch
    t = ttnn.from_torch(torch.ones(1, 1, FENCE_DIM, FENCE_DIM), layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16, device=dev)

    def fence():
        for _ in range(FENCE_N):
            ttnn.exp(t)
        ttnn.synchronize_device(dev)
    return fence


def arg_at(g, i, name):
    """Positional or keyword, whichever the grab caught."""
    if i < len(g["args"]):
        return g["args"][i], ("pos", i)
    return g["kwargs"].get(name), ("kw", name)


def row_slice(ttnn, t, d, axis):
    """Rows [0, d) of `t` on `axis`, as a fresh tensor. None passes through."""
    if t is None:
        return None
    if int(t.shape[axis]) == d:
        return ttnn.clone(t)
    sl = [slice(None)] * len(t.shape)
    sl[axis] = slice(0, d)
    return ttnn.clone(t[tuple(sl)])


def build_operands(ttnn, g, d):
    """The grabbed call with its MSA row axis cut to `d`. z is cloned: the layer adds into it."""
    args = list(g["args"])
    kw = dict(g["kwargs"])
    z, _ = arg_at(g, 0, "z")
    m, mloc = arg_at(g, 1, "m")
    mm, mmloc = arg_at(g, 4, "msa_mask")
    rowaxis = 1 if len(m.shape) == 4 else 0
    new_m = row_slice(ttnn, m, d, rowaxis)
    new_z = ttnn.clone(z)
    if mloc[0] == "pos":
        args[1] = new_m
    else:
        kw[mloc[1]] = new_m
    args[0] = new_z
    if mm is not None:
        mmaxis = 1 if len(mm.shape) == 4 else 0
        new_mm = row_slice(ttnn, mm, d, mmaxis)
        if mmloc[0] == "pos":
            args[4] = new_mm
        else:
            kw[mmloc[1]] = new_mm
    return args, kw


class _Timed:
    """Sync-bracketed sub-unit. Adds 2 syncs per sub-unit per call; on a 238 ms call with 782
    programs of 341 us mean the host is fully ahead, so the syncs report the device and do not
    create a gap the device would not otherwise have had."""

    def __init__(self, inner, ttnn, dev, name, acc):
        self._i, self._t, self._d, self._n, self._a = inner, ttnn, dev, name, acc

    def __call__(self, *a, **k):
        self._t.synchronize_device(self._d)
        t0 = time.perf_counter()
        r = self._i(*a, **k)
        self._t.synchronize_device(self._d)
        self._a[self._n].append(time.perf_counter() - t0)
        return r

    def __getattr__(self, n):
        return getattr(self._i, n)


def mark(ttnn, dev, g, acc):
    obj = g["obj"]
    saved = {n: getattr(obj, n) for n in SUBUNITS}
    for n, inner in saved.items():
        setattr(obj, n, _Timed(inner, ttnn, dev, n, acc))

    def undo():
        for n, inner in saved.items():
            setattr(obj, n, inner)
    return undo


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1 - ss_res / ss_tot if ss_tot else float("nan"))


def free(ttnn, args, kw):
    for t in list(args) + list(kw.values()):
        if isinstance(t, ttnn.Tensor):
            try:
                ttnn.deallocate(t)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="512")
    ap.add_argument("--depths", default="128,256,512,768,1024")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--sub", action="store_true", help="also fit the per-sub-unit curve")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global OUT_PATH
    if a.out:
        OUT_PATH = Path(a.out)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    OUT["env"].update({"visible": os.environ.get("TT_VISIBLE_DEVICES"),
                       "depths": a.depths, "reps": a.reps, "blocks": a.blocks})
    dev, g = grab(ttnn, T, B, a.size)
    fence = make_fence(ttnn, dev)
    depths = [int(x) for x in a.depths.split(",")]
    OUT["layer_curve"], OUT["sub_curve"] = {}, {}

    for d in depths:
        args, kw = build_operands(ttnn, g, d)
        # warm: kernels for this shape class compile on the first call
        for _ in range(2):
            z, m = g["obj"](*args, **kw)
            args[0], args[1] = z, m
        ttnn.synchronize_device(dev)
        fence()
        walls = []
        for _ in range(a.blocks):
            t0 = time.perf_counter()
            for _ in range(a.reps):
                z, m = g["obj"](*args, **kw)
                args[0], args[1] = z, m
            ttnn.synchronize_device(dev)
            walls.append((time.perf_counter() - t0) / a.reps)
        fence()
        OUT["layer_curve"][str(d)] = {"ms": round(1e3 * st.median(walls), 4),
                                      "all": [round(1e3 * w, 4) for w in walls]}
        print(f"  depth {d:5d}: {1e3*st.median(walls):9.4f} ms/call", flush=True)
        dump()

        if a.sub:
            acc = {n: [] for n in SUBUNITS}
            undo = mark(ttnn, dev, g, acc)
            try:
                for _ in range(a.reps):
                    z, m = g["obj"](*args, **kw)
                    args[0], args[1] = z, m
                ttnn.synchronize_device(dev)
            finally:
                undo()
            OUT["sub_curve"][str(d)] = {n: round(1e3 * st.median(v), 4) for n, v in acc.items()}
            OUT["sub_curve"][str(d)]["_sum"] = round(
                sum(1e3 * st.median(v) for v in acc.values()), 4)
            print("        " + "  ".join(f"{n[:6]}={OUT['sub_curve'][str(d)][n]:8.3f}"
                                         for n in SUBUNITS), flush=True)
            dump()
        free(ttnn, args[:2], {})

    xs = [float(d) for d in depths]
    ys = [OUT["layer_curve"][str(d)]["ms"] for d in depths]
    a0, b0, r2 = fit(xs, ys)
    at1024 = a0 + b0 * 1024.0
    OUT["fit"] = {"a_ms": round(a0, 5), "b_ms_per_row": round(b0, 8), "r2": round(r2, 6),
                  "t_at_1024_ms": round(at1024, 4),
                  "constant_fraction_pct": round(100.0 * a0 / at1024, 3)}
    print(f"\nLAYER FIT: t = {a0:.4f} ms + {b0:.6f} ms/row   R2 {r2:.6f}")
    print(f"CONSTANT-FRACTION at 1024 rows: {100.0*a0/at1024:.3f} %")
    if OUT["sub_curve"]:
        OUT["sub_fit"] = {}
        for n in SUBUNITS:
            sy = [OUT["sub_curve"][str(d)][n] for d in depths]
            sa, sb, sr2 = fit(xs, sy)
            s1024 = sa + sb * 1024.0
            OUT["sub_fit"][n] = {"a_ms": round(sa, 5), "b_ms_per_row": round(sb, 8),
                                 "r2": round(sr2, 6), "t_at_1024_ms": round(s1024, 4),
                                 "constant_fraction_pct": round(100.0 * sa / s1024, 3)}
            print(f"  {n:26s} a={sa:8.3f} ms  b={sb:.6f}  R2={sr2:.5f}  "
                  f"const={100.0*sa/s1024:6.2f} %  t(1024)={s1024:8.3f}")
    for nm, st_ in (("PWA_DEPTH_STATS", "PWA_DEPTH_STATS"), ("OPM_ROW_STATS", "OPM_ROW_STATS"),
                    ("OPM_SMALL_DEPTH_STATS", "OPM_SMALL_DEPTH_STATS")):
        if hasattr(T, st_):
            OUT.setdefault("path_stats", {})[nm] = str(getattr(T, st_))
    dump()
    print("\n" + json.dumps(OUT.get("fit", {}), indent=1))


if __name__ == "__main__":
    main()
