#!/usr/bin/env python3
"""Which layout moves a 512 aa fold actually issues, attributed to the source line that asks.

`c14-unpriced-block` priced Transpose+Permute at 0.5997 s in situ over 12,696 calls and refused to
open a lever, for a stated reason: a ttnn graph capture records operand shapes but NOT op arguments
(`arguments: []` on every function_start), so the permutation ORDER is unrecorded, a shape-
preserving permute reads as the identity, and its arm and the fold disagreed 1.74x on this class.

A capture cannot answer that. A wrapper can: it sees the real `dims`, the real caller, and the
real tensor identities. This records, per call,

  * the permutation order, so an identity permute is visible as an identity rather than inferred
  * the tt_bio source stack, so 12,696 anonymous calls become N named sites with counts
  * the producer of the input, so an `a -> b -> a` pair across two ops is detected mechanically
    rather than by reading hopefully

It is a COUNT, not a second: clock-immune and pair-immune, so it is legal on a contended chip and
on qb1 (`c14-unpriced-block`: a fold whose clock collapsed to 800 MHz still produced a trustworthy
call count while producing no usable wall time). No wall time here is a number of record.

Control: two consecutive folds in one process. The per-site counts must agree exactly, or the
census is not settled and nothing below it may be quoted.
"""
from __future__ import annotations

import argparse, json, os, socket, sys, time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/gpu_vs_tt"), str(ROOT / "perf/other512")]

WRAP = ["permute", "transpose", "reshape", "to_layout"]
MAXF = 6


def _shape(x):
    try:
        return tuple(int(d) for d in x.shape)
    except Exception:
        return None


def _stack():
    """The tt_bio-only call stack, innermost first, as file:line strings."""
    out = []
    f = sys._getframe(2)
    while f is not None and len(out) < MAXF:
        fn = f.f_code.co_filename
        if "/tt_bio/" in fn and "/_vendor/" not in fn:
            out.append("%s:%d" % (fn.split("/tt_bio/")[-1], f.f_lineno))
        f = f.f_back
    return out


def _memcfg(t):
    try:
        mc = t.memory_config()
        return str(mc.buffer_type).split(".")[-1]
    except Exception:
        return "?"


class Census:
    def __init__(self):
        self.rows = defaultdict(int)     # key -> calls
        self.meta = {}                   # key -> descriptor
        self.chain = defaultdict(int)    # (producer_key, consumer_key) -> calls
        self.produced = {}               # id(tensor) -> (key, shape)
        self.on = False

    def record(self, op, args, kw, out):
        ins = [a for a in args if _shape(a) is not None]
        t = ins[0] if ins else None
        ish = _shape(t)
        osh = _shape(out)
        if op == "permute":
            dims = kw.get("dims", args[1] if len(args) > 1 else None)
        elif op == "transpose":
            d = [kw.get("dim1", args[1] if len(args) > 1 else None),
                 kw.get("dim2", args[2] if len(args) > 2 else None)]
            dims = tuple(d)
        else:
            dims = None
        try:
            dims = tuple(int(x) for x in dims) if dims is not None else None
        except Exception:
            dims = None
        st = _stack()
        site = st[0] if st else "?"
        ident = None
        if op == "permute" and dims is not None:
            ident = tuple(dims) == tuple(range(len(dims)))
        if op == "transpose" and dims is not None and ish is not None:
            n = len(ish)
            a, b = (dims[0] % n, dims[1] % n)
            ident = (a == b)
        key = "%s|%s|dims=%s|in=%s|out=%s" % (
            op, site, ",".join(map(str, dims)) if dims else "-",
            "x".join(map(str, ish)) if ish else "-",
            "x".join(map(str, osh)) if osh else "-")
        self.rows[key] += 1
        if key not in self.meta:
            self.meta[key] = {"op": op, "site": site, "stack": st, "dims": dims,
                              "in_shape": ish, "out_shape": osh,
                              "identity_perm": ident, "in_mem": _memcfg(t),
                              "out_mem": _memcfg(out),
                              "shape_preserving": (ish == osh) if (ish and osh) else None}
        if t is not None:
            prev = self.produced.get(id(t))
            if prev is not None and prev[1] == ish:
                self.chain[(prev[0], key)] += 1
        if osh is not None:
            self.produced[id(out)] = (key, osh)

    def dump(self):
        return {"sites": dict(sorted(self.rows.items(), key=lambda kv: -kv[1])),
                "meta": self.meta,
                "chains": {"%s >>> %s" % k: v
                           for k, v in sorted(self.chain.items(), key=lambda kv: -kv[1])}}


def install(ttnn, cen):
    for name in WRAP:
        fn = getattr(ttnn, name)

        def counted(*args, _fn=fn, _n=name, **kw):
            if not cen.on:
                return _fn(*args, **kw)
            out = _fn(*args, **kw)
            try:
                cen.record(_n, args, kw, out)
            except Exception as e:
                cen.rows["ERROR|%s|%r" % (_n, e)] += 1
            return out
        setattr(ttnn, name, counted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--node", type=int, required=True)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import torch, ttnn
    import tt_bio.tenstorrent as T
    assert Path(T.__file__).resolve().is_relative_to(ROOT), T.__file__
    import tt_baseline as B
    from fold_ab_multi import patch_boltz2_cfg

    R = {"host": socket.gethostname(), "node": a.node, "size": a.size, "pid": os.getpid(),
         "ttnn": ttnn.__file__, "tt_bio": T.__file__, "started_utc": time.time(),
         "folds": [], "note": "CALL COUNTS ONLY. No wall time here is a number of record."}

    def save():
        (a.out / "census.json").write_text(json.dumps(R, indent=1, default=str))

    B.RECYCLING_STEPS = 3
    B.SAMPLING_STEPS = 200
    B.DIFFUSION_SAMPLES = 1
    B.SEED = 0
    patch_boltz2_cfg()
    fixture = ROOT / ("perf/size512/fixtures/cdk2x2_%d" % a.size)
    target, msa = fixture.with_suffix(".yaml"), fixture.with_suffix(".a3m")
    _f, meta, state = B.build_fold("boltz2", a.out / "msa", target, msa, instrument=False,
                                   hoist=False, fast=False, trace=False, recycling_steps=3)
    dev = T.get_device()
    R["n_msa"] = meta.get("n_msa")
    save()

    cen = Census()
    install(ttnn, cen)
    for i in range(a.folds):
        cen.rows.clear(); cen.meta.clear(); cen.chain.clear(); cen.produced.clear()
        cen.on = True
        ttnn.synchronize_device(dev)
        t0 = time.monotonic()
        state.predict_one(target, meta["job_cfg"])
        ttnn.synchronize_device(dev)
        el = time.monotonic() - t0
        cen.on = False
        d = cen.dump()
        (a.out / ("fold%d.json" % i)).write_text(json.dumps(d, indent=1, default=str))
        R["folds"].append({"i": i, "wall_s_NOT_OF_RECORD": round(el, 3),
                           "n_calls": sum(d["sites"].values()), "n_sites": len(d["sites"])})
        save()
        print(json.dumps(R["folds"][-1]), flush=True)
    if a.folds >= 2:
        f0 = json.loads((a.out / "fold0.json").read_text())["sites"]
        f1 = json.loads((a.out / "fold1.json").read_text())["sites"]
        R["settled_control"] = {"identical": f0 == f1,
                                "keys_only_in_fold0": sorted(set(f0) - set(f1))[:20],
                                "keys_only_in_fold1": sorted(set(f1) - set(f0))[:20],
                                "count_diffs": {k: [f0.get(k), f1.get(k)]
                                                for k in set(f0) | set(f1)
                                                if f0.get(k) != f1.get(k)}}
    R["completed"] = True
    save()
    print("settled:", R.get("settled_control", {}).get("identical"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
