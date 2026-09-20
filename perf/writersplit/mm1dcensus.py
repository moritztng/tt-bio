#!/usr/bin/env python3
"""How many matmul calls in one fold would take the 1D mcast_in1 path, counted at run time.

`eligibility-firing-condition-is-not-a-code-fact`: the build row left the count unconfirmed
("17,920 is unconfirmed and the fold implies ~11.7k").  This wraps ttnn.matmul and ttnn.linear,
runs a real fold, and records every call's operands, then classifies each against the shipped
chooser's own rule rather than against a flag.

The rule, transcribed from
ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config.cpp at v0.67.4
(`get_matmul_program_config`, :378-505) for the no-explicit-program_config case:

  input_b_is_batched  batch_size_b > 1              -> not 1D
  any_size_within_tile  k <= 32 or m <= 32 or n <= 32
  is_narrow_shape(h, w)  max(h, w) / min(h, w) > 8   (all_dram=false at this call site)
  1D is chosen when (is_narrow_shape or any_size_within_tile) and a is not block-sharded
  mcast_in1 when is_tall, i.e. (batch_a * m) / 32 > n / 32

A call with an explicit program_config is classified off the config object instead.  The last
column, `ours`, is the stricter question: of the mcast_in1 calls, how many fall inside what
tt_bio/mm1d_generic.py actually covers -- interleaved DRAM throughout, no bias, no fused
activation, uniform bf16.

Counting is not timing, so this does not force or sample the clock and does not need a quiet host.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "other512"))

TILE = 32


def batch_size(shape):
    n = 1
    for d in shape[:-2]:
        n *= d
    return n


def is_narrow_shape(h, w):
    return (h // w if h > w else w // h) > 8


def classify(a_shape, b_shape, a_sharded, b_sharded, pcfg_name, mcast_in0):
    """(path, reason) for one call."""
    if pcfg_name is not None:
        if pcfg_name == "MatmulMultiCoreReuseMultiCast1DProgramConfig":
            return ("mcast_in0" if mcast_in0 else "mcast_in1"), "explicit program_config"
        return pcfg_name, "explicit program_config"
    k, m, n = a_shape[-1], a_shape[-2], b_shape[-1]
    ba, bb = batch_size(a_shape), batch_size(b_shape)
    if bb > 1:
        return "reuse_batched", "input b batched"
    if a_sharded or b_sharded:
        return "sharded", "a or b sharded"
    h, w = ba * m, n
    within_tile = k <= TILE or m <= TILE or n <= TILE
    if is_narrow_shape(h, w) or within_tile:
        return ("mcast_in1" if h // TILE > w // TILE else "mcast_in0"), (
            "narrow" if is_narrow_shape(h, w) else "within tile")
    return "mcast_2d", "neither narrow nor within tile"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--model", default="boltz2")
    ap.add_argument("--folds", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "mm1dcensus.json"))
    a = ap.parse_args()

    import ttnn
    import tt_bio
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    from fold_ab_multi import patch_boltz2_cfg
    patch_boltz2_cfg()
    assert Path(tt_bio.__file__).resolve().is_relative_to(ROOT), (
        "imported tt_bio from %s, not this worktree" % tt_bio.__file__)

    calls = collections.Counter()
    active = [False]

    def record(name, orig):
        def wrapped(*args, **kw):
            if active[0] and len(args) >= 2:
                try:
                    x, w = args[0], args[1]
                    pc = kw.get("program_config")
                    calls[(name,
                           tuple(int(d) for d in x.padded_shape),
                           tuple(int(d) for d in w.padded_shape),
                           str(x.dtype), str(w.dtype),
                           x.is_sharded(), w.is_sharded(),
                           type(pc).__name__ if pc is not None else None,
                           bool(getattr(pc, "mcast_in0", False)),
                           kw.get("bias") is not None or (len(args) > 2 and args[2] is not None),
                           kw.get("activation") is not None,
                           str(kw.get("dtype")))] += 1
                except Exception as e:                      # never break a fold to count it
                    calls[("ERR", repr(e)[:80])] += 1
            return orig(*args, **kw)
        return wrapped

    for nm in ("matmul", "linear"):
        setattr(ttnn, nm, record(nm, getattr(ttnn, nm)))

    fixdir = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, *_ = B.build_fold(
        a.model, ROOT / (".msa_mm1dcensus_%s_%d" % (a.model, a.size)),
        fixdir / ("cdk2x2_%d.yaml" % a.size), fixdir / ("cdk2x2_%d.a3m" % a.size))

    active[0] = True
    for i in range(a.folds):
        fold_s, m = one_fold()
        print("fold %d: %.3f s, %d distinct matmul signatures so far"
              % (i, fold_s, len(calls)), flush=True)
    active[0] = False

    rows = []
    for key, n in calls.items():
        if key[0] == "ERR":
            rows.append({"error": key[1], "n": n})
            continue
        (nm, xs, ws, xd, wd, xsh, wsh, pcn, mc0, bias, act, odt) = key
        path, why = classify(xs, ws, xsh, wsh, pcn, mc0)
        ours = (path == "mcast_in1" and not xsh and not wsh and not bias and not act
                and xd == "DataType.BFLOAT16" and wd == "DataType.BFLOAT16")
        rows.append({"op": nm, "a": list(xs), "b": list(ws), "a_dtype": xd, "b_dtype": wd,
                     "a_sharded": xsh, "b_sharded": wsh, "program_config": pcn,
                     "bias": bias, "activation": act, "out_dtype": odt,
                     "path": path, "why": why, "covered_by_mm1d_generic": ours, "n": n})
    rows.sort(key=lambda r: -r.get("n", 0))

    tot = sum(r.get("n", 0) for r in rows)
    by_path = collections.Counter()
    for r in rows:
        by_path[r.get("path", "ERR")] += r.get("n", 0)
    ours = sum(r["n"] for r in rows if r.get("covered_by_mm1d_generic"))
    print("\n%d matmul/linear calls per fold, %d distinct signatures" % (tot, len(rows)))
    for p, n in by_path.most_common():
        print("  %-16s %7d  %5.1f %%" % (p, n, 100.0 * n / tot))
    print("  covered by mm1d_generic: %d (%.1f %% of all calls, %.1f %% of mcast_in1)"
          % (ours, 100.0 * ours / tot,
             100.0 * ours / by_path["mcast_in1"] if by_path["mcast_in1"] else 0.0))
    print("\ntop mcast_in1 signatures:")
    for r in [r for r in rows if r.get("path") == "mcast_in1"][:12]:
        print("  n=%-6d a=%s b=%s covered=%s" % (r["n"], r["a"], r["b"],
                                                 r["covered_by_mm1d_generic"]))

    Path(a.out).write_text(json.dumps(
        {"model": a.model, "size": a.size, "folds": a.folds, "total_calls": tot,
         "by_path": dict(by_path), "covered_by_mm1d_generic": ours,
         "git_head": os.popen("git -C %s rev-parse HEAD" % ROOT).read().strip(),
         "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
