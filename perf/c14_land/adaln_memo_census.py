#!/usr/bin/env python3
"""How much work does the hit-driven AdaLN memo actually delete? A COUNTER census, not a clock.

`BOLTZ2_ADALN_S_MEMO` ships ON. It keys on the object identity of the caller's `s`, so on RF3 --
which rebuilds `s` every call (`rf3/atom_encoder.py:201`) -- the identity check missed on every
call while the retain ran unconditionally: two `to_memory_config` copies to DRAM per AdaLN call,
for a pair that was stored, never read, and dropped on the next call. `2b80fdaad` makes the retain
hit-driven, so RF3 stores nothing and boltz-2 gives up one hit out of 400.

The orchestrator priced this at nothing on purpose and told the lander not to price it either.
This census does not price it. It COUNTS the deleted `to_memory_config` calls and the bytes they
moved, so the item can be screened against the campaign's byte-deletion unit before anyone spends
a board pair on it. A count is admissible on a contended chip: the shared resource is the board
power budget, which moves the AICLK and therefore the seconds, and a call count does not move with
the clock. Nothing in this file is a measurement of seconds and no field here may be quoted as one.

INSTRUMENT: the wrapper OBSERVES the effect rather than transcribing the decision. It reads
`_s_memo` / `_s_memo_src` before and after the real `s_terms` and infers:

    hit    = the call returned the pre-existing memo object
    store  = the memo object was replaced and now keys on this call's `s`

so a change to the store condition in production cannot silently desynchronise the census the way
a copied `if` would. The two arms are the shipped code (`TT_BIO_ADALN_MEMO_EAGER=0`) and the
pre-2026-09-18 unconditional retain (`=1`), which `test_memo_statemachine.py` test 7 proves equal
to the old code call for call.

LINEARITY, executed rather than assumed: every model is folded at two `--sampling_steps` so the
per-step call count is measured, not extrapolated from one point. The production figure is then
stated at the protocol it belongs to instead of being read off a short fold.

    adaln_memo_census.py --model rf3 --size 512 --steps 8,16 --out perf/c14_land/adaln_census_rf3.json
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "perf" / "size512" / "fixtures"
FLAG = "TT_BIO_ADALN_MEMO_EAGER"
READER = "_B2_ADALN_MEMO_EAGER"
ARMS = {"new": "0", "eager": "1"}


def _bytes(t) -> int:
    """Logical bytes of a ttnn tensor: product of shape times the dtype width."""
    n = 1
    for d in tuple(t.shape):
        n *= int(d)
    name = str(t.dtype).lower()
    if "float32" in name or "uint32" in name or "int32" in name:
        return n * 4
    if "bfloat8" in name:
        return n  # bfp8_b, 1 byte per datum plus a shared exponent per 16
    return n * 2  # bfloat16 and friends


def child() -> int:
    """One fold. The COUNTER lives in `hook/sitecustomize.py`, not here.

    The first version of this census patched `AdaLN.s_terms` in this process and read 0 calls in
    all four arms: `tt_bio/main.py:1135` folds in a `spawn` worker, which re-imports every module
    and discards a parent-side patch. The patch now travels by PYTHONPATH so it reaches the worker.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    os.environ[FLAG] = a.value
    sys.path.insert(0, str(REPO))
    from tt_bio import tenstorrent as T

    want = a.value not in ("0", "", "false", "False")
    got = getattr(T, READER)
    assert got == want, f"{FLAG}={a.value} did not reach {READER} (module reads {got})"
    assert T._B2_ADALN_S_MEMO, "BOLTZ2_ADALN_S_MEMO is off; the census would compare nothing"

    argv = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
    from tt_bio.main import cli

    t0 = time.perf_counter()
    rc = 0
    try:
        cli(argv, standalone_mode=False)
    except SystemExit as e:
        rc = e.code or 0
    a.report.write_text(json.dumps({
        "rc": rc,
        # Named so no reader can quote it: this fold ran on a contended box on purpose.
        "fold_s_NOT_A_MEASUREMENT": round(time.perf_counter() - t0, 3),
        "reader": bool(got)}))
    return rc


def run(model: str, size: int, steps: int, arm: str, msa_dir: Path, recycles: int) -> dict:
    hook = Path(__file__).resolve().parent / "hook"
    with tempfile.TemporaryDirectory(prefix=f"adalncensus_{arm}_") as work:
        out_dir = Path(work) / "out"
        report = Path(work) / "report.json"
        countdir = Path(work) / "counts"
        countdir.mkdir()
        env = dict(os.environ)
        env["ADALN_CENSUS_DIR"] = str(countdir)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(hook)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child",
               "--value", ARMS[arm], "--report", str(report), "--",
               "predict", str(FIX / f"cdk2x2_{size}.yaml"), "--model", model,
               "--seed", "0", "--out_dir", str(out_dir), "--msa_dir", str(msa_dir),
               "--output_format", "cif",
               "--sampling_steps", str(steps), "--recycling_steps", str(recycles)]
        print(f"[census] {model} {size}aa steps={steps} arm={arm}", flush=True)
        rc = subprocess.call(cmd, env=env)
        assert report.exists(), f"{arm} produced no report (rc={rc})"
        rep = json.loads(report.read_text())
        assert rep["rc"] == 0 and rc == 0, f"{arm} fold failed rc={rc}/{rep['rc']}"
        procs = [json.loads(f.read_text()) for f in sorted(countdir.glob("*.json"))]

    agg = {k: sum(p[k] for p in procs) for k in ("calls", "atom_calls", "token_calls",
                                                 "hits", "stores")}
    pair = next((p for p in procs if p["pair_bytes"]), None)
    rep.update(agg, model=model, size=size, steps=steps, arm=arm, recycles=recycles,
               procs=[{k: p[k] for k in ("pid", "calls", "atom_calls", "stores", "hits",
                                         "eager", "memo_on")} for p in procs],
               pair_bytes=pair["pair_bytes"] if pair else None,
               pair_shapes=pair["pair_shapes"] if pair else None,
               pair_dtype=pair["pair_dtype"] if pair else None)
    # POSITIVE CONTROL. A census that saw nothing is a dark instrument, not a null result, and
    # the first version of this one was exactly that. Fail loudly instead of reporting a zero.
    assert rep["calls"] > 0, (
        f"{arm}: the counter saw 0 AdaLN.s_terms calls across {len(procs)} process(es). "
        "The hook did not reach the fold worker -- this is a dark instrument, not a null.")
    # The arm must have reached the worker too, or both arms run the same code.
    assert any(p["eager"] == (ARMS[arm] == "1") for p in procs if p["calls"]), (
        f"{arm}: no counting process saw {READER}={ARMS[arm]}")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rf3")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", default="8,16")
    ap.add_argument("--recycles", type=int, default=1)
    ap.add_argument("--production-steps", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    assert FLAG not in os.environ, f"{FLAG} is pinned; both arms would run the same code"

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))
    import ab_flag_levers as AB

    steps = [int(s) for s in a.steps.split(",")]
    runs = []
    with tempfile.TemporaryDirectory(prefix="adalncensus_msa_") as msa:
        msa_dir = Path(msa)
        AB._seed_msa(FIX / f"cdk2x2_{a.size}.yaml",
                     (FIX / f"cdk2x2_{a.size}.a3m").read_text(), msa_dir)
        for n in steps:
            for arm in ("new", "eager"):
                runs.append(run(a.model, a.size, n, arm, msa_dir, a.recycles))

    by = {(r["arm"], r["steps"]): r for r in runs}
    # Deleted work = the stores the eager arm did that the shipped arm does not, times the two
    # `to_memory_config` calls each store makes. Per step, from two points, so the slope is
    # measured rather than divided out of one fold.
    per_step = {}
    if len(steps) >= 2:
        lo, hi = min(steps), max(steps)
        for arm in ("new", "eager"):
            d_stores = by[(arm, hi)]["stores"] - by[(arm, lo)]["stores"]
            d_calls = by[(arm, hi)]["atom_calls"] - by[(arm, lo)]["atom_calls"]
            per_step[arm] = {
                "stores_per_step": d_stores / (hi - lo),
                "atom_calls_per_step": d_calls / (hi - lo),
                "intercept_stores": by[(arm, lo)]["stores"] - d_stores / (hi - lo) * lo,
                "intercept_atom_calls": by[(arm, lo)]["atom_calls"] - d_calls / (hi - lo) * lo,
            }

    summary = {"model": a.model, "size": a.size}
    pair_bytes = next((r["pair_bytes"] for r in runs if r["pair_bytes"]), None)
    summary["pair_bytes"] = pair_bytes
    summary["pair_shapes"] = next((r["pair_shapes"] for r in runs if r["pair_shapes"]), None)
    summary["pair_dtype"] = next((r["pair_dtype"] for r in runs if r["pair_dtype"]), None)
    if per_step and pair_bytes:
        n = a.production_steps
        st = {arm: per_step[arm]["stores_per_step"] * n + per_step[arm]["intercept_stores"]
              for arm in per_step}
        deleted = st["eager"] - st["new"]
        summary["stores_at_production"] = st
        summary["deleted_stores_at_production"] = deleted
        # Each store is two `to_memory_config` calls, one per tensor of the pair.
        summary["deleted_to_memory_config_calls"] = 2 * deleted
        # Bytes WRITTEN to DRAM by those copies. The matching L1 read is the same volume, so the
        # traffic a rate would be applied to is twice this. Stated both ways, priced neither.
        summary["deleted_dram_write_bytes"] = deleted * pair_bytes
        summary["deleted_traffic_bytes_read_plus_write"] = 2 * deleted * pair_bytes

    report = {"host": socket.gethostname(), "flag": FLAG,
              "note": "COUNTER census. No field here is a measurement of seconds. "
                      "fold_s_NOT_A_MEASUREMENT was taken on a deliberately contended box.",
              "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("TT_")},
              "runs": runs, "per_step": per_step, "summary": summary}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))

    print("\n=== AdaLN memo census:", a.model, a.size, "aa ===")
    for r in runs:
        print(f"  steps={r['steps']:>3} arm={r['arm']:<5} atom_calls={r['atom_calls']:>6} "
              f"stores={r['stores']:>6} hits={r['hits']:>6}")
    for arm, v in per_step.items():
        print(f"  {arm:<5} stores/step {v['stores_per_step']:.2f}  "
              f"atom_calls/step {v['atom_calls_per_step']:.2f}")
    print(f"  pair bytes/store: {summary.get('pair_bytes')}  shapes {summary.get('pair_shapes')}")
    if "deleted_dram_write_bytes" in summary:
        print(f"  at {a.production_steps} steps: deleted stores "
              f"{summary['deleted_stores_at_production']:.0f}, "
              f"DRAM writes deleted {summary['deleted_dram_write_bytes']/1e9:.3f} GB, "
              f"read+write {summary['deleted_traffic_bytes_read_plus_write']/1e9:.3f} GB")
    return 0


if __name__ == "__main__":
    if "--child" in sys.argv:
        sys.argv.remove("--child")
        sys.exit(child())
    sys.exit(main())
