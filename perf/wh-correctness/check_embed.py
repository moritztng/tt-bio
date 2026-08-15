#!/usr/bin/env python3
"""Decide whether an embedding artifact is real, the way check_structure.py decides a fold.

"The job succeeded and returned a file" is not a pass on this surface either. An
embedding's garbage modes are a constant tensor, a NaN, a vector that belongs to a
different sequence, and a pooling knob that is silently ignored -- none of which any
existing check would have seen.

The format is fixed by tt_bio/esmc.py:write_npz / write_parquet:

  npz      one file per sequence: per_residue [L, d_model] float32, pooled [d_model]
           float32, sequence (0-d string array), and logits only when asked for.
  parquet  one table: id, sequence, length, pooled (list per row).

The <cls>/<eos> special tokens are stripped from per_residue (esmc.py:1035-1036 and
the artifact note at :1539), so `per_residue.shape[0] == len(sequence)` is an exact
equality. That is the composition check for embeddings: it catches an off-by-one in
the token handling and a vector returned for the wrong sequence.

    check_embed.py emb.npz --sequence MKT... --pool mean --json report.json
    check_embed.py embeddings.parquet --expect-ids a,b,c --json report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Tolerance for "pooled is the mean of per_residue". float32 accumulation over up to
# 2000 rows, so rtol has to be loose enough not to fire on summation order alone;
# 1e-4 is ~3 orders above the float32 epsilon and still ~4 orders below the spread
# between two genuinely different pooling modes.
POOL_RTOL = 1e-4
POOL_ATOL = 1e-5


def _finite(name: str, arr: np.ndarray, fail: list) -> bool:
    bad = int((~np.isfinite(arr)).sum())
    if bad:
        fail.append(f"{name} has {bad} non-finite value(s)")
        return False
    return True


def check_npz(path: Path, sequence: str | None, pool: str, rep: dict) -> None:
    fail, warn, checks = rep["fail"], rep["warn"], rep["checks"]
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        checks["keys"] = sorted(keys)
        for want in ("per_residue", "pooled", "sequence"):
            if want not in keys:
                fail.append(f"npz is missing '{want}'")
        if fail:
            return
        per_residue = np.asarray(z["per_residue"])
        pooled = np.asarray(z["pooled"])
        seq_out = str(z["sequence"])

    checks["shape"] = list(per_residue.shape)
    checks["d_model"] = int(pooled.shape[-1]) if pooled.ndim else 0
    checks["length_out"] = len(seq_out)

    ok = _finite("per_residue", per_residue, fail)
    ok &= _finite("pooled", pooled, fail)

    if per_residue.ndim != 2:
        fail.append(f"per_residue is {per_residue.ndim}-D, expected [L, d_model]")
    if pooled.ndim != 1:
        fail.append(f"pooled is {pooled.ndim}-D, expected [d_model]")
    if per_residue.ndim == 2 and pooled.ndim == 1 and per_residue.shape[1] != pooled.shape[0]:
        fail.append(f"per_residue d_model {per_residue.shape[1]} != pooled d_model {pooled.shape[0]}")

    # The composition check. <cls>/<eos> are stripped, so this is an equality.
    if per_residue.ndim == 2 and per_residue.shape[0] != len(seq_out):
        fail.append(f"per_residue has {per_residue.shape[0]} rows for a {len(seq_out)}-residue "
                    f"sequence -- one row per residue is the documented contract")
    if sequence is not None and seq_out != sequence:
        why = (f"{len(seq_out)} residues out, {len(sequence)} in" if len(seq_out) != len(sequence)
               else f"same length, {sum(x != y for x, y in zip(seq_out, sequence))} residue(s) differ")
        fail.append(f"the returned sequence is not the submitted one ({why})")

    if not ok:
        return

    # A constant tensor is this surface's garbage signature, the same role the
    # constant-confidence check plays on the fold side. Measure it with peak-to-peak,
    # not std: float32 std over identical values is 3.6e-07, not 0 (measured on a
    # tiled control while calibrating this checker), so `std == 0` misses a perfectly
    # constant tensor while `max - min == 0` is exact and needs no invented tolerance.
    checks["pooled_ptp"] = float(pooled.max() - pooled.min())
    checks["per_residue_ptp"] = float(per_residue.max() - per_residue.min())
    if checks["pooled_ptp"] == 0.0:
        fail.append("pooled is constant -- a dead embedding")
    if checks["per_residue_ptp"] == 0.0:
        fail.append("per_residue is constant -- a dead embedding")
    # Every residue identical is the other constant mode: the tensor varies across
    # dimensions but not along the sequence, which is what a broadcast bug looks like.
    if per_residue.ndim == 2 and per_residue.shape[0] > 1:
        row_spread = float((per_residue.max(axis=0) - per_residue.min(axis=0)).max())
        checks["row_spread"] = row_spread
        if row_spread == 0.0:
            fail.append("every residue has the identical embedding -- broadcast, not a forward pass")

    # The pooling knob must do what it says.
    if pool == "mean" and per_residue.ndim == 2 and pooled.ndim == 1 and per_residue.shape[0]:
        want = per_residue.mean(axis=0)
        if not np.allclose(pooled, want, rtol=POOL_RTOL, atol=POOL_ATOL):
            dev = float(np.abs(pooled - want).max())
            fail.append(f"pool=mean but pooled is not the mean of per_residue (max |diff| {dev:.3g})")
        else:
            checks["pool_mean_matches"] = True


def _read_parquet(path: Path) -> dict:
    """The parquet table as plain column lists. pyarrow if it is there, pandas
    otherwise -- the system python3 on pc has neither, so this raises a message that
    names the missing module rather than one that reads like a service failure."""
    try:
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pydict()
    except ImportError:
        pass
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError(
            "reading a parquet artifact needs pyarrow or pandas and this interpreter has "
            "neither -- run this cell with an interpreter that does (INSTRUMENT, not the "
            "service). This is not a verdict on the cell.")
    return {c: list(v) for c, v in pd.read_parquet(path).items()}


def check_parquet(path: Path, expect_ids: list[str] | None, rep: dict) -> None:
    fail, checks = rep["fail"], rep["checks"]
    tbl = _read_parquet(path)
    checks["columns"] = sorted(tbl)
    for col in ("id", "sequence", "length", "pooled"):
        if col not in tbl:
            fail.append(f"parquet is missing column '{col}'")
    if fail:
        return
    ids = [str(v) for v in tbl["id"]]
    seqs = [str(v) for v in tbl["sequence"]]
    checks["rows"] = len(ids)
    if expect_ids is not None and ids != expect_ids:
        fail.append(f"ids {ids} do not match the submitted ids {expect_ids}")
    mat = np.array([np.asarray(v, dtype=np.float64) for v in tbl["pooled"]])
    if not _finite("pooled", mat, fail):
        return
    checks["d_model"] = int(mat.shape[1]) if mat.ndim == 2 else 0
    for i, row in enumerate(mat):
        if float(row.max() - row.min()) == 0.0:
            fail.append(f"row {i} ('{ids[i]}') pooled is constant -- a dead embedding")
    for i, (seq, length) in enumerate(zip(seqs, tbl["length"])):
        if int(length) != len(seq):
            fail.append(f"row {i}: length {int(length)} != len(sequence) {len(seq)}")
    # Two different sequences must not give the same vector.
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            if seqs[i] != seqs[j] and np.array_equal(mat[i], mat[j]):
                fail.append(f"rows {i} and {j} are different sequences with an identical "
                            f"pooled vector")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", type=Path)
    ap.add_argument("--sequence", help="the submitted sequence, for the composition check")
    ap.add_argument("--pool", default="mean", choices=("mean", "max", "cls"))
    ap.add_argument("--expect-ids", help="comma-separated ids, parquet only")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    rep: dict = {"struct": str(a.artifact), "checks": {}, "fail": [], "warn": []}
    try:
        if a.artifact.suffix == ".parquet":
            ids = a.expect_ids.split(",") if a.expect_ids else None
            check_parquet(a.artifact, ids, rep)
        else:
            check_npz(a.artifact, a.sequence, a.pool, rep)
    except Exception as e:  # a checker that dies is a failed cell, not a missing one
        rep["fail"].append(f"{type(e).__name__} reading the artifact: {e}")

    rep["verdict"] = "FAIL" if rep["fail"] else ("WARN" if rep["warn"] else "PASS")
    if a.json:
        a.json.write_text(json.dumps(rep, indent=1))
    if not a.quiet:
        print(f"{rep['verdict']}  {a.artifact}")
        for f in rep["fail"]:
            print(f"  FAIL {f}")
    return 1 if rep["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
