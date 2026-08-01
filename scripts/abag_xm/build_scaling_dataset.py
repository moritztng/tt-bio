"""Turn a directory of AbAg-XM folds into the sample-scaling dataset (parquet).

Two tables, because they answer different questions and have different row counts:

  * `<out>_samples.parquet` -- ONE ROW PER SAMPLE. The raw record: every diffusion sample's
    confidence metrics, plus the target geometry and the code version that produced it. This is
    the publishable artifact; everything else is derivable from it.
  * `<out>_curve.parquet` -- one row per (model, target, metric, k). The scaling curve itself:
    the expected best-of-k for k = 1..N.

WHY best-of-k IS COMPUTED, NOT MEASURED
---------------------------------------
The obvious way to ask "how does quality scale with sample count" is to fold the same target at
N = 1, 2, 5, 10, ... and compare. That is enormously wasteful and it is also unnecessary.

Diffusion samples within one fold are i.i.d. draws (verified: the N=8 pilot produced N distinct
structures, §115). Exchangeable draws mean a random k-subset of an N-sample run has exactly the
same distribution as an independent k-sample run. So the expected best-of-k follows from the order
statistic of the N values already in hand:

    E[max of a random k-subset] = sum_i  v_i * C(N-i-1, k-1) / C(N, k)      (v sorted descending)

i.e. v_i is the subset maximum exactly when the other k-1 members come from the N-i-1 values
ranked below it. One N-sample fold therefore yields the whole curve up to k=N, exactly -- not
approximately, not by bootstrap. The identity is unit-tested against brute-force enumeration over
all C(N,k) subsets in `_selftest`.

This also settles the sample-ORDER question the campaign plan left open (contiguous blocks vs
random subsets): under exchangeability the answer is that order carries no information, so
`all_runs` being rank-ordered rather than generation-ordered costs nothing.

CAVEAT worth stating in the paper: this measures scaling of the SELECTION metric (confidence),
which is what a user without ground truth can actually act on. It is not the same as scaling of
true accuracy. DockQ requires a reference structure, and only 38 of the 164 panel targets ship one
(`examples/ground_truth_structures/`), so true-accuracy scaling is available on that subset only.

Usage:
    python3 scripts/abag_xm/build_scaling_dataset.py <folds_dir> --model opendde-abag \
        --out /path/to/abag_xm_scaling
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from math import comb

# The metrics worth a scaling curve. confidence_score is the model's own ranking key, so it is the
# one a sample-selection strategy would actually use; iptm is the interface-specific signal and the
# one the Ab-Ag trust-signal work found predictive (AUC 0.73-0.84).
CURVE_METRICS = ("confidence_score", "iptm", "ptm", "plddt")


def best_of_k(values: list[float], k: int) -> float:
    """Expected maximum of a uniformly random k-subset of `values`. Exact, not sampled."""
    v = sorted(values, reverse=True)
    n = len(v)
    if not 1 <= k <= n:
        raise ValueError(f"k={k} out of range for n={n}")
    return sum(v[i] * comb(n - i - 1, k - 1) for i in range(n - k + 1)) / comb(n, k)


def _selftest() -> None:
    """Brute-force the identity. A closed form that is silently wrong would poison every curve."""
    from itertools import combinations
    for vals in ([0.31, 0.29, 0.288, 0.354, 0.348, 0.324, 0.318, 0.36], [1.0, 0.0], [5.0]):
        n = len(vals)
        for k in range(1, n + 1):
            got = best_of_k(vals, k)
            want = sum(max(c) for c in combinations(vals, k)) / comb(n, k)
            assert abs(got - want) < 1e-12, f"best_of_k({vals},{k}) {got} != {want}"


def code_version(repo: pathlib.Path) -> str:
    """The git SHA that produced the folds. A dataset without it cannot be reproduced, and this
    campaign has already had two numerics changes land mid-flight."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        sha = out.stdout.strip()
        return sha if sha else "unknown"
    except Exception:                                        # noqa: BLE001
        return "unknown"


def read_folds(folds_dir: pathlib.Path, model: str, sha: str) -> list[dict]:
    """Collect one row per sample from every results.json under `folds_dir`."""
    rows: list[dict] = []
    for rj in sorted(folds_dir.glob("*/*/results.json")):
        try:
            recs = json.loads(rj.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"  skip {rj}: {e}", file=sys.stderr)
            continue
        for rec in recs:
            runs = rec.get("all_runs") or []
            if not runs:
                # A fold with no per-sample block cannot contribute to a scaling curve; recording
                # it as a zero-sample row would silently flatten the curve instead.
                print(f"  {rec.get('id')}: no all_runs, skipped", file=sys.stderr)
                continue
            for run in runs:
                rows.append({
                    "model": model,
                    "target": rec.get("id"),
                    "status": rec.get("status"),
                    "n_samples": rec.get("samples") or len(runs),
                    "rank": run.get("rank"),
                    "confidence_score": run.get("confidence_score"),
                    "iptm": run.get("iptm"),
                    "ptm": run.get("ptm"),
                    "plddt": run.get("plddt"),
                    "complex_plddt": run.get("complex_plddt"),
                    "n_tokens": rec.get("n_tokens"),
                    "n_chains": rec.get("n_chains"),
                    "n_residues": rec.get("n_residues"),
                    "n_atoms": rec.get("n_atoms"),
                    "msa": bool(rec.get("msa")),
                    "fold_runtime_s": rec.get("runtime_s"),
                    "code_sha": sha,
                })
    return rows


def build_curve(rows: list[dict]) -> list[dict]:
    by_target: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_target.setdefault((r["model"], r["target"]), []).append(r)
    out: list[dict] = []
    for (model, target), rs in sorted(by_target.items()):
        for metric in CURVE_METRICS:
            vals = [r[metric] for r in rs if r.get(metric) is not None]
            if not vals:
                continue
            for k in range(1, len(vals) + 1):
                out.append({"model": model, "target": target, "metric": metric,
                            "k": k, "n_samples": len(vals),
                            "expected_best_of_k": best_of_k(vals, k)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folds_dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="path prefix; _samples.parquet/_curve.parquet appended")
    ap.add_argument("--repo", default=str(pathlib.Path(__file__).resolve().parents[2]))
    args = ap.parse_args()

    _selftest()
    import pyarrow as pa
    import pyarrow.parquet as pq

    sha = code_version(pathlib.Path(args.repo))
    rows = read_folds(pathlib.Path(args.folds_dir), args.model, sha)
    if not rows:
        print("no sample rows found -- refusing to write an empty manifest", file=sys.stderr)
        return 1
    curve = build_curve(rows)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sp, cp = out.with_name(out.name + "_samples.parquet"), out.with_name(out.name + "_curve.parquet")
    pq.write_table(pa.Table.from_pylist(rows), sp)
    pq.write_table(pa.Table.from_pylist(curve), cp)

    targets = sorted({r["target"] for r in rows})
    print(f"code_sha       {sha}")
    print(f"targets        {len(targets)}: {' '.join(targets)}")
    print(f"sample rows    {len(rows)}  -> {sp} ({sp.stat().st_size} B)")
    print(f"curve rows     {len(curve)}  -> {cp} ({cp.stat().st_size} B)")
    for t in targets:
        rs = [r for r in rows if r["target"] == t]
        n = len(rs)
        print(f"  {t}: n={n}  best-of-1 {best_of_k([r['confidence_score'] for r in rs], 1):.4f}"
              f"  best-of-{n} {best_of_k([r['confidence_score'] for r in rs], n):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
