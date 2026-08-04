#!/usr/bin/env python3
"""Build the deep-N saturation community asset (per-model parquets) from the drained trees.

One command per model, run on qb1 (post-drain or mid-drain -- read-only):

    python3 scripts/abag_xm/build_deepn_dataset.py --model boltz2 --out ~/abag_xm/deepn/asset/boltz2_deepn

  * `<out>_samples.parquet` -- ONE ROW PER SAMPLE in the galaxy deep-N spine (plus the
    ARK-restated N=16 rung for the models that have one): results.json confidence
    metrics joined with the ARK-interface labels.json by rank, plus per-chunk
    provenance (seed, mps, wall seconds) from galaxy/fleet_results.jsonl.
  * `<out>_curve.parquet` -- one row per (target, rung, metric, k): the EXACT
    E[best-of-k] over the pooled rung (order-statistic identity; exchangeable draws,
    computed not measured -- see build_scaling_dataset's docstring). metric=oracle
    ranks by DockQ; metric=user ranks by the model's confidence selector and carries
    DockQ as the value (the deep-N user sees all N samples and picks by confidence).

Exclusions match the analysis exactly (GALAXY_EXCLUDE / N16_ARK_EXCLUDE imported from
abag_xm_deepn_analysis -- that script is the canonical owner): pipeline artifacts are
not model behavior. The galaxy N=16 parquet arm (global_dockq flavor) is NOT packaged
here; the restated rung supersedes it for bz/esm/px and opendde's N=16 stays a
datasheet-only flavor-flagged row (DATASHEET section 9).
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abag_xm_deepn_analysis import (  # noqa: E402
    GALAXY_EXCLUDE, MODELS, N16_ARK, N16_ARK_EXCLUDE, N16_ARK_OK)

BASE = Path.home() / "abag_xm" / "deepn"
GALAXY = BASE / "galaxy"


def fleet_index():
    """(target, rung, chunk) -> (seed, mps, seconds); last-attempt-wins, rc=0 only."""
    out = {}
    fj = GALAXY / "fleet_results.jsonl"
    if not fj.exists():
        return out
    for line in fj.read_text().splitlines():
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("rc") != 0:
            continue
        out[(r.get("target"), r.get("rung"), r.get("chunk"))] = (
            r.get("seed"), r.get("mps"), r.get("seconds"))
    return out


def getdq(s):
    d = s.get("dockq")
    return d if isinstance(d, dict) else None


def fold_rows(model, out_dir, target, rung, chunk, hardware, sha, prov):
    """One fold dir -> sample rows (results.json x labels.json joined by rank)."""
    prefix, _tp, sel = MODELS[model]
    rj = out_dir / f"{prefix}_results_{target}" / "results.json"
    lj = out_dir / "labels.json"
    if not rj.exists() or not lj.exists():
        return []
    try:
        runs = json.loads(rj.read_text())[0].get("all_runs", [])
        labs = {int(s["rank"]): s for s in json.loads(lj.read_text()).get("samples", [])}
    except Exception:
        return []
    seed, mps, secs = prov.get((target, rung, chunk), (None, None, None))
    rows = []
    for r in runs:
        rank = r.get("rank")
        if rank is None:
            continue
        s = labs.get(int(rank), {})
        dq = getdq(s) or {}
        il = s.get("interface_lddt") or {}
        cdr = (s.get("cdr_rmsd") or {}).get("cdrs") or {}
        ej = s.get("epitope_jaccard") or {}
        rows.append({
            "model": model, "target": target, "rung": rung,
            "chunk": chunk if chunk is not None else -1, "rank": int(rank),
            "selector": r.get(sel), "confidence_score": r.get("confidence_score"),
            "ptm": r.get("ptm"), "iptm": r.get("iptm"),
            "complex_plddt": r.get("complex_plddt"),
            "dockq": dq.get("dockq"), "irmsd": dq.get("iRMSD"),
            "lrmsd": dq.get("LRMSD"), "fnat": dq.get("fnat"),
            "interface_lddt": il.get("interface_lddt"),
            "cdr_h1_rmsd": cdr.get("H1"), "cdr_h2_rmsd": cdr.get("H2"),
            "cdr_h3_rmsd": cdr.get("H3"),
            "epitope_jaccard": ej.get("epitope_jaccard"),
            "seed": seed, "mps": int(mps) if mps is not None else None,
            "wall_s": secs, "hardware": hardware, "code_sha": sha})
    return rows


def collect(model, sha):
    prov = fleet_index()
    rows = []
    for root, hardware, excl in ((GALAXY / MODELS[model][0], "wh-galaxy",
                                  GALAXY_EXCLUDE.get(model, ())),
                                 (N16_ARK / MODELS[model][0], "wh-galaxy-ark",
                                  N16_ARK_EXCLUDE.get(model, ()))):
        if hardware == "wh-galaxy-ark" and model not in N16_ARK_OK:
            continue
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            try:
                t, rest = d.name.split("_n")
                rung = int(rest.split("_c")[0])
                chunk = int(rest.split("_c")[1]) if "_c" in rest else None
            except (ValueError, IndexError):
                continue
            if t in excl:
                continue
            rows += fold_rows(model, d, t, rung, chunk, hardware, sha, prov)
    return rows


def bok_ordered(ordered_vals, k):
    """E[value of the rank-top-1 of a random k-subset]; ordered_vals pre-sorted by rank."""
    n = len(ordered_vals)
    return sum(ordered_vals[i] * comb(n - i - 1, k - 1)
               for i in range(n - k + 1)) / comb(n, k)


def curve_rows(rows):
    out = []
    pools = {}
    for r in rows:
        if r["dockq"] is None or r["selector"] is None:
            continue
        pools.setdefault((r["model"], r["target"], r["rung"]), []).append(r)
    for (m, t, rung), rs in sorted(pools.items()):
        by_dq = sorted((r["dockq"] for r in rs), reverse=True)
        by_sel = [r["dockq"] for r in
                  sorted(rs, key=lambda r: r["selector"], reverse=True)]
        n = len(by_dq)
        for k in range(1, n + 1):
            out.append({"model": m, "target": t, "rung": rung, "metric": "oracle",
                        "k": k, "n_samples": n, "expected_best_of_k": bok_ordered(by_dq, k)})
            out.append({"model": m, "target": t, "rung": rung, "metric": "user",
                        "k": k, "n_samples": n, "expected_best_of_k": bok_ordered(by_sel, k)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--out", required=True, help="path prefix; _samples/_curve.parquet appended")
    ap.add_argument("--sha", default=None, help="code sha stamped on every row "
                    "(default: git HEAD of this repo)")
    a = ap.parse_args()
    sha = a.sha
    if sha is None:
        import subprocess
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = collect(a.model, sha)
    if not rows:
        print("no sample rows found -- refusing to write an empty manifest", file=sys.stderr)
        return 1
    curve = curve_rows(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sp, cp = out.with_name(out.name + "_samples.parquet"), out.with_name(out.name + "_curve.parquet")
    pq.write_table(pa.Table.from_pylist(rows), sp)
    pq.write_table(pa.Table.from_pylist(curve), cp)
    targets = sorted({r["target"] for r in rows})
    rungs = sorted({r["rung"] for r in rows})
    labeled = sum(1 for r in rows if r["dockq"] is not None)
    print(f"code_sha    {sha}")
    print(f"targets     {len(targets)}  rungs {rungs}")
    print(f"sample rows {len(rows)} ({labeled} with ARK DockQ) -> {sp} ({sp.stat().st_size} B)")
    print(f"curve rows  {len(curve)} -> {cp} ({cp.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
