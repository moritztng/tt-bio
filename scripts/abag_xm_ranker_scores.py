#!/usr/bin/env python3
"""Phase 5 ranker-score driver for the AbAg-XM campaign.

Builds the per-sample ranker-score table for a completed (target, generator)
fold by JOINing:

  - native generator confidences (ipTM, pTM, ranking/confidence score, complex
    pLDDT) from the harness `results.json[0]["all_runs"][rank]`;
  - closed-form rankers (pDockQ2, ipSAE, AntiConf) from the Phase-4 labels
    `samples[rank]["pae_metrics"]` (already computed by abag_pae_metrics.py
    with source-verified constants);
  - PSS (per-sample) = mean pairwise interface DockQ over the ensemble, computed
    from `pairwise_matrix["matrix"]` (row-mean of the dockq column for pairs
    involving sample rank);
  - the oracle label (DockQ on the ARK-declared interface) +
    epitope_jaccard, interface_lddt, per-CDR RMSD from the Phase-4 labels;
  - (optional) learned rankers: DeepRank-Ab (per-sample predicted_dockq via the
    deeprank-ab-predict CLI) and ABAG-Rank (per-sample model_predicted_dockq via
    scripts/abag_xm_abagrank_adapter.py --run_inference).

Output: one row per (target, generator, rank) with all ranker columns + the
oracle label, written as CSV (append mode for --all). This is the Phase 5
deliverable: the ranker-transfer table every downstream comparison is built on.

Usage (one fold):
    python scripts/abag_xm_ranker_scores.py \\
        --fold_dir ~/abag_xm/tier_a/protenix_v2/protenix_results_21av \\
        --target 21av --gen protenix-v2 \\
        --labels ~/abag_xm/tier_a/labels/protenix_v2_21av.json \\
        --out ~/abag_xm/tier_a/ranker_scores.csv

Usage (all completed folds):
    python scripts/abag_xm_ranker_scores.py --all --out ~/abag_xm/tier_a/ranker_scores.csv
"""
import argparse, csv, json, os, subprocess, sys, tempfile
from pathlib import Path
from collections import OrderedDict

ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = Path.home() / "abag_xm" / "tier_a" / "labels"
TIERA_DIR = Path.home() / "abag_xm" / "tier_a"

# gen dir name -> generator id
GEN_DIRS = {"protenix_v2": "protenix-v2", "boltz2": "boltz2", "opendde_abag": "opendde-abag"}

COLUMNS = [
    "target", "gen", "rank",
    # native (generator confidences)
    "iptm", "ptm", "ranking_score", "complex_plddt",
    # closed-form (from pae_metrics)
    "pdockq2", "ipsae", "anticonf",
    # consensus
    "pss",
    # learned (optional)
    "deeprank_ab", "abag_rank",
    # oracle / labels
    "dockq", "epitope_jaccard", "interface_lddt", "cdr_h3_rmsd",
]


def _per_sample_pss(pairwise):
    """Per-sample PSS = mean pairwise interface DockQ over pairs involving sample i."""
    m = pairwise.get("matrix", [])
    n = pairwise.get("n_samples", 0)
    if not m or n == 0:
        return [0.0] * n
    s = [0.0] * n
    c = [0] * n
    for row in m:
        i, j, dq = row["i"], row["j"], row["dockq"]
        s[i] += dq; c[i] += 1
        s[j] += dq; c[j] += 1
    return [s[i] / c[i] if c[i] else 0.0 for i in range(n)]


def _native_confidences(fold_dir):
    """Return list of per-rank native confidence dicts from results.json all_runs."""
    with open(fold_dir / "results.json") as f:
        r = json.load(f)
    assert isinstance(r, list) and len(r) == 1
    return r[0]["all_runs"]


def _score_one_fold(fold_dir, target, gen, labels_path, with_deeprank, with_abagrank,
                    deeprank_venv, abagrank_dir):
    """Return list of row dicts for one fold."""
    with open(labels_path) as f:
        labels = json.load(f)
    all_runs = _native_confidences(Path(fold_dir))
    samples = labels["samples"]
    n = labels["n_samples"]
    if not (len(all_runs) == len(samples) == n):
        raise RuntimeError(f"length mismatch: all_runs={len(all_runs)} "
                           f"samples={len(samples)} n={n}")
    pss = _per_sample_pss(labels["pairwise_matrix"])

    # Optional learned rankers (run once per fold, merge by rank)
    deeprank_scores = {}
    if with_deeprank:
        deeprank_scores = _run_deeprank(fold_dir, target, gen, deeprank_venv)
    abagrank_scores = {}
    if with_abagrank:
        abagrank_scores = _run_abagrank(fold_dir, target, gen, abagrank_dir)

    rows = []
    for k in range(n):
        run = all_runs[k]
        s = samples[k]
        pm = s.get("pae_metrics", {}) or {}
        dq = s.get("dockq", {}) or {}
        cdr = (s.get("cdr_rmsd", {}) or {}).get("cdrs", {}) or {}
        rows.append(OrderedDict([
            ("target", target), ("gen", gen), ("rank", k),
            ("iptm", run.get("iptm")), ("ptm", run.get("ptm")),
            ("ranking_score", run.get("confidence_score")),
            ("complex_plddt", run.get("complex_plddt")),
            ("pdockq2", pm.get("pdockq2")), ("ipsae", pm.get("ipsae")),
            ("anticonf", pm.get("anticonf")),
            ("pss", pss[k] if k < len(pss) else None),
            ("deeprank_ab", deeprank_scores.get(k)),
            ("abag_rank", abagrank_scores.get(k)),
            ("dockq", _scalar(dq, "dockq")),
            ("epitope_jaccard", _scalar(s.get("epitope_jaccard"), "epitope_jaccard")),
            ("interface_lddt", _scalar(s.get("interface_lddt"), "interface_lddt")),
            ("cdr_h3_rmsd", cdr.get("H3")),
        ]))
    return rows


def _run_deeprank(fold_dir, target, gen, deeprank_venv):
    """Batched DeepRank-Ab: build one ensemble PDB from all 50 sample CIFs,
    run `deeprank-ab-predict` ONCE (ESM loaded once via dedup), parse the
    output CSV -> {rank: predicted_dockq}. ~2-4 min/fold vs ~30-50 min per-sample.
    """
    wrapper = ROOT / "scripts" / "abag_xm_deeprank_batch.py"
    py = os.path.join(deeprank_venv, "bin", "python3")
    out_json = tempfile.mktemp(prefix=f"deeprank_{target}_{gen}_", suffix=".json")
    r = subprocess.run([py, str(wrapper),
                        "--fold_dir", str(fold_dir),
                        "--target", target, "--gen", gen,
                        "--out_json", out_json,
                        "--deeprank_venv", deeprank_venv],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[deeprank] {target}/{gen} batch wrapper failed rc={r.returncode}",
              file=sys.stderr)
        print(r.stderr[-1200:], file=sys.stderr)
        return {}
    try:
        with open(out_json) as f:
            scores = json.load(f)
        return {int(k): float(v) for k, v in scores.items()}
    except Exception as e:
        print(f"[deeprank] {target}/{gen} parse failed: {e}", file=sys.stderr)
        return {}


def _run_abagrank(fold_dir, target, gen, abagrank_dir):
    """Run the ABAG-Rank adapter on a fold; return {rank: model_predicted_dockq}."""
    yaml = ROOT / "examples" / "abag_xm" / f"{target}.yaml"
    h5 = tempfile.mktemp(suffix=".h5", prefix=f"abagrank_{target}_{gen}_")
    scores_json = tempfile.mktemp(suffix=".json", prefix=f"abagrank_sc_{target}_{gen}_")
    cmd = [sys.executable, str(ROOT / "scripts" / "abag_xm_abagrank_adapter.py"),
           "--fold_dir", str(fold_dir), "--target", target, "--yaml", str(yaml),
           "--out_h5", h5, "--run_inference", "--out_scores", scores_json,
           "--abagrank_dir", abagrank_dir]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-2000:])
        return {}
    if os.path.exists(scores_json):
        with open(scores_json) as f:
            d = json.load(f)
        # keys are "seed-0_sample-k"
        out = {}
        for sk, v in d.items():
            try:
                k = int(sk.split("sample-")[-1])
                out[k] = v
            except Exception:
                pass
        return out
    return {}


def _scalar(d, key):
    """Extract a scalar from a label field that may be a dict or already a scalar."""
    if isinstance(d, dict):
        return d.get(key)
    return d


def _all_completed_folds():
    """Yield (fold_dir, target, gen_id, labels_path) for every labeled fold.

    Labels filenames use the UNDERSCORE generator dir name
    (e.g. protenix_v2_21av.json, opendde_abag_21av.json). The fold dir is
    tier_a/<gen_dir>/<gen_dir>_results_<target>. The output gen id uses the
    dash form (protenix-v2) for consistency with progress.jsonl.
    """
    for lj in sorted(LABELS_DIR.glob("*.json")):
        stem = lj.stem
        idx = stem.rfind("_")
        if idx < 0:
            continue
        gen_dir, target = stem[:idx], stem[idx + 1:]
        # fold dir uses the model SHORT name (protenix/boltz2/opendde), not the
        # gen dir name (protenix_v2/opendde_abag) — glob to be robust.
        matches = list((TIERA_DIR / gen_dir).glob(f"*_results_{target}"))
        if len(matches) != 1:
            continue
        gen_id = GEN_DIRS.get(gen_dir, gen_dir)  # underscore -> dash
        yield matches[0], target, gen_id, lj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dir")
    ap.add_argument("--target")
    ap.add_argument("--gen")
    ap.add_argument("--labels")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--with_deeprank", action="store_true")
    ap.add_argument("--with_abagrank", action="store_true")
    ap.add_argument("--deeprank_venv",
                    default=os.path.expanduser("~/.deeprank_ab_venv"))
    ap.add_argument("--abagrank_dir",
                    default=os.path.expanduser("~/ABAG-Rank"))
    args = ap.parse_args()

    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    folds = []
    if args.all:
        folds = list(_all_completed_folds())
    else:
        if not (args.fold_dir and args.target and args.gen and args.labels):
            ap.error("--fold_dir/--target/--gen/--labels required (or --all)")
        folds = [(Path(args.fold_dir), args.target, args.gen, Path(args.labels))]

    print(f"[ranker_scores] scoring {len(folds)} folds; "
          f"deeprank={args.with_deeprank} abagrank={args.with_abagrank}")
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            w.writeheader()
        for fold_dir, target, gen, labels_path in folds:
            try:
                rows = _score_one_fold(fold_dir, target, gen, labels_path,
                                       args.with_deeprank, args.with_abagrank,
                                       args.deeprank_venv, args.abagrank_dir)
                for r in rows:
                    w.writerow(r)
                f.flush()
                print(f"[ranker_scores] {target} {gen}: {len(rows)} rows")
            except Exception as e:
                print(f"[ranker_scores] ERROR {target} {gen}: {e}", file=sys.stderr)
    print(f"[ranker_scores] wrote {args.out}")


if __name__ == "__main__":
    main()
