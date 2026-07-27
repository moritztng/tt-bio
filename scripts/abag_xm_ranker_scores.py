#!/usr/bin/env python3
"""Phase 5 ranker-score driver for the AbAg-XM campaign.

Builds the per-sample ranker-score table for a completed (target, generator)
fold by JOINing native confidences, closed-form PAE rankers, PSS, oracle
labels, and optionally learned rankers (DeepRank-Ab, ABAG-Rank).

In --all mode the CSV is self-healing: rows whose (target,gen) no longer has
a label file are pruned, so purging labels (e.g. compromised folds) cleans the
CSV on the next run. --clean forces a full rebuild.
"""
import argparse, csv, json, os, subprocess, sys, tempfile
from pathlib import Path
from collections import OrderedDict

ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = Path.home() / "abag_xm" / "tier_a" / "labels"
TIERA_DIR = Path.home() / "abag_xm" / "tier_a"

GEN_DIRS = {"protenix_v2": "protenix-v2", "boltz2": "boltz2", "opendde_abag": "opendde-abag"}

COLUMNS = [
    "target", "gen", "rank",
    "iptm", "ptm", "ranking_score", "complex_plddt",
    "pdockq2", "ipsae", "anticonf",
    "pss",
    "deeprank_ab", "abag_rank",
    "dockq", "epitope_jaccard", "interface_lddt", "cdr_h3_rmsd",
]


def _per_sample_pss(pairwise):
    m = pairwise.get("matrix", [])
    n = pairwise.get("n_samples", 0)
    if not m or n == 0:
        return [0.0] * n
    s = [0.0] * n
    c = [0] * n
    for row in m:
        i, j, dq = row["i"], row["j"], row["dockq"]
        if dq is None:
            continue  # DockQ failed on this pair (e.g. chain mapping); skip, PSS uses available pairs
        s[i] += dq; c[i] += 1
        s[j] += dq; c[j] += 1
    return [s[i] / c[i] if c[i] else 0.0 for i in range(n)]


def _native_confidences(fold_dir):
    with open(fold_dir / "results.json") as f:
        r = json.load(f)
    assert isinstance(r, list) and len(r) == 1
    return r[0]["all_runs"]


def _score_one_fold(fold_dir, target, gen, labels_path, with_deeprank, with_abagrank,
                    deeprank_venv, abagrank_dir):
    with open(labels_path) as f:
        labels = json.load(f)
    all_runs = _native_confidences(Path(fold_dir))
    samples = labels["samples"]
    n = labels["n_samples"]
    if not (len(all_runs) == len(samples) == n):
        raise RuntimeError(f"length mismatch: all_runs={len(all_runs)} "
                           f"samples={len(samples)} n={n}")
    pss = _per_sample_pss(labels["pairwise_matrix"])

    # A learned ranker that fails returns {} and the fold is still written -- 50 complete-looking
    # rows with one empty column, indistinguishable from "that ranker was not requested". ABAG-Rank
    # did exactly this for want of h5py: the run reported success and the downstream signal script
    # said "learned rankers: NOT run yet". Requested-but-empty is an error, and it says so.
    deeprank_scores = {}
    if with_deeprank:
        deeprank_scores = _run_deeprank(fold_dir, target, gen, deeprank_venv)
        if not deeprank_scores:
            _EMPTY_LEARNED.append((target, gen, "deeprank_ab"))
            print(f"[ranker_scores] !! {target}/{gen}: deeprank_ab requested but produced NO "
                  f"scores -- that column will be empty", file=sys.stderr, flush=True)
    abagrank_scores = {}
    if with_abagrank:
        abagrank_scores = _run_abagrank(fold_dir, target, gen, abagrank_dir)
        if not abagrank_scores:
            _EMPTY_LEARNED.append((target, gen, "abag_rank"))
            print(f"[ranker_scores] !! {target}/{gen}: abag_rank requested but produced NO "
                  f"scores -- that column will be empty", file=sys.stderr, flush=True)

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


_FORCE_DEEPRANK = False
# (target, gen, column) for every learned ranker that was asked for and returned nothing.
_EMPTY_LEARNED = []


def _device_folds_running():
    """True if a tt_bio predict is in flight on this host."""
    try:
        r = subprocess.run(["pgrep", "-f", "tt_bio.main predict"],
                           capture_output=True, text=True)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def _run_deeprank(fold_dir, target, gen, deeprank_venv):
    """Score a fold with DeepRank-Ab. Run this ONLY when the host is not folding.

    Measured on qb1 (32 cores, 4 folds in flight): scoring one fold took the load average
    from 15.6 to 29.7 with ~38 processes, and to >45 on a second attempt with ~70. It
    starves generation.

    There is no way to contain it from the outside, which is why this takes no budget
    argument:
      * ``deeprank-ab-predict`` exposes only chain IDs -- no worker, thread or batch flag;
      * its pool ignores OMP/MKL/OpenBLAS caps;
      * ``taskset`` does not hold. The wrapper does inherit the mask (verified: affinity
        0-3), but the ``deeprank-ab-predict`` grandchild comes up with 0-31 -- it resets
        its own affinity, so the pin is discarded exactly where the workers are spawned;
      * an unprivileged ``systemd-run --user --scope -p AllowedCPUs=0-3`` was likewise not
        enforced on this host (the scope started, affinity inside was still 0-31).
    A delegated cpuset cgroup would hold, but that needs root. Schedule it instead.
    """
    wrapper = ROOT / "scripts" / "abag_xm_deeprank_batch.py"
    py = os.path.join(deeprank_venv, "bin", "python3")
    out_json = tempfile.mktemp(prefix=f"deeprank_{target}_{gen}_", suffix=".json")
    if _device_folds_running() and not _FORCE_DEEPRANK:
        print("[deeprank] REFUSING: device folds are in flight on this host and DeepRank-Ab "
              "cannot be contained (see _run_deeprank). Score after the cards go idle, or "
              "pass --force_deeprank if you accept starving them.", file=sys.stderr)
        return {}
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
    if isinstance(d, dict):
        return d.get(key)
    return d


def _all_completed_folds():
    for lj in sorted(LABELS_DIR.glob("*.json")):
        stem = lj.stem
        idx = stem.rfind("_")
        if idx < 0:
            continue
        gen_dir, target = stem[:idx], stem[idx + 1:]
        matches = list((TIERA_DIR / gen_dir).glob(f"*_results_{target}"))
        if len(matches) != 1:
            continue
        gen_id = GEN_DIRS.get(gen_dir, gen_dir)
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
    ap.add_argument("--force_deeprank", action="store_true",
                    help="score with DeepRank-Ab even while this host is folding. It cannot "
                         "be contained (no worker flag, ignores OMP caps, resets its own "
                         "taskset affinity) and will starve the folds.")
    ap.add_argument("--clean", action="store_true",
                    help="delete the CSV before scoring (full rebuild)")
    args = ap.parse_args()
    global _FORCE_DEEPRANK
    _FORCE_DEEPRANK = args.force_deeprank

    folds = []
    if args.all:
        folds = list(_all_completed_folds())
    else:
        if not (args.fold_dir and args.target and args.gen and args.labels):
            ap.error("--fold_dir/--target/--gen/--labels required (or --all)")
        folds = [(Path(args.fold_dir), args.target, args.gen, Path(args.labels))]

    current_pairs = {(fd[1], fd[2]) for fd in folds}

    # Self-healing prune: in --all mode, keep only existing CSV rows whose
    # (target,gen) still has a label file. Prevents stale contamination when
    # labels are purged (e.g. compromised opendde folds). --clean skips this
    # and starts fresh.
    surviving_rows = []
    pruned = 0
    if args.all and not args.clean and os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        total = 0
        try:
            with open(args.out, newline="") as f:
                for r in csv.DictReader(f):
                    total += 1
                    if (r.get("target"), r.get("gen")) in current_pairs:
                        surviving_rows.append(r)
        except Exception:
            surviving_rows = []
        pruned = total - len(surviving_rows)

    already = {(r.get("target"), r.get("gen")) for r in surviving_rows}
    new_folds = [fd for fd in folds if (fd[1], fd[2]) not in already]
    skipped = len(folds) - len(new_folds)

    msg = f"[ranker_scores] scoring {len(new_folds)} folds (skipped {skipped} already-scored)"
    if pruned:
        msg += f"; pruned {pruned} orphaned rows"
    if args.clean:
        msg += "; --clean (full rebuild)"
    print(msg + f"; deeprank={args.with_deeprank} abagrank={args.with_abagrank}")

    # --all mode: overwrite with surviving + new. single-fold: append.
    if args.all or args.clean:
        mode = "w"
        write_header = True
    else:
        mode = "a"
        write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0

    with open(args.out, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            w.writeheader()
        if mode == "w":
            for r in surviving_rows:
                w.writerow(r)
        for fold_dir, target, gen, labels_path in new_folds:
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
    if _EMPTY_LEARNED:
        by_col = {}
        for _t, _g, _c in _EMPTY_LEARNED:
            by_col.setdefault(_c, []).append(f"{_t}/{_g}")
        for _c, _folds in sorted(by_col.items()):
            print(f"[ranker_scores] !! {_c}: EMPTY for {len(_folds)} fold(s) that requested it "
                  f"-- e.g. {', '.join(_folds[:4])}", file=sys.stderr)
        print("[ranker_scores] !! the CSV is complete in every other column, so this will look "
              "like the ranker was simply never run. Fix before using it.", file=sys.stderr)


if __name__ == "__main__":
    main()
