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

GEN_DIRS = {"protenix_v2": "protenix-v2", "boltz2": "boltz2", "opendde_abag": "opendde-abag",
            "esmfold2": "esmfold2"}

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
                    deeprank_venv, abagrank_dir, deeprank_precomputed=None):
    with open(labels_path) as f:
        labels = json.load(f)
    all_runs = _native_confidences(Path(fold_dir))
    samples = labels["samples"]
    n = labels["n_samples"]
    if not (len(all_runs) == len(samples) == n):
        raise RuntimeError(f"length mismatch: all_runs={len(all_runs)} "
                           f"samples={len(samples)} n={n}")
    # Index the labels by their OWN rank, never by list position. Every other per-sample source
    # here is rank-ordered -- all_runs is written rank 0..N-1, pairwise_matrix builds its cifs with
    # `for k in range(1, n)`, deeprank_batch returns {rank: score} -- but labels.py::_samples()
    # sorts the model files by FILENAME, so its list runs 0, 1, 10, 11, ..., 19, 2, 20, ... The old
    # code paired all_runs[k] with samples[k], which is only correct for ranks 0 and 1 and pairs
    # every other confidence with a different structure's labels.
    #
    # It hid because it damages exactly the quantity nobody had computed yet. Between-target signal
    # survives a within-target permutation untouched, so global Spearman stayed a healthy 0.79 while
    # the per-target median -- the number this dataset exists to report -- collapsed to 0.06, which
    # reads as "confidence cannot rank samples" rather than as a join bug.
    # all_runs is written rank-ordered, but keyed by its own `rank` field it does not have to be.
    runs_by_rank = {r.get("rank", i): r for i, r in enumerate(all_runs)}
    by_rank = {}
    for i, s in enumerate(samples):
        r = s.get("rank")
        if r is None:
            raise RuntimeError(f"{target}/{gen}: label sample {i} has no rank; cannot join by rank")
        by_rank[r] = s
    missing = [k for k in range(n) if k not in by_rank]
    if missing:
        raise RuntimeError(f"{target}/{gen}: label samples missing ranks {missing[:5]} "
                           f"(have {len(by_rank)} of {n})")
    pss = _per_sample_pss(labels["pairwise_matrix"])

    # A learned ranker that fails returns {} and the fold is still written -- 50 complete-looking
    # rows with one empty column, indistinguishable from "that ranker was not requested". ABAG-Rank
    # did exactly this for want of h5py: the run reported success and the downstream signal script
    # said "learned rankers: NOT run yet". Requested-but-empty is an error, and it says so.
    deeprank_scores = {}
    if with_deeprank:
        # Scored up front for the whole work list, batched -- see _run_deeprank_batched. A fold
        # missing from that dict is a real hole and falls through to the empty-column error below.
        deeprank_scores = (deeprank_precomputed or {}).get((target, gen), {})
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

    # An all-empty column is loud (above); a PARTIAL one used to be silent. Under CPU contention
    # ABAG-Rank returned a short score map and the missing ranks became blank cells that look
    # exactly like "not requested" -- 153 of 164 folds landed that way on 2026-07-30 and were only
    # noticed because _needs_score re-queued them. Coverage is checked, so a hole names itself.
    for _col, _sc in (("deeprank_ab", deeprank_scores), ("abag_rank", abagrank_scores)):
        if not _sc:
            continue
        _missing = [k for k in range(n) if k not in _sc]
        if _missing:
            _PARTIAL_LEARNED.append((target, gen, _col, tuple(_missing)))
            print(f"[ranker_scores] !! {target}/{gen}: {_col} covered {len(_sc)}/{n} samples -- "
                  f"ranks {_missing[:6]} will be EMPTY", file=sys.stderr, flush=True)

    rows = []
    for k in range(n):
        run = runs_by_rank.get(k, {})
        s = by_rank[k]
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
# Selftests point this at a temp dir; production uses TIERA_DIR / "deeprank_json_cache".
_DEEPRANK_CACHE_DIR = None
# (target, gen, column) for every learned ranker that was asked for and returned nothing.
_EMPTY_LEARNED = []
# (target, gen, column, missing_ranks) where a learned ranker scored SOME samples but not all.
_PARTIAL_LEARNED = []


def _device_folds_running():
    """True if a tt_bio predict is in flight on this host."""
    try:
        r = subprocess.run(["pgrep", "-f", "tt_bio.main predict"],
                           capture_output=True, text=True)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


# DeepRank-Ab cannot be confined from the outside, which is why the guard below schedules it away
# from the folding window instead of throttling it. Measured on qb1 (32 cores, 4 folds in flight):
# scoring one fold took the load average from 15.6 to 29.7 with ~38 processes, and past 45 on a
# second attempt with ~70. Every containment route was tried and none held: deeprank-ab-predict
# exposes only chain IDs (no worker/thread/batch flag), its pool ignores OMP/MKL/OpenBLAS caps, and
# `taskset` is discarded exactly where it matters -- the wrapper inherits the mask (verified:
# affinity 0-3) but the deeprank-ab-predict grandchild comes up 0-31, resetting its own affinity as
# it spawns the workers. An unprivileged `systemd-run --user --scope -p AllowedCPUs=0-3` was not
# enforced either. A delegated cpuset cgroup would hold, but that needs root.
def _run_deeprank_batched(folds, deeprank_venv, max_batch=5):
    """Score EVERY fold's DeepRank-Ab column in batched CLI calls. {(target, gen): {rank: score}}.

    One invocation per fold pays the model-load cost 492 times. Measured on qb1:
    T(n) = 79.3 s + 1.88 s/model over n = 5/25/50 models, so 46% of a 50-model fold is fixed
    overhead. Handing the wrapper the whole work list instead drops 173 s/fold to a measured
    81.9 s/fold at 5 folds per call -- the slab goes from 23.7 h to 11.2 h on one host.

    The wrapper does the chain-signature grouping, because the generators label chains differently
    and getting that wrong scores models against a chain that does not exist. Callers just say what
    they want scored.
    """
    if not folds:
        return {}
    wrapper = ROOT / "scripts" / "abag_xm_deeprank_batch.py"
    py = os.path.join(deeprank_venv, "bin", "python3")
    # Per-fold JSONs go to a STABLE cache, not a per-run temp dir. The DeepRank phase holds every
    # score in memory and the CSV is only rewritten after it completes, so a crash/reboot mid-phase
    # (qb1 lost one to unattended-upgrades) used to orphan hours of scoring: the JSONs survived in
    # /tmp but the relaunch built a fresh work dir and recomputed everything. A cached JSON is
    # reused only when it parses to a non-empty dict (a kill mid-write leaves a truncated file,
    # which just re-scores) AND is newer than every fold input (results.json + CIFs), so a refold
    # is never served a stale score.
    cache = Path(_DEEPRANK_CACHE_DIR) if _DEEPRANK_CACHE_DIR else \
        TIERA_DIR / "deeprank_json_cache"
    cache.mkdir(parents=True, exist_ok=True)

    def _cache_hit(path, fold_dir):
        try:
            with open(path) as f:
                scores = {int(k): float(v) for k, v in json.load(f).items()}
            if not scores:
                return None
            inputs = [Path(fold_dir) / "results.json"] + \
                list((Path(fold_dir) / "structures").glob("*.cif"))
            newest_in = max(os.path.getmtime(p) for p in inputs if p.exists())
            return scores if os.path.getmtime(path) >= newest_in else None
        except Exception:
            return None

    out = {}
    manifest = []
    for fold_dir, target, gen in folds:
        cj = cache / f"{target}__{gen}.json"
        hit = _cache_hit(cj, fold_dir)
        if hit is not None:
            out[(target, gen)] = hit
        else:
            manifest.append({"target": target, "gen": gen, "fold_dir": str(fold_dir),
                             "out_json": str(cj)})
    cached = len(out)
    if cached:
        print(f"[deeprank] {cached}/{len(folds)} folds reused from {cache}", flush=True)

    # The guard belongs HERE, not ahead of the cache pass. A cache hit is a file read and cannot
    # starve a folding card, but the old placement returned {} for the WHOLE work list the moment
    # any device fold appeared -- blanking 153 already-cached columns and leaving a CSV that looks
    # like the ranker was never run. Only uncached folds are real DeepRank-Ab work, so only they
    # wait for the cards.
    if manifest and _device_folds_running() and not _FORCE_DEEPRANK:
        print(f"[deeprank] REFUSING to score {len(manifest)} uncached fold(s): device folds are in "
              "flight on this host and DeepRank-Ab cannot be contained (see the comment above "
              "_run_deeprank_batched). Score after the cards idle, or pass --force_deeprank if you "
              "accept starving them. Cached folds are unaffected.", file=sys.stderr)
        manifest = []

    if manifest:
        work = tempfile.mkdtemp(prefix="deeprank_manifest_")
        man_path = os.path.join(work, "manifest.json")
        with open(man_path, "w") as f:
            json.dump(manifest, f)

        print(f"[deeprank] scoring {len(manifest)} folds in batches of {max_batch}", flush=True)
        r = subprocess.run([py, str(wrapper), "--manifest", man_path,
                            "--max_batch", str(max_batch),
                            "--deeprank_venv", deeprank_venv],
                           capture_output=True, text=True)
        for ln in (r.stdout or "").splitlines():
            if ln.startswith("[deeprank-batch]"):
                print("  " + ln, flush=True)
        if r.returncode != 0:
            # Partial output is still usable: the wrapper writes per-fold and only skips the folds
            # it could not complete. Read what landed and let the empty check name the holes.
            print(f"[deeprank] wrapper exited rc={r.returncode} -- reading whatever landed",
                  file=sys.stderr)
            print((r.stderr or "")[-1200:], file=sys.stderr)

        for m in manifest:
            try:
                with open(m["out_json"]) as f:
                    out[(m["target"], m["gen"])] = \
                        {int(k): float(v) for k, v in json.load(f).items()}
            except Exception:
                continue
    print(f"[deeprank] got scores for {len(out)}/{len(folds)} folds", flush=True)
    return out


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
    ap.add_argument("--deeprank_batch", type=int, default=5,
                    help="folds per DeepRank-Ab CLI invocation (default 5; measured 81.9 s/fold "
                         "against 173 s one-at-a-time)")
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

    rows_by_pair = {}
    for r in surviving_rows:
        rows_by_pair.setdefault((r.get("target"), r.get("gen")), []).append(r)

    def _needs_score(pair):
        rows = rows_by_pair.get(pair)
        if not rows:
            return True
        # Rows from a physics-only pass (or a ranker run whose env was broken, cf. qb2's
        # missing pyyaml leaving every learned cell empty) are holes, not scores. Without
        # this, "already-scored" locked the requested column empty forever.
        if args.with_deeprank and any(not r.get("deeprank_ab") for r in rows):
            return True
        if args.with_abagrank and any(not r.get("abag_rank") for r in rows):
            return True
        return False

    new_folds = [fd for fd in folds if _needs_score((fd[1], fd[2]))]
    rescored = {(fd[1], fd[2]) for fd in new_folds} & set(rows_by_pair)
    if rescored:
        surviving_rows = [r for r in surviving_rows
                          if (r.get("target"), r.get("gen")) not in rescored]
    skipped = len(folds) - len(new_folds)

    msg = (f"[ranker_scores] scoring {len(new_folds)} folds (skipped {skipped} fully-scored"
           f", re-scoring {len(rescored)} with missing requested columns)")
    if pruned:
        msg += f"; pruned {pruned} orphaned rows"
    if args.clean:
        msg += "; --clean (full rebuild)"
    print(msg + f"; deeprank={args.with_deeprank} abagrank={args.with_abagrank}")

    # DeepRank-Ab up front for every fold that needs it, in batched CLI calls. Doing it inside the
    # per-fold loop paid the model-load cost once per fold, which is 46% of each invocation.
    deeprank_precomputed = {}
    if args.with_deeprank and new_folds:
        deeprank_precomputed = _run_deeprank_batched(
            [(fd[0], fd[1], fd[2]) for fd in new_folds], args.deeprank_venv, args.deeprank_batch)

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
                                       args.deeprank_venv, args.abagrank_dir,
                                       deeprank_precomputed)
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
    if _PARTIAL_LEARNED:
        _by_col = {}
        for _t, _g, _c, _mr in _PARTIAL_LEARNED:
            _by_col.setdefault(_c, []).append(f"{_t}/{_g}({len(_mr)})")
        for _c, _f in sorted(_by_col.items()):
            print(f"[ranker_scores] !! {_c}: PARTIAL for {len(_f)} fold(s) -- blank cells at the "
                  f"unscored ranks. e.g. {', '.join(_f[:4])}", file=sys.stderr)


if __name__ == "__main__":
    main()
