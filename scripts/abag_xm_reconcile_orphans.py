#!/usr/bin/env python3
"""Record folds whose artifacts are complete but whose driver died before logging them.

`progress.jsonl` is written by the generate.py driver when a fold returns. If the driver dies
while a fold is running -- the fold keeps going, it is a separate process -- the fold completes,
writes its CIFs and results.json, and is never recorded. `done_pairs()` then treats it as
outstanding and it gets folded a second time. That cost this campaign ~3.1 card-hours once.

Everything written is derived from the artifacts on disk, never inferred. `wall_s`, `device` and
`host_threads` are null: the driver held them and it is gone, and a fabricated timing would poison
the rate tables the stall scan and the adaptive timeout are built on. Every record is flagged
`recovered: true`.

PROVENANCE is the delicate part, and getting it wrong in either direction is expensive. Claim this
worktree's engine tree for a fold some earlier driver ran and the claim may be false. Leave it null
and the fold is unpublishable, so the resume refolds it and this script has saved nothing -- which
is exactly what it did before: it wrote `tt_bio_commit: null`, and once "done" was tightened to mean
"defensible", every record it produced was rejected. So establish provenance when it can be
established: a fold loads `tt_bio/` at startup and writes its first artifact strictly later, so if
no file under `tt_bio/` has changed since that artifact appeared, the tree on disk now IS the tree
the fold ran. Otherwise say so and let it be refolded.

    python3 scripts/abag_xm_reconcile_orphans.py            # report only
    python3 scripts/abag_xm_reconcile_orphans.py --write     # append the records
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _gen():
    argv, sys.argv = sys.argv, ["abag_xm_generate"]
    try:
        spec = importlib.util.spec_from_file_location(
            "abag_xm_generate", ROOT / "scripts" / "abag_xm_generate.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    finally:
        sys.argv = argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="append the records (default: report)")
    a = ap.parse_args()
    g = _gen()

    recorded_ok = set()
    if g.PROGRESS.exists():
        for line in g.PROGRESS.open():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok":
                recorded_ok.add((r["target"], r["model"]))

    found, n_written, n_unprovenanced = [], 0, 0
    for model in g.MODELS:
        out_dir = g.OUT_BASE / model.replace("-", "_")
        if not out_dir.is_dir():
            continue
        prefix = g.RESULT_PREFIX[model]
        for rd in sorted(out_dir.glob(f"{prefix}_results_*")):
            target = rd.name[len(f"{prefix}_results_"):]
            if (target, model) in recorded_ok:
                continue
            entry = g.results_entry(rd)
            if entry is None or not (rd / "structures").is_dir():
                continue
            # Count exactly as the harness does -- the loose `*_pae.npz` glob this used to
            # carry is what produced 8 records claiming 51 PAEs on a 50-sample campaign.
            cifs, paes = g.count_artifacts(rd, target)
            # And apply the harness's own failed-results cross-check, which this used to skip
            # entirely: a fold whose results.json says failed, or which silently dropped
            # samples, was being recorded `ok` from its leftover files.
            n_runs = len(entry.get("all_runs") or [])
            complete = (entry.get("status") == "ok"
                        and n_runs == g.N_SAMPLES
                        and len(cifs) == g.N_SAMPLES
                        and len(paes) == g.N_SAMPLES)
            if not complete:
                found.append((target, model, len(cifs), len(paes),
                              f"INCOMPLETE (results={entry.get('status')} runs={n_runs}) "
                              f"- not recorded"))
                continue
            # Oldest artifact: the fold had already loaded tt_bio/ before this was written.
            oldest = min(cifs + paes, key=lambda p: p.stat().st_mtime)
            tree = g._head_tree() if g.tree_unchanged_since(oldest) else None
            rec = {
                "target": target, "model": model,
                "wall_s": None, "device": None, "host_threads": None,
                "n_samples": g.N_SAMPLES, "mps": g.MPS,
                "host": g._HOST,
                "tt_bio_commit": g._head_commit() if tree else None,
                "tt_bio_tree": tree,
                "msa_sha": g._msa_sha(target)[0],
                "paired_msa": False, "status": "ok",
                "n_cifs": len(cifs), "n_paes": len(paes),
                "result_dir": str(rd),
                "recovered": True,
                "recovered_note": "driver died before logging; fields it held are null",
            }
            if tree:
                note = "recovered, provenance established"
            else:
                note = "recovered, PROVENANCE UNSTATEABLE -> will be refolded"
                n_unprovenanced += 1
            found.append((target, model, len(cifs), len(paes), note))
            if a.write:
                with g.PROGRESS.open("a") as fp:
                    fp.write(json.dumps(rec) + "\n")
                n_written += 1

    if not found:
        print("nothing to reconcile: every complete fold already has an ok record")
        return 0
    for t, m, nc, npae, note in found:
        print(f"  {t:6s} {m:14s} cifs={nc:3d} paes={npae:3d}  {note}")
    n_ok = sum(1 for f in found if f[4].startswith("recovered"))
    print(f"\n{n_ok} recoverable ({n_ok - n_unprovenanced} with provenance, "
          f"{n_unprovenanced} without), {len(found) - n_ok} incomplete"
          f"{f' -- {n_written} WRITTEN' if a.write else ' -- report only, pass --write to append'}")
    if n_unprovenanced:
        print(f"the {n_unprovenanced} without provenance are recorded so their artifacts and "
              f"labels are not lost, but the resume will refold them: tt_bio/ has changed since "
              f"they ran, so this worktree's tree cannot honestly be attributed to them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
