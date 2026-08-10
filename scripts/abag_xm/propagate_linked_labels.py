#!/usr/bin/env python3
"""Propagate DockQ labels across seed-nested rungs with provably identical content.

The campaign's ladder is seed-nested: rung 2N chunk j carries the same seed block as rung
N chunk j, and the galaxy link gate attests the engine tree is unchanged, so the CIFs are
bit-identical. A label depends only on (structures, GT, yaml, scorer), so a labeled dir's
labels.json is valid for ANY same-model same-target dir carrying the same per-rank CIF md5
map. The p28/p29 windows land ~1600 linked chunks as full duplicate dirs; propagation saves
~20-25h of label CPU per window and shortens the post-window label tail that gates the
refresh.

Modes:
  audit      -- group all fold dirs by content key; where >=2 dirs in a group are BOTH
                labeled, per-sample DockQ must agree exactly. Any disagreement means
                scorer-era mixing or nondeterminism -- a link-premise violation worth
                stopping for. Run pre-window.
  propagate  -- for each unlabeled fold dir whose content key matches a labeled dir (same
                model, same target), copy labels.json with a _propagated_from note.
                Never overwrites. Run post-harvest, before the labelers drain the rung.

usage: propagate_linked_labels.py [--base ~/abag_xm/deepn/galaxy] [--dry-run] audit|propagate
"""
import argparse, hashlib, json, sys
from collections import defaultdict
from pathlib import Path

MODELS = ("opendde", "protenix", "boltz2", "esmfold2")


def md5key(rd):
    # Keyed on (filename, md5) pairs, not on the bare md5 set. A label is joined to a
    # structure BY RANK, and rank lives in the filename (<target>_model_<rank>.cif), so
    # set equality alone would license propagating labels between two dirs that hold the
    # same structures under permuted ranks -- every DockQ silently attached to the wrong
    # selector. Link-copied dirs are hardlinks and never permute, so this costs nothing
    # and removes the gap between what the premise says and what it checks.
    st = rd / "structures"
    h = sorted((p.name, hashlib.md5(p.read_bytes()).hexdigest()) for p in st.glob("*.cif"))
    return tuple(h) if h else None


def getdq(s):
    d = s.get("dockq")
    return d.get("dockq") if isinstance(d, dict) else d


def results_dir(out_dir):
    rds = list(out_dir.glob("*_results_*"))
    return rds[0] if rds else None


def scan(base):
    """(model, target, suffix, out_dir, key, labels_or_None) for every fold dir."""
    rows = []
    for m in MODELS:
        mdir = base / m
        if not mdir.is_dir():
            continue
        for out_dir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            rd = results_dir(out_dir)
            if rd is None:
                continue
            target = rd.name.split("results_")[1]
            key = md5key(rd)
            if key is None:
                continue
            lj = out_dir / "labels.json"
            labels = None
            if lj.exists():
                try:
                    labels = json.loads(lj.read_text())
                except Exception:
                    pass
            rows.append((m, target, out_dir.name, out_dir, key, labels))
    return rows


def do_audit(rows):
    groups = defaultdict(list)
    for m, t, name, out_dir, key, labels in rows:
        groups[(m, t, key)].append((name, labels))
    n_groups = n_multi = n_bad = 0
    for (m, t, key), members in sorted(groups.items()):
        labeled = [(n, l) for n, l in members if l]
        if len(labeled) < 2:
            continue
        n_multi += 1
        ref_name, ref = labeled[0]
        ref_dq = [getdq(s) for s in ref.get("samples", [])]
        for name, lab in labeled[1:]:
            dq = [getdq(s) for s in lab.get("samples", [])]
            n_groups += 1
            if len(dq) != len(ref_dq) or any(
                    (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9)
                    for a, b in zip(dq, ref_dq)):
                n_bad += 1
                print(f"MISMATCH {m}/{t}: {ref_name} vs {name}")
    print(f"audit: {n_groups} labeled-dir comparisons across {n_multi} shared-content "
          f"groups, {n_bad} mismatches")
    return n_bad


def do_propagate(rows, dry):
    by_key = defaultdict(list)
    for m, t, name, out_dir, key, labels in rows:
        if labels:
            by_key[(m, t, key)].append((name, labels))
    n = 0
    for m, t, name, out_dir, key, labels in rows:
        if labels is not None:
            continue
        srcs = by_key.get((m, t, key))
        if not srcs:
            continue
        src_name, src_labels = srcs[0]
        n += 1
        if dry:
            print(f"would propagate {m}/{name} <- {src_name}")
            continue
        out = dict(src_labels)
        out["_propagated_from"] = src_name
        (out_dir / "labels.json").write_text(json.dumps(out))
        print(f"propagated {m}/{name} <- {src_name}")
    print(f"propagate: {n} dirs {'(dry-run)' if dry else 'written'}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("audit", "propagate"))
    ap.add_argument("--base", type=Path, default=Path.home() / "abag_xm" / "deepn" / "galaxy")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rows = scan(a.base)
    print(f"scanned {len(rows)} fold dirs under {a.base}", flush=True)
    bad = do_audit(rows) if a.mode == "audit" else do_propagate(rows, a.dry_run)
    sys.exit(1 if a.mode == "audit" and bad else 0)


if __name__ == "__main__":
    main()
