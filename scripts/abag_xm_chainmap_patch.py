#!/usr/bin/env python3
"""Patch labels for folds flagged by the chain-map audit (addendum A3).

The 9q1l-class bug: `_build_seq_map`'s old CA-count fallback mapped a declared
native light chain to the model's HEAVY chain on sequence-less
OpenDDE/Protenix CIFs, scoring every sample of the fold as a false failure
(shipped max DockQ 0.094 vs corrected 0.96 on the declared interface). The
fixed `abag_xm_dockq_interface.py` derives sequences from _atom_site and
matches by alignment identity; single-copy folds reproduce shipped values
bit-for-bit (21du/9u5r regression).

This driver recomputes the dockq block of every sample of each flagged fold
with the SAME declared chains through the fixed script, updates the labels
JSONs in place (backup .bak_chainmap), then syncs the dockq projection in
ranker_scores.csv. Nothing else is touched; other label blocks
(interface_lddt, cdr_rmsd, rankers) are chain-map-independent... except the
ranker columns that read the same model chains, which the release-table
rebuild re-derives from the patched JSONs.

    python3 abag_xm_chainmap_patch.py recompute --folds opendde_abag_9q1l protenix_v2_9q1l [--workers 4]
    python3 abag_xm_chainmap_patch.py sync_csv [--csv ~/abag_xm/tier_a/ranker_scores.csv]
"""
import argparse, csv, json, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
LABELS_DIR = Path.home() / "abag_xm" / "tier_a" / "labels"
GT_DIR = Path.home() / "abag_xm" / "ground_truth"
DIR_TO_GEN = {"protenix_v2": "protenix-v2", "boltz2": "boltz2",
              "opendde_abag": "opendde-abag"}

DOCKQ_COLS = ("dockq",)


def _split_stem(stem):
    gen_dir = next((d for d in DIR_TO_GEN if stem.startswith(d + "_")), None)
    if gen_dir is None:
        raise ValueError(f"unrecognised generator prefix: {stem}")
    return gen_dir, stem[len(gen_dir) + 1:]


def _rerun(cif, native, chain1, chain2):
    r = subprocess.run([sys.executable, str(SCRIPTS / "abag_xm_dockq_interface.py"),
                        cif, native, chain1, chain2],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {"_error": (r.stderr.strip() or r.stdout.strip())[:400]}
    try:
        return json.loads(r.stdout)
    except Exception:
        return {"_raw": r.stdout.strip()[:400]}


def cmd_recompute(a):
    jobs = []  # (labels_path, sample_idx, cif, native, chain1, chain2)
    for stem in a.folds:
        f = LABELS_DIR / f"{stem}.json"
        gen_dir, target = _split_stem(stem)
        d = json.loads(f.read_text())
        native = GT_DIR / f"{target}.cif"
        for i, s in enumerate(d.get("samples", [])):
            blk = s.get("dockq") or {}
            if "model_chain1" not in blk:
                continue  # unresolved/no-interface blocks have no assignment to fix
            cif = Path(blk.get("model", ""))
            if not cif.exists():
                cif = Path(s["cif"])
            jobs.append((f, i, str(cif), str(native),
                         blk["chain1"], blk["chain2"]))
    print(f"[chainmap_patch] {len(jobs)} samples to recompute across "
          f"{len({j[0] for j in jobs})} label files")

    def work(j):
        f, i, cif, native, c1, c2 = j
        return j, _rerun(cif, native, c1, c2)

    results = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for j, block in ex.map(work, jobs):
            results[(j[0], j[1])] = block

    by_file = {}
    for (f, i), block in results.items():
        by_file.setdefault(f, []).append((i, block))
    bad = 0
    for f, patches in by_file.items():
        d = json.loads(f.read_text())
        for i, block in patches:
            if "_error" in block or "_raw" in block:
                bad += 1
                continue  # leave the old block; inspect before shipping
            old = d["samples"][i]["dockq"]
            merged = {**old, **block}
            d["samples"][i]["dockq"] = merged
        bak = f.with_suffix(f.suffix + ".bak_chainmap")
        if not bak.exists():
            shutil.copy2(f, bak)
        f.write_text(json.dumps(d, indent=2))
        dq = [s["dockq"]["dockq"] for s in d["samples"]
              if isinstance(s.get("dockq"), dict) and "dockq" in s["dockq"]]
        ok = sum(1 for x in dq if x is not None and x >= 0.23)
        print(f"[chainmap_patch] {f.stem}: {ok}/{len(dq)} samples DockQ>=0.23 "
              f"after patch (max {max(dq):.3f})")
    print(f"[chainmap_patch] {bad} samples failed to recompute (left untouched)")
    return 1 if bad else 0


def cmd_sync_csv(a):
    csv_path = Path(a.csv)
    proj = {}  # (target, gen, rank) -> dockq projection
    for f in sorted(LABELS_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        try:
            gen_dir, target = _split_stem(f.stem)
        except ValueError:
            continue  # sibling-workstream labels (esmfold2); not in the release csv
        gen = DIR_TO_GEN[gen_dir]
        for s in d.get("samples", []):
            blk = s.get("dockq") or {}
            proj[(target, gen, s.get("rank"))] = {
                "dockq": blk.get("dockq"),
                "fnat": blk.get("fnat"),
                "irmsd": blk.get("iRMSD"),
                "lrmsd": blk.get("LRMSD"),
            }
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys())
    missing = [c for c in DOCKQ_COLS if c not in fields]
    if missing:
        print(f"[chainmap_patch] csv lacks columns {missing}; aborting")
        return 1
    changed = set()
    for r in rows:
        p = proj.get((r["target"], r["gen"], int(r["rank"])))
        if p is None:
            continue
        for col in DOCKQ_COLS:
            v = p[col]
            cell = "" if v is None else str(v)
            if r[col] != cell:
                r[col] = cell
                changed.add((r["target"], r["gen"]))
    bak = csv_path.with_suffix(".bak_chainmap")
    if not bak.exists():
        shutil.copy2(csv_path, bak)
    tmp = csv_path.with_suffix(".tmp_sync")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(csv_path)
    print(f"[chainmap_patch] csv dockq cells updated in "
          f"{len(changed)} (target,gen) folds: {sorted(changed)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recompute")
    r.add_argument("--folds", nargs="+", required=True)
    r.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("sync_csv")
    s.add_argument("--csv", default=str(Path.home() / "abag_xm/tier_a/ranker_scores.csv"))
    a = ap.parse_args()
    return cmd_recompute(a) if a.cmd == "recompute" else cmd_sync_csv(a)


if __name__ == "__main__":
    sys.exit(main())
