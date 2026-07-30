#!/usr/bin/env python3
"""Recompute recoverable per-sample label blocks in place, then sync the csv.

Two failure classes in the shipped labels are harness artifacts, not data
(abag-xm-label-census.md): interface_lddt blocks carrying `_error` from the
exact-match chain-resolution bug (9ly2/9ly3/9lz2/9mz8 class, fixed by the
near-identity sliding-window match in abag_xm_interface_lddt.py), and
cdr_rmsd blocks carrying `_raw` because an ANARCI species-limit warning on
stdout broke JSON parsing (9lwc class, fixed by stdout hygiene in
abag_xm_cdr_rmsd.py). Genuine nulls (pose docked away, native unresolved)
have valid blocks and are untouched.

Usage:
    python3 scripts/abag_xm_label_patch.py recompute [--workers 4]
    python3 scripts/abag_xm_label_patch.py sync_csv [--csv ~/abag_xm/tier_a/ranker_scores.csv]
"""
import argparse, csv, json, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
LABELS_DIR = Path.home() / "abag_xm" / "tier_a" / "labels"
GT_DIR = Path.home() / "abag_xm" / "ground_truth"
YAML_DIR = ROOT / "examples" / "abag_xm"
SCRIPT_FOR = {"interface_lddt": "abag_xm_interface_lddt",
              "cdr_rmsd": "abag_xm_cdr_rmsd"}
DIR_TO_GEN = {"protenix_v2": "protenix-v2", "boltz2": "boltz2",
              "opendde_abag": "opendde-abag"}


def _split_stem(stem):
    gen_dir = next((d for d in DIR_TO_GEN if stem.startswith(d + "_")), None)
    if gen_dir is None:
        raise ValueError(f"unrecognised generator prefix: {stem}")
    return gen_dir, stem[len(gen_dir) + 1:]


def _resolve_native_yaml(d, target):
    """Labels JSONs recorded paths inside the since-collected p4 worktree; remap to the
    stable ground-truth dir and this worktree's yamls when the recorded path is gone."""
    native, yaml_ = Path(d.get("native", "")), Path(d.get("yaml", ""))
    if not native.exists():
        native = GT_DIR / f"{target}.cif"
    if not yaml_.exists():
        yaml_ = YAML_DIR / f"{target}.yaml"
    if not native.exists() or not yaml_.exists():
        raise FileNotFoundError(f"{target}: native={native} yaml={yaml_}")
    return str(native), str(yaml_)


def _broken(column, block):
    if not isinstance(block, dict):
        return False
    if column == "interface_lddt":
        return "_error" in block
    return "_raw" in block or "_error" in block


def _rerun(script, cif, native, yaml_):
    r = subprocess.run([sys.executable, str(SCRIPTS / f"{script}.py"),
                        cif, native, yaml_], capture_output=True, text=True)
    if r.returncode != 0:
        return {"_error": r.stderr.strip()[:400]}
    try:
        return json.loads(r.stdout)
    except Exception:
        return {"_raw": r.stdout.strip()[:400]}


def cmd_recompute(a):
    jobs = []  # (labels_path, sample_idx, column, cif, native, yaml)
    for f in sorted(LABELS_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        _, target = _split_stem(f.stem)
        try:
            native, yaml_ = _resolve_native_yaml(d, target)
        except FileNotFoundError:
            continue  # nothing to patch without inputs; reported by the census, not here
        for i, s in enumerate(d.get("samples", [])):
            for col in SCRIPT_FOR:
                if _broken(col, s.get(col)):
                    jobs.append((f, i, col, s["cif"], native, yaml_))
    print(f"[label_patch] {len(jobs)} broken sample blocks across "
          f"{len({j[0] for j in jobs})} label files")

    def work(j):
        f, i, col, cif, native, yaml_ = j
        return j, _rerun(SCRIPT_FOR[col], cif, native, yaml_)

    results = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for j, block in ex.map(work, jobs):
            results[(j[0], j[1], j[2])] = block

    by_file = {}
    for (f, i, col), block in results.items():
        by_file.setdefault(f, []).append((i, col, block))
    still_broken, recovered = 0, 0
    for f, patches in by_file.items():
        d = json.loads(f.read_text())
        for i, col, block in patches:
            d["samples"][i][col] = block
            if _broken(col, block):
                still_broken += 1
            else:
                recovered += 1
        bak = f.with_suffix(f.suffix + ".bak_labelpatch")
        if not bak.exists():
            shutil.copy2(f, bak)
        f.write_text(json.dumps(d, indent=2))
    print(f"[label_patch] recovered {recovered} blocks; {still_broken} still broken "
          f"(rerun did not clear the signature -- inspect before shipping)")
    return 1 if still_broken else 0


def _scalar(block, key):
    return block.get(key) if isinstance(block, dict) else block


def cmd_sync_csv(a):
    csv_path = Path(a.csv)
    # project the two columns from the labels JSONs for every fold
    proj = {}  # (target, gen, rank) -> {interface_lddt, cdr_h3_rmsd}
    for f in sorted(LABELS_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        gen_dir, target = _split_stem(f.stem)
        gen = DIR_TO_GEN[gen_dir]
        for s in d.get("samples", []):
            cdrs = (s.get("cdr_rmsd") or {}).get("cdrs") or {}
            proj[(target, gen, s.get("rank"))] = {
                "interface_lddt": _scalar(s.get("interface_lddt"), "interface_lddt"),
                "cdr_h3_rmsd": cdrs.get("H3"),
            }
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = rows[0].keys()
    changed = set()
    for r in rows:
        p = proj.get((r["target"], r["gen"], int(r["rank"])))
        if p is None:
            continue
        for col in ("interface_lddt", "cdr_h3_rmsd"):
            v = p[col]
            cell = "" if v is None else str(v)
            if r[col] != cell:
                r[col] = cell
                changed.add((r["target"], r["gen"], col))
    bak = csv_path.with_suffix(".bak_labelpatch")
    if not bak.exists():
        shutil.copy2(csv_path, bak)
    tmp = csv_path.with_suffix(".tmp_sync")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(csv_path)  # atomic on the same filesystem
    print(f"[label_patch] csv cells updated in {len(changed)} (target,gen,column) groups:")
    for t, g, c in sorted(changed):
        print(f"  {t}/{g} {c}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recompute")
    r.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("sync_csv")
    s.add_argument("--csv", default=str(Path.home() / "abag_xm/tier_a/ranker_scores.csv"))
    a = ap.parse_args()
    return cmd_recompute(a) if a.cmd == "recompute" else cmd_sync_csv(a)


if __name__ == "__main__":
    sys.exit(main())
