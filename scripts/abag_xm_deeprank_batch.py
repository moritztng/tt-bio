#!/usr/bin/env python3
"""Batched DeepRank-Ab scoring for one abag-xm fold.

Builds a single ensemble PDB from all 50 sample CIFs (MODEL/ENDMDL + REMARK
lines), runs `deeprank-ab-predict` ONCE (ESM loaded once via dedup, then
per-model graph + inference), parses the output CSV, and writes
{rank: predicted_dockq} JSON.

This replaces the per-sample driver path (50 CLI calls × ~30-60s ESM reload
each = ~30-50 min/fold) with one CLI call (~2-4 min/fold after ESM cached).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO


def _cif_to_pdb_lines(cif_path, target):
    """Convert a sample CIF to a list of PDB ATOM/END lines (no MODEL wrapper)."""
    parser = MMCIFParser(QUIET=True)
    struct = parser.get_structure(target, str(cif_path))
    io = PDBIO()
    io.set_structure(struct)
    buf = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    buf.close()
    io.save(buf.name)
    with open(buf.name) as f:
        lines = f.readlines()
    os.unlink(buf.name)
    # keep only ATOM/HETATM/TER/END; drop HEADER/CRYST1 (ensemble has one set)
    out = [ln for ln in lines if ln[:6] in ("ATOM  ", "HETATM", "TER\n", "TER \n",
                                            "END\n", "END \n")]
    # drop trailing END (we add per-model ENDMDL instead)
    out = [ln for ln in out if ln.strip() != "END"]
    return out


def build_ensemble_pdb(fold_dir, target, out_pdb):
    """Concatenate all 50 sample CIFs into one ensemble PDB with REMARK names."""
    sdir = Path(fold_dir) / "structures"
    cifs = [sdir / f"{target}.cif"] + [sdir / f"{target}_model_{k}.cif"
                                       for k in range(1, 50)]
    present = [(k, c) for k, c in enumerate(cifs) if c.exists()]
    if not present:
        raise RuntimeError(f"no sample CIFs in {sdir}")
    with open(out_pdb, "w") as f:
        for k, cif in present:
            name = f"r{k}"
            f.write(f"REMARK  4      MODEL     {k+1} FROM x/{name}.pdb\n")
            f.write(f"MODEL        {k+1}\n")
            for ln in _cif_to_pdb_lines(cif, target):
                f.write(ln if ln.endswith("\n") else ln + "\n")
            f.write("ENDMDL\n")
        f.write("END\n")
    return [k for k, _ in present]


def find_predictions_csv(stdout, work_dirs):
    """Locate the predictions CSV from CLI stdout or by scanning work dirs."""
    # stdout often has "Output written → <path>" or "predictions.csv"
    for line in stdout.splitlines():
        m = re.search(r"([^\s]+predictions\.csv)", line)
        if m:
            p = Path(m.group(1))
            if p.exists():
                return p
        m = re.search(r"→\s*([^\s]+\.csv)", line)
        if m:
            p = Path(m.group(1))
            if p.exists():
                return p
    # scan recent work dirs for *predictions.csv
    import glob
    for wd in work_dirs:
        for csv in sorted(glob.glob(str(wd) + "**/*predictions.csv", recursive=True),
                          key=os.path.getmtime, reverse=True):
            return Path(csv)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dir", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--deeprank_venv",
                    default=os.path.expanduser("~/.deeprank_ab_venv"))
    args = ap.parse_args()

    cli = Path(args.deeprank_venv) / "bin" / "deeprank-ab-predict"
    py = Path(args.deeprank_venv) / "bin" / "python3"

    work = tempfile.mkdtemp(prefix=f"deeprank_batch_{args.target}_{args.gen}_")
    ensemble = Path(work) / f"{args.target}_{args.gen}_ensemble.pdb"
    ranks = build_ensemble_pdb(args.fold_dir, args.target, ensemble)
    print(f"[deeprank-batch] {args.target}/{args.gen}: {len(ranks)} models -> "
          f"{ensemble.name}", flush=True)

    # deeprank-ab-predict auto-detects chains via ANARCI, which needs hmmscan/hmmsearch
    # (HMMER3 built from source into ~/.local/bin). Prepend it to PATH so the subprocess
    # works regardless of the parent env.
    _env = {**os.environ, "PATH": os.path.expanduser("~/.local/bin") + os.pathsep + os.environ.get("PATH", "")}
    r = subprocess.run([str(cli), str(ensemble)], capture_output=True,
                       text=True, cwd=str(Path(work)), env=_env)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print(f"[deeprank-batch] CLI failed rc={r.returncode}", file=sys.stderr)
        print(r.stderr[-1500:], file=sys.stderr)
        sys.exit(1)

    # find CSV: scan work + /tmp for predictions.csv
    import glob
    work_dirs = [work, "/tmp", "/tmp/tmp*"]
    csv_path = find_predictions_csv(r.stdout, work_dirs)
    if csv_path is None:
        # last resort: newest *predictions.csv under /tmp
        cs = sorted(glob.glob("/tmp/**/*predictions.csv", recursive=True),
                    key=os.path.getmtime, reverse=True)
        if cs:
            csv_path = Path(cs[0])
    if csv_path is None:
        print(f"[deeprank-batch] no predictions CSV found", file=sys.stderr)
        sys.exit(2)
    print(f"[deeprank-batch] predictions CSV: {csv_path}", flush=True)

    import csv as csvmod
    scores = {}
    with open(csv_path) as f:
        for row in csvmod.DictReader(f):
            pid = row.get("pdb_id") or row.get("mol") or ""
            mk = re.search(r"r(\d+)", pid)
            if mk:
                k = int(mk.group(1))
            else:
                # fallback: row order
                continue
            v = row.get("predicted_dockq") or row.get("dockq") or 0.0
            scores[str(k)] = float(v)
    # if no r<k> match, map by row order to sorted present ranks
    if not scores:
        with open(csv_path) as f:
            rows = list(csvmod.DictReader(f))
        for k, row in zip(sorted(ranks), rows):
            v = row.get("predicted_dockq") or row.get("dockq") or 0.0
            scores[str(k)] = float(v)

    with open(args.out_json, "w") as f:
        json.dump(scores, f)
    print(f"[deeprank-batch] wrote {len(scores)} scores -> {args.out_json}",
          flush=True)


if __name__ == "__main__":
    main()
