#!/usr/bin/env python3
"""Fetch the smallest OpenFold3 dataset that exercises every featurization path.

Four structures, pinned by upstream, one per molecule class they care about:

    7ohe  DNA duplex            24 tokens
    7kud  RNA                   13 tokens
    7vus  1 protein + 3 ligand  87 tokens
    7fb8  protein homodimer     53 tokens

That set is not ours; it is `SMOKE_VALIDATION_PDB_IDS` from upstream's
`scripts/datasets/pdb_subset_helpers.py`, chosen by their
`select_smoke_validation_set.py` and used by their own training test. Taking their
choice rather than picking our own keeps the coverage argument theirs too.

Everything is pulled unsigned from the public `s3://openfold3-data` bucket. The S3
layout constants below mirror that same helper module.

Why this does not just call upstream's script: their `stream_subset` parses the cache
with `ijson`, which rejects the bare `NaN` literals their own published
`validation_cache_with_templates.json` contains (59 of them, in `resolution` fields).
`json` accepts `NaN`, so this loads the cache with the stdlib instead. The subset file
written here is otherwise the same document their `write_subset` would produce.

Usage:
    python scripts/of3_port/build_of3_subset.py --target-dir <dir>
    python scripts/of3_port/build_of3_subset.py --target-dir <dir> --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "openfold3-data"
S3_PREFIX = "pdb_training_set"
FULL_VALIDATION_CACHE = "validation_cache_with_templates.json"
FULL_VALIDATION_KEY = f"{S3_PREFIX}/dataset_caches/{FULL_VALIDATION_CACHE}"

# upstream pdb_subset_helpers.SMOKE_VALIDATION_PDB_IDS
PINNED = {
    "7ohe": "dna",
    "7kud": "rna",
    "7vus": "protein_ligand",
    "7fb8": "multimer",
}
# upstream download_subset.py:115 keys this off the split, not the cache contents.
TEMPLATE_CACHE_SUBDIR = {"train": "train_template_cache", "val": "val_template_cache"}


def _client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_full_cache(target: Path) -> Path:
    out = target / FULL_VALIDATION_CACHE
    if out.exists():
        return out
    target.mkdir(parents=True, exist_ok=True)
    print(f"downloading s3://{BUCKET}/{FULL_VALIDATION_KEY} ...")
    _client().download_file(BUCKET, FULL_VALIDATION_KEY, str(out))
    return out


def build_subset_cache(full: Path, target: Path, keep_ccd: set[str] | None = None) -> Path:
    """Write the 4-structure subset cache.

    `reference_molecule_data` is keyed by CCD code and covers the whole corpus (68k
    entries, most of the 36 MB). It is pruned to `keep_ccd` when that is given. The
    right key set is NOT the chains' `reference_mol_id`: that field is only set for
    standalone ligand chains, so every ordinary polymer residue (RNA `G`, and so on)
    would be dropped and the conformer pipeline raises `KeyError` on the first one.
    The caller therefore passes the residue names actually present in the downloaded
    structures, which is why this runs in two passes.
    """
    out = target / f"{full.stem}_subset_{len(PINNED)}.json"
    cache = json.loads(full.read_text())
    sd = cache["structure_data"]
    missing = sorted(set(PINNED) - set(sd))
    if missing:
        raise SystemExit(f"pinned ids absent from {full.name}: {missing}")

    ids = sorted(PINNED)
    refmol = cache["reference_molecule_data"]
    if keep_ccd is not None:
        absent = sorted(keep_ccd - set(refmol))
        if absent:
            print(f"  note: {len(absent)} residue code(s) have no reference-mol entry: "
                  f"{', '.join(absent[:8])}")
        refmol = {k: v for k, v in refmol.items() if k in keep_ccd}

    payload = {
        "_type": cache["_type"],
        "name": f"{cache.get('name', full.stem)}-pinned-{len(ids)}",
        "reference_molecule_data": refmol,
        "structure_data": {pid: sd[pid] for pid in ids},
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out.name} ({out.stat().st_size/1e3:.1f} kB, {len(ids)} structures, "
          f"{len(refmol)} reference mols)")
    return out


def extract_ids(cache_path: Path) -> dict[str, set[str]]:
    cache = json.loads(cache_path.read_text())
    out = {k: set() for k in ("pdb_ids", "alignment_rep_ids", "template_ids", "reference_mol_ids")}
    for pdb_id, entry in cache["structure_data"].items():
        out["pdb_ids"].add(pdb_id)
        for ch in entry["chains"].values():
            if rep := ch.get("alignment_representative_id"):
                out["alignment_rep_ids"].add(rep)
            for t in ch.get("template_ids") or []:
                out["template_ids"].add(t)
            if rm := ch.get("reference_mol_id"):
                out["reference_mol_ids"].add(rm)
    return out


def manifest(ids: dict[str, set[str]], root: Path, split: str = "val") -> list[tuple[str, Path]]:
    m: list[tuple[str, Path]] = []
    std = root / "preprocessed_pdb_data" / "standard"
    for pid in sorted(ids["pdb_ids"]):
        m.append((f"{S3_PREFIX}/preprocessed_pdb_data/standard/structure_files/{pid}/{pid}.npz",
                  std / "structure_files" / pid / f"{pid}.npz"))
    for rep in sorted(ids["alignment_rep_ids"]):
        m.append((f"{S3_PREFIX}/alignment_arrays/{rep}.npz",
                  root / "alignment_arrays" / f"{rep}.npz"))
        sub = TEMPLATE_CACHE_SUBDIR[split]
        m.append((f"{S3_PREFIX}/templates/{sub}/{rep}.npz",
                  root / "templates" / sub / f"{rep}.npz"))
    tmpl_pdbs = set()
    for tid in sorted(ids["template_ids"]):
        tp = tid.split("_")[0]
        tmpl_pdbs.add(tp)
        m.append((f"{S3_PREFIX}/templates/template_structure_arrays/{tp}/{tid}.npz",
                  root / "templates" / "template_structure_arrays" / tp / f"{tid}.npz"))
    for tp in sorted(tmpl_pdbs):
        m.append((f"{S3_PREFIX}/templates/template_structure_arrays/{tp}/chain_id_to_moltype.npz",
                  root / "templates" / "template_structure_arrays" / tp / "chain_id_to_moltype.npz"))
    return m


def download(items: list[tuple[str, Path]], workers: int = 16) -> tuple[int, list[str]]:
    s3 = _client()
    missing: list[str] = []
    got = 0

    def one(item: tuple[str, Path]) -> None:
        nonlocal got
        key, dest = item
        if dest.exists():
            got += 1
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            s3.download_file(BUCKET, key, str(dest))
            got += 1
        except Exception:
            missing.append(key)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, items))
    return got, missing


def scan_residue_names(structure_paths: list[Path]) -> set[str]:
    """CCD codes actually present in the downloaded structures.

    The cache's `reference_mol_id` is only set for standalone ligand chains, so
    modified residues inside a polymer (a methylated cysteine, say) are invisible to
    it and still need their own reference conformer.
    """
    import numpy as np

    names: set[str] = set()
    for p in structure_paths:
        if not p.exists():
            continue
        with np.load(p, allow_pickle=True) as z:
            for field in ("res_name", "residue_name", "comp_id"):
                if field in z:
                    names |= {str(x) for x in np.unique(z[field])}
                    break
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-dir", type=Path, required=True)
    ap.add_argument("--verify", action="store_true",
                    help="Re-hash an existing subset and print the manifest digest.")
    args = ap.parse_args()

    target = args.target_dir
    root = target / "pdb_training_set"

    full = fetch_full_cache(target)
    subset = build_subset_cache(full, target)
    ids = extract_ids(subset)
    print("  ".join(f"{k}={len(v)}" for k, v in ids.items()))

    items = manifest(ids, root)
    print(f"fetching {len(items)} files ...")
    got, missing = download(items)
    print(f"  {got}/{len(items)} present, {len(missing)} absent on S3")
    for k in missing[:10]:
        print(f"    absent: {k}")

    struct_paths = [d for k, d in items if "/structure_files/" in k]
    ccds = scan_residue_names(struct_paths) | ids["reference_mol_ids"]
    # Second pass: now that the structures are on disk we know every residue code
    # the pipeline will look up, so the metadata table can be pruned to those.
    subset = build_subset_cache(full, target, keep_ccd=ccds)
    ref_items = [
        (f"{S3_PREFIX}/preprocessed_pdb_data/standard/reference_mols/{c}.sdf",
         root / "preprocessed_pdb_data" / "standard" / "reference_mols" / f"{c}.sdf")
        for c in sorted(ccds)
    ]
    print(f"fetching {len(ref_items)} reference mol SDFs ...")
    rgot, rmissing = download(ref_items)
    print(f"  {rgot}/{len(ref_items)} present, {len(rmissing)} absent")

    # A manifest digest over (relative path, sha256) makes the corpus itself a hashed
    # artifact, so "the same data" is checkable on another host rather than asserted.
    entries = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            entries.append((p.relative_to(root).as_posix(), sha256(p)))
    entries.append((subset.name, sha256(subset)))
    digest = hashlib.sha256(
        "\n".join(f"{n}  {h}" for n, h in entries).encode()
    ).hexdigest()
    mf = target / "MANIFEST.sha256"
    mf.write_text("".join(f"{h}  {n}\n" for n, h in entries) + f"\n# corpus-digest {digest}\n")
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    print(f"\n{len(entries)} files, {total/1e6:.1f} MB")
    print(f"corpus-digest sha256 {digest}")
    print(f"manifest {mf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
