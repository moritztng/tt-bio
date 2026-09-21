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

The `--split train` mode does the same for a random 8-structure sample of the
1.68 GB training cache, drawn exactly as upstream's `sample_subset_cache` draws it
(`random.Random(seed).sample(enumerate_structure_ids(cache), n)`, then sorted). That
cache is streamed, never loaded: `--drop-full-cache` deletes it once the subset is
written, which matters on a host with a few GB free.

Usage:
    python scripts/of3_port/build_of3_subset.py --target-dir <dir>
    python scripts/of3_port/build_of3_subset.py --target-dir <dir> --split train
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "openfold3-data"
S3_PREFIX = "pdb_training_set"
FULL_CACHE = {
    "val": "validation_cache_with_templates.json",
    "train": "training_cache_with_templates.json",
}

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


# A bare `NaN` value, only where JSON would accept a value: after `:` `,` `[` or space,
# and followed by a delimiter. Restricting it this way keeps the substring "NaN" inside a
# real string value (a ligand name, a SMILES) from being rewritten.
_NAN_VALUE = re.compile(rb"(?<=[:\[,\s])NaN(?=[,\]\}\s])")
_NAN_SENTINEL = b'"@@NaN@@"'
NAN_SENTINEL = "@@NaN@@"


class _NanShim(io.RawIOBase):
    """Byte stream with bare `NaN` values swapped for a string sentinel.

    Their published caches carry bare `NaN` in `resolution`, which every ijson backend
    rejects, so their own generate_subset_cache.py cannot read their own data
    (LEDGER K10). `json` accepts NaN but would have to hold the whole 1.68 GB training
    cache in memory at once.

    The sentinel is a STRING and is turned back into `float("nan")` by `_unshim` after
    parsing, because NaN and null are NOT interchangeable here: `set_loss_weights`
    (`pipelines/featurization/loss_weights.py:45-48`) zeroes every confidence loss when
    `resolution is None`, whereas NaN fails both range comparisons and keeps them. A
    NaN -> null rewrite silently turns the confidence losses off for every structure of
    unknown resolution, which is a different batch. Measured: it moved `loss_weights` on
    3 of the 4 corpus structures.
    """

    def __init__(self, fh, chunk: int = 1 << 20) -> None:
        self._fh = fh
        self._chunk = chunk
        self._buf = b""
        self._carry = b""
        self._eof = False

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        want = len(b)
        while len(self._buf) < want and not self._eof:
            data = self._fh.read(self._chunk)
            if not data:
                self._eof = True
                self._buf += _NAN_VALUE.sub(_NAN_SENTINEL, self._carry)
                self._carry = b""
                break
            data = self._carry + data
            # Hold back the last few bytes so a NaN token, or the delimiter the
            # lookahead needs, cannot be split across a chunk boundary.
            data, self._carry = data[:-4], data[-4:]
            self._buf += _NAN_VALUE.sub(_NAN_SENTINEL, data)
        n = min(want, len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n


def _unshim(obj):
    """Turn the sentinel back into a real NaN, and ijson Decimals into floats."""
    if isinstance(obj, str):
        return float("nan") if obj == NAN_SENTINEL else obj
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _unshim(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unshim(v) for v in obj]
    return obj


def _open_filtered(path: Path):
    return io.BufferedReader(_NanShim(open(path, "rb")))


def enumerate_structure_ids(path: Path) -> list[str]:
    """Every key under `structure_data`, streamed. Mirrors upstream's helper."""
    import ijson

    ids = []
    with _open_filtered(path) as f:
        for prefix, event, value in ijson.parse(f):
            if prefix == "structure_data" and event == "map_key":
                ids.append(value)
    return ids


def stream_subset(path: Path, selected: set[str]) -> tuple[dict, dict]:
    """(metadata, structure_data subset), streamed. Mirrors upstream's helper."""
    import ijson

    metadata = {}
    with _open_filtered(path) as f:
        for key, value in ijson.kvitems(f, ""):
            if key != "structure_data":
                metadata[key] = _unshim(value)

    data = {}
    with _open_filtered(path) as f:
        for pdb_id, entry in ijson.kvitems(f, "structure_data"):
            if pdb_id in selected:
                data[pdb_id] = _unshim(entry)
                if len(data) == len(selected):
                    break
    return metadata, data


def fetch_full_cache(target: Path, split: str) -> Path:
    name = FULL_CACHE[split]
    out = target / name
    if out.exists():
        return out
    target.mkdir(parents=True, exist_ok=True)
    key = f"{S3_PREFIX}/dataset_caches/{name}"
    print(f"downloading s3://{BUCKET}/{key} ...")
    _client().download_file(BUCKET, key, str(out))
    return out


def build_subset_cache(full: Path, target: Path, keep_ccd: set[str] | None = None,
                       ids: list[str] | None = None) -> Path:
    """Write the 4-structure subset cache.

    `reference_molecule_data` is keyed by CCD code and covers the whole corpus (68k
    entries, most of the 36 MB). It is pruned to `keep_ccd` when that is given. The
    right key set is NOT the chains' `reference_mol_id`: that field is only set for
    standalone ligand chains, so every ordinary polymer residue (RNA `G`, and so on)
    would be dropped and the conformer pipeline raises `KeyError` on the first one.
    The caller therefore passes the residue names actually present in the downloaded
    structures, which is why this runs in two passes.
    """
    if ids is None:
        ids = sorted(PINNED)
    out = target / f"{full.stem}_subset_{len(ids)}.json"
    metadata, sd = stream_subset(full, set(ids))
    missing = sorted(set(ids) - set(sd))
    if missing:
        raise SystemExit(f"ids absent from {full.name}: {missing}")

    cache = metadata
    refmol = cache.get("reference_molecule_data", {})
    if keep_ccd is not None:
        absent = sorted(keep_ccd - set(refmol))
        if absent:
            print(f"  note: {len(absent)} residue code(s) have no reference-mol entry: "
                  f"{', '.join(absent[:8])}")
        refmol = {k: v for k, v in refmol.items() if k in keep_ccd}

    payload = {
        **{k: v for k, v in cache.items() if k != "reference_molecule_data"},
        "name": f"{cache.get('name', full.stem)}-subset-{len(ids)}",
        "reference_molecule_data": refmol,
        "structure_data": {pid: sd[pid] for pid in ids},
    }
    # _unshim has already turned Decimals into floats and the sentinel back into NaN;
    # json.dumps writes NaN as a bare `NaN`, which is exactly what upstream published.
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
    ap.add_argument("--split", choices=("val", "train"), default="val")
    ap.add_argument("--train-size", type=int, default=8,
                    help="Structures to sample for --split train (upstream default 8).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Sampling seed for --split train (upstream default 42).")
    ap.add_argument("--ids", default=None,
                    help="Comma-separated PDB ids to take instead of the seeded sample. "
                         "The ids must already be in the split's cache, so this selects "
                         "from upstream's own corpus rather than adding to it.")
    ap.add_argument("--drop-full-cache", action="store_true",
                    help="Delete the full cache once the subset is written.")
    args = ap.parse_args()

    target = args.target_dir
    root = target / "pdb_training_set"

    full = fetch_full_cache(target, args.split)
    ids = None
    if args.ids:
        ids = sorted({i.strip().lower() for i in args.ids.split(",") if i.strip()})
        print(f"explicit ids: {' '.join(ids)}")
    elif args.split == "train":
        # Exactly upstream's sample_subset_cache draw.
        all_ids = enumerate_structure_ids(full)
        print(f"{len(all_ids)} structures in {full.name}; "
              f"sampling {args.train_size} with seed={args.seed}")
        ids = sorted(random.Random(args.seed).sample(all_ids, args.train_size))
        print("  " + " ".join(ids))
    subset = build_subset_cache(full, target, ids=ids)
    ids_map = extract_ids(subset)
    print("  ".join(f"{k}={len(v)}" for k, v in ids_map.items()))

    items = manifest(ids_map, root, args.split)
    print(f"fetching {len(items)} files ...")
    got, missing = download(items)
    print(f"  {got}/{len(items)} present, {len(missing)} absent on S3")
    for k in missing[:10]:
        print(f"    absent: {k}")

    struct_paths = [d for k, d in items if "/structure_files/" in k]
    ccds = scan_residue_names(struct_paths) | ids_map["reference_mol_ids"]
    # Second pass: now that the structures are on disk we know every residue code
    # the pipeline will look up, so the metadata table can be pruned to those.
    subset = build_subset_cache(full, target, keep_ccd=ccds, ids=ids)
    if args.drop_full_cache:
        full.unlink()
        print(f"removed {full.name}")
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
    mf = target / f"MANIFEST.{args.split}.sha256"
    mf.write_text("".join(f"{h}  {n}\n" for n, h in entries) + f"\n# corpus-digest {digest}\n")
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    print(f"\n{len(entries)} files, {total/1e6:.1f} MB")
    print(f"corpus-digest sha256 {digest}")
    print(f"manifest {mf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
