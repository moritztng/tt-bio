"""Host-side input pipeline for Nesso-1: YAML in, featurized batch out.

Upstream splits this across ``nesso/main.py`` (a click CLI wrapped around a Lightning
``predict``) and the vendored data package. tt-bio does not ship Lightning, so the
preprocessing steps live here as plain functions: parse the YAMLs into a manifest,
compute the ESM-2 embeddings the featurizer reads off disk, then hand back an
``InferenceDataset``.

One deliberate difference from upstream: ``num_workers=0`` means "no worker
processes" and runs the parse inline. Upstream passes 0 straight into
``ProcessPoolExecutor`` and dies with ``max_workers must be greater than 0``.
"""
from __future__ import annotations

import hashlib
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml as pyyaml
from rdkit import Chem
from safetensors.torch import save_file

from tt_bio._vendor.nesso.data.types import Manifest, Record

# RDKit drops atom properties when UNPICKLING too, not just when pickling, so this has
# to be set before ccd.pkl is read or every standard-residue mol comes back without its
# ``name`` prop and the featurizer dies with KeyError: 'name'. The vendored
# ``featurizer`` module sets the same thing as an import side-effect, which makes the
# whole pipeline import-order dependent; set it here where the load happens.
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AtomProps)

ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"

# What the shipped CLI passes into Nesso1.predict; the code defaults differ (a 196
# token crop budget instead of 256), so a comparison against upstream numbers has to
# set these explicitly.
CLI_PREDICT_ARGS = {
    "pose_protein_cutoff": 15.0,
    "affinity_protein_cutoff": 15.0,
    "refine_protein_inference": True,
    "refine_protein_cutoff": 22.0,
    "refine_protein_tokens_budget": 256,
}


@dataclass
class Paths:
    processed: Path
    mol_dir: Path
    esm_dir: Path
    manifest_path: Path
    structures_dir: Path
    records_dir: Path


def resolve_paths(out_dir: Path) -> Paths:
    processed = out_dir / "processed"
    p = Paths(
        processed=processed,
        mol_dir=processed / "rdkit_conformers",
        esm_dir=processed / "esm_embeddings",
        manifest_path=processed / "manifest.json",
        structures_dir=processed / "structures",
        records_dir=processed / "records",
    )
    for d in (p.mol_dir, p.esm_dir, p.structures_dir, p.records_dir):
        d.mkdir(parents=True, exist_ok=True)
    return p


def find_yamls(data: Path) -> list[Path]:
    if data.is_dir():
        hits = sorted(q for q in data.iterdir() if q.suffix.lower() in (".yaml", ".yml"))
        if not hits:
            raise ValueError(f"no .yaml/.yml files in {data}")
        return hits
    if data.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(f"expected a .yaml/.yml file, got {data.suffix!r}")
    return [data]


def protein_blocks(yaml_paths: list[Path]):
    """Every ``protein:`` block with a sequence, across the inputs, in file order."""
    for yp in yaml_paths:
        schema = pyyaml.safe_load(yp.read_text())
        if not isinstance(schema, dict):
            continue
        for item in schema.get("sequences", []) or []:
            block = (item or {}).get("protein")
            if block and "sequence" in block:
                yield block


def collect_esm(yaml_paths: list[Path]) -> tuple[dict[str, str], dict[str, str]]:
    """``({md5(seq): seq}, {md5(seq): precomputed_esm_path})`` over protein entities."""
    seqs: dict[str, str] = {}
    given: dict[str, str] = {}
    for block in protein_blocks(yaml_paths):
        seq = str(block["sequence"])
        mid = hashlib.md5(seq.encode("utf-8")).hexdigest()
        seqs.setdefault(mid, seq)
        if block.get("esm"):
            given.setdefault(mid, str(block["esm"]))
    return seqs, given


# The only keys ``_parse_entity`` reads off a ``protein:`` block. The vendored parser drops
# everything else without a word, and we tell users the affinity YAML is the Boltz-2 one,
# which carries ``msa:``. Ignoring that key is right — Nesso-1 conditions on ESM-2, never on
# an alignment — but ignoring it silently is the defect we already fixed once for ``esm:``.
PROTEIN_KEYS = frozenset({"id", "sequence", "esm", "pocket_mask"})

_IGNORED_KEY_WHY = {
    "msa": "Nesso-1 conditions on ESM-2 embeddings, not on an alignment",
}


def warn_ignored_protein_keys(yaml_paths: list[Path]) -> list[str]:
    """Warn once per run about ``protein:`` keys the featurizer reads and then drops."""
    ignored = sorted({k for b in protein_blocks(yaml_paths) for k in b} - PROTEIN_KEYS)
    if ignored:
        named = ", ".join(
            f"{k} ({_IGNORED_KEY_WHY[k]})" if k in _IGNORED_KEY_WHY else k
            for k in ignored
        )
        warnings.warn(
            f"ignoring protein keys Nesso-1 does not read: {named}. "
            f"It reads {', '.join(sorted(PROTEIN_KEYS))}.",
            stacklevel=2,
        )
    return ignored


def link_given_esm(given: dict[str, str], esm_dir: Path) -> int:
    """Put a user-supplied ``esm:`` embedding where the featurizer looks for it.

    The featurizer reads ``<esm_dir>/<md5(sequence)>.safetensors``, so an ``esm:`` path in
    the YAML only has an effect if it lands there before ``run_esm``; otherwise the 650M
    encoder recomputes the embedding and the supplied file is ignored. Upstream symlinks
    and swallows every failure, which turns a typo in the path into a silently different
    model input, so a missing path raises here instead.
    """
    linked = 0
    for mid, src in sorted(given.items()):
        source = Path(src).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"esm: {src} does not exist")
        dst = esm_dir / f"{mid}.safetensors"
        if dst.exists():
            continue
        try:
            dst.symlink_to(source.resolve())
        except OSError:
            dst.write_bytes(source.read_bytes())
        linked += 1
    return linked


def run_esm(
    seqs: dict[str, str],
    esm_dir: Path,
    model_name: str = ESM2_MODEL,
    cache_dir: Path | None = None,
) -> int:
    """Write ``[1, L+2, 1280]`` final-layer ESM-2 embeddings, skipping ones on disk."""
    missing = {m: s for m, s in seqs.items() if not (esm_dir / f"{m}.safetensors").exists()}
    if not missing:
        return 0
    from tt_bio._vendor.nesso.data.esm import extract_esm_embedding, setup_esm_model

    model, tokenizer = setup_esm_model(model_name, torch.device("cpu"), cache_dir=cache_dir)
    for mid, seq in sorted(missing.items()):
        emb = extract_esm_embedding(seq, model, tokenizer)
        save_file({"embeddings": emb}, esm_dir / f"{mid}.safetensors")
    return len(missing)


_CCD: dict | None = None


def _init_worker(ccd_pkl: Path | None) -> None:
    global _CCD
    if ccd_pkl is not None:
        from tt_bio._vendor.nesso.data.yaml_input import load_ccd_mol_dict

        _CCD = load_ccd_mol_dict(ccd_pkl)


def _parse_one(yp: Path, mol_dir: Path, structures_dir: Path, records_dir: Path) -> Record:
    from tt_bio._vendor.nesso.data.yaml_input import parse_yaml

    struct, rec, _, _ = parse_yaml(yp, mol_dir, ccd_dict=_CCD)
    struct.dump(structures_dir / f"{rec.id}.npz")
    rec.dump(records_dir / f"{rec.id}.json")
    return rec


def preprocess(
    yaml_paths: list[Path],
    paths: Paths,
    ccd_pkl: Path | None,
    num_workers: int = 2,
) -> tuple[Manifest, list[str]]:
    """Parse YAMLs into a manifest. ``num_workers=0`` parses inline, no pool."""
    records: list[Record] = []
    failed: list[str] = []
    if num_workers <= 0:
        _init_worker(ccd_pkl)
        for yp in yaml_paths:
            try:
                records.append(_parse_one(yp, paths.mol_dir, paths.structures_dir, paths.records_dir))
            except Exception as exc:  # noqa: BLE001 - report and keep going, like upstream
                failed.append(yp.stem)
                print(f"error processing {yp.name}: {exc}")
    else:
        with ProcessPoolExecutor(
            max_workers=num_workers, initializer=_init_worker, initargs=(ccd_pkl,)
        ) as pool:
            futures = {
                pool.submit(_parse_one, yp, paths.mol_dir, paths.structures_dir, paths.records_dir): yp
                for yp in yaml_paths
            }
            for fut in as_completed(futures):
                try:
                    records.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    failed.append(futures[fut].stem)
                    print(f"error processing {futures[fut].name}: {exc}")
    records.sort(key=lambda r: r.id)
    manifest = Manifest(records)
    manifest.dump(paths.manifest_path)
    return manifest, failed


def build_dataset(paths: Paths, manifest: Manifest, ccd_pkl: Path, max_dist: float = 22.0):
    from tt_bio._vendor.nesso.data.featurizer import NessoFeaturizer
    from tt_bio._vendor.nesso.data.inference import InferenceDataset

    return InferenceDataset(
        manifest=manifest,
        target_dir=paths.processed,
        ligand_dir=paths.mol_dir,
        ccd_pkl=ccd_pkl,
        use_esm_all_layers=False,
        num_dist_bins=64,
        min_dist=2.0,
        max_dist=max_dist,
        atoms_per_window_queries=32,
        featurizer=NessoFeaturizer(
            esm_emb_dir=paths.esm_dir, esm_emb_dim=1280, esm_num_layers=33
        ),
    )


def find_ccd(cache_dir: Path | None = None) -> Path:
    """Locate the ``ccd.pkl`` shipped alongside the checkpoint (413 MB, never committed).

    Searches every place it could be rather than only the first one that is set, and the error
    names all of them plus the two flags that override it. A caller that has the checkpoint in
    the default cache and passes ``--cache`` for the ESM-2 weights used to get "no ccd.pkl under
    <the ESM cache>", which reads as a missing download rather than a lookup that never looked
    where the file is.
    """
    roots, seen = [], set()
    for cand in (cache_dir, os.environ.get("NESSO_CACHE"), os.environ.get("HF_HOME"),
                 "~/.cache/huggingface"):
        if not cand:
            continue
        r = Path(cand).expanduser()
        if r not in seen:
            seen.add(r)
            roots.append(r)
    for root in roots:
        hits = sorted(root.rglob("models--recursionpharma--nesso/snapshots/*/ccd.pkl"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "no ccd.pkl found. Looked under: " + ", ".join(str(r) for r in roots)
        + ". It ships with the checkpoint; point NESSO_CACHE (or HF_HOME) at the HuggingFace "
          "cache holding it, pass --cache, or name the file directly with --ccd")


def prepare(
    data: Path,
    out_dir: Path,
    ccd_pkl: Path | None = None,
    num_workers: int = 2,
    esm_cache: Path | None = None,
    max_dist: float = 22.0,
):
    """YAML path or directory -> (dataset, manifest, failed stems)."""
    yaml_paths = find_yamls(data)
    warn_ignored_protein_keys(yaml_paths)
    paths = resolve_paths(out_dir)
    # esm_cache is the HuggingFace cache the caller named (`--cache`), which is where the
    # checkpoint and so ccd.pkl live too. Not passing it made `--cache` a documented no-op for
    # ccd discovery.
    ccd = Path(ccd_pkl) if ccd_pkl else find_ccd(esm_cache)
    manifest, failed = preprocess(yaml_paths, paths, ccd, num_workers=num_workers)
    seqs, given = collect_esm(yaml_paths)
    link_given_esm(given, paths.esm_dir)
    run_esm(seqs, paths.esm_dir, cache_dir=esm_cache)
    return build_dataset(paths, manifest, ccd, max_dist=max_dist), manifest, failed


def collate(item: dict) -> dict:
    """Single-item batch, the only batch shape Nesso-1 supports."""
    return {
        k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v])
        for k, v in item.items()
    }
