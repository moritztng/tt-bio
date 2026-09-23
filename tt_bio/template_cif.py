"""A structure template given as an mmCIF, turned into the per-chain alignment npz.

Boltz-2 reads a top-level `templates:` block (`cif:` a structure file, `chain_id:` the query
chain(s) it templates, optionally `template_id:` the template chain(s) to use) with its own
parser. Protenix-v2, OpenDDE and the OpenFold3 family take a template as a per-chain alignment
npz instead: `{f"{pdb_id}_{chain}": {"index", "release_date", "idx_map"}}` with coordinates read
from `<pdb_id>.cif` in the shared template-structure cache (`protenix_template`,
`openfold3_data`). This module is the bridge, so one YAML templates the same way on all of them.

It follows Boltz-2's own reading of the block (`data/parse.py`): chains are named by
`label_asym_id`, the template sequence is the entity's full sequence indexed by
`label_seq_id`, a query chain without a `template_id` takes the template protein chain with
the best global alignment score, and every query chain named must be a protein chain. The
alignment itself is BLASTP-scored and local, as Boltz-2's is, but gapped: the npz carries an
arbitrary residue map (upstream's own npz files come from a gapped search), so a template with
an indel relative to the query is used on both sides of the indel instead of on the longest
ungapped segment only.

The cif is copied into the structure cache under a content hash, so a user file never shadows a
same-named RCSB entry the cache already holds, and two different files never share a name.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from tt_bio.cache import cached, staged

#: The npz field is carried for OpenFold3's cache format; nothing at inference reads it.
_RELEASE_DATE = "1900-01-01"


def _one_letter(res_name: str) -> str:
    from tt_bio.data import const

    return const.prot_token_to_letter.get(res_name, "M" if res_name == "MSE" else "X")


def template_chains(cif_path) -> dict[str, str]:
    """{label_asym_id: sequence} for the protein chains of an mmCIF.

    Position i of each sequence is `label_seq_id` i + 1, which is the numbering the npz's
    template column and both featurizers use. The entity sequence is read where the file has
    one, so a residue with no coordinates keeps its place; a file without `entity_poly_seq`
    (some predictors write none) falls back to the residues in `atom_site`, unobserved
    positions filled with X.
    """
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    block = pdbx.CIFFile.read(str(cif_path)).block
    per_chain: dict[str, dict[int, str]] = {}
    if all(k in block for k in ("entity_poly", "entity_poly_seq", "struct_asym")):
        ep = block["entity_poly"]
        protein = {e for e, t in zip(ep["entity_id"].as_array(), ep["type"].as_array())
                   if str(t).startswith("polypeptide")}
        eps = block["entity_poly_seq"]
        by_entity: dict[str, dict[int, str]] = {}
        for e, num, mon in zip(eps["entity_id"].as_array(), eps["num"].as_array(),
                               eps["mon_id"].as_array()):
            by_entity.setdefault(str(e), {}).setdefault(int(num), str(mon))
        sa = block["struct_asym"]
        for asym, e in zip(sa["id"].as_array(), sa["entity_id"].as_array()):
            if str(e) in protein and str(e) in by_entity:
                per_chain[str(asym)] = by_entity[str(e)]
    else:
        arr = pdbx.get_structure(pdbx.CIFFile.read(str(cif_path)), model=1,
                                 use_author_fields=False)
        arr = arr[struc.filter_amino_acids(arr)]
        for c, r, n in zip(arr.chain_id, arr.res_id, arr.res_name):
            per_chain.setdefault(str(c), {}).setdefault(int(r), str(n))
    return {c: "".join(_one_letter(res.get(i, "UNK")) for i in range(1, max(res) + 1))
            for c, res in per_chain.items() if res and min(res) >= 1}


def align(query: str, template: str) -> np.ndarray:
    """(n, 2) int64 map of 1-indexed (query residue, template residue) aligned pairs."""
    from Bio import Align

    aligner = Align.PairwiseAligner(scoring="blastp")
    aligner.mode = "local"
    aln = aligner.align(query, template)[0]
    pairs = [(q + 1, t + 1)
             for (qs, qe), (ts, te) in zip(*aln.aligned)
             for q, t in zip(range(qs, qe), range(ts, te))]
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _global_score(query: str, template: str) -> float:
    from tt_bio.data.parse import get_global_alignment_score

    return get_global_alignment_score(query, template)


def _as_list(v):
    return None if v is None else [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]


def _cache_cif(cif_path: Path, struct_dir: Path) -> str:
    """Copy the cif into the structure cache under its content hash; returns the entry id."""
    data = cif_path.read_bytes()
    entry = "cif" + hashlib.sha256(data).hexdigest()[:16]     # no "_": it splits the npz key
    dest = struct_dir / f"{entry}.cif"
    if not cached(dest):
        with staged(dest) as tmp:
            Path(tmp).write_bytes(data)
    return entry


def structure_template_npz(blocks, chains, struct_dir, model: str) -> dict[str, str]:
    """Per-chain alignment npz paths built from a top-level `templates:` list.

    ``chains`` is the shared reader's chain list. Every block becomes one npz entry on each
    query chain it names, in the order given (the order is the rank the featurizers read).
    A block this cannot honour is a hard error, never a template-free fold.
    """
    struct_dir = Path(struct_dir)
    seqs = {c[0]: "".join(c[1].split()) for c in chains if c[3] == "protein"}
    per_chain: dict[str, dict[str, dict]] = {}
    for n, block in enumerate(blocks or []):
        if not isinstance(block, dict):
            raise RuntimeError(f"--model {model}: `templates:` entry {n + 1} is not a mapping.")
        if "cif" not in block:
            form = "a pdb file" if "pdb" in block else "no `cif:` path"
            raise RuntimeError(
                f"--model {model}: `templates:` entry {n + 1} gives {form}; this model reads "
                f"a template structure as mmCIF (`cif:`). Convert it, or use --model boltz2.")
        if block.get("force"):
            raise RuntimeError(
                f"--model {model}: `force:` on a template steers Boltz-2's sampler with a "
                f"potential no other model has. Drop it to use the template as a feature, "
                f"or use --model boltz2.")
        cif = Path(str(block["cif"])).expanduser()
        if not cif.is_file():
            raise RuntimeError(f"--model {model}: template file {cif} does not exist.")
        query_ids = _as_list(block.get("chain_id")) or list(seqs)
        bad = [c for c in query_ids if c not in seqs]
        if bad:
            raise RuntimeError(
                f"--model {model}: template {cif.name} names chain(s) {bad}, which are not "
                f"protein chains of this input.")
        tchains = template_chains(cif)
        wanted = _as_list(block.get("template_id"))
        if wanted is not None:
            if len(wanted) != len(query_ids):
                raise RuntimeError(
                    f"--model {model}: template {cif.name} gives {len(wanted)} template_id(s) "
                    f"for {len(query_ids)} chain_id(s); they pair up one to one.")
            missing = [t for t in wanted if t not in tchains]
            if missing:
                raise RuntimeError(
                    f"--model {model}: template {cif.name} has no protein chain {missing} "
                    f"(label_asym_id; it has {sorted(tchains)}).")
        elif not tchains:
            raise RuntimeError(f"--model {model}: template {cif.name} has no protein chain.")
        entry = _cache_cif(cif, struct_dir)
        for i, qc in enumerate(query_ids):
            tc = wanted[i] if wanted else max(
                tchains, key=lambda t: _global_score(seqs[qc], tchains[t]))
            idx_map = align(seqs[qc], tchains[tc])
            if not len(idx_map):
                raise RuntimeError(
                    f"--model {model}: template {cif.name} chain {tc} does not align to "
                    f"chain {qc}.")
            per_chain.setdefault(qc, {})[f"{entry}_{tc}"] = {
                "index": len(per_chain.get(qc, {})), "release_date": _RELEASE_DATE,
                "idx_map": idx_map}
    out: dict[str, str] = {}
    for qc, entries in per_chain.items():
        digest = hashlib.sha256(repr((seqs[qc], sorted(
            (k, v["index"], v["idx_map"].tobytes()) for k, v in entries.items()))).encode())
        npz = struct_dir / "alignments" / f"{digest.hexdigest()[:24]}.npz"
        if not cached(npz):
            with staged(npz) as tmp:
                np.savez(tmp, **{k: np.array(v, dtype=object) for k, v in entries.items()})
        out[qc] = str(npz)
    return out


def chain_ca(query_len: int, npz, struct_dir, model: str):
    """``(ca (L, 3), mask (L,))`` for one query chain from its alignment npz.

    For a model that templates by coordinate with one template per chain (RF3), where the
    embedders above take up to four. Read through the same alignment and structure reader
    the Protenix featurizer uses, so a residue counts only when that one would use it.
    """
    from tt_bio.protenix_template import chain_template_arrays, read_alignment_entries

    entries = read_alignment_entries(npz, max_templates=1 << 30)
    if len(entries) > 1:
        raise RuntimeError(
            f"--model {model} takes one template per chain; {Path(npz).name} gives "
            f"{len(entries)}. Keep the one you want, or use a model that takes up to four.")
    _aatype, pos, mask = chain_template_arrays(query_len, entries, struct_dir)
    return pos[0, :, 1], mask[0, :, 1]
