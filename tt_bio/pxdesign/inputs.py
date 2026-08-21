"""PXDesign's input path: a target structure file to the model-ready 18-key input dict.

`ProtenixDesign.design` eats exactly the keys `capture_ref_design_f.py:MODEL_INPUT_KEYS`
lists. Most of them are plain Protenix features that `tt_bio.protenix_data` already builds
bit-exact against the reference; what is design-specific is the 36-way `restype` with the
binder placeholder, the `hotspot` channel and the conditioning distogram
(`tt_bio.pxdesign.featurize`).

Going through gemmi rather than through protenix's own parser is not just about dropping a
pinned dependency. PXDesign reads a user's CIF with protenix's `DistillationMMCIFParser`,
a parallel entry point that skips the `remove_water` / `remove_hydrogens` the production
`MMCIFParser.get_bioassembly` runs, so hydrogens reach the CCD atom-name match and a residue
with at least as many hydrogens as heavy atoms is thrown away. This path filters first, so
the bug cannot happen here: `examples/5o45.cif` conditions on all 116 residues, not 55.

`ref_pos` is the one feature that cannot be compared against a capture. Upstream builds the
`Featurizer` with `ref_pos_augment` left at its default `True`, which applies a random
rotation and translation per residue from the unseeded global numpy RNG, so upstream's own
`ref_pos` does not reproduce run to run. This path emits the committed conformer table
unaugmented, which is strictly more reproducible.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..protenix_data import build_complex_features, structure_token_coords
from .featurize import BINDER_PLACEHOLDER, RESTYPE_VOCAB, condition_template, restype_onehot

MODEL_INPUT_KEYS = (
    "ref_pos", "ref_charge", "ref_element", "ref_atom_name_chars", "ref_mask",
    "ref_space_uid", "atom_to_token_idx",
    "restype", "hotspot", "deletion_mean",
    "asym_id", "residue_index", "entity_id", "sym_id", "token_index",
    "conditional_templ", "conditional_templ_mask",
)


def read_design_yaml(path) -> dict:
    """A PXDesign target YAML to `{structure, chains, crop, hotspots, binder_length}`.

    Schema per `pxdesign/utils/inputs.py`: `target.file`, `target.chains.<label_asym_id>`
    with optional `crop` / `hotspots` / `msa`, and a top-level `binder_length`. A chain
    mapped to null, `all` or `full` means the whole chain. `msa` is accepted and ignored:
    PXDesign-d has no trunk, so generation reads no alignment.
    """
    import yaml

    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    target = cfg.get("target")
    if not target or not target.get("file"):
        raise ValueError(f"{path}: missing target.file")
    if not target.get("chains"):
        raise ValueError(f"{path}: missing target.chains")
    structure = Path(target["file"])
    if not structure.is_absolute():
        # Upstream resolves against the working directory; a committed fixture YAML wants to
        # resolve against its own directory. Take whichever exists, so both work.
        beside = (path.parent / structure).resolve()
        structure = beside if beside.exists() else structure.resolve()
    if not structure.exists():
        raise FileNotFoundError(f"{path}: target.file {target['file']} not found")
    crop, hotspots = {}, {}
    for cid, props in target["chains"].items():
        cid = str(cid)
        if props is None or (isinstance(props, str) and props.lower() in ("all", "full")):
            props = {}
        if props.get("crop") is not None:
            crop[cid] = props["crop"]
        if props.get("hotspots"):
            hotspots[cid] = [int(h) for h in props["hotspots"]]
    binder_length = cfg.get("binder_length")
    if not binder_length:
        raise ValueError(f"{path}: missing binder_length")
    return {"structure": structure, "chains": [str(c) for c in target["chains"]],
            "crop": crop, "hotspots": hotspots, "binder_length": int(binder_length)}


def design_inputs(structure, chains, binder_length: int, crop=None, hotspots=None) -> dict:
    """Target structure plus a binder length to the 18-key `ProtenixDesign.design` input.

    Target chains come first in `chains` order, the designed binder last, which is the
    token order upstream produces and the order every committed capture carries.
    """
    toks = structure_token_coords(structure, chains, crop)
    hotspots = hotspots or {}
    if binder_length < 1:
        raise ValueError(f"design_inputs: binder_length must be positive, got {binder_length}")

    entries = [toks[c] for c in chains]
    feats = build_complex_features(
        [(e["sequence"], None, e["mol_type"][0]) for e in entries]
        + [("G" * binder_length, None, "protein")],
        # The binder is built from a sequence, so it carries a real C-terminus; a target
        # chain carries OXT only where the structure file shows one, which a mid-sequence
        # crop never does.
        oxt=[e["has_oxt"] for e in entries] + [None])

    res_name = [n for e in entries for n in e["res_name"]] + [BINDER_PLACEHOLDER] * binder_length
    mol_type = [m for e in entries for m in e["mol_type"]] + ["protein"] * binder_length
    coord = torch.cat([e["coord"] for e in entries]
                      + [torch.zeros(binder_length, 3)], dim=0)
    is_resolved = torch.cat([e["is_resolved"] for e in entries]
                            + [torch.ones(binder_length, dtype=torch.bool)], dim=0)
    # The binder is not a condition, so its placeholder name excludes it from the distogram
    # (`condition_template`) regardless of the flag; `is_resolved` stays True to match
    # upstream, whose design chains are annotated resolved.

    residue_index = torch.tensor(
        [s for c in chains for s in toks[c]["label_seq"]] + list(range(1, binder_length + 1)),
        dtype=torch.long)
    hotspot = torch.tensor(
        [1.0 if s in set(hotspots.get(c, ())) else 0.0
         for c in chains for s in toks[c]["label_seq"]] + [0.0] * binder_length)

    n_token = len(res_name)
    out = {k: feats[k] for k in ("ref_pos", "ref_charge", "ref_element",
                                 "ref_atom_name_chars", "ref_mask", "ref_space_uid",
                                 "atom_to_token_idx", "asym_id", "entity_id", "sym_id")}
    out["restype"] = restype_onehot(
        [n if n in RESTYPE_VOCAB else "UNK" for n in res_name])
    out["hotspot"] = hotspot
    out["deletion_mean"] = torch.zeros(n_token)
    out["residue_index"] = residue_index
    out["token_index"] = torch.arange(n_token)
    out.update(condition_template(coord, res_name, mol_type, is_resolved))
    # Not model inputs: what a scorer needs to check the generated fold against the target.
    out["condition"] = {"coord": coord, "res_name": res_name, "mol_type": mol_type,
                        "is_resolved": is_resolved}
    out["distogram_rep_atom_mask"] = feats["distogram_rep_atom_mask"]

    if out["restype"].shape[0] != n_token:
        raise AssertionError("restype/token count disagree")
    for k in MODEL_INPUT_KEYS:
        if k not in out:
            raise AssertionError(f"design_inputs did not produce {k}")
    return out


def design_inputs_from_yaml(path) -> dict:
    """`read_design_yaml` then `design_inputs`. This is the whole user-facing input path."""
    spec = read_design_yaml(path)
    return design_inputs(spec["structure"], spec["chains"], spec["binder_length"],
                         crop=spec["crop"], hotspots=spec["hotspots"])
