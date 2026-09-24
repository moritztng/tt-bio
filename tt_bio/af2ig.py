"""AF2-IG as a predict model: a designed complex in, its interface confidence out.

The trunk, the featurizer and the confidence heads were already here -- `af2.py` runs the
evoformer on ttnn, `af2_data.complex_features` is ColabDesign's `protocol="binder"` input
exactly, and `af2_confidence.confidence_scalars` returns what the design loop logs. What was
missing was the route from a request to them, which is this module plus its rows in
`capabilities.py`, `main.py` and `worker.py`.

**What AF2-IG answers.** Not "what shape is this sequence" -- it re-predicts a complex you
already built, starting from that complex's own coordinates (the initial guess), and the
interface numbers say whether the design is real. So its input is a designed complex: a
structure carrying the target chain AND the binder backbone, plus the binder's sequence. That
is one file more than every other model here takes, and it is why af2ig gets its own reader
rather than riding `_read_bio_chains`:

    target:
      structure: |          # or `file: target.pdb`, for a CLI user
        ATOM      1  N   MET A   1      ...
      chain: A
    binder:
      sequence: SPEDEIQALEEKNAQLKQEIAALEEKIQALKY
      chain: B

The binder chain must be present in the structure and the same length as the sequence: the
backbone is what the model starts from, and the sequence is what it is asked to place on it.
A sequence with no backbone under it is a different question (fold it with one of the
cofolders); the reader says so rather than guessing.

**No sampling, no MSA.** Four forward passes with the recycling state threaded, single
sequence on both chains, no diffusion and no seed -- the same input gives the same structure.
"""
from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_TARGET_CHAIN = "A"
DEFAULT_BINDER_CHAIN = "B"

#: Every key a submission may carry. Anything else is refused rather than ignored, so no option
#: can be sent and silently dropped. af2ig has none: it always starts from the design's own
#: coordinates, and starting from zeros would be single-sequence AF2 under this name.
_KEYS = {"the document": frozenset({"target", "binder"}),
         "target": frozenset({"structure", "file", "chain"}),
         "binder": frozenset({"sequence", "chain"})}

#: `af2_data.CHAIN_INDEX_GAP`, re-derived here only to undo it when writing the structure back.
_CHAIN_GAP = 50


@dataclass(frozen=True)
class AF2IGInput:
    """One af2ig submission: the designed complex, and the sequence to place on its binder."""

    structure: str
    binder_sequence: str
    target_chain: str = DEFAULT_TARGET_CHAIN
    binder_chain: str = DEFAULT_BINDER_CHAIN


def _as_text(value, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _chain_id(value, where: str, default: str) -> str:
    if value is None:
        return default
    cid = str(value).strip()
    if len(cid) != 1 or cid not in string.ascii_letters + string.digits:
        raise ValueError(f"{where} must be a single-character chain id, got {cid!r}")
    return cid


def _looks_like_cif(text: str) -> bool:
    return any(line.startswith(("data_", "_atom_site.", "loop_"))
               for line in text.splitlines()[:200])


def to_pdb_text(text: str) -> str:
    """The structure as PDB text, converting mmCIF when that is what arrived.

    A participant designing on this platform gets mmCIF back (`tt-bio design` writes CIFs),
    so refusing mmCIF here would break the one route that matters most: design, then score
    what you designed.
    """
    if not _looks_like_cif(text):
        return text
    import io

    import biotite.structure.io.pdb as _pdb
    import biotite.structure.io.pdbx as _pdbx

    arr = _pdbx.get_structure(_pdbx.CIFFile.read(io.StringIO(text)), model=1)
    out = _pdb.PDBFile()
    out.set_structure(arr)
    buf = io.StringIO()
    out.write(buf)
    return buf.getvalue()


def read_af2ig_input(path) -> AF2IGInput:
    """Parse an af2ig YAML. Raises ValueError naming what is missing, never a KeyError."""
    import yaml

    path = Path(path)
    try:
        doc = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ValueError(f"{path.name} is not valid YAML: {e}") from None
    if not isinstance(doc, dict):
        raise ValueError(f"{path.name} must be a YAML mapping with `target:` and `binder:`")
    target, binder = doc.get("target"), doc.get("binder")
    if not isinstance(target, dict) or not isinstance(binder, dict):
        missing = [k for k in ("target", "binder") if not isinstance(doc.get(k), dict)]
        hint = ("--model af2ig scores a complex you already designed, so it takes a "
                "structure plus the binder's sequence, not a sequence list. "
                "To fold a sequence on its own, use one of the cofolders.")
        raise ValueError(f"{path.name} has no `{'`/`'.join(missing)}:` block. {hint}")
    for where, block in (("the document", doc), ("target", target), ("binder", binder)):
        extra = sorted(map(str, set(block) - _KEYS[where]))
        if extra:
            raise ValueError(f"{path.name}: {where} does not take {', '.join(extra)} "
                             f"(only {', '.join(sorted(_KEYS[where]))}); af2ig has no options")
    if target.get("structure") is not None and target.get("file") is not None:
        raise ValueError("target: give either `structure:` (inline text) or `file:`, not both")
    if target.get("structure") is not None:
        text = _as_text(target["structure"], "target.structure")
    elif target.get("file") is not None:
        ref = Path(_as_text(target["file"], "target.file")).expanduser()
        ref = ref if ref.is_absolute() else (path.parent / ref)
        if not ref.is_file():
            raise ValueError(f"target.file {ref} does not exist")
        text = ref.read_text()
    else:
        raise ValueError("target: needs `structure:` (inline PDB/mmCIF text) or `file:`")
    sequence = _as_text(binder.get("sequence"), "binder.sequence").strip().upper()
    unknown = sorted(set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"))
    if unknown:
        raise ValueError(f"binder.sequence has non-standard residue(s) {''.join(unknown)}; "
                         "AF2-IG places one of the 20 standard types on each position")
    return AF2IGInput(structure=to_pdb_text(text), binder_sequence=sequence,
                      target_chain=_chain_id(target.get("chain"), "target.chain",
                                             DEFAULT_TARGET_CHAIN),
                      binder_chain=_chain_id(binder.get("chain"), "binder.chain",
                                             DEFAULT_BINDER_CHAIN))


def features(spec: AF2IGInput) -> dict[str, np.ndarray]:
    """`af2_data.complex_features` on this submission, with its errors made actionable.

    The structure is written to a scratch file because the featurizer parses a path: it is
    AlphaFold's own PDB reader (float32 coordinate rounding included), and re-implementing it
    against text would be a second parser to keep bit-exact.
    """
    import tempfile

    from tt_bio.af2_data import complex_features

    with tempfile.TemporaryDirectory(prefix="af2ig-") as tmp:
        pdb = Path(tmp) / "complex.pdb"
        pdb.write_text(spec.structure)
        try:
            return complex_features(str(pdb), spec.binder_sequence,
                                    spec.target_chain, spec.binder_chain)
        except ValueError as e:
            raise ValueError(str(e).replace(str(pdb), "the target structure")) from None


def token_count(spec: AF2IGInput) -> int:
    """Residues the trunk will see: target chain plus binder. Used to refuse before a load."""
    return int(len(features(spec)["residue_index"]))


@dataclass(frozen=True)
class AF2IGPrediction:
    """One fold: the complex as a biotite AtomArray, and the scalars ColabDesign logs."""

    atom_array: object
    coords: np.ndarray            # (atoms, 3) float32, in atom_array order
    b_factors: np.ndarray         # (atoms,) float32, per-residue pLDDT on the 0-100 scale
    metrics: dict[str, float]
    tokens: int
    binder_length: int


def _residue_numbers(feats: dict[str, np.ndarray], num_target: int) -> np.ndarray:
    """The input's own residue numbering, with the featurizer's +50 chain jump undone.

    `af2_data._concat_chains` offsets the binder so the trunk sees a chain break; a user
    reading the output back should see the numbering their file carried.
    """
    index = np.asarray(feats["residue_index"], np.int64).copy()
    index[num_target:] -= index[num_target - 1] + _CHAIN_GAP
    return index


def _atom_array(spec: AF2IGInput, feats: dict[str, np.ndarray], positions: np.ndarray,
                plddt: np.ndarray):
    """The predicted complex in atom37, as the AtomArray the worker's writer takes."""
    import biotite.structure as struc

    from tt_bio.af2_data import ATOM_TYPES
    from tt_bio._vendor.esm.utils import residue_constants as rc

    aatype = np.asarray(feats["aatype"], np.int64)
    exists = np.asarray(feats["atom37_atom_exists"], np.float32) > 0
    asym = np.asarray(feats["asym_id"], np.int64)
    num_target = int((asym == asym[0]).sum())
    numbers = _residue_numbers(feats, num_target)
    chains = np.where(asym == asym[0], spec.target_chain, spec.binder_chain)

    rows, coords, b_factors = [], [], []
    for i, restype in enumerate(aatype):
        letter = rc.restypes[restype] if restype < len(rc.restypes) else "X"
        res_name = rc.restype_1to3.get(letter, "UNK")
        for a in np.flatnonzero(exists[i]):
            rows.append((chains[i], int(numbers[i]), res_name, ATOM_TYPES[a]))
            coords.append(positions[i, a])
            b_factors.append(plddt[i])
    arr = struc.AtomArray(len(rows))
    arr.chain_id = np.array([r[0] for r in rows], dtype="U4")
    arr.res_id = np.array([r[1] for r in rows], dtype=int)
    arr.res_name = np.array([r[2] for r in rows], dtype="U5")
    arr.atom_name = np.array([r[3] for r in rows], dtype="U6")
    arr.element = np.array([r[3][0] for r in rows], dtype="U2")
    arr.coord = np.asarray(coords, np.float32)
    return arr, np.asarray(coords, np.float32), np.asarray(b_factors, np.float32)


def fold(model, spec: AF2IGInput, *, recycles: int = 3, progress=None) -> AF2IGPrediction:
    """`recycles + 1` passes with the recycling state threaded, then the confidence scalars.

    The same sequence `scripts/af2_port/fold_timing.py` and `filter_tolerance.py --arm device`
    run, which is what makes the committed device floor
    (`docs/implementation-parity-data/af2ig-trunk-device.json`) a statement about this path.
    """
    import torch

    from tt_bio.af2_confidence import confidence_scalars, plddt_per_residue
    from tt_bio.af2_data import initial_recycle_state

    feats_np = features(spec)
    to_torch = lambda a: (torch.from_numpy(a) if a.dtype == np.bool_
                          else torch.from_numpy(a.astype(np.int64)) if a.dtype.kind in "iu"
                          else torch.from_numpy(a.astype(np.float32)))
    feats = {k: to_torch(v) for k, v in feats_np.items()}
    # The initial guess is the whole model: without the design's coordinates in prev_pos this is
    # single-sequence AF2 under the af2ig name. So it is passed literally, never a parameter, and
    # initial_recycle_state's zero fallback for a feature dict without positions is refused here.
    if "batch/all_atom_positions" not in feats_np:
        raise ValueError("af2ig needs the design's coordinates for its initial guess, "
                         "and the featurized complex carries none")
    prev = {k: to_torch(v) for k, v in initial_recycle_state(feats_np, initial_guess=True).items()}

    out = None
    with torch.no_grad():
        for cycle in range(recycles + 1):
            if progress is not None:
                progress(cycle, recycles + 1)
            out = model(feats, prev)
            prev = {"prev_msa_first_row": out["msa_first_row"], "prev_pair": out["pair"],
                    "prev_pos": out["structure"]["final_atom_positions"]}

    binder_len = len(spec.binder_sequence)
    scalars = confidence_scalars(out["plddt_logits"], out["pae_logits"], out["pae_breaks"],
                                 feats["seq_mask"], feats["asym_id"], binder_len=binder_len)
    plddt = plddt_per_residue(out["plddt_logits"]).detach().cpu().numpy().astype(np.float32)
    positions = (out["structure"]["final_atom_positions"]
                 .detach().cpu().to(torch.float32).numpy())
    arr, coords, b_factors = _atom_array(spec, feats_np, positions, plddt * 100.0)
    return AF2IGPrediction(atom_array=arr, coords=coords, b_factors=b_factors,
                           metrics=metrics(scalars), tokens=len(plddt),
                           binder_length=binder_len)


def metrics(scalars: dict[str, float]) -> dict[str, float]:
    """The scalars as the results row publishes them.

    `confidence_scalars` names them the way ColabDesign's loss does; these are the names the
    rest of this repo's result rows use, plus AF2-IG's two interface numbers. `i_pae` is the
    0-1 normalised interface pAE the design loop minimises and `interface_pae` the same value
    in Angstrom, because a bare 0.28 means nothing to a reader and 8.7 A does.
    """
    out = {"plddt": round(float(scalars["plddt"]), 4),
           "ptm": round(float(scalars["ptm"]), 4),
           "iptm": round(float(scalars["i_ptm"]), 4),
           "pae": round(float(scalars["pae"]), 4)}
    if "i_pae" in scalars:
        out["ipae"] = round(float(scalars["i_pae"]), 4)
        out["interface_pae"] = round(float(scalars["unscaled_i_pae"]), 4)
    return out
