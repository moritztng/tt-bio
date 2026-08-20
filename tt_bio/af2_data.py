"""AF2-IG input features: the dict ColabDesign hands AlphaFold2, built without JAX.

Two configurations, matching PXDesign's two AF2 stages:

* `complex_features` -- `protocol="binder"` with templates on, the target's coordinates as the
  initial guess (`pxdbench/tools/af2/main_af2_complex.py:130-146`).
* `monomer_features` -- `protocol="hallucination"`, templates off, no structure at all
  (`main_af2_monomer.py:120-128`).

Scored bit-exact, key by key, against a captured production forward pass
(`scripts/af2_port/parity_artifacts/laczc128_b80/ref_inputs.npz`, 33/33). Three details carry
that bit-exactness and are easy to lose:

1. **Coordinates go decimal -> float32 -> float64.** AlphaFold parses PDBs through Biopython,
   whose `Atom.coord` is float32, then stores it in a float64 array. Every geometric quantity
   derived on top (the virtual CB below) is computed in float64 from float32-rounded inputs, and
   only rounded to float32 on the way into the feature dict. Parsing straight to float64, or
   doing the geometry in float32, both move the virtual CB by one float32 ulp.
2. **Glycines get a virtual CB** built from N/CA/C (`colabdesign/af/prep.py:397-407`), and it
   feeds `template_all_atom_positions` and `template_pseudo_beta`.
3. **`restype_atom{14,37}_mask` must be AF2's, not the vendored ESM copy's.** The two differ on
   restype 20: AF2 gives `UNK` no atoms, ESM gives it N/CA/C. The row is unreachable from a
   20-letter design sequence, which is exactly why it gets pinned here rather than trusted.

Chain concatenation is target-then-binder with a +50 jump in `residue_index` at the break, and
that is the only thing the trunk knows about the chain boundary -- the monomer config has no asym
embedding, so `asym_id`/`sym_id`/`entity_id` exist for i_pTM alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tt_bio._vendor.esm.utils import residue_constants as _rc

ATOM_TYPES: list[str] = list(_rc.atom_types)
ATOM_ORDER: dict[str, int] = dict(_rc.atom_order)
NUM_ATOM = len(ATOM_TYPES)
NUM_RESTYPE = 20
UNKNOWN_RESTYPE = 20
TEMPLATE_MASKED_AATYPE = 21
CHAIN_INDEX_GAP = 50

# CB placement from N/CA/C: length, angle, dihedral (colabdesign/shared/protein.py:195).
_CB_GEOM = (1.522, 1.927, -2.143)


def _atom_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """AF2's four `update_aatype` tables (`colabdesign/af/inputs.py:130-137`).

    Rows 0-19 come from the atom14 name lists, which the vendored ESM copy shares with AF2
    bit-for-bit. Row 20 (`UNK`) is AF2's: no atoms, both index maps zero
    (`residue_constants.py:782-792`). ESM's `UNK` has N/CA/C, so it must not be read here.
    """
    atom14_mask = np.zeros((21, 14), np.float32)
    atom37_mask = np.zeros((21, NUM_ATOM), np.float32)
    atom14_to_atom37, atom37_to_atom14 = [], []
    for restype in _rc.restypes:
        names = _rc.restype_name_to_atom14_names[_rc.restype_1to3[restype]]
        index14 = {name: i for i, name in enumerate(names) if name}
        atom14_to_atom37.append([ATOM_ORDER[n] if n else 0 for n in names])
        atom37_to_atom14.append([index14.get(n, 0) for n in ATOM_TYPES])
        row = len(atom14_to_atom37) - 1
        for i, name in enumerate(names):
            if name:
                atom14_mask[row, i] = 1.0
                atom37_mask[row, ATOM_ORDER[name]] = 1.0
    atom14_to_atom37.append([0] * 14)
    atom37_to_atom14.append([0] * NUM_ATOM)
    return (atom14_mask, atom37_mask,
            np.array(atom14_to_atom37, np.int32), np.array(atom37_to_atom14, np.int32))


RESTYPE_ATOM14_MASK, RESTYPE_ATOM37_MASK, RESTYPE_ATOM14_TO_ATOM37, RESTYPE_ATOM37_TO_ATOM14 = \
    _atom_tables()


@dataclass(frozen=True)
class Chain:
    """One parsed PDB chain in atom37 layout. Coordinates are float64 of float32 values."""

    aatype: np.ndarray          # (L,) int64, 20 = unknown residue
    positions: np.ndarray       # (L, 37, 3) float64
    mask: np.ndarray            # (L, 37) float64
    residue_index: np.ndarray   # (L,) int64, the PDB residue numbers

    def __len__(self) -> int:
        return len(self.aatype)


def parse_pdb_chain(path: str, chain_id: str) -> Chain:
    """Read one chain of a PDB into atom37, matching `protein.from_pdb_string`.

    Only `ATOM` records of the named chain, first model, first altloc of each atom name. Atoms
    outside the 37 canonical names are dropped, non-standard residues become `UNK`, and a residue
    with no recognised atom at all is skipped. Insertion codes are rejected rather than silently
    collapsed, as upstream does.
    """
    residues: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    model = 1
    with open(path, "rb") as fh:
        for raw in fh:
            line = raw.decode("utf-8", "ignore").rstrip("\n")
            if line[:5] == "MODEL":
                model = int(line[5:])
            if model != 1 or line[:4] != "ATOM" or line[21:22] != chain_id:
                continue
            if line[26:27].strip():
                raise ValueError(f"insertion code at chain {chain_id} residue {line[22:27]!r}")
            atom_name, res_name, res_seq = line[12:16].strip(), line[17:20], line[22:26].strip()
            key = (res_seq, res_name)
            if (res_seq, res_name, atom_name) in seen:
                continue
            seen.add((res_seq, res_name, atom_name))
            if key not in residues:
                residues[key] = {"pos": np.zeros((NUM_ATOM, 3)), "mask": np.zeros(NUM_ATOM),
                                 "res_name": res_name, "res_seq": int(res_seq)}
                order.append(key)
            if atom_name not in ATOM_ORDER:
                continue
            i = ATOM_ORDER[atom_name]
            # decimal -> float32 (Biopython's Atom.coord) -> float64 (AlphaFold's pos array)
            residues[key]["pos"][i] = [np.float64(np.float32(float(line[30 + 8 * k:38 + 8 * k])))
                                       for k in range(3)]
            residues[key]["mask"][i] = 1.0

    aatype, positions, mask, residue_index = [], [], [], []
    for key in order:
        res = residues[key]
        if res["mask"].sum() < 0.5:
            continue
        letter = _rc.restype_3to1.get(res["res_name"], "X")
        aatype.append(_rc.restype_order.get(letter, UNKNOWN_RESTYPE))
        positions.append(res["pos"])
        mask.append(res["mask"])
        residue_index.append(res["res_seq"])
    if not aatype:
        raise ValueError(f"chain {chain_id!r} of {path} has no residues")
    return Chain(np.array(aatype), np.array(positions), np.array(mask), np.array(residue_index))


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.square(x).sum(-1, keepdims=True) + 1e-8)


def add_virtual_cb(chain: Chain) -> Chain:
    """Fill in CB where the structure has none, most often glycine.

    `colabdesign/af/prep.py:397-407`. Runs in float64; the mask gains CB wherever N, CA and C are
    all present, and a real CB is never overwritten.
    """
    length, angle, dihedral = _CB_GEOM
    c, n, ca = chain.positions[:, 2], chain.positions[:, 0], chain.positions[:, 1]
    bc = _unit(n - ca)
    normal = _unit(np.cross(n - c, bc))
    virtual = ca + sum([length * np.cos(angle) * bc,
                        length * np.sin(angle) * np.cos(dihedral) * np.cross(normal, bc),
                        length * np.sin(angle) * np.sin(dihedral) * -normal])
    cb = ATOM_ORDER["CB"]
    have_backbone = chain.mask[:, 0] * chain.mask[:, 1] * chain.mask[:, 2]
    positions, mask = chain.positions.copy(), chain.mask.copy()
    positions[:, cb, :] = np.where(mask[:, cb, None].astype(bool), positions[:, cb, :], virtual)
    mask[:, cb] = (mask[:, cb] + have_backbone) > 0
    return Chain(chain.aatype, positions, mask, chain.residue_index)


def _concat_chains(path: str, chain_ids: list[str]) -> tuple[Chain, list[int]]:
    """Parse and concatenate chains, jumping `residue_index` by +50 at each break.

    Residues with no N are dropped (`prep_pdb(..., ignore_missing=True)`; upstream's docstring
    says CA but the code tests atom37 index 0, which is N).
    """
    parts, lengths, last = [], [], 0
    for chain_id in chain_ids:
        chain = add_virtual_cb(parse_pdb_chain(path, chain_id))
        keep = chain.mask[:, 0] == 1
        index = chain.residue_index[keep] + last
        last = index[-1] + CHAIN_INDEX_GAP
        parts.append(Chain(chain.aatype[keep], chain.positions[keep], chain.mask[keep], index))
        lengths.append(int(keep.sum()))
    cat = lambda attr: np.concatenate([getattr(p, attr) for p in parts], 0)
    return Chain(cat("aatype"), cat("positions"), cat("mask"), cat("residue_index")), lengths


def _one_hot20(aatype: np.ndarray) -> np.ndarray:
    """One-hot over the 20 standard types. An unknown residue is all-zero, as upstream's
    `jax.nn.one_hot(20, 20)` is."""
    out = np.zeros((len(aatype), NUM_RESTYPE), np.float32)
    known = aatype < NUM_RESTYPE
    out[np.arange(len(aatype))[known], aatype[known]] = 1.0
    return out


def aatype_from_sequence(sequence: str) -> np.ndarray:
    return np.array([_rc.restype_order[c] for c in sequence], np.int64)


def _sequence_features(one_hot: np.ndarray) -> dict[str, np.ndarray]:
    """`update_seq` + `update_aatype` (`colabdesign/af/inputs.py:109-140`).

    The design loop's soft sequence collapses to a one-hot here: production runs `hard=1.0` and
    `pssm_hard=True`, so the profile block of `msa_feat` is the same one-hot as its sequence
    block.
    """
    num_res = len(one_hot)
    aatype = one_hot.argmax(-1).astype(np.int32)
    padded = np.zeros((1, num_res, 22), np.float32)
    padded[0, :, :NUM_RESTYPE] = one_hot
    msa_feat = np.zeros((1, num_res, 49), np.float32)
    msa_feat[..., 0:22] = padded
    msa_feat[..., 25:47] = padded
    keep = np.ones((num_res, 1), bool)
    return {
        "aatype": aatype,
        "target_feat": one_hot,
        "msa_feat": msa_feat,
        "atom14_atom_exists": np.where(keep, RESTYPE_ATOM14_MASK[aatype], 0).astype(np.float32),
        "atom37_atom_exists": np.where(keep, RESTYPE_ATOM37_MASK[aatype], 0).astype(np.float32),
        "residx_atom14_to_atom37":
            np.where(keep, RESTYPE_ATOM14_TO_ATOM37[aatype], 0).astype(np.int32),
        "residx_atom37_to_atom14":
            np.where(keep, RESTYPE_ATOM37_TO_ATOM14[aatype], 0).astype(np.int32),
    }


def _blank_features(num_res: int) -> dict[str, np.ndarray]:
    """The parts of `prep_input_features` the trunk still reads when nothing fills them.

    AF2-IG is single-sequence, so the extra-MSA block is all zeros with one row -- including
    `extra_msa_mask`, which makes the extra-MSA stack's own MSA track dead while its pair track
    still runs.
    """
    return {
        "seq_mask": np.ones(num_res, np.float32),
        "msa_mask": np.ones((1, num_res), np.float32),
        "msa_row_mask": np.ones(1, np.float32),
        "extra_deletion_value": np.zeros((1, num_res), np.float32),
        "extra_has_deletion": np.zeros((1, num_res), np.float32),
        "extra_msa": np.zeros((1, num_res), np.int32),
        "extra_msa_mask": np.zeros((1, num_res), np.float32),
        "extra_msa_row_mask": np.zeros(1, np.float32),
        "all_atom_positions": np.zeros((1, NUM_ATOM, 3), np.float32),
    }


def _blank_template(num_res: int) -> dict[str, np.ndarray]:
    return {
        "template_aatype": np.zeros((1, num_res), np.int32),
        "template_all_atom_mask": np.zeros((1, num_res, NUM_ATOM), np.float32),
        "template_all_atom_positions": np.zeros((1, num_res, NUM_ATOM, 3), np.float32),
        "template_mask": np.zeros(1, np.float32),
        "template_pseudo_beta": np.zeros((1, num_res, 3), np.float32),
        "template_pseudo_beta_mask": np.zeros((1, num_res), np.float32),
    }


def _chain_ids(lengths: list[int]) -> np.ndarray:
    return np.concatenate([np.full(n, i, np.int32) for i, n in enumerate(lengths)])


def complex_features(pdb_path: str, binder_sequence: str, target_chain: str = "A",
                     binder_chain: str = "B", *, rm_target_seq: bool = True,
                     rm_target_sc: bool = False, rm_binder_seq: bool = True,
                     rm_binder_sc: bool = True,
                     rm_template_interchain: bool = True) -> dict[str, np.ndarray]:
    """Features for the templated two-chain pass, PXDesign's `af2_complex` stage.

    The defaults are the production call. `rm_target_sc=False` is deliberately kept as an
    argument even though it cannot do anything: `inputs.py:61` computes
    `rm_sc = where(rm_seq, True, rm_sc)`, so dropping the template sequence drops the sidechains
    with it and the template is backbone+CB on both chains either way.
    """
    chain, lengths = _concat_chains(pdb_path, [target_chain, binder_chain])
    num_target, num_res = lengths[0], sum(lengths)
    binder_aatype = aatype_from_sequence(binder_sequence)
    if len(binder_aatype) != lengths[1]:
        raise ValueError(f"binder sequence is {len(binder_aatype)} residues, chain "
                         f"{binder_chain!r} of {pdb_path} has {lengths[1]}")

    one_hot = np.concatenate([_one_hot20(chain.aatype[:num_target]),
                              _one_hot20(binder_aatype)], 0)
    per_chain = lambda target, binder: np.concatenate(
        [np.full(num_target, target, bool), np.full(num_res - num_target, binder, bool)])
    rm = np.zeros(num_res, bool)
    rm_seq = np.where(rm, True, per_chain(rm_target_seq, rm_binder_seq))
    rm_sc_arg = per_chain(rm_target_sc, rm_binder_sc)
    rm_sc = np.where(rm_seq, True, rm_sc_arg)

    positions = chain.positions.astype(np.float32)
    mask = chain.mask.astype(np.float32)
    template_mask = mask.copy()
    template_mask[..., 5:] = np.where(rm_sc[:, None], 0, template_mask[..., 5:])
    template_mask = np.where(rm[:, None], 0, template_mask)
    # pseudo_beta_fn is handed `where(rm_seq, 0, aatype)`, so a masked-out sequence reads as
    # alanine and every residue takes CB -- glycine included, which is why the virtual CB above
    # has to exist (`inputs.py:75-78`, `modules.py:1221-1234`).
    is_glycine = np.where(rm_seq, 0, chain.aatype) == _rc.restype_order["G"]
    cb = ATOM_ORDER["CB"]
    ca = ATOM_ORDER["CA"]

    features = {
        **_blank_features(num_res),
        **_sequence_features(one_hot),
        "residue_index": chain.residue_index.astype(np.int32),
        "template_aatype":
            np.where(rm_seq, TEMPLATE_MASKED_AATYPE, chain.aatype)[None].astype(np.int32),
        "template_all_atom_mask": template_mask[None],
        "template_all_atom_positions": positions[None],
        "template_mask": np.ones(1, np.float32),
        "template_pseudo_beta":
            np.where(is_glycine[:, None], positions[:, ca, :], positions[:, cb, :])[None],
        "template_pseudo_beta_mask":
            np.where(is_glycine, mask[:, ca], mask[:, cb]).astype(np.float32)[None],
        "batch/aatype": chain.aatype.astype(np.int32),
        "batch/all_atom_positions": positions,
        "batch/all_atom_mask": mask,
        "rm_template": rm,
        "rm_template_seq": rm_seq,
        "rm_template_sc": rm_sc_arg,
        "mask_template_interchain": np.array(rm_template_interchain),
    }
    chain_id = _chain_ids(lengths)
    features.update(asym_id=chain_id, sym_id=chain_id, entity_id=chain_id)
    return features


def monomer_features(binder_sequence: str) -> dict[str, np.ndarray]:
    """Features for the template-free single-chain pass, PXDesign's `af2_monomer` stage.

    `protocol="hallucination"` takes no structure, so there is no `batch`, no template and no
    initial guess. `residue_index` is 0-based here where the complex path is 1-based: upstream
    uses `np.arange(length)` rather than PDB numbering (`prep.py:167`).
    """
    num_res = len(binder_sequence)
    one_hot = _one_hot20(aatype_from_sequence(binder_sequence))
    chain_id = np.zeros(num_res, np.int32)
    return {
        **_blank_features(num_res),
        **_blank_template(num_res),
        **_sequence_features(one_hot),
        "residue_index": np.arange(num_res, dtype=np.int32),
        "asym_id": chain_id, "sym_id": chain_id, "entity_id": chain_id,
        "mask_template_interchain": np.array(False),
    }


def initial_recycle_state(features: dict[str, np.ndarray], *,
                          initial_guess: bool = True) -> dict[str, np.ndarray]:
    """The `prev` dict for the first of four forward passes (`colabdesign/af/design.py:160-172`).

    With `use_initial_guess` the pair and MSA state start at zero but the atom positions start at
    the design's own coordinates. That is the "IG" in AF2-IG.
    """
    num_res = len(features["residue_index"])
    positions = (features["batch/all_atom_positions"]
                 if initial_guess and "batch/all_atom_positions" in features
                 else np.zeros((num_res, NUM_ATOM, 3), np.float32))
    return {
        "prev_msa_first_row": np.zeros((num_res, 256), np.float32),
        "prev_pair": np.zeros((num_res, num_res, 128), np.float32),
        "prev_pos": positions,
    }
