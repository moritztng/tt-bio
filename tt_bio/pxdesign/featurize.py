"""Host featurization for PXDesign, for the features Protenix does not already produce.

PXDesign conditions a Protenix diffusion module on a target structure, so most of its input
is the Protenix feature set tt-bio already builds. What is new is four things, and only the
first is subtle:

  * ``conditional_templ`` / ``conditional_templ_mask`` -- a 64-bin distogram over the target,
    written ONLY into the sub-block of tokens that are both resolved and not the binder
    placeholder. Get the placeholder wrong and the model is handed the answer it is supposed
    to design, and it will still return a plausible structure, so this is gated rather than
    eyeballed (``scripts/pxdesign_port/parity_gate.py``).
  * ``restype`` -- 36-way, not Protenix's 32-way: four design placeholders on the end.
  * ``hotspot`` -- a per-token 0/1 channel.

``plddt``, ``add_feat1`` and ``add_feat2`` are deliberately absent: upstream's
``InputFeatureEmbedderDesign.forward`` synthesises them from ``deletion_mean`` at model entry
rather than in the featurizer, and this module keeps that split.

Reference: ``pxdesign/data/featurizer.py:847 DesignFeaturizer.get_condition_template_feature``
and ``pxdesign/utils/design.py``.
"""
from __future__ import annotations

import torch

# The placeholder residue name for the binder being designed. Every conditioning mask
# negates it: the target conditions the model, the binder is what the model produces.
BINDER_PLACEHOLDER = "xpb"

# Protenix's 32-way residue vocabulary plus PXDesign's four design placeholders. Order is
# load-bearing -- it is the one-hot column order the checkpoint's `input_map` was trained on.
RESTYPE_VOCAB = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK", "A", "G", "C", "U", "N", "DA", "DG", "DC", "DT", "DN", "-",
    "xpb", "xpa", "rbb", "raa",
)
_RESTYPE_INDEX = {n: i for i, n in enumerate(RESTYPE_VOCAB)}

# 64 bins over 2-22 A, so 63 interior boundaries and `bin = #(dist > boundary)`.
N_TEMPL_BINS = 64
TEMPL_BIN_MIN = 2.0
TEMPL_BIN_MAX = 22.0


def restype_onehot(res_names) -> torch.Tensor:
    """[N_token, 36] one-hot over `RESTYPE_VOCAB`, from per-token canonical residue names."""
    res_names = list(res_names)
    if not res_names:
        raise ValueError("restype_onehot: empty residue list -- no tokens to featurize")
    unknown = sorted({n for n in res_names if n not in _RESTYPE_INDEX})
    if unknown:
        raise ValueError(f"restype_onehot: residue names outside the 36-way design "
                         f"vocabulary: {unknown}")
    out = torch.zeros(len(res_names), len(RESTYPE_VOCAB))
    out[torch.arange(len(res_names)), [_RESTYPE_INDEX[n] for n in res_names]] = 1.0
    return out


def condition_template(coord, res_name, mol_type, is_resolved, templ_token_mask=None,
                       ignore_ligand_only_condition: bool = True) -> dict:
    """The structural conditioning: a 64-bin distogram over the conditioned tokens.

    Inputs are per-DISTOGRAM-REPRESENTATIVE-ATOM, i.e. one entry per token: `coord`
    [N_token, 3], `res_name` and `mol_type` as sequences of strings, `is_resolved` a boolean
    mask. `templ_token_mask` optionally narrows the conditioning further.

    Returns ``conditional_templ`` [N, N] int64 of bin indices and ``conditional_templ_mask``
    [N, N] marking where a bin was actually written. Both are all-zero outside the
    conditioned sub-block, and `ConditionTemplateEmbedder` looks up ``mask * (1 + bin)``, so
    row 0 of its 65-row embedding means "no condition here" -- see `condition_template_index`.

    Two early returns are upstream's, reproduced deliberately: an all-ligand condition is not
    given as a template at all, and neither is a condition with no resolved non-placeholder
    token. Both leave the feature all-zero, which reads at the model as "design freely".
    """
    coord = torch.as_tensor(coord, dtype=torch.float32)
    if coord.ndim != 2 or coord.shape[-1] != 3:
        raise ValueError(f"condition_template: coord must be [N_token, 3], got "
                         f"{tuple(coord.shape)}")
    n = coord.shape[0]
    if n == 0:
        raise ValueError("condition_template: no distogram representative atoms -- the "
                         "target parsed to zero tokens, which is a malformed input, not an "
                         "unconditioned design")
    res_name = list(res_name)
    mol_type = list(mol_type)
    if not (len(res_name) == len(mol_type) == n):
        raise ValueError(f"condition_template: got {n} coords but {len(res_name)} res_names "
                         f"and {len(mol_type)} mol_types")
    is_resolved = torch.as_tensor(is_resolved).bool().reshape(-1)
    if is_resolved.shape[0] != n:
        raise ValueError(f"condition_template: is_resolved has {is_resolved.shape[0]} "
                         f"entries for {n} tokens")

    templ = torch.zeros((n, n), dtype=torch.long)
    mask = torch.zeros((n, n), dtype=torch.bool)
    # The early returns hand back a LONG mask where the main path hands back a BOOL one.
    # That is upstream's own asymmetry and it is kept, because the gate compares dtypes: both
    # promote identically under `mask * (1 + bin)`, so nothing downstream can tell them apart.
    unconditioned = {"conditional_templ": templ, "conditional_templ_mask": mask.long()}

    not_placeholder = torch.tensor([r != BINDER_PLACEHOLDER for r in res_name])
    if ignore_ligand_only_condition:
        is_ligand_condition = not_placeholder & torch.tensor([m == "ligand" for m in mol_type])
        if bool(is_ligand_condition.all()):
            return unconditioned

    conditioned = not_placeholder & is_resolved
    if templ_token_mask is not None:
        conditioned &= torch.as_tensor(templ_token_mask).bool().reshape(-1)
    if not bool(conditioned.any()):
        return unconditioned

    sub = coord[conditioned]
    boundaries = torch.linspace(TEMPL_BIN_MIN, TEMPL_BIN_MAX, N_TEMPL_BINS - 1)
    bins = torch.sum(torch.cdist(sub, sub).unsqueeze(-1) > boundaries, dim=-1)
    idx = torch.nonzero(conditioned, as_tuple=True)[0]
    ii, jj = torch.meshgrid(idx, idx, indexing="ij")
    templ[ii, jj] = bins
    mask[ii, jj] = True
    return {"conditional_templ": templ, "conditional_templ_mask": mask}


def condition_template_index(conditional_templ, conditional_templ_mask) -> torch.Tensor:
    """``mask * (1 + bin)``: the row index into the 65-row conditioning embedding.

    64 bins need 65 rows because row 0 is reserved for "no condition at this pair", which is
    why the shift cannot be folded into the bin computation.
    """
    return torch.as_tensor(conditional_templ_mask).long() * (
        1 + torch.as_tensor(conditional_templ).long())
