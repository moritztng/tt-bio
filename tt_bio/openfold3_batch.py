"""Batch-container rules for the OpenFold3 features whose batch axis is a Python list.

Almost every entry of an OpenFold3 feature dict is a tensor, so a batch of them is the
same dict with one extra leading dimension and a tensor tree map is the right tool.
ref_space_uid_to_perm is the exception, and it is the reason this module exists.

The feature maps a reference-space uid to the symmetry-equivalent atom permutations of
that conformer: {int uid: int32[n_perm, n_atoms_in_conformer]}. Different
conformers have different atom counts and different permutation counts, so the entries
cannot be stacked. Upstream therefore gives the feature a **list** batch axis:
openfold_batch_collator sets batch[key] to a list of the per-sample mappings,
and multi_chain_permutation_alignment recovers a sample with value[i] on that
list. atom_array is carried the same way.

Three rules follow. Every one of them has been broken here at least once, each time by
a different caller writing its own workaround:

1. **A tensor tree map must not descend into it.** tensor_tree_map treats
   {uid: tensor} as a tree and maps every permutation tensor, so a sample-dim
   expansion silently gives each permutation a phantom axis and a collate stacks
   mappings that must not be stacked.
2. **A transform must not strip it from the caller's dict.** The usual workaround pops
   the key, maps the rest, and puts the key back -- but the map returns a *new* dict, so
   the restore lands there and the caller's dict has lost the key for good. A second
   pass over that dict raises KeyError: 'ref_space_uid_to_perm'.
3. **Its int keys are the uids in ref_space_uid.** A sample whose mapping is missing
   a uid that its ref_space_uid names is malformed, whatever its type looks like.

Both failures land in safe_multi_chain_permutation_alignment, which catches them and
falls back to naive alignment with one log line, so nothing crashes and the run trains
against a differently-aligned ground truth.

None of this reaches inference: build_openfold3_features calls the conformer
featurizer with add_ref_space_uid_to_perm=False, so the key is training-only.
"""

import torch

from tt_bio._vendor.openfold3.core.utils.tensor_utils import tensor_tree_map

PERM_KEY = "ref_space_uid_to_perm"

#: Features whose batch axis is a Python list of per-sample objects rather than a
#: tensor dimension. No tensor tree map may descend into these.
LIST_AXIS_FEATURES = frozenset({PERM_KEY, "atom_array"})


def map_batch_tensors(batch: dict, fn) -> dict:
    """Map fn over every feature tensor, carrying the list-axis features across.

    Use this instead of tensor_tree_map on a feature dict. It obeys rules 1 and 2:
    it does not descend into the list-axis features, and it leaves batch itself
    untouched, so the caller can hand the same dict to a second forward.
    """
    carried = {k: batch[k] for k in batch if k in LIST_AXIS_FEATURES}
    mapped = tensor_tree_map(fn, {k: v for k, v in batch.items() if k not in carried})
    mapped.update(carried)
    return mapped


def collate_list_axis(samples: list[dict]) -> tuple[list[dict], dict]:
    """Split the list-axis features off a list of samples so the rest can be stacked.

    Returns the samples with those keys removed (copies, the inputs are untouched) and
    the batch-level entries to merge back in after stacking. This is what
    openfold_batch_collator does, minus the mutation of its inputs.
    """
    stripped = [{k: v for k, v in s.items() if k not in LIST_AXIS_FEATURES} for s in samples]
    keys = {k for s in samples for k in s if k in LIST_AXIS_FEATURES}
    batched = {k: [s[k] for s in samples] for k in keys}
    return stripped, batched


def check_perm_contract(
    batch: dict, batch_size: int | None = None, required: bool = False
) -> None:
    """Raise ValueError if ref_space_uid_to_perm violates its contract.

    batch_size=None checks a single unbatched sample, where the value is the mapping
    itself; an int checks a batch, where the value is a list of that many mappings.
    ``required`` asserts the key is present, which is how a training harness catches
    a transform that popped it off the dict; leave it False for a batch that may
    legally not carry it, as an inference batch does not.

    This is the check that catches both shipped defects before the alignment swallows
    them: a recursed mapping fails the type, shape and uid rules, a popped one fails
    the presence rule.
    """
    if PERM_KEY not in batch:
        if required:
            raise ValueError(f"{PERM_KEY} is missing from a batch that requires it")
        return

    value = batch[PERM_KEY]
    if batch_size is None:
        mappings, uids = [value], [batch.get("ref_space_uid")]
    else:
        if not isinstance(value, list):
            raise ValueError(
                f"{PERM_KEY} must be a list of {batch_size} per-sample mappings, "
                f"got {type(value).__name__} -- a tree map descended into it"
            )
        if len(value) != batch_size:
            raise ValueError(
                f"{PERM_KEY} has {len(value)} entries for batch size {batch_size}"
            )
        mappings = value
        all_uids = batch.get("ref_space_uid")
        uids = [None] * batch_size if all_uids is None else list(all_uids)

    for i, mapping in enumerate(mappings):
        if not isinstance(mapping, dict):
            raise ValueError(
                f"{PERM_KEY}[{i}] must be a uid -> permutations mapping, "
                f"got {type(mapping).__name__}"
            )
        for uid, perms in mapping.items():
            if not isinstance(uid, int) or not torch.is_tensor(perms):
                raise ValueError(
                    f"{PERM_KEY}[{i}] must map int uids to tensors, "
                    f"got {type(uid).__name__} -> {type(perms).__name__}"
                )
            if perms.ndim != 2:
                raise ValueError(
                    f"{PERM_KEY}[{i}][{uid}] must be [n_perm, n_atom], "
                    f"got shape {tuple(perms.shape)} -- a tree map added an axis"
                )
        if uids[i] is None:
            continue
        missing = sorted({int(u) for u in uids[i].flatten().tolist()} - set(mapping))
        if missing:
            raise ValueError(
                f"{PERM_KEY}[{i}] has no entry for ref_space_uid {missing[:8]}"
            )
