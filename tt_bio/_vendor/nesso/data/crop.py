from collections import defaultdict

import numpy as np


def select_pocket_token_indices(
    protein_token_indices: np.ndarray,
    protein_asym_ids: np.ndarray,
    protein_res_idxs: np.ndarray,
    dists_to_ligand: np.ndarray,
    ligand_token_indices: np.ndarray,
    max_tokens: int,
    threshold_distance: float,
) -> list[int]:
    """Select token indices (pocket + ligand) using cutoff, budget, and sequence expansion.

    All ``*_token_indices`` are indices into the same token table (row positions or ``token_idx``).

    Parameters
    ----------
    protein_token_indices
        Shape ``[P]``, one entry per protein token considered.
    protein_asym_ids, protein_res_idxs
        Shape ``[P]``, chain and residue id per protein token (aligned with above).
    dists_to_ligand
        Shape ``[P]``, minimum distance (Å or expected Å) from each protein token to the ligand(s)
        used for the cutoff (caller defines the distance definition).
    ligand_token_indices
        Shape ``[L]``, token indices to always retain (e.g. one target ligand or all ligands).
    max_tokens
        Hard cap on total tokens kept.
    threshold_distance
        Keep protein tokens with ``dists_to_ligand <= threshold_distance``.

    Reference: We follow algorithm 1 from [Terrabind](https://arxiv.org/pdf/2602.07735).

    Returns
    -------
    list[int]
        Sorted token indices to keep.
    """
    if ligand_token_indices.size == 0:
        msg = "select_pocket_token_indices requires at least one ligand token index"
        raise ValueError(msg)

    within_cutoff = dists_to_ligand <= threshold_distance
    valid_non_target_idx = protein_token_indices[within_cutoff]
    valid_dists = dists_to_ligand[within_cutoff]

    sorted_local = np.argsort(valid_dists)
    sorted_indices = valid_non_target_idx[sorted_local]

    cropped: set[int] = set()
    ligand_token_indices = np.atleast_1d(ligand_token_indices).astype(
        np.int64, copy=False
    )
    cropped.update(int(x) for x in ligand_token_indices.tolist())

    num_ligand_tokens = len(ligand_token_indices)
    max_protein_tokens = max_tokens - num_ligand_tokens
    if max_protein_tokens < 0:
        # Rare: ligand alone exceeds budget — keep ligand only (truncate pocket)
        return sorted(cropped)

    if len(sorted_indices) <= max_protein_tokens:
        protein_kept = set[int](int(x) for x in sorted_indices.tolist())
    else:
        protein_kept = set(int(x) for x in sorted_indices[:max_protein_tokens].tolist())
    cropped.update(protein_kept)

    # Contiguous sequence expansion (if under budget)
    if len(cropped) < max_tokens:
        protein_tokens_arr = np.empty(
            len(protein_token_indices),
            dtype=[
                ("token_idx", np.int64),
                ("asym_id", np.int64),
                ("res_idx", np.int64),
            ],
        )
        protein_tokens_arr["token_idx"] = protein_token_indices
        protein_tokens_arr["asym_id"] = protein_asym_ids
        protein_tokens_arr["res_idx"] = protein_res_idxs

        # Select tokens that have been cropped
        cropped_prot = protein_tokens_arr[
            np.isin(protein_tokens_arr["token_idx"], list(cropped))
        ]

        # Chains that have been cropped
        chains_with_cropped = np.unique(cropped_prot["asym_id"])

        # (Chain ID, Residue Index in sequence) -> List of token indices
        res_to_tokens: dict[tuple[int, int], np.ndarray] = defaultdict(list)
        for row in protein_tokens_arr:
            res_to_tokens[(int(row["asym_id"]), int(row["res_idx"]))].append(
                int(row["token_idx"])
            )
        res_to_tokens = {k: np.array(v) for k, v in res_to_tokens.items()}

        candidates: list[tuple[float, int, int]] = []  # Line 18
        for chain_id in chains_with_cropped:  # Line 19
            # Select all tokens that have been cropped in this chain
            chain_cropped = cropped_prot[cropped_prot["asym_id"] == chain_id]
            # Extract sequence index of all cropped residues in this chain
            cropped_res = np.unique(chain_cropped["res_idx"])

            # Get the set of uncropped residues, and then extract their sequence index
            chain_all = protein_tokens_arr[protein_tokens_arr["asym_id"] == chain_id]
            all_res = np.unique(chain_all["res_idx"])

            # just do argsort in sequence space to subselect uncropped residues
            uncropped_res = np.setdiff1d(all_res, cropped_res)
            if len(uncropped_res) == 0:
                continue

            # sort the sequence/residue indices
            sorted_cropped_res = np.sort(cropped_res)  # [N]
            # compute the running difference (minus 1 to compute residue gap)
            running_diff = np.diff(sorted_cropped_res) - 1  # [N - 1]
            # indices i where the gap *after* sorted index i exceeds 3 (cluster break)
            break_after = np.where(running_diff > 3)[0]
            # cluster k spans sorted indices [starts[k], ends[k]] (inclusive)
            starts = np.concatenate(([0], break_after + 1))
            ends = np.concatenate((break_after, [len(sorted_cropped_res) - 1]))
            # union of cluster start/end residue ids — distance to these matches
            # min_k(min(|r - s_k|, |r - e_k|)) from Algorithm 1, line 23
            boundary_res = np.unique(
                np.concatenate((sorted_cropped_res[starts], sorted_cropped_res[ends]))
            )
            # compute minimum distance to cluster boundaries all collated
            d_seq = np.min(
                np.abs(uncropped_res[:, None].astype(np.float64) - boundary_res[None]),
                axis=1,
            )

            # based on d_seq calculated update the candidates
            for j, res_idx in enumerate(uncropped_res):
                candidates.append((float(d_seq[j]), int(chain_id), int(res_idx)))

        candidates.sort(key=lambda x: x[0])
        for _, asym_id, res_idx in candidates:
            if len(cropped) >= max_tokens:
                break
            cropped.update(res_to_tokens.get((asym_id, res_idx), ()))

    return sorted(cropped)
