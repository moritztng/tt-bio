from typing import Any, ClassVar

import numpy as np
import torch

from tt_bio._vendor.atomworks.ml.transforms._checks import check_contains_keys, check_is_instance
from tt_bio._vendor.atomworks.ml.transforms.base import Transform
from tt_bio._vendor.atomworks.ml.utils.token import get_token_starts


class PackageConfidenceFeats(Transform):
    """
    Restructures all the confidence information so it's included in the confidence_feats dictionary.
    Converts sequence to torch tensor. Properly indexes atom_frames to only include atomized tokens.

    Adds:
    - confidence_feats: Dict[str, torch.Tensor]
    """

    requires_previous_transforms: ClassVar[list[str | Transform]] = [
        "EncodeAtomArray",
        "AddAtomFrames",
        "AddIsRealAtom",
        "AddPolymerFrameIndices",
    ]

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(
            data,
            [
                "encoded",
                "rf2aa_atom_frames",
                "is_real_atom",
                "pae_frame_idx_token_lvl_from_atom_lvl",
            ],
        )
        check_is_instance(data, "is_real_atom", torch.Tensor)
        check_is_instance(data, "pae_frame_idx_token_lvl_from_atom_lvl", torch.Tensor)

    def forward(self, data: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
        atom_array = data["atom_array"]
        rf2aa_atom_frames = data["rf2aa_atom_frames"]  # [n_tokens_across_chains, 3, 2] (int)

        # Index to only the atomized tokens
        token_starts = get_token_starts(atom_array)
        token_wise_atom_array = atom_array[token_starts]
        atomized_tokens = token_wise_atom_array.atomize

        if np.any(atomized_tokens):
            rf2aa_atom_frames = rf2aa_atom_frames[atomized_tokens]  # [n_atomized_tokens, 3, 2] (int)
        else:
            # If there are no atomized tokens, we need to add a dummy atom frame
            rf2aa_atom_frames = torch.zeros((0, 3, 2), dtype=torch.int64)

        confidence_feats = {
            "rf2aa_seq": torch.from_numpy(data["encoded"]["seq"]),
            "atom_frames": rf2aa_atom_frames,
            "is_real_atom": data["is_real_atom"],
            "pae_frame_idx_token_lvl_from_atom_lvl": data["pae_frame_idx_token_lvl_from_atom_lvl"],
        }
        data["confidence_feats"] = confidence_feats

        return data
