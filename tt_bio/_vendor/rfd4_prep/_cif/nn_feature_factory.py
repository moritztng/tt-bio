# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 2db07bb444d91695f8c10e6acb7648e9297f9221.
import logging
import math
from functools import lru_cache
from typing import Any, Literal

import einops
import torch
from tt_bio._vendor.atomworks.ml.conditions import C_CRD, C_SEQ
from jaxtyping import Float
from torch.nn import functional as F

from .conditions import (
    C_CATH,
    C_CENTROID,
    C_CRDGRP,
    C_HB,
    C_HOT,
    C_HPHOB,
    C_IDXGRP,
    C_INTCRD,
    C_MCOORD,
    C_PICATION,
    C_PISTACK,
    C_RAS,
    C_SALT,
    C_SEC,
    SEQUENCE_PROBABILITY_CONDITIONS,
    HBondEnum,
    MetalCoordEnum,
    PiCationEnum,
    RasaEnum,
    SaltBridgeEnum,
)
from .datapipes_schema_proteina import NUM_TOKEN_TYPES, TokenType
from .encoding import UNIFIED_ATOM37_ENCODING
from .nn_modules_norm import make_norm
from .nn_noise_alignment import align_chirality, align_internal_coords
from .transforms_chirality_feature import MAX_OVERLAPPING_CENTERS

logger = logging.getLogger(__name__)

RECYCLE_GRAD_DEPTHS = {"last": 1, "last2": 2}

# Full-graph compilation cannot trace tensor comparisons with IntEnum members
# (Equivalent plain integers keep the atom37 feature forwards traceable)
_HB_DONOR, _HB_ACCEPTOR, _HB_BOTH = int(HBondEnum.DONOR), int(HBondEnum.ACCEPTOR), int(HBondEnum.BOTH)
_PC_AROMATIC, _PC_CATION, _PC_BOTH = int(PiCationEnum.AROMATIC), int(PiCationEnum.CATION), int(PiCationEnum.BOTH)
_SALT_NONE = int(SaltBridgeEnum.NONE)
_MCOORD_NONE = int(MetalCoordEnum.NONE)


def _require_condition_in_batch(batch: dict, full_name: str, feature_name: str) -> None:
    """Raise if a requested conditioning annotation is absent from the batch."""
    if full_name not in batch:
        raise KeyError(
            f"{feature_name}: required conditioning annotation '{full_name}' is missing from the batch. "
            f"This feature is requested in the model's feats_seq but the annotation was not produced "
            f"upstream. Remove it from feats_seq if it should not be provided."
        )


@lru_cache(maxsize=128)
def _warn_once(msg: str) -> None:
    """Log ``msg`` at WARNING level, once per unique string per process (bounded cache)."""
    logger.warning(msg)


################################
# # Some auxiliary functions # #
################################


def get_time_embedding(
    t: Float[torch.Tensor, "b"], edim: int, max_positions: int = 2000
) -> Float[torch.Tensor, "b embdim"]:
    """
    Code from Frameflow, which got it from
    https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py

    Creates embedding for a given vector of times t.

    Args:
        t: vector of times (float) of shape [b].
        edim: dimension of the embeddings.
        max_positions: ...

    Returns:
        Embedding for the vector t of shape [b, edim]
    """
    assert len(t.shape) == 1
    t = t * max_positions
    half_dim = edim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
    emb = t.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if edim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (t.shape[0], edim)
    return emb


def bin_pairwise_distances(x: torch.Tensor, min_dist: float, max_dist: float, dim: int) -> torch.Tensor:
    """
    Takes coordinates and bins the pairwise distances.

    Args:
        x: Coordinates of shape [b, n, 3]
        min_dist: Right limit of first bin
        max_dist: Left limit of last bin
        dim: Dimension of the final one hot vectors

    Returns:
        Tensor of shape [b, n, n, dim] consisting of one-hot vectors
    """
    pair_dists_nm = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)  # [b, n, n]
    bin_limits = torch.linspace(min_dist, max_dist, dim - 1, device=x.device)  # Open left and right
    return bin_and_one_hot(pair_dists_nm, bin_limits)  # [b, n, n, pair_dist_dim]


def one_hot_pair_feature(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Create one-hot features in the autocast dtype, or FP32 outside autocast."""
    device_type = indices.device.type
    dtype = torch.get_autocast_dtype(device_type) if torch.is_autocast_enabled(device_type) else torch.float32
    out = torch.zeros((*indices.shape, num_classes), device=indices.device, dtype=dtype)
    return out.scatter_(-1, indices.unsqueeze(-1), 1.0)


def bin_and_one_hot(tensor: torch.Tensor, bin_limits: torch.Tensor) -> torch.Tensor:
    """
    Converts a tensor of shape [*] to a tensor of shape [*, d] using the given bin limits.

    Args:
        tensor (Tensor): Input tensor of shape [*]
        bin_limits (Tensor): bin limits [l1, l2, ..., l_{d-1}]. d-1 limits define
            d-2 bins, and the first one is <l1, the last one is >l_{d-1}, giving a total of d bins.

    Returns:
        torch.Tensor: Output tensor of shape [*, d] where d = len(bin_limits) + 1
    """
    bin_indices = torch.bucketize(tensor, bin_limits)
    return one_hot_pair_feature(bin_indices, len(bin_limits) + 1)


################################
# # Classes for each feature # #
################################


class Feature(torch.nn.Module):
    """Base class for features."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def get_dim(self) -> int:
        return self.dim

    def forward(self, batch: dict) -> torch.Tensor:
        pass  # Implemented by each class

    def extract_bs_and_n(self, batch: dict) -> tuple[int, int]:
        """
        Extracts batch size and n from input batch.
        Once we add more modalities just try modaliites until you hit one.
        Other option is to pass arguments to nn which modalities are used.
        """
        if "x_t" in batch:
            if "bb_ca" in batch["x_t"]:
                v = batch["x_t"]["bb_ca"]  # [b, n, 3]
        elif "coords" in batch:
            v = batch["coords"]  # [b, n, 37, 3]
        elif "z_latent" in batch:
            v = batch["z_latent"]  # [b, n, latent_dim]
        else:
            raise OSError("Don't know how to extract batch size and n from batch...")
        bs, n = v.shape[0], v.shape[1]
        return bs, n

    def extract_device(self, batch: dict) -> torch.device:
        """Extracts device from input batch."""
        if "x_t" in batch:
            if "bb_ca" in batch["x_t"]:
                v = batch["x_t"]["bb_ca"]  # [b, n, 3]
        elif "coords" in batch:
            v = batch["coords"]  # [b, n, 37, 3]
        elif "z_latent" in batch:
            v = batch["z_latent"]  # [b, n, latent_dim]
        else:
            raise OSError("Don't know how to extract device from batch...")
        return v.device

    def assert_defaults_allowed(self, batch: dict, ftype: str) -> None:
        """Raises error if default features should not be used to fill-up missing features in the current batch."""
        if batch.get("strict_feats"):
            raise OSError(
                f"{ftype} feature requested but no appropriate feature provided. "
                "Make sure to include the relevant transform in the data config."
            )


class ZeroFeat(Feature):
    """Computes empty feature (zero) of shape [b, n, dim] or [b, n, n, dim],
    depending on sequence or pair features."""

    def __init__(
        self,
        dim_feats_out: int = 128,
        mode: Literal["seq", "pair"] = "seq",
        name: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(dim=128)
        self.mode = mode

    def forward(self, batch: dict) -> torch.Tensor:
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)
        if self.mode == "seq":
            return torch.zeros((b, n, self.dim), device=device)
        elif self.mode == "pair":
            torch.zeros((b, n, n, self.dim_feats_out), device=device)
        else:
            raise OSError(f"Mode {self.mode} wrong for zero feature")


class FoldEmbeddingSeqFeat(Feature):
    """Computes fold class embedding and returns as sequence feature of shape [b, n, fold_emb_dim * 3]."""

    def __init__(
        self,
        cath_emb_dim: int = 32,
        **kwargs: Any,
    ):
        """
        Args:
            cath_emb_dim (int): Dimension of the embedding
        """
        super().__init__(dim=cath_emb_dim * 3)

        self.register_buffer("_device_param", torch.tensor(0), persistent=False)

        # We add an extra class for null values
        self.embedding_C = torch.nn.Embedding(5 + 1, cath_emb_dim)
        self.embedding_A = torch.nn.Embedding(43 + 1, cath_emb_dim)
        self.embedding_T = torch.nn.Embedding(1472 + 1, cath_emb_dim)

    @property
    def device(self) -> torch.device:
        return next(self.buffers()).device

    def forward(self, batch: dict) -> torch.Tensor:
        # Add 1 because padding/null values are set to -1
        cath_codes = batch[C_CATH.full_name] + 1  # [b, n, 3]

        cath_embeddings = torch.cat(
            [
                self.embedding_C(cath_codes[:, :, 0]),
                self.embedding_A(cath_codes[:, :, 1]),
                self.embedding_T(cath_codes[:, :, 2]),
            ],
            dim=-1,
        )  # [b, n, fold_emb_dim * 3]

        return cath_embeddings  # [b, n, fold_emb_dim * 3]


class TimeEmbeddingBatchFeat(Feature):
    """Compute one time embedding per example with shape [b, t_emb_dim]."""

    def __init__(self, data_mode_use: str, t_emb_dim: int, **kwargs: Any) -> None:
        super().__init__(dim=t_emb_dim)
        self.data_mode_use = data_mode_use

    def forward(self, batch: dict) -> torch.Tensor:
        return get_time_embedding(batch["t"][self.data_mode_use], edim=self.dim)


class RecycleRemainingBatchFeat(Feature):
    """Encode remaining recycle passes independently of backpropagation depth.

    ``recycle.conditioning_dim`` sets the feature width and must match the trained
    checkpoint. For width 2, zero remaining passes gives [1, 0], one or more gives
    [0, 1], and absent conditioning gives [0, 0]. Counts saturate at width minus one.
    Changing ``recycle.grad`` need not change this learned input representation.
    When no width is specified, it defaults to the gradient depth (or
    ``training_max_recycles`` for ``grad=all``).
    """

    def __init__(self, recycle: dict, **kwargs: Any) -> None:
        grad = recycle.get("grad", "last")
        if grad == "all":
            grad_depth = recycle.get("training_max_recycles")
            if not isinstance(grad_depth, int) or grad_depth < 1:
                raise ValueError("recycle_remaining with grad=all requires positive nn.recycle.training_max_recycles")
        elif grad in RECYCLE_GRAD_DEPTHS:
            grad_depth = RECYCLE_GRAD_DEPTHS[grad]
        else:
            raise ValueError(f"nn.recycle.grad must be one of last|last2|all, got {grad!r}")
        conditioning_dim = recycle.get("conditioning_dim", grad_depth)
        if not isinstance(conditioning_dim, int) or conditioning_dim < 1:
            raise ValueError("nn.recycle.conditioning_dim must be a positive integer")
        super().__init__(dim=conditioning_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        mask = batch["padding_mask"]
        remaining = batch.get("recycle_remaining")
        if remaining is None:
            return batch["x_t"]["bb_ca"].new_zeros((mask.shape[0], self.dim))
        indices = torch.full((mask.shape[0],), min(remaining, self.dim - 1), device=mask.device, dtype=torch.long)
        return one_hot_pair_feature(indices, self.dim)


class XtBBCASeqFeat(Feature):
    """Computes feature from backbone CA x_t coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=3)

    def forward(self, batch: dict) -> torch.Tensor:
        data_modes_avail = list(batch["x_t"])
        assert (
            "bb_ca" in data_modes_avail
        ), f"`bb_ca` seq feat feature requested but key not available in data modes {data_modes_avail}"
        return batch["x_t"]["bb_ca"]  # [b, n, 3]


class XtLocalLatentsSeqFeat(Feature):
    """Computes feature from backbone CA x_t coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, latent_dim: int, **kwargs: Any) -> None:
        super().__init__(dim=latent_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        data_modes_avail = list(batch["x_t"])
        assert (
            "local_latents" in data_modes_avail
        ), f"`local_latents` seq feat feature requested but key not available in data modes {data_modes_avail}"
        return batch["x_t"]["local_latents"]  # [b, n, latent_dim]


class CaCoorsNanometersSeqFeat(Feature):
    """Computes feature from ca coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=3)

    def forward(self, batch: dict) -> torch.Tensor:
        assert (
            "ca_coors_nm" in batch or "coords_nm" in batch
        ), "`ca_coors_nm` nor `coords_nm` in batch, cannot compute CaCoorsNanometersSeqFeat"
        if "ca_coors_nm" in batch:
            return batch["ca_coors_nm"]  # [b, n, 3]
        else:
            return batch["coords_nm"][:, :, 1, :]  # [b, n, 3]


class Atom37ElementChargeEmbeddingSeqFeat(Feature):
    """Combine element and flow charge embeddings at every occupied atom37 slot."""

    def __init__(self, charge_emb_dim: int = 16, **kwargs: Any) -> None:
        super().__init__(dim=37 * (8 + charge_emb_dim))
        self.element_embedding = torch.nn.Embedding(119, 8)
        self.charge_embedding = AtomChargeSeqFeat(charge_emb_dim=charge_emb_dim).embedding

    def forward(self, batch: dict) -> torch.Tensor:
        mask = batch["atom37_mask"].bool()
        elements = batch["ground_truth_element"].long().masked_fill(~mask, 0)
        charges = batch["charge"].long().masked_fill(~mask, 0)
        encoded = torch.cat(
            (self.element_embedding(elements), AtomChargeSeqFeat.embed_charge(self.charge_embedding, charges)), dim=-1
        )
        return (encoded * mask.unsqueeze(-1)).flatten(start_dim=-2)


class IncidentBondTypeCountsSeqFeat(Feature):
    """Flatten per-Atom37-slot counts of incident Biotite bond types."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=37 * 10)

    def forward(self, batch: dict) -> torch.Tensor:
        counts = batch["incident_bond_type_counts"].float()
        return (counts * batch["atom37_mask"].unsqueeze(-1)).flatten(start_dim=-2)


class ConditionalResidueTypeSeqFeat(Feature):
    """Compute a feature from the canonical conditional residue type.

    Used by the AE decoder and released flow checkpoints. Generalized flow configs
    use ``SequenceProbabilitiesSeqFeat`` instead.

    Uses `C_SEQ.full_name` if available, otherwise returns zeros.
    This prevents leakage of ground truth sequence information during training.

    The encoded mask token in the annotation indicates which tokens are unconditioned.

    The final feature dimension matches ``UNIFIED_ATOM37_ENCODING``.
    """

    def __init__(
        self,
        num_classes: int | None = None,
        learnable: bool = False,
        **kwargs: Any,
    ) -> None:
        if num_classes is None:
            num_classes = kwargs.get("n_residue_types", UNIFIED_ATOM37_ENCODING.n_tokens)
        super().__init__(dim=num_classes)
        self._has_logged = False
        self.num_classes = num_classes

        # Create embedding layer initialized as identity matrix (one-hot behavior)
        self.embedding = torch.nn.Embedding(num_classes, num_classes)
        if not learnable:
            # Initialize as identity matrix for one-hot behavior
            self.embedding.weight.data = torch.eye(num_classes)
            self.embedding.weight.requires_grad = False

    def forward(self, batch: dict) -> torch.Tensor:
        if C_SEQ.full_name in batch:
            rtype = batch[C_SEQ.full_name]  # [b, n] - contains <M> for unconditioned tokens
            seq_cond_mask = rtype != UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"]

            # Apply sequence conditioning mask
            rtype = rtype * seq_cond_mask  # [b, n]

            # Validate indices before embedding lookup
            if not torch.compiler.is_compiling():
                invalid_mask = (rtype < 0) | (rtype >= self.num_classes)
                if invalid_mask.any():
                    invalid_indices = torch.where(invalid_mask)
                    invalid_values = rtype[invalid_mask]
                    example_ids = batch.get("example_id_batch", ["unknown"] * rtype.shape[0])
                    batch_indices = invalid_indices[0].tolist()
                    token_indices = invalid_indices[1].tolist()
                    raise ValueError(
                        f"ConditionalResidueTypeSeqFeat: Invalid residue type indices for embedding "
                        f"(expected 0-{self.num_classes - 1}, got values: {invalid_values.tolist()}). "
                        f"Batch indices: {batch_indices[:10]}{'...' if len(batch_indices) > 10 else ''}, "
                        f"Token indices: {token_indices[:10]}{'...' if len(token_indices) > 10 else ''}, "
                        f"Example IDs: {[example_ids[i] for i in set(batch_indices[:5])]}"
                    )

            # Use embedding layer (acts like one-hot if initialized as identity)
            rtype_embedded = self.embedding(rtype)  # [b, n, num_classes]
            rtype_embedded = rtype_embedded * seq_cond_mask[..., None]
            return rtype_embedded
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(f"No {C_SEQ.full_name} in batch, returning zeros for ConditionalResidueTypeSeqFeat")
                self._has_logged = True
            return torch.zeros(b, n, self.num_classes, device=device)


class TokenTypeSeqFeat(Feature):
    """Unmasked token type embedding for the unified vocabulary.

    Reads token_type from batch (populated by collate) and embeds it.

    Token types:
        0: protein (standard amino acids)
        1: RNA
        2: DNA
        3: atomized
        4: unknown (mask/padding)
        5: glycan
        6: NCAA/PTM

    Args:
        token_type_emb_dim: Embedding dimension. Defaults to 32.
    """

    def __init__(
        self,
        token_type_emb_dim: int = 32,
        num_token_types: int | None = None,
        **kwargs,
    ):
        super().__init__(dim=token_type_emb_dim)
        self.token_type_emb_dim = token_type_emb_dim
        if num_token_types is None:
            num_token_types = NUM_TOKEN_TYPES
        self.embedding = torch.nn.Embedding(num_token_types, token_type_emb_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        assert "token_type" in batch, "`token_type` not in batch"
        token_type = batch["token_type"]  # [b, n] integer token-type indices

        # Validate indices before embedding lookup
        if not torch.compiler.is_compiling():
            invalid_mask = (token_type < 0) | (token_type >= self.embedding.num_embeddings)
            if invalid_mask.any():
                invalid_indices = torch.where(invalid_mask)
                invalid_values = token_type[invalid_mask]
                example_ids = batch.get("example_id_batch", ["unknown"] * token_type.shape[0])
                batch_indices = invalid_indices[0].tolist()
                raise ValueError(
                    f"TokenTypeSeqFeat: Invalid token_type indices for embedding "
                    f"(expected 0-{self.embedding.num_embeddings - 1}, got values: {invalid_values.tolist()[:20]}). "
                    f"Example IDs: {[example_ids[i] for i in set(batch_indices[:5])]}"
                )

        return self.embedding(token_type)  # [b, n, token_type_emb_dim]


class Atom37NanometersCoorsSeqFeat(Feature):
    """
    Computes feature from the atom37 representation (in Å), feature of shape [b, n, 37 * 4].

    Atom37 has shape [b, n, 37, 3], and the appropriate mask (for the residue type) has shape
    [b, n, 37]. This feature concatenates the flattened mask (shape [b, n, 37]) with the flattened coordinates (of shape
    [b, n, 37 * 3])

    Note that in residue type the padding is done with a -1, but this function
    multiplies with the mask.
    """

    def __init__(self, rel: bool = False, **kwargs: Any) -> None:
        super().__init__(dim=(37 * 4))
        # 37 * 4, 37 * 3 for the coordinates, + 37 for the mask
        self.rel = rel  # whether to get features relative to CA or absolute

    def forward(self, batch: dict) -> torch.Tensor:
        assert "coords_nm" in batch, "`coords_nm` not in batch, cannot compute Atom37NanometersCoorsSeqFeat"
        coors = batch["coords_nm"]  # [b, n, 37, 3]
        if "atom37_mask" in batch:
            coors_mask = batch["atom37_mask"]  # [b, n, 37]
        elif "coord_mask" in batch:
            coors_mask = batch["coord_mask"]  # [b, n, 37]
        else:
            raise ValueError("Neither 'atom37_mask' nor 'coord_mask' found in batch")
        coors = coors * coors_mask[..., None]  # Zero-out non-atoms

        if self.rel:
            # If relative remove CA coordinates
            ca_coors = coors[:, :, 1, :]  # [b, n, 3]
            coors = coors - ca_coors[:, :, None, :]  # [b, n, 37, 3]
            coors = coors * coors_mask[..., None]

        coors_flat = einops.rearrange(coors, "b n a t -> b n (a t)")  # [b, n, 37, 3] -> [b, n, 37 * 3]
        feat = torch.cat([coors_flat, coors_mask.float()], dim=-1)  # [b, n, 37 * 4]
        return feat


class ConditionalAtom37NanometersCoorsSeqFeat(Feature):
    """
    Computes feature from conditional atom37 coordinates, feature of shape [b, n, 37 * 4].

    Uses `C_CRD.full_name` if available, otherwise returns zeros.
    This prevents leakage of ground truth coordinate information during training.

    Finite coordinates indicate absolute coordinate conditioning; unconditioned values are NaN.
    """

    def __init__(self, rel: bool = False, **kwargs: Any) -> None:
        super().__init__(dim=(37 * 4))
        self.rel = rel  # whether to get features relative to CA or absolute
        self._has_logged = False

    def forward(self, batch: dict) -> torch.Tensor:
        if C_CRD.full_name in batch:
            # C_CRD.full_name coordinates are in Angstroms from collate, convert to nm
            conditioned_coords = batch[C_CRD.full_name]
            coors_mask = torch.isfinite(conditioned_coords).any(dim=-1)
            coors = torch.nan_to_num(conditioned_coords, nan=0.0) / 10.0
            coors = coors * coors_mask[..., None]

            if self.rel:
                # If relative remove CA coordinates
                ca_coors = coors[:, :, 1, :]  # [b, n, 3]
                coors = coors - ca_coors[:, :, None, :]  # [b, n, 37, 3]
                coors = coors * coors_mask[..., None]

            coors_flat = einops.rearrange(coors, "b n a t -> b n (a t)")  # [b, n, 37, 3] -> [b, n, 37 * 3]
            feat = torch.cat([coors_flat, coors_mask.float()], dim=-1)  # [b, n, 37 * 4]
            return feat
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    f"No {C_CRD.full_name} in batch, returning zeros for ConditionalAtom37NanometersCoorsSeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class ConditionalAtom37NanometersInternalCoorsSeqFeat(Feature):
    """
    Computes feature from conditional atom37 coordinates, feature of shape [b, n, 37 * 4].

    Uses `C_INTCRD.full_name` if available, otherwise returns zeros.
    This prevents leakage of ground truth coordinate information during training.

    Positive ``C_CRDGRP`` values indicate internal-coordinate conditioning;
    ``C_INTCRD`` contains the materialized coordinate payload.
    """

    def __init__(
        self,
        noise_align_internal_coords: bool = False,
        noise_align_internal_coords_translate: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(dim=(37 * 4))
        self._has_logged = False
        self.noise_align = noise_align_internal_coords
        self.translate = noise_align_internal_coords_translate

    def forward(self, batch: dict) -> torch.Tensor:
        if C_INTCRD.full_name in batch and C_CRDGRP.full_name in batch:
            # C_INTCRD.full_name coordinates are in Angstroms from collate, convert to nm
            conditioned_internal_coords = batch[C_INTCRD.full_name]
            coord_groups = batch[C_CRDGRP.full_name]
            internal_coords_mask = (coord_groups != C_CRDGRP.UNKNOWN_COORDS) & (
                coord_groups != C_CRDGRP.KNOWN_ABSOLUTE_COORDS
            )
            internal_coords = torch.nan_to_num(conditioned_internal_coords) / 10.0
            internal_coords = internal_coords * internal_coords_mask[..., None]

            if self.noise_align or self.translate:
                selected = internal_coords_mask & batch["padding_mask"][..., None]
                if not torch.isfinite(conditioned_internal_coords[selected]).all():
                    raise ValueError("Internal-coordinate conditioning must be finite within each supplied group")
                internal_coords = align_internal_coords(
                    internal_coords,
                    coord_groups,
                    internal_coords_mask,
                    batch["x_t"]["bb_ca"],
                    batch["padding_mask"],
                    translate=self.translate,
                    rotate=self.noise_align,
                )

            internal_coords_flat = einops.rearrange(
                internal_coords, "b n a t -> b n (a t)"
            )  # [b, n, 37, 3] -> [b, n, 37 * 3]
            feat = torch.cat([internal_coords_flat, internal_coords_mask.float()], dim=-1)  # [b, n, 37 * 4]
            return feat
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            if not self._has_logged:
                logger.warning(
                    f"Missing {C_INTCRD.full_name} or {C_CRDGRP.full_name} in batch, "
                    f"returning zeros for ConditionalAtom37NanometersInternalCoorsSeqFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, self.dim, device=device)


class ChiralitySeqFeat(Feature):
    """Packed chirality vectors, optionally oriented toward the current noisy state.

    Output: ``[b, n, MAX_OVERLAPPING_CENTERS * 3]``.
    """

    def __init__(
        self, noise_align_chirality: bool = False, noise_align_chirality_translate: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(dim=MAX_OVERLAPPING_CENTERS * 3)
        self.noise_align = noise_align_chirality
        self.translate = noise_align_chirality_translate

    def forward(self, batch: dict) -> torch.Tensor:
        if "chirality" in batch:
            feature = batch["chirality"].float()
            if self.noise_align or self.translate:
                return align_chirality(feature, batch, translate=self.translate, rotate=self.noise_align)
            return feature

        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)
        _warn_once("No chirality in batch, returning zeros")
        return torch.zeros(b, n, self.dim, device=device)


class CoordGroupSeqFeat(Feature):
    """Binary feature for whether two atoms are in the same coordinate group."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=37)

    def forward(self, batch: dict) -> torch.Tensor:
        coord_group = batch[C_CRDGRP.full_name].long().clone()  # [b, n, 37]

        # Assign a random value to the non-padding index groups, to avoid the model learning spurious correlations
        non_padding_mask = coord_group != C_CRDGRP.UNKNOWN_COORDS
        unique_groups, inverse_indices = torch.unique(coord_group[non_padding_mask], return_inverse=True)
        permuted_index_groups = torch.randperm(len(unique_groups), device=coord_group.device)
        coord_group[non_padding_mask] = permuted_index_groups[inverse_indices]

        return coord_group.float()  # [b, n, 37]


class LatentVariableSeqFeat(Feature):
    """Returns sequence feature from latent variable."""

    def __init__(self, latent_z_dim: int, **kwargs: Any) -> None:
        super().__init__(dim=latent_z_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        assert "z_latent" in batch, "`z_latent` not in batch, cannot compute LatentVariableSeqFeat"
        return batch["z_latent"]  # [b, n, latent_dim]


class RasaBinnedSeqFeat(Feature):
    """RASA (surface accessibility) binned feature: 0=BURIED, 1=INTERMEDIATE, 2=EXPOSED, 3=UNSPECIFIED.

    Returns atom37 representation [b, n, 37] as feature.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=37)

    def forward(self, batch: dict) -> torch.Tensor:
        if C_RAS.full_name not in batch:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            return torch.zeros(b, n, 37, device=device)

        rasa_atom37 = batch[C_RAS.full_name].float()  # [b, n, 37]
        atom37_mask = batch["atom37_mask"]  # [b, n, 37]

        # Set masked atoms to UNSPECIFIED
        rasa_atom37[~atom37_mask.bool()] = RasaEnum.UNSPECIFIED
        return rasa_atom37


class ChainCentroidXyzSeqFeat(Feature):
    """Per-token chain centroid xyz in nm, ``[b, n, 3]``; zero where unincluded."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=3)

    def forward(self, batch: dict) -> torch.Tensor:
        if C_CENTROID.full_name not in batch:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            _warn_once("No C_CENTROID in batch, returning zeros for ChainCentroidXyzSeqFeat")
            return torch.zeros(b, n, self.dim, device=device)

        centroid_angstrom = batch[C_CENTROID.full_name].float()  # [b, n, 3]
        included = torch.isfinite(centroid_angstrom).all(dim=-1)  # [b, n]
        centroid_nm = torch.where(
            included.unsqueeze(-1),
            centroid_angstrom / 10.0,
            torch.zeros_like(centroid_angstrom),
        )
        return centroid_nm  # [b, n, 3]


class SecondaryStructureSeqFeat(Feature):
    """Secondary structure: 0=NONE, 1=HELIX, 2=SHEET, 3=LOOP."""

    def __init__(self, embed_dim: int = 16, **kwargs: Any) -> None:
        super().__init__(dim=embed_dim)
        self.ss_embedding = torch.nn.Embedding(4, embed_dim, padding_idx=0)

    def forward(self, batch: dict) -> torch.Tensor:
        if C_SEC.full_name not in batch:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            # Dummy forward so embedding participates in DDP backward pass
            zeros = torch.zeros(b, n, self.dim, device=device)
            return zeros + 0.0 * self.ss_embedding.weight.sum()

        ss_type = batch[C_SEC.full_name].long()  # [b, n] - already token-level

        # Validate indices before embedding lookup
        if not torch.compiler.is_compiling():
            invalid_mask = (ss_type < 0) | (ss_type >= 4)
            if invalid_mask.any():
                invalid_indices = torch.where(invalid_mask)
                invalid_values = ss_type[invalid_mask]
                example_ids = batch.get("example_id_batch", ["unknown"] * ss_type.shape[0])
                batch_indices = invalid_indices[0].tolist()
                raise ValueError(
                    f"SecondaryStructureSeqFeat: Invalid ss_type indices for embedding "
                    f"(expected 0-3, got values: {invalid_values.tolist()[:20]}). "
                    f"Example IDs: {[example_ids[i] for i in set(batch_indices[:5])]}"
                )

        return self.ss_embedding(ss_type)  # [b, n, embed_dim]


class ModernizedHydrogenBondSeqFeat(Feature):
    """Hydrogen bonds as two binary atom37 channels: ``[is_donor (37), is_acceptor (37)]``.

    ``HBondEnum.BOTH`` (donor in one bond, acceptor in another) sets both channels. Returns
    ``[b, n, 74]``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=74)

    def forward(self, batch: dict) -> torch.Tensor:
        _require_condition_in_batch(batch, C_HB.full_name, "ModernizedHydrogenBondSeqFeat")
        hb = batch[C_HB.full_name]  # [b, n, 37]
        atom37_mask = batch["atom37_mask"].float()  # [b, n, 37]
        is_donor = ((hb == _HB_DONOR) | (hb == _HB_BOTH)).float() * atom37_mask  # [b, n, 37]
        is_acceptor = ((hb == _HB_ACCEPTOR) | (hb == _HB_BOTH)).float() * atom37_mask  # [b, n, 37]
        return torch.cat([is_donor, is_acceptor], dim=-1)  # [b, n, 74]


class PiStackingSeqFeat(Feature):
    """Pi-stacking contacts as a binary atom37 feature.

    Returns ``[b, n, 37]``: 1 for atoms involved in pi-stacking, 0 otherwise.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=37)

    def forward(self, batch: dict) -> torch.Tensor:
        _require_condition_in_batch(batch, C_PISTACK.full_name, "PiStackingSeqFeat")

        pistack = batch[C_PISTACK.full_name].float()  # [b, n, 37] bool
        atom37_mask = batch["atom37_mask"].float()  # [b, n, 37]
        return pistack * atom37_mask  # [b, n, 37]


class PiCationSeqFeat(Feature):
    """Pi-cation contacts as two binary atom37 channels: ``[is_aromatic (37), is_cation (37)]``.

    ``PiCationEnum.BOTH`` (aromatic in one interaction, cation in another) sets both channels.
    Returns ``[b, n, 74]``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=74)

    def forward(self, batch: dict) -> torch.Tensor:
        _require_condition_in_batch(batch, C_PICATION.full_name, "PiCationSeqFeat")
        pc = batch[C_PICATION.full_name]  # [b, n, 37]
        atom37_mask = batch["atom37_mask"].float()  # [b, n, 37]
        is_aromatic = ((pc == _PC_AROMATIC) | (pc == _PC_BOTH)).float() * atom37_mask  # [b, n, 37]
        is_cation = ((pc == _PC_CATION) | (pc == _PC_BOTH)).float() * atom37_mask  # [b, n, 37]
        return torch.cat([is_aromatic, is_cation], dim=-1)  # [b, n, 74]


class SaltBridgeSeqFeat(Feature):
    """Salt bridge contacts as a binary atom37 feature.

    Returns ``[b, n, 37]``: 1 for atoms involved in a salt bridge, 0 otherwise.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=37)

    def forward(self, batch: dict) -> torch.Tensor:
        _require_condition_in_batch(batch, C_SALT.full_name, "SaltBridgeSeqFeat")

        salt = batch[C_SALT.full_name].long()  # [b, n, 37]
        atom37_mask = batch["atom37_mask"].float()  # [b, n, 37]
        return (salt != _SALT_NONE).float() * atom37_mask  # [b, n, 37]


class HydrophobicSeqFeat(Feature):
    """Hydrophobic contacts as a binary atom37 feature.

    Returns ``[b, n, 37]``: 1 for atoms involved in hydrophobic contacts, 0 otherwise.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=37)

    def forward(self, batch: dict) -> torch.Tensor:
        _require_condition_in_batch(batch, C_HPHOB.full_name, "HydrophobicSeqFeat")

        hphob = batch[C_HPHOB.full_name].float()  # [b, n, 37] bool
        atom37_mask = batch["atom37_mask"].float()  # [b, n, 37]
        return hphob * atom37_mask  # [b, n, 37]


class MetalCoordinationSeqFeat(Feature):
    """Metal coordination as a binary atom37 feature.

    Returns ``[b, n, 37]``: 1 for atoms involved in metal coordination, 0 otherwise.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=37)

    def forward(self, batch: dict) -> torch.Tensor:
        _require_condition_in_batch(batch, C_MCOORD.full_name, "MetalCoordinationSeqFeat")

        mcoord = batch[C_MCOORD.full_name].long()  # [b, n, 37]
        atom37_mask = batch["atom37_mask"].float()  # [b, n, 37]
        return (mcoord != _MCOORD_NONE).float() * atom37_mask  # [b, n, 37]


class HotspotSeqFeat(Feature):
    """Hotspot annotation: 0=not hotspot, 1=hotspot.

    Returns atom37 representation [b, n, 37] as feature.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=37)

    def forward(self, batch: dict) -> torch.Tensor:
        if C_HOT.full_name not in batch:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            return torch.zeros(b, n, 37, device=device)

        hotspot_atom37 = batch[C_HOT.full_name].float()  # [b, n, 37] boolean
        atom37_mask = batch["atom37_mask"]  # [b, n, 37]

        # Zero out invalid atoms
        hotspot_feat = hotspot_atom37 * atom37_mask.float()  # [b, n, 37]
        return hotspot_feat


class SequenceProbabilitiesSeqFeat(Feature):
    """Canonical-polymer probabilities and masks in unified-encoding order.

    The first half contains probability values and the second half indicates
    which values were explicitly specified. Thus an explicit zero is distinct
    from an unconstrained sequence-token probability.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=2 * len(SEQUENCE_PROBABILITY_CONDITIONS))

    def forward(self, batch: dict) -> torch.Tensor:
        probabilities = torch.stack(
            [batch[condition.full_name] for condition in SEQUENCE_PROBABILITY_CONDITIONS], dim=-1
        ).float()
        masks = ~torch.isnan(probabilities)
        return torch.cat((probabilities.masked_fill(~masks, 0.0), masks.float()), dim=-1)


class IndexGroupPairFeat(Feature):
    """Binary feature for whether two tokens are in the same index group."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=1)

    def forward(self, batch: dict) -> torch.Tensor:
        idx_group = batch[C_IDXGRP.full_name].long().clone()  # [b, n] - already token-level
        idx_group_pair_feat = idx_group[:, :, None] == idx_group[:, None, :]  # [b, n, n] boolean
        return idx_group_pair_feat.unsqueeze(-1).float()  # [b, n, n, 1]


class AtomChargeSeqFeat(Feature):
    """Learned embedding for formal charge on atomized tokens.

    Formal charges are mapped to classes: -4→0, -3→1, ..., 0→4, ..., +6→10.
    Out-of-range values are clamped. Non-atomized tokens get zeros.

    The embedding table is initialized from sin/cos of actual charge values so
    that nearby charges start with similar representations, but the weights are
    learnable and can specialize during training.

    .. note::
       For atomized residues each atom occupies one token, and the per-atom
       scalars (charge, element, atom-name, ...) are written into ``slot 1`` of
       the atom37 channel by ``ConvertToAtom37Format``. Slot 0 is reserved for
       the residue-level CA position (zero / unused for atomized tokens), so
       reading ``batch["charge"][:, :, 1]`` is the atom37 convention for
       fetching the per-atom value of an atomized token.

    Shape: [b, n, embed_dim].
    """

    _CHARGE_OFFSET = 4  # maps charge -4..+6 to index 0..10
    _NUM_CLASSES = 11

    def __init__(self, charge_emb_dim: int = 16, **kwargs):
        super().__init__(dim=charge_emb_dim)
        self._has_logged = False
        self.embedding = torch.nn.Embedding(self._NUM_CLASSES, charge_emb_dim)
        self._init_embedding_from_charge_values(charge_emb_dim)

    def _init_embedding_from_charge_values(self, dim: int) -> None:
        """Initialize embedding rows from sin/cos of actual charge values.

        Args:
          dim: Embedding dimension (must be even).
        """
        charges = torch.arange(
            -self._CHARGE_OFFSET,
            self._NUM_CLASSES - self._CHARGE_OFFSET,
            dtype=torch.float32,
        )
        with torch.no_grad():
            half = dim // 2
            i = torch.arange(half, dtype=torch.float32)
            freqs = 1.0 / (10.0 ** (2 * i / dim))
            for idx, c in enumerate(charges):
                self.embedding.weight[idx, 0::2] = torch.sin(c * freqs)
                self.embedding.weight[idx, 1::2] = torch.cos(c * freqs)

    @classmethod
    def embed_charge(cls, embedding: torch.nn.Embedding, charges: torch.Tensor) -> torch.Tensor:
        """Embed formal charges of any shape using the saturated flow charge vocabulary."""
        indices = (charges.long() + cls._CHARGE_OFFSET).clamp(0, cls._NUM_CLASSES - 1)
        return embedding(indices)

    def forward(self, batch: dict) -> torch.Tensor:
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        if "charge" in batch and "token_type" in batch:
            charge_slot1 = batch["charge"][:, :, 1].long()  # [b, n]
            emb = self.embed_charge(self.embedding, charge_slot1)  # [b, n, charge_emb_dim]
            is_atomized = (batch["token_type"] == TokenType.ATOMIZED).unsqueeze(-1)  # [b, n, 1]
            return emb * is_atomized
        if not self._has_logged:
            logger.warning("No charge/token_type in batch, returning zeros for AtomChargeSeqFeat")
            self._has_logged = True
        # Dummy forward so embedding participates in DDP backward pass
        zeros = torch.zeros(b, n, self.dim, device=device)
        return zeros + 0.0 * self.embedding.weight.sum()


class AtomicElementSeqFeat(Feature):
    """Learned embedding for atomic element on atomized tokens.

    For atomized tokens: Embeds element from atom37 slot 1 (atomized atoms
    are placed in slot 1 by ConvertToAtom37Format).
    For non-atomized tokens (proteins): Returns zero vector.
    Shape: [b, n, embed_dim].
    """

    def __init__(self, max_atomic_num: int = 128, element_emb_dim: int = 16, **kwargs):
        super().__init__(dim=element_emb_dim)
        self._has_logged = False
        self.max_atomic_num = max_atomic_num
        self.embedding = torch.nn.Embedding(max_atomic_num, element_emb_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        if "element" in batch and "token_type" in batch:
            # Atomized tokens place their atom in slot 1
            element_slot1 = batch["element"][:, :, 1].long()  # [b, n]
            element_slot1 = element_slot1.clamp(0, self.max_atomic_num - 1)
            emb = self.embedding(element_slot1)  # [b, n, embed_dim]
            is_atomized = (batch["token_type"] == TokenType.ATOMIZED).unsqueeze(-1)  # [b, n, 1]
            return emb * is_atomized
        if not self._has_logged:
            logger.warning("No element/token_type in batch, returning zeros for AtomicElementSeqFeat")
            self._has_logged = True

        # Dummy forward so embedding participates in DDP backward pass
        zeros = torch.zeros(b, n, self.dim, device=device)
        return zeros + 0.0 * self.embedding.weight.sum()


class ChainIdxSeqFeat(Feature):
    """Gets chain idx feature (-1 for padding) and returns feature of shape [b, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch: dict) -> torch.Tensor:
        if "chain_iid" not in batch:
            raise ValueError("'chain_iid' not found in batch")
        return batch["chain_iid"].unsqueeze(-1).float()  # [b, n, 1]


class TokenBondPathLengthPairFeat(Feature):
    """One-hot encode token bond-graph path lengths.

    The data pipeline stores true (or pipeline-cutoff-bounded) hop counts in
    ``token_bond_path_length``:
      - 0: same token, cross-IndexGroup, or disconnected pair
      - k > 0: bond-graph hop count (clamped to ``pipeline_cutoff + 1`` if a cutoff was set)

    This feature collapses values into ``seq_sep_dim`` bins at tensor time:
      - bin 0: disconnected / cross-group / self
      - bins 1 .. seq_sep_dim - 2: exact path lengths
      - bin seq_sep_dim - 1: overflow (any hop count >= seq_sep_dim - 1)

    Because the clamp happens here, ``seq_sep_dim`` is independent of the data pipeline
    and can be changed without regenerating features. For the overflow bin to receive
    any data, the pipeline's BFS cutoff (``path_length_search_cutoff``) should be set
    to at least ``seq_sep_dim - 2``.
    """

    def __init__(self, seq_sep_dim: int, **kwargs: Any) -> None:
        super().__init__(dim=seq_sep_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        path_lengths = batch["token_bond_path_length"].long()  # [b, n, n]
        clipped = path_lengths.clamp(max=self.dim - 1)  # collapse high distances into overflow bin
        return one_hot_pair_feature(clipped, num_classes=self.dim)  # [b, n, n, dim]


class XtBBCAPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(self, xt_pair_dist_dim: int, xt_pair_dist_min: float, xt_pair_dist_max: float, **kwargs: Any) -> None:
        super().__init__(dim=xt_pair_dist_dim)
        self.min_dist = xt_pair_dist_min
        self.max_dist = xt_pair_dist_max

    def forward(self, batch: dict) -> torch.Tensor:
        data_modes_avail = list(batch["x_t"])
        assert (
            "bb_ca" in data_modes_avail
        ), f"`bb_ca` pair dist feature requested but key not available in data modes {data_modes_avail}"
        return bin_pairwise_distances(
            x=batch["x_t"]["bb_ca"],
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            dim=self.dim,
        )  # [b, n, n, pair_dist_dim]


class CaCoorsNanometersPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=30)
        self.min_dist = 0.1
        self.max_dist = 3.0

    def forward(self, batch: dict) -> torch.Tensor:
        assert (
            "ca_coors_nm" in batch or "coords_nm" in batch
        ), "`ca_coors_nm` pair dist feature requested but key `ca_coors_nm` nor `coords_nm` not available"
        if "ca_coors_nm" in batch:
            ca_coors = batch["ca_coors_nm"]
        else:
            ca_coors = batch["coords_nm"][:, :, 1, :]
        return bin_pairwise_distances(
            x=ca_coors,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            dim=self.dim,
        )  # [b, n, n, pair_dist_dim]


class BackbonePairDistancesNanometerPairFeat(Feature):
    """
    Computes pairwise distances between backbone atoms.

    Position (i, j) encodes the distance between CA_i and
    {N_j, CA_j, C_j, CB_j}.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=(4 * 21))  # 84

    def forward(self, batch: dict) -> torch.Tensor:
        assert "coords_nm" in batch, "`coords_nm` not in batch, cannot comptue BackbonePairDistancesNanometerPairFeat"
        coords = batch["coords_nm"]
        if "atom37_mask" in batch:
            atom_mask = batch["atom37_mask"]  # [b, n, 37]
        elif "coord_mask" in batch:
            atom_mask = batch["coord_mask"]  # [b, n, 37]
        else:
            raise ValueError("Neither 'atom37_mask' nor 'coord_mask' found in batch")
        mask = atom_mask[:, :, 1]  # [b, n]
        pair_mask = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
        has_cb = atom_mask[:, :, 3]  # [b, n] boolean, indicates if corresponding

        N = coords[:, :, 0, :]  # [b, n, 3]
        CA = coords[:, :, 1, :]  # [b, n, 3]
        C = coords[:, :, 2, :]  # [b, n, 3]
        CB = coords[:, :, 3, :]  # [b, n, 3]

        CA_i = CA[:, :, None, :]  # [b, n, 1, 3]
        N_j, CA_j, C_j, CB_j = (v[:, None, :, :] for v in (N, CA, C, CB))  # Each [b, 1, n, 3]

        CA_N, CA_CA, CA_C, CA_CB = (
            torch.linalg.norm(v[0] - v[1], dim=-1) for v in ((CA_i, N_j), (CA_i, CA_j), (CA_i, C_j), (CA_i, CB_j))
        )  # Each shape [b, n, n]
        # CA_X[..., i, j] has distance (nm) between CA[..., i] and X[..., j]

        # Accomodate residues without CB
        CA_CB = CA_CB * has_cb[:, None, :]  # [b, n, n]

        # Fix for mask
        CA_N, CA_CA, CA_C, CA_CB = (v * pair_mask for v in (CA_N, CA_CA, CA_C, CA_CB))  # Each shape [b, n, n]

        bin_limits = torch.linspace(0.1, 2, 20, device=coords.device)
        CA_N_feat, CA_CA_feat, CA_C_feat, CA_CB_feat = (
            bin_and_one_hot(v, bin_limits=bin_limits) for v in (CA_N, CA_CA, CA_C, CA_CB)
        )  # Each [b, n, n, 21]

        feat = torch.cat([CA_N_feat, CA_CA_feat, CA_C_feat, CA_CB_feat], dim=-1)  # [b, n, n, 4 * 21]
        feat = feat * pair_mask[..., None]
        return feat


class MoleculeIdxPairFeat(Feature):
    """Binary feature for whether two tokens belong to the same molecule instance."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=1)

    def forward(self, batch: dict) -> torch.Tensor:
        molecule_iid = batch["molecule_iid"].long()  # [b, n]
        same_molecule = molecule_iid[:, :, None] == molecule_iid[:, None, :]  # [b, n, n] boolean
        return same_molecule.unsqueeze(-1).float()  # [b, n, n, 1]


class MoleculeEntityPairFeat(Feature):
    """Binary feature for whether two tokens belong to the same molecule *entity*."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=1)

    def forward(self, batch: dict) -> torch.Tensor:
        entity = batch["molecule_entity_id"].long()  # [b, n]
        same_entity = entity[:, :, None] == entity[:, None, :]  # [b, n, n] boolean
        return same_entity.unsqueeze(-1).float()  # [b, n, n, 1]


class CanonicalPolymerOrderingPairFeat(Feature):
    """Directional ordering of canonical polymer backbone linkages.

    Passes through the ``canonical_polymer_ordering`` array built by
    :py:func:`~rfproteina.datapipes.bonds.compute_token_bond_features`, which encodes
    the chemical direction of backbone linkages (C-N / O3'-P) between same-type
    canonical polymer residues (protein/RNA/DNA):

    - ``+1`` at ``[downstream, upstream]``, where the upstream residue contributes the
      leaving atom (C / O3') and the downstream residue the entering atom (N / P).
    - ``-1`` at the transpose.

    Direction is read from the bonded atom pair, so a head-to-tail cyclisation is
    encoded correctly even though its downstream residue sits at the lower token index.

    All other pairs (non-bonded, cross-type, cross-IndexGroup, or non-canonical tokens) get
    ``0``. Shape: ``[b, n, n, 1]``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=1)
        self._has_logged = False

    def forward(self, batch: dict) -> torch.Tensor:
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        if "canonical_polymer_ordering" in batch:
            return batch["canonical_polymer_ordering"][..., None].float()  # [b, n, n, 1]
        else:
            if not self._has_logged:
                logger.warning(
                    "No canonical_polymer_ordering in batch, returning zeros for CanonicalPolymerOrderingPairFeat"
                )
                self._has_logged = True
            return torch.zeros(b, n, n, 1, device=device)


class BondMaskPairFeat(Feature):
    """Binary bond existence mask for atomized molecules.

    Non-atomized token pairs get zeros.
    Shape: [b, n, n, 1]
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=1)
        self._has_logged = False

    def forward(self, batch: dict) -> torch.Tensor:
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        if "bond_mask" in batch:
            # [b, n, n] -> [b, n, n, 1]
            return batch["bond_mask"][..., None].float()
        else:
            if not self._has_logged:
                logger.warning("No bond_mask in batch, returning zeros for BondMaskPairFeat")
                self._has_logged = True
            return torch.zeros(b, n, n, 1, device=device)


class BondOrderPairFeat(Feature):
    """Bond order one-hot encoding for atomized molecules (Kekulé assignments).

    Bond types: 0=none, 1=single, 2=double, 3=triple, 4=coordination.
    Aromatic bonds receive their Kekulé bond order (single/double/triple).
    Non-atomized token pairs get zeros.
    Shape: [b, n, n, 5]
    """

    def __init__(self, num_bond_orders: int = 5, **kwargs: Any) -> None:
        super().__init__(dim=num_bond_orders)
        self._has_logged = False
        self.num_bond_orders = num_bond_orders

    def forward(self, batch: dict) -> torch.Tensor:
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        if "bond_order" in batch:
            # [b, n, n] int -> [b, n, n, 5] one-hot
            bond_order = batch["bond_order"].long()
            return one_hot_pair_feature(bond_order, num_classes=self.num_bond_orders)
        else:
            if not self._has_logged:
                logger.warning("No bond_order in batch, returning zeros for BondOrderPairFeat")
                self._has_logged = True
            return torch.zeros(b, n, n, self.num_bond_orders, device=device)


class IsAromaticPairFeat(Feature):
    """Binary aromaticity flag for atomized molecule bonds.

    Dropout is handled upstream by :class:`DropoutAromaticFlag` transform.
    Shape: [b, n, n, 1]
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=1)
        self._has_logged = False

    def forward(self, batch: dict) -> torch.Tensor:
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        if "is_aromatic" in batch:
            return batch["is_aromatic"][..., None].float()  # [b, n, n, 1]
        else:
            if not self._has_logged:
                logger.warning("No is_aromatic in batch, returning zeros for IsAromaticPairFeat")
                self._has_logged = True
            return torch.zeros(b, n, n, 1, device=device)


class NucleicSecondaryStructurePairFeat(Feature):
    """Nucleic-acid secondary-structure pair feature with explicit mask channels.

    Expects:
      - ``na_ss_pair``: binary [b, n, n] for positive base-pair edges
      - ``na_ss_pair_mask``: binary [b, n, n] for specified entries

    Returns:
      - concatenated [value, mask] channels with shape [b, n, n, 2]
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=2)
        self._has_logged = False

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)
        if "na_ss_pair" in batch and "na_ss_pair_mask" in batch:
            na_ss_pair = batch["na_ss_pair"].to(device=device, dtype=torch.float32)[..., None]
            na_ss_pair_mask = batch["na_ss_pair_mask"].to(device=device, dtype=torch.float32)[..., None]
            return torch.cat([na_ss_pair, na_ss_pair_mask], dim=-1)
        if not self._has_logged:
            logger.warning("No na_ss_pair in batch, returning zeros for NucleicSecondaryStructurePairFeat")
            self._has_logged = True
        return torch.zeros(b, n, n, self.dim, device=device)


class NucleicSecondaryStructureStateSeqFeat(Feature):
    """Nucleic-acid 1-body secondary-structure state with explicit mask channel.

    Expects:
      - ``na_ss_state``: int [b, n] with BasePairEnum values
      - ``na_ss_state_mask``: binary [b, n] for specified tokens

    Returns:
      - concatenated [value, mask] channels with shape [b, n, 2]
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(dim=2)
        self._has_logged = False

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)
        if "na_ss_state" in batch and "na_ss_state_mask" in batch:
            na_ss_state = batch["na_ss_state"].to(device=device, dtype=torch.float32)[..., None]
            na_ss_state_mask = batch["na_ss_state_mask"].to(device=device, dtype=torch.float32)[..., None]
            return torch.cat([na_ss_state, na_ss_state_mask], dim=-1)
        if not self._has_logged:
            logger.warning("No na_ss_state in batch, returning zeros for NucleicSecondaryStructureStateSeqFeat")
            self._has_logged = True
        return torch.zeros(b, n, self.dim, device=device)


####################################
# # Class that produces features # #
####################################


class FeatureFactory(torch.nn.Module):
    def __init__(
        self,
        feats: list[str],
        dim_feats_out: int,
        use_ln_out: bool,
        mode: Literal["seq", "pair"],
        compact: bool = False,
        **kwargs,
    ):
        """Project selected features to ``dim_feats_out`` channels.

        Modes return [b, n, d] (seq) or [b, n, n, d] (pair). With ``compact=True``,
        per-example features stay [b, d] until broadcast by the caller.
        """
        super().__init__()
        self.mode = mode
        self.compact = compact
        self.ret_zero = bool(feats is None or len(feats) == 0)
        if self.ret_zero:
            logger.info("No features requested")
            self.zero_creator = ZeroFeat(dim_feats_out=dim_feats_out, mode=mode)
            return

        self.feat_creators = torch.nn.ModuleList([self.get_creator(f, **kwargs) for f in feats])
        self.ln_out = make_norm(dim_feats_out) if use_ln_out else torch.nn.Identity()
        self.linear_out = torch.nn.Linear(sum([c.get_dim() for c in self.feat_creators]), dim_feats_out, bias=False)

    def get_creator(self, f: str, **kwargs: Any) -> Feature:
        """Returns the right class for the requested feature f (a string)."""

        if f == "time_emb_bb_ca":
            return TimeEmbeddingBatchFeat(data_mode_use="bb_ca", **kwargs)
        if f == "time_emb_local_latents":
            return TimeEmbeddingBatchFeat(data_mode_use="local_latents", **kwargs)

        if self.mode == "seq":
            if f == "recycle_remaining":
                return RecycleRemainingBatchFeat(**kwargs)

            # Position and indexing
            if f == "chain_idx_seq":
                return ChainIdxSeqFeat(**kwargs)

            # Basic residue information
            elif f == "token_type":
                return TokenTypeSeqFeat(**kwargs)
            elif f == "incident_bond_type_counts":
                return IncidentBondTypeCountsSeqFeat(**kwargs)
            elif f == "atom37_element_charge_embedding":
                return Atom37ElementChargeEmbeddingSeqFeat(**kwargs)

            # Raw coordinate features
            elif f == "ca_coors_nm":
                return CaCoorsNanometersSeqFeat(**kwargs)
            elif f == "x1_a37coors_nm":
                return Atom37NanometersCoorsSeqFeat(**kwargs)
            elif f == "x1_a37coors_nm_rel":
                return Atom37NanometersCoorsSeqFeat(rel=True, **kwargs)

            # Diffusion/sampling coordinates
            elif f == "xt_bb_ca":
                return XtBBCASeqFeat(**kwargs)
            elif f == "xt_local_latents":
                return XtLocalLatentsSeqFeat(**kwargs)

            # Latent variables
            elif f == "z_latent_seq":
                return LatentVariableSeqFeat(**kwargs)

            # 1D Condition annotations
            elif f == "rasa_binned":
                return RasaBinnedSeqFeat(**kwargs)
            elif f == "chain_centroid_xyz":
                return ChainCentroidXyzSeqFeat(**kwargs)
            elif f == "secondary_structure":
                return SecondaryStructureSeqFeat(**kwargs)
            elif f == "hbond_modernized":
                return ModernizedHydrogenBondSeqFeat(**kwargs)
            elif f == "pistack":
                return PiStackingSeqFeat(**kwargs)
            elif f == "pication":
                return PiCationSeqFeat(**kwargs)
            elif f == "saltbridge":
                return SaltBridgeSeqFeat(**kwargs)
            elif f == "hydrophobic":
                return HydrophobicSeqFeat(**kwargs)
            elif f == "metal_coord":
                return MetalCoordinationSeqFeat(**kwargs)
            elif f == "hotspot":
                return HotspotSeqFeat(**kwargs)
            elif f == "sequence_probabilities":
                return SequenceProbabilitiesSeqFeat(**kwargs)
            elif f == "na_ss_state":
                return NucleicSecondaryStructureStateSeqFeat(**kwargs)
            elif f == "coord_group_seq":
                return CoordGroupSeqFeat(**kwargs)
            elif f == "cath_code":
                return FoldEmbeddingSeqFeat(**kwargs)

            # Ligand/atomized features
            elif f == "atom_charge":
                return AtomChargeSeqFeat(**kwargs)
            elif f == "atomic_element":
                return AtomicElementSeqFeat(**kwargs)
            # Conditional features (prevent leakage)
            elif f == "cond_residue_type":
                return ConditionalResidueTypeSeqFeat(**kwargs)
            elif f == "cond_a37coors_nm":
                return ConditionalAtom37NanometersCoorsSeqFeat(**kwargs)
            elif f == "cond_a37coors_internal_nm":
                return ConditionalAtom37NanometersInternalCoorsSeqFeat(**kwargs)

            # Chirality
            elif f == "chirality":
                return ChiralitySeqFeat(**kwargs)

            else:
                raise OSError(f"Sequence feature {f} not implemented.")

        elif self.mode == "pair":
            # Sequence separation
            if f == "token_bond_path_length":
                return TokenBondPathLengthPairFeat(**kwargs)

            # Distance features
            elif f == "xt_bb_ca_pair_dists":
                return XtBBCAPairwiseDistancesPairFeat(**kwargs)
            elif f == "ca_coors_nm_pair_dists":
                return CaCoorsNanometersPairwiseDistancesPairFeat(**kwargs)
            elif f == "x1_bb_pair_dists_nm":
                return BackbonePairDistancesNanometerPairFeat(**kwargs)

            # Chain and indexing features
            elif f == "molecule_idx_pair":
                return MoleculeIdxPairFeat(**kwargs)
            elif f == "molecule_entity_pair":
                return MoleculeEntityPairFeat(**kwargs)
            elif f == "index_group_pair":
                return IndexGroupPairFeat(**kwargs)

            # Bond features (for atomized molecules/ligands)
            elif f == "bond_mask":
                return BondMaskPairFeat(**kwargs)
            elif f == "bond_order":
                return BondOrderPairFeat(**kwargs)
            elif f == "is_aromatic":
                return IsAromaticPairFeat(**kwargs)
            elif f == "na_ss_pair":
                return NucleicSecondaryStructurePairFeat(**kwargs)

            # Canonical polymer ordering (default off; add to feats_pair_repr to enable)
            elif f == "canonical_polymer_ordering":
                return CanonicalPolymerOrderingPairFeat(**kwargs)

            else:
                raise OSError(f"Pair feature {f} not implemented.")

        else:
            raise OSError(f"Wrong feature mode (creator): {self.mode}. Should be 'seq' or 'pair'.")

    def apply_padding_mask(self, feature_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply a [b, n] padding mask to sequence or pair features."""
        if feature_tensor.ndim == 2:
            return feature_tensor * mask.any(dim=-1, keepdim=True)
        elif self.mode == "seq":
            return feature_tensor * mask[..., None]  # [b, n, d]
        elif self.mode == "pair":
            mask_pair = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
            return feature_tensor * mask_pair[..., None]  # [b, n, n, d]
        else:
            raise OSError(f"Wrong feature mode (pad mask): {self.mode}. Should be 'seq' or 'pair'.")

    def forward(self, batch: dict) -> torch.Tensor:
        """Return masked sequence, pair, or per-example features in the projected dtype."""
        # If no features requested just return the zero tensor of appropriate dimensions
        if self.ret_zero:
            return self.zero_creator(batch)

        device_type = batch["padding_mask"].device.type
        autocast_enabled = torch.is_autocast_enabled(device_type)
        feature_tensors = []
        for fcreator in self.feat_creators:
            feature = fcreator(batch)
            if autocast_enabled:
                feature = feature.to(torch.get_autocast_dtype(device_type))
            feature_tensors.append(feature)

        if not (self.compact and all(feature.ndim == 2 for feature in feature_tensors)):
            n = batch["padding_mask"].shape[-1]
            spatial_shape = {"seq": (n,), "pair": (n, n)}[self.mode]
            feature_tensors = [
                feature.reshape(feature.shape[0], *((1,) * len(spatial_shape)), -1).expand(-1, *spatial_shape, -1)
                if feature.ndim == 2
                else feature
                for feature in feature_tensors
            ]

        # Concatenate features and mask
        features = torch.cat(feature_tensors, dim=-1)  # [b, n, dim_f] or [b, n, n, dim_f]
        features = self.apply_padding_mask(features, batch["padding_mask"])  # [b, n, dim_f] or [b, n, n, dim_f]

        # Linear layer and mask
        projected = self.linear_out(features)
        features_proc = self.ln_out(projected).to(projected.dtype)
        return self.apply_padding_mask(features_proc, batch["padding_mask"])  # [b, n, dim_f] or [b, n, n, dim_f]
