"""Nesso-1: coarse-grained protein-ligand affinity prediction.

Torch reference for the Tenstorrent port. Nesso-1 has no structure module: the only
3D quantity it predicts is a soft distogram, and both the pocket crop and the affinity
ensemble consume that distogram rather than coordinates. The whole device workload is
one pair tensor ``z: [1, N, N, 128]`` plus per-token ``s_inputs: [1, N, 384]``.

88% of the 41.3 M checkpoint parameters load key-for-key into classes tt-bio already
ships, so this module assembles rather than reimplements: the 48-block trunk and both
8-block affinity pairformers are :class:`tt_bio.reference.PairformerNoSeqModule`, the
input embedder is :class:`tt_bio.boltz2.InputEmbedder` with the MSA profile track off,
and the relative-position and pairwise-conditioning blocks come from Boltz-2 unchanged.
New here: the ESM pair update, the soft-distogram affinity head, and the top level.

Upstream: https://github.com/recursionpharma/nesso (Apache-2.0), weights
``recursionpharma/nesso``. The host featurization pipeline is vendored verbatim under
``tt_bio/_vendor/nesso``; see ``scripts/nesso1_port/`` for the parity gate.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.nn import LayerNorm, Linear

from tt_bio._vendor.nesso.data import const
from tt_bio._vendor.nesso.data.crop import select_pocket_token_indices
from tt_bio.boltz2 import (
    AffinityHeadsTransformer,
    InputEmbedder,
    LinearNoBias,
    PairwiseConditioning,
    RelativePositionEncoder,
    init,
    tenstorrent,
)
from tt_bio.reference import PairformerNoSeqModule
from tt_bio import weights as _weights

HPARAMS_NAME = "hparams.json"
WEIGHTS_NAME = "model.safetensors"

MIN_DIST = 2.0


# ---------------------------------------------------------------------------
# distogram utilities
# ---------------------------------------------------------------------------


def distogram_bin_centers(
    num_bins: int = 64,
    min_dist: float = MIN_DIST,
    max_dist: float = 22.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Bin centers for a distogram. The two outer bins are open-ended."""
    boundaries = torch.linspace(
        min_dist, max_dist, num_bins - 1, device=device, dtype=dtype
    )
    centers = torch.empty(num_bins, device=device, dtype=dtype)
    centers[0] = 1.5
    centers[-1] = 24.5
    centers[1:-1] = (boundaries[:-1] + boundaries[1:]) * 0.5
    return centers


def compute_expected_distance(
    pdistogram: Tensor,
    min_dist: float = MIN_DIST,
    max_dist: float = 22.0,
) -> Tensor:
    """Expected pairwise distance ``[B, N, N]`` from distogram logits ``[B, N, N, bins]``."""
    probs = pdistogram.softmax(dim=-1).float()
    centers = distogram_bin_centers(
        num_bins=pdistogram.shape[-1],
        min_dist=min_dist,
        max_dist=max_dist,
        device=pdistogram.device,
        dtype=probs.dtype,
    )
    return torch.einsum("...b,b->...", probs, centers)


def compute_distogram_entropy(
    pdistogram: Tensor,
    feats: dict[str, Tensor],
    custom_mask: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Normalized distogram entropy, with protein-protein / protein-ligand / ligand-ligand means."""
    pred = pdistogram.float()
    n_bins = pred.shape[-1]

    base_mask = feats["token_disto_mask"] if custom_mask is None else custom_mask
    base_mask = base_mask.float()
    base_mask = base_mask[:, None, :] * base_mask[:, :, None]
    base_mask = base_mask * (
        1 - torch.eye(base_mask.shape[1], device=base_mask.device)[None]
    )

    mol_type = feats["mol_type"].long()
    is_protein = (mol_type == const.chain_type_ids["PROTEIN"]).float()
    is_ligand = (mol_type == const.chain_type_ids["NONPOLYMER"]).float()

    pp_mask = base_mask * (is_protein[:, :, None] * is_protein[:, None, :])
    ll_mask = base_mask * (is_ligand[:, :, None] * is_ligand[:, None, :])
    pl_mask = base_mask * (
        is_protein[:, :, None] * is_ligand[:, None, :]
        + is_ligand[:, :, None] * is_protein[:, None, :]
    )

    log_q = torch.nn.functional.log_softmax(pred, dim=-1)
    q = log_q.exp()
    h_pair = -torch.sum(q * log_q, dim=-1) / math.log(n_bins)

    def masked_mean(val: Tensor, m: Tensor) -> Tensor:
        return torch.sum(val * m, dim=(-1, -2)) / torch.sum(m, dim=(-1, -2)).clamp(min=1.0)

    return {
        "entropy_pair": h_pair,
        "entropy_pp": masked_mean(h_pair, pp_mask),
        "entropy_pl": masked_mean(h_pair, pl_mask),
        "entropy_ll": masked_mean(h_pair, ll_mask),
    }


# ---------------------------------------------------------------------------
# new modules
# ---------------------------------------------------------------------------


def _pairformer(use_tenstorrent: bool, fp32: bool, token_z: int, args: dict):
    """The pair-only pairformer, on host or on device.

    ``transform_s=False`` with ``affinity=True`` IS the no-seq pairformer: the affinity
    flag selects the cross-chain ``pair_mask`` path over a 1D token mask, which is what
    both of Nesso-1's stacks want. ``Fp32PairformerModule`` takes ``pair_mask`` directly
    and needs no such flag.
    """
    if not use_tenstorrent:
        return PairformerNoSeqModule(token_z, **args)
    if fp32:
        return tenstorrent.Fp32PairformerModule(
            args["num_blocks"], 32, 4, None, None, False
        )
    return tenstorrent.PairformerModule(
        args["num_blocks"], 32, 4, None, None, False, affinity=True
    )


class ESMModule(nn.Module):
    """Fold the ESM-2 sequence embedding into the pair track.

    Takes the place of Boltz-2's MSA module: project ESM and ``s_inputs`` to a single
    representation, then add its outer sum into ``z``. No attention, no pair stack.
    """

    def __init__(
        self,
        token_s: int,
        token_z: int,
        esm_embed_dim: int,
        use_esm_all_layers: bool = False,
        esm_num_layers: int = 37,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_esm_all_layers = use_esm_all_layers
        self.esm_num_layers = esm_num_layers
        if use_esm_all_layers:
            self.esm_layer_weights = nn.Parameter(torch.zeros(esm_num_layers))

        # Dropout is inert at inference but holds index 4 in the state dict.
        self.esm_mlp = nn.Sequential(
            nn.LayerNorm(esm_embed_dim),
            nn.Linear(esm_embed_dim, token_s),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(token_s, token_s),
        )
        self.s_inputs_proj = nn.Linear(token_s, token_s)

        self.esm_z_1 = nn.Linear(token_s, token_z, bias=False)
        self.esm_z_2 = nn.Linear(token_s, token_z, bias=False)
        init.gating_init_(self.esm_z_1.weight)
        init.gating_init_(self.esm_z_2.weight)
        init.gating_init_(self.s_inputs_proj.weight)

    def forward(
        self,
        z: Tensor,
        s_inputs: Tensor,
        s_esm: Tensor,
        pair_mask: Tensor,
        use_kernels: bool = False,
    ) -> Tensor:
        if s_esm.dim() == 4 and self.use_esm_all_layers:
            s_esm = torch.einsum("bnld,l->bnd", s_esm, self.esm_layer_weights.softmax(0))

        s = self.s_inputs_proj(s_inputs) + self.esm_mlp(s_esm)
        delta_z = self.esm_z_1(s)[:, :, None, :] + self.esm_z_2(s)[:, None, :, :]
        return z + delta_z * pair_mask.unsqueeze(-1)


class AffinityModule(nn.Module):
    """Affinity head over the cropped pocket.

    The Boltz-2 affinity module bins a ``cdist`` over predicted coordinates into an
    embedding. Nesso-1 has no coordinates, so it projects the softmaxed distogram
    instead, and adds an ESM term to ``s_inputs``.
    """

    def __init__(
        self,
        token_s: int,
        token_z: int,
        pairformer_args: dict,
        transformer_args: dict,
        num_dist_bins: int = 64,
        max_dist: float = 22.0,
        esm_embed_dim: int = 1280,
        use_tenstorrent: bool = False,
        fp32: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.distogram_proj = LinearNoBias(num_dist_bins, token_z)
        init.gating_init_(self.distogram_proj.weight)

        self.esm_proj = nn.Sequential(
            nn.LayerNorm(esm_embed_dim),
            nn.Linear(esm_embed_dim, token_s),
            nn.ReLU(),
            nn.Linear(token_s, token_s),
        )
        init.gating_init_(self.esm_proj[-1].weight)
        init.bias_init_zero_(self.esm_proj[-1].bias)

        self.z_norm = nn.LayerNorm(token_z)
        self.z_linear = LinearNoBias(token_z, token_z)

        self.s_to_z_prod_in1 = LinearNoBias(token_s, token_z)
        self.s_to_z_prod_in2 = LinearNoBias(token_s, token_z)

        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=2,
        )

        self.pairformer_stack = _pairformer(
            use_tenstorrent, fp32, token_z, pairformer_args
        )
        self.use_tenstorrent = use_tenstorrent

        self.affinity_heads = AffinityHeadsTransformer(
            token_z=token_z,
            input_token_s=token_s,
            num_blocks=transformer_args.get("num_blocks", 0),
            num_heads=transformer_args.get("num_heads", 0),
            activation_checkpointing=False,
            return_repr=True,
        )

    def forward(
        self,
        s_inputs: Tensor,
        z: Tensor,
        pdistogram: Tensor,
        feats: dict[str, Tensor],
        use_kernels: bool = False,
    ) -> dict[str, Tensor]:
        pad_token_mask = feats["token_pad_mask"]
        is_protein = (
            (feats["mol_type"] == const.chain_type_ids["PROTEIN"]).float()
            * pad_token_mask
        ).unsqueeze(-1)

        s_inputs = s_inputs + self.esm_proj(feats["s_esm"]) * is_protein

        z = self.z_linear(self.z_norm(z.float()))
        z = (
            z
            + self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
            + self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
        )
        z = z + self.pairwise_conditioner(
            z_trunk=z, token_rel_pos_feats=self.distogram_proj(pdistogram)
        )

        rec_mask = (
            feats["mol_type"] == const.chain_type_ids["PROTEIN"]
        ).float() * pad_token_mask
        lig_mask = feats["affinity_token_mask"].float() * pad_token_mask
        cross_pair_mask = (
            lig_mask[:, :, None] * rec_mask[:, None, :]
            + rec_mask[:, :, None] * lig_mask[:, None, :]
            + lig_mask[:, :, None] * lig_mask[:, None, :]
        )

        if self.use_tenstorrent:
            _, z = self.pairformer_stack(None, z, pair_mask=cross_pair_mask)
        else:
            z = self.pairformer_stack(
                z, pair_mask=cross_pair_mask, use_kernels=use_kernels
            )
        return self.affinity_heads(z=z, feats=feats)


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


class Nesso1(nn.Module):
    """Nesso-1, inference only."""

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        embedder_args: dict[str, Any],
        atom_feature_dim: int = 128,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        predict_args: Optional[dict[str, Any]] = None,
        num_dist_bins: int = 64,
        max_dist: float = 22.0,
        pairformer_model_args: Optional[dict[str, Any]] = None,
        use_kernels: bool = False,
        esm_module_args: Optional[dict[str, Any]] = None,
        affinity_prediction: bool = False,
        affinity_model_args: Optional[dict[str, Any]] = None,
        affinity_model_args2: Optional[dict[str, Any]] = None,
        use_tenstorrent: bool = False,
        trunk_fp32: bool = False,
        affinity_fp32: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.predict_args = dict(predict_args or {})
        self.use_tenstorrent = use_tenstorrent
        self.affinity_prediction = affinity_prediction
        self.use_kernels = use_kernels
        self.num_dist_bins = num_dist_bins
        self.max_dist = max_dist

        embedder_args = dict(embedder_args)
        embedder_args.pop("pocket_conditioning", None)  # parsed upstream, no weights
        self.input_embedder = InputEmbedder(
            atom_s,
            atom_z,
            token_s,
            token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            use_msa_profile=False,
            **embedder_args,
        )

        self.rel_pos = RelativePositionEncoder(token_z)

        self.z_init_1 = Linear(token_s, token_z, bias=False)
        self.z_init_2 = Linear(token_s, token_z, bias=False)
        self.token_bonds = Linear(1, token_z, bias=False)
        self.token_bonds_type = nn.Embedding(len(const.bond_types) + 1, token_z)
        init.gating_init_(self.z_init_1.weight)
        init.gating_init_(self.z_init_2.weight)
        init.gating_init_(self.token_bonds.weight)
        init.gating_init_(self.token_bonds_type.weight)

        self.z_norm = LayerNorm(token_z)
        self.z_recycle = Linear(token_z, token_z, bias=False)
        init.gating_init_(self.z_recycle.weight)

        self.esm_module = ESMModule(
            token_s=token_s, token_z=token_z, **dict(esm_module_args or {})
        )
        self.pairformer_module = _pairformer(
            use_tenstorrent, trunk_fp32, token_z, pairformer_model_args
        )
        self.distogram_head = Linear(token_z, num_dist_bins)

        if affinity_prediction:
            self.affinity_module = AffinityModule(
                token_s=token_s,
                token_z=token_z,
                use_tenstorrent=use_tenstorrent,
                fp32=affinity_fp32,
                **(affinity_model_args or {}),
            )
            self.affinity_module2 = AffinityModule(
                token_s=token_s,
                token_z=token_z,
                use_tenstorrent=use_tenstorrent,
                fp32=affinity_fp32,
                **(affinity_model_args2 or affinity_model_args or {}),
            )

    # -- pocket selection ---------------------------------------------------

    def get_pocket_mask(
        self, z_exp: Tensor, feats: dict[str, Any], cutoff: float = 15.0
    ) -> Tensor:
        """Tokens within ``cutoff`` of any ligand token, plus the ligand itself."""
        mol_type = feats["mol_type"].long()
        is_protein = mol_type == const.chain_type_ids["PROTEIN"]
        is_ligand = mol_type == const.chain_type_ids["NONPOLYMER"]
        valid = feats["token_pad_mask"].bool()

        has_lig = is_ligand.any(dim=-1, keepdim=True)
        large = torch.full((), 1e6, device=z_exp.device, dtype=z_exp.dtype)
        min_to_ligand = torch.where(is_ligand[:, None, :], z_exp, large).min(dim=-1).values
        pocket_with_lig = (is_ligand | (is_protein & (min_to_ligand <= cutoff))) & valid
        pocket = torch.where(has_lig.expand(-1, z_exp.shape[1]), pocket_with_lig, valid)
        return pocket.float()

    @staticmethod
    def _crop_feats_by_indices(
        feats: dict[str, Any], mask: Tensor, keys: tuple[str, ...]
    ) -> dict[str, Any]:
        new = dict(feats)
        for k in keys:
            v = feats.get(k)
            if isinstance(v, Tensor):
                new[k] = v[:, mask].contiguous()
        return new

    def _select_pocket_indices(
        self,
        pdistogram: Tensor,
        feats: dict[str, Any],
        *,
        cutoff: float,
        max_tokens: int,
    ) -> Tensor:
        d_exp = compute_expected_distance(pdistogram, max_dist=self.max_dist)
        mol_type = feats["mol_type"][0]
        pad = feats["token_pad_mask"][0].bool()
        prot_pos = torch.where((mol_type == const.chain_type_ids["PROTEIN"]) & pad)[0]
        lig_pos = torch.where((mol_type == const.chain_type_ids["NONPOLYMER"]) & pad)[0]
        if prot_pos.numel() == 0 or lig_pos.numel() == 0:
            raise ValueError("No protein or ligand tokens found in the batch")

        min_d = d_exp[0][prot_pos[:, None], lig_pos[None, :]].min(dim=1).values
        asym_np = feats["asym_id"][0].cpu().numpy()
        res_np = feats["residue_index"][0].cpu().numpy()
        prot_np = prot_pos.cpu().numpy()

        keep = select_pocket_token_indices(
            protein_token_indices=prot_np,
            protein_asym_ids=asym_np[prot_np],
            protein_res_idxs=res_np[prot_np],
            dists_to_ligand=min_d.float().cpu().numpy(),
            ligand_token_indices=lig_pos.cpu().numpy(),
            max_tokens=max_tokens,
            threshold_distance=cutoff,
        )
        return torch.tensor(keep, device=pdistogram.device, dtype=torch.long)

    def _invalidate_device_masks(self) -> None:
        """Drop cached device masks on every pairformer that has them.

        Device modules cache their pad and pair masks on the first forward, which is
        right for a model whose token count is fixed for the life of the module.
        Nesso-1 breaks that invariant twice per prediction: it crops to the pocket
        after the first trunk pass, and the next prediction starts back at the full
        count. Both transitions have to invalidate, and the affinity stacks need it as
        much as the trunk because two inputs rarely crop to the same size.
        """
        if not self.use_tenstorrent:
            return
        from tt_bio import tenstorrent

        # A typed call, not `getattr(module, ..., None)`. The soft form was silently a
        # no-op the moment the method it looked for was not there, and a merge that
        # dropped it produced a broadcast failure 300 lines away instead of an
        # AttributeError here.
        for module in self.modules():
            if isinstance(module, tenstorrent.TorchWrapper):
                module.reset_static_cache()

    def pocket_crop(self, z: Tensor, feats: dict[str, Any]) -> tuple[Tensor, Tensor]:
        pdistogram_full = self.distogram_head(z + z.transpose(1, 2))
        keep_t = self._select_pocket_indices(
            pdistogram_full,
            feats,
            cutoff=self.predict_args.get("refine_protein_cutoff", 22.0),
            max_tokens=self.predict_args.get("refine_protein_tokens_budget", 196),
        )
        return keep_t, pdistogram_full

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        feats: dict[str, Tensor],
        recycling_steps: int = 0,
        refine_protein_inference: bool = False,
    ) -> dict[str, Tensor]:
        dict_out: dict[str, Tensor] = {}

        s_inputs = self.input_embedder(feats)
        z_init = self.z_init_1(s_inputs)[:, :, None] + self.z_init_2(s_inputs)[:, None, :]
        z_init = z_init + self.rel_pos(feats)
        z_init = z_init + self.token_bonds(feats["token_bonds"].float())
        z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())

        z = torch.zeros_like(z_init)
        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]
        s_esm = feats["s_esm"]

        pdistogram_full: Optional[Tensor] = None
        keep_for_merge: Optional[Tensor] = None
        refine_pocket = refine_protein_inference and recycling_steps >= 1

        # The trunk always starts at the full token count, but a previous prediction
        # left its device masks at the cropped count. Restore the invariant here, not
        # only at the crop: without this, prediction 1 works and prediction 2 dies in
        # an eltwise broadcast, padded-N against the crop budget.
        self._invalidate_device_masks()

        for i in range(recycling_steps + 1):
            z = z_init + self.z_recycle(self.z_norm(z))
            z = self.esm_module(
                z,
                s_inputs=s_inputs,
                s_esm=s_esm,
                pair_mask=pair_mask,
                use_kernels=self.use_kernels,
            )
            z = self._trunk(z, mask, pair_mask)

            if refine_pocket and i == 0:
                # One crop, after the first trunk pass: the remaining recycles run on
                # at most refine_protein_tokens_budget tokens.
                keep_for_merge, pdistogram_full = self.pocket_crop(z, feats=feats)
                dict_out["keep_indices"] = keep_for_merge
                dict_out["crop_protein_tokens"] = keep_for_merge

                keep_t = keep_for_merge
                z = z[:, keep_t][:, :, keep_t].contiguous()
                z_init = z_init[:, keep_t][:, :, keep_t].contiguous()
                s_inputs = s_inputs[:, keep_t].contiguous()

                feats = self._crop_feats_by_indices(
                    feats,
                    mask=keep_t,
                    keys=(
                        "mol_type",
                        "token_pad_mask",
                        "s_esm",
                        "asym_id",
                        "residue_index",
                        "affinity_token_mask",
                    ),
                )
                s_esm = feats["s_esm"]
                mask = feats["token_pad_mask"].float()
                pair_mask = mask[:, :, None] * mask[:, None, :]
                # N just changed, so the cached device masks no longer describe
                # this input.
                self._invalidate_device_masks()

        pdistogram_local = self.distogram_head(z + z.transpose(1, 2))

        if pdistogram_full is not None and keep_for_merge is not None:
            km = keep_for_merge
            pdistogram_full[0, km[:, None], km[None, :], :] = pdistogram_local[0]
            dict_out["pdistogram"] = pdistogram_full
        else:
            dict_out["pdistogram"] = pdistogram_local
        dict_out["z"] = z

        if self.affinity_prediction:
            dict_out.update(
                self._affinity(s_inputs, z, pdistogram_local, feats)
            )
        return dict_out

    def _trunk(self, z: Tensor, mask: Tensor, pair_mask: Tensor) -> Tensor:
        if self.use_tenstorrent:
            _, z = self.pairformer_module(None, z, mask=mask, pair_mask=pair_mask)
            return z
        return self.pairformer_module(
            z, pair_mask=pair_mask, use_kernels=self.use_kernels
        )

    def _affinity(
        self,
        s_inputs: Tensor,
        z: Tensor,
        pdistogram_local: Tensor,
        feats: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        cutoff = self.predict_args.get("affinity_protein_cutoff", 15.0)
        z_exp = compute_expected_distance(pdistogram_local, max_dist=self.max_dist)
        pocket = self.get_pocket_mask(z_exp, feats, cutoff=cutoff)[0].bool()
        keep_t = pocket.nonzero(as_tuple=True)[0]

        s_aff = s_inputs[:, keep_t].contiguous()
        z_aff = z[:, keep_t][:, :, keep_t].contiguous()
        pdistogram_aff = (
            pdistogram_local[:, keep_t][:, :, keep_t].contiguous().softmax(dim=-1)
        )
        feats_aff = self._crop_feats_by_indices(
            feats,
            mask=keep_t,
            keys=(
                "mol_type",
                "token_pad_mask",
                "s_esm",
                "affinity_token_mask",
                "asym_id",
                "residue_index",
            ),
        )

        out: dict[str, Tensor] = {"affinity_keep_indices": keep_t}
        members = [
            m(
                s_inputs=s_aff,
                z=z_aff,
                pdistogram=pdistogram_aff,
                feats=feats_aff,
                use_kernels=self.use_kernels,
            )
            for m in (self.affinity_module, self.affinity_module2)
        ]
        out.update(members[0])
        out["affinity_pred_value1"] = members[0]["affinity_pred_value"]
        out["affinity_pred_value2"] = members[1]["affinity_pred_value"]
        out["affinity_pred_value"] = (
            out["affinity_pred_value1"] + out["affinity_pred_value2"]
        ) / 2.0

        p = (
            torch.sigmoid(members[0]["affinity_logits_binary"])
            + torch.sigmoid(members[1]["affinity_logits_binary"])
        ) / 2.0
        out["affinity_probability_binary"] = p
        out["affinity_logits_binary"] = torch.logit(p.clamp(1e-6, 1 - 1e-6))
        return out

    # -- prediction ---------------------------------------------------------

    @torch.no_grad()
    def predict(self, feats: dict[str, Tensor], save_metadata: bool = False) -> dict[str, Any]:
        """Run one prediction and derive everything the CLI reports."""
        recycling_steps = self.predict_args.get("recycling_steps", 3)
        out = self(
            feats,
            recycling_steps=recycling_steps,
            refine_protein_inference=self.predict_args.get(
                "refine_protein_inference", False
            ),
        )

        pred: dict[str, Any] = {
            "pdistogram": out["pdistogram"],
            "token_pad_mask": feats["token_pad_mask"],
        }
        z_exp = compute_expected_distance(out["pdistogram"], max_dist=self.max_dist)
        pocket_mask = self.get_pocket_mask(
            z_exp, feats, cutoff=self.predict_args.get("pose_protein_cutoff", 15.0)
        )

        pred.update(compute_distogram_entropy(out["pdistogram"], feats))
        entropy_crop = compute_distogram_entropy(
            out["pdistogram"],
            feats,
            custom_mask=feats["token_pad_mask"].float() * pocket_mask,
        )
        for key in ("pp", "pl", "ll"):
            pred[f"entropy_crop_{key}"] = entropy_crop[f"entropy_{key}"]

        pred["pocket_mask"] = pocket_mask
        pred["token_mask"] = feats["token_pad_mask"].float() * pocket_mask

        if self.affinity_prediction and "affinity_pred_value" in out:
            for key in (
                "affinity_pred_value",
                "affinity_pred_value1",
                "affinity_pred_value2",
                "affinity_logits_binary",
                "affinity_probability_binary",
            ):
                if key in out:
                    pred[key] = out[key].squeeze(-1)

        if save_metadata:
            n = feats["token_pad_mask"].shape[1]
            refine_mask = torch.zeros(n, dtype=torch.bool, device=out["z"].device)
            if "keep_indices" in out:
                refine_mask[out["keep_indices"]] = True
            else:
                refine_mask = feats["token_pad_mask"][0].bool().clone()
            pred["z_full"] = out["z"][0].to(torch.bfloat16)
            pred["expected_distances_full"] = z_exp[0, refine_mask][:, refine_mask]
            pred["refine_mask_full"] = refine_mask
            pred["pocket_mask_full"] = pocket_mask[0].bool() & refine_mask
            pred["mol_type_refined"] = feats["mol_type"][0][refine_mask]
        return pred

    # -- weights ------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | Path = _weights.NESSO_REPO,
        *,
        revision: str = _weights.NESSO_REVISION,
        cache_dir: str | Path | None = None,
        use_tenstorrent: bool = False,
        **model_kwargs,
    ) -> "Nesso1":
        """Load from a local directory or the ``recursionpharma/nesso`` Hub repo.

        The Hub layout nests both files under the revision tag, e.g.
        ``v1.0.0/hparams.json``.
        """
        from safetensors.torch import load_file

        local = Path(path_or_repo)
        if local.is_dir():
            hparams_path, weights_path = local / HPARAMS_NAME, local / WEIGHTS_NAME
        else:
            from huggingface_hub import hf_hub_download

            hparams_path, weights_path = (
                Path(
                    hf_hub_download(
                        repo_id=str(path_or_repo),
                        filename=f"{revision}/{name}",
                        revision=revision,
                        cache_dir=cache_dir,
                    )
                )
                for name in (HPARAMS_NAME, WEIGHTS_NAME)
            )

        hparams = json.loads(Path(hparams_path).read_text())
        model = cls(**hparams, use_tenstorrent=use_tenstorrent, **model_kwargs)
        model.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
        model.eval()
        return model

# ---------------------------------------------------------------------------
# batch driver
# ---------------------------------------------------------------------------

REPORTED_SCALARS = (
    "affinity_pred_value",
    "affinity_pred_value1",
    "affinity_pred_value2",
    "affinity_logits_binary",
    "affinity_probability_binary",
    "entropy_pp",
    "entropy_pl",
    "entropy_ll",
    "entropy_crop_pp",
    "entropy_crop_pl",
    "entropy_crop_ll",
)

DEFAULT_SEED = 20260820


def screen(
    data: "Path | str",
    out_dir: "Path | str",
    weights: str = _weights.NESSO_REPO,
    *,
    use_tenstorrent: bool = True,
    trunk_fp32: bool = False,
    affinity_fp32: bool = True,
    recycling_steps: int = 5,
    tokens_budget: int = 256,
    num_workers: int = 0,
    ccd_pkl: "Path | None" = None,
    cache: "Path | None" = None,
    seed: int | None = DEFAULT_SEED,
    progress=None,
) -> list[dict]:
    """Score every YAML under ``data``, one record at a time, and return the rows.

    The model loads once and stays resident, which is the whole point for a screen:
    one target against many ligands pays the weight load and the kernel compile once.

    ``seed`` pins the featurization draw. It is not cosmetic: the featurizer applies a
    random roto-translation to each conformer off the global torch RNG, so upstream is
    not reproducible run to run (the reference differed on 64/64 affinity values, max
    0.058). Seeding makes a tt-bio screen repeatable. Pass None for upstream behaviour.
    """
    from tt_bio.nesso1_input import CLI_PREDICT_ARGS, collate, prepare

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    dataset, manifest, failed = prepare(
        Path(data).expanduser(),
        out,
        ccd_pkl=ccd_pkl,
        num_workers=num_workers,
        esm_cache=cache,
    )
    model = Nesso1.from_pretrained(
        weights,
        use_tenstorrent=use_tenstorrent,
        trunk_fp32=trunk_fp32,
        affinity_fp32=affinity_fp32,
        cache_dir=cache,
    )
    # The checkpoint ships use_kernels: true, which routes the triangle ops through
    # cuEquivariance. There is no cueq here and none is wanted: tt-bio's own
    # TriangleMultiplication and TriangleAttention (both with fused kernels, both
    # default-on) are what replace it on device, and upstream force-disables the flag on
    # CPU for the same reason.
    model.use_kernels = False
    model.predict_args.update(CLI_PREDICT_ARGS)
    model.predict_args["recycling_steps"] = recycling_steps
    model.predict_args["refine_protein_tokens_budget"] = tokens_budget

    rows: list[dict] = []
    for idx, record in enumerate(manifest.records):
        if seed is not None:
            torch.manual_seed(seed)
        item = dataset[idx]
        if item.get("exception"):
            rows.append({"id": record.id, "error": "featurizer raised"})
            continue
        feats = collate(item)
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model.predict(feats)
        row = {
            "id": record.id,
            "n_tokens": int(feats["token_pad_mask"].shape[-1]),
            "seconds": time.perf_counter() - t0,
        }
        row.update(
            {k: float(pred[k].reshape(-1)[0]) for k in REPORTED_SCALARS if k in pred}
        )
        (out / f"{record.id}_affinity.json").write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
        if progress is not None:
            progress(row)
    for stem in failed:
        rows.append({"id": stem, "error": "could not be parsed"})
    return rows
