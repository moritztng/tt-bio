"""OpenFold3's training adapter: one registration, the shipped forward, upstream's featuriser.

This is the first entry in ``tt_bio.train.catalogue``, and it is deliberately thin. The
catalogue's contract is ``adapter(path, tokens=None) -> (forward, dataset)`` and nothing here
widens it: the forward composes the SHIPPED device modules, the featuriser is upstream's own,
and the loss is the registered ``af3`` objective. There is no second OpenFold3 forward and no
second loss.

What the forward is, and why it is not ``OpenFold3.fold()`` under a tape
-----------------------------------------------------------------------

Upstream's training forward runs the diffusion rollout under ``torch.no_grad()``
(``of3pkg043/openfold3/projects/of3_all_atom/model.py:381``), and in training it is the MINI
rollout (``no_mini_rollout_steps``, ``model.py:360``). Its coordinates feed ``aux_heads``,
which is where the gradient starts. ``OF3ConfidenceHead.distance_onehot`` says the same thing
from our side: *the confidence head is trained on a rolled-out structure that upstream
detaches*.

The shipped sampler agrees by construction. ``OF3SampleDiffusion.__call__`` reads the denoised
coordinates back to host inside the rollout loop and does the EDM update there
(``openfold3_sample_diffusion.py:171``), so no tape survives the recursion whatever the tape
does. That is consistent with upstream rather than a defect, and it is why a training forward
composes the modules rather than wrapping ``fold()``.

So one forward is:

  trunk (taped)
    -> rollout (no_grad, the shipped sampler, unchanged)
    -> confidence heads on the rolled-out structure (taped)

and ``OF3ConfidenceHead.forward_device`` -- which its own docstring calls "the training entry
point" -- returns five of the objective's seven outputs, distogram included.

What a batch can actually score
-------------------------------

``objectives.af3_loss`` keys eight terms off seven outputs and SKIPS a term whose inputs are
absent, recording it as skipped. Two are skipped here and both are properties of the contract
rather than of this model:

``plddt``   ``_TERMS["plddt"]`` reads ``per_atom_lddt`` and ``per_atom_weight`` from the BATCH,
            and both are functions of the PREDICTION -- upstream builds them inside the loss
            under ``no_grad`` and ``losses.atom_bespoke_lddt`` is the same function. A
            featuriser cannot emit them. The objective's contract assumes every label is
            batch-side and pLDDT's is not.

``mse`` /   these read ``pred_xyz`` and ``pred_dist``, which come from the one-step denoise arm
``bond`` /  (upstream's diffusion training objective) rather than from the rollout. That arm is
``smooth_lddt``  not wired here yet; ``forward(..., denoise=True)`` is where it lands, and until
            it does those three terms are reported skipped rather than silently zero.

Both are visible in the objective's own breakdown, which is the designed behaviour: a zero for
a missing term moves the weighted sum and is how a run reports a healthy loss while training on
half its terms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import catalogue, losses

__all__ = ["adapter", "OpenFold3Dataset", "OpenFold3Forward", "MODEL"]

MODEL = "openfold3"

#: The objective's output names this forward produces today, in the order it produces them.
OUTPUTS = ("distogram_logits", "plddt_logits", "pde_logits", "pae_logits", "resolved_logits")

#: Labels the af3 objective reads out of the batch, and what each one is for.
LABELS = ("true_xyz", "coord_mask", "true_dist", "lddt_pair_mask", "bond_mask",
          "frame_atom_index", "is_dna", "is_rna", "is_ligand")


# ------------------------------------------------------------------------------- dataset

def _one(v, i: int = 0):
    """One sample out of a leading batch axis. Upstream collates with a leading 1."""
    if torch.is_tensor(v):
        return v[i]
    if isinstance(v, dict):
        return {k: _one(x, i) for k, x in v.items()}
    if isinstance(v, (list, tuple)) and len(v) > i:
        return v[i]
    return v


def _frame_atom_index(n_token: int) -> np.ndarray:
    """The token triple ``(i-1, i, i+1)``, clamped at the ends.

    Upstream's alignment frame for a polymer token, and the convention the campaign's own
    census records for the frozen batch (``perf/of3t_gradients/coverage_census.json``:
    *"frames": the token triple (i-1, i, i+1), clamped at the ends*).
    """
    i = np.arange(n_token)
    return np.stack([np.clip(i - 1, 0, n_token - 1), i,
                     np.clip(i + 1, 0, n_token - 1)], axis=-1)


class OpenFold3Dataset:
    """Upstream's own featuriser output, one sample per index, plus the af3 labels.

    ``path`` is a ``.pt`` file holding a batch upstream's featuriser produced, or a directory
    of them. Featurisation is NOT generalised here and that is the catalogue's own position:
    each family's cropping and MSA handling is the part that is genuinely different, so a
    model arrives with its featuriser rather than with a shared data layer.

    Four members, no base class: ``__len__``, ``tokens``, ``device``, ``batch``. ``device`` is
    resolved lazily on first use, because a data-parallel run is one process per chip and the
    process that spawns the ranks has to reach the launcher holding no card.
    """

    def __init__(self, path, *, tokens: Optional[int] = None):
        path = Path(path)
        self.paths = (sorted(path.glob("*.pt")) if path.is_dir() else [path])
        if not self.paths:
            raise FileNotFoundError(f"no featurised batch under {path}")
        self._device = None
        self._cache: dict[int, dict] = {}
        self._tokens = tokens

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def tokens(self) -> int:
        if self._tokens is None:
            self._tokens = int(self._features(0)["token_mask"].shape[0])
        return self._tokens

    @property
    def device(self):
        if self._device is None:
            from ..tenstorrent import get_device
            self._device = get_device()
        return self._device

    def _features(self, index: int) -> dict:
        if index not in self._cache:
            raw = torch.load(self.paths[index], map_location="cpu", weights_only=False)
            self._cache[index] = {k: _one(v) for k, v in raw.items()}
        return self._cache[index]

    def batch(self, indices) -> dict:
        """One sample as ``{features, labels...}``. More than one index is refused.

        The loop accumulates per sample and clips per sample -- upstream's
        ``per_sample_clipping: True`` is a different ALGORITHM from clipping the batch once,
        not a different constant -- so it never asks for more than one, and collating an OF3
        batch on the token axis would be a second featuriser.
        """
        indices = list(indices)
        if len(indices) != 1:
            raise ValueError(f"OpenFold3 batches one sample at a time; got {indices}. The "
                             f"loop clips per sample, which is upstream's own shipped "
                             f"default, so it never needs more than one")
        f = self._features(indices[0])
        gt = f.get("ground_truth", {})
        tok = f["token_mask"].float()
        n = int(tok.shape[0])

        # Token scope, one representative atom per token: upstream's own convention for this
        # batch is `ground_truth.start_atom_index`, each token's first atom.
        #
        # THE TWO AXES ARE DIFFERENT LENGTHS and conflating them is the whole of this
        # function's risk. The features are the CROP -- 384 tokens on the pinned batch, of
        # which `token_mask` marks 56 real -- while `ground_truth` is the real tokens only,
        # 56 of them. Every label has to be scattered onto the crop's axis, because that is
        # the axis the model's outputs live on; a label built at the ground truth's length
        # would silently score the first 56 logits against all 56 labels and read healthy.
        rep = gt["start_atom_index"].long() if "start_atom_index" in gt \
            else f["start_atom_index"].long()
        xyz = gt["atom_positions"].float()
        resolved = gt.get("atom_resolved_mask", gt.get("atom_mask")).float()
        real = np.flatnonzero(tok.numpy() > 0)
        if len(real) != int(rep.shape[0]):
            raise ValueError(
                f"{int(tok.sum())} real tokens on the crop axis but {int(rep.shape[0])} in "
                f"ground_truth; the scatter below would put a label on the wrong token")
        true_xyz = np.zeros((n, 3), np.float64)
        coord_mask = np.zeros(n, np.float64)
        true_xyz[real] = xyz[rep].numpy().astype(np.float64)
        coord_mask[real] = resolved[rep].numpy().astype(np.float64)

        true_dist = losses._pdist(true_xyz)
        pair = coord_mask[:, None] * coord_mask[None, :]
        is_dna = f["is_dna"].float().numpy().astype(np.float64)
        is_rna = f["is_rna"].float().numpy().astype(np.float64)
        is_ligand = f["is_ligand"].float().numpy().astype(np.float64)
        lddt_pair_mask = losses.lddt_mask(true_dist, pair, is_dna + is_rna)

        bonds = f["token_bonds"].float().numpy().astype(np.float64)
        return {
            "features": f,
            "true_xyz": true_xyz,
            "coord_mask": coord_mask,
            "true_dist": true_dist,
            "lddt_pair_mask": lddt_pair_mask,
            "bond_mask": bonds * pair,
            "frame_atom_index": _frame_atom_index(n),
            "is_dna": is_dna, "is_rna": is_rna, "is_ligand": is_ligand,
            # `af3` is OpenFold3's and Protenix-v2's mol_type convention. Naming it is what
            # keeps an unresolved dna/rna split from being a silent weighting error.
            "mol_type_convention": "af3",
            "pdb_id": f.get("pdb_id"),
        }


# ------------------------------------------------------------------------------- forward

class OpenFold3Forward:
    """The objective's outputs from an OF3 batch, on device, on the tape.

    Holds the built model, so ``recipes.train_loop`` can take the parameter set from a WALK of
    it rather than from the call-site census -- the census cannot see a weight a module fuses
    in its own ``__init__``, which is 2119 of 2531 on this trunk.
    """

    def __init__(self, checkpoint, *, device=None, num_cycles: int = 1, rollout: int = 20,
                 seed: int = 0):
        self.checkpoint = Path(checkpoint)
        self.rollout = int(rollout)
        self.num_cycles = int(num_cycles)
        self.seed = int(seed)
        self._device = device
        self._model = None
        self._registered = False

    @property
    def device(self):
        if self._device is None:
            from ..tenstorrent import get_device
            self._device = get_device()
        return self._device

    @property
    def model(self):
        """The shipped ``OpenFold3`` module, built once on first use."""
        if self._model is None:
            import ttnn
            from ..openfold3_fold import OpenFold3
            dev = self.device
            sd = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
            ckc = ttnn.init_device_compute_kernel_config(
                dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True, packer_l1_acc=True)
            self._model = OpenFold3(sd, ckc, num_cycles=self.num_cycles)
            # The confidence heads upload their device weights lazily inside
            # `forward_device`, so a walk run before the first forward would not find them
            # and the optimizer's parameter set would be short by the whole confidence head.
            from ..openfold3_confidence import OF3ConfidenceHead
            self._model.confidence_head = OF3ConfidenceHead(
                self._model._confidence_sd, dev, ckc)
            self._model.confidence_head.materialize_device_weights()
        return self._model

    def parameters(self) -> dict:
        """Every device weight the built model reaches, registered as a taped leaf.

        A WALK, not the call-site census: the census sees a weight only where the forward
        routes through ``tt_bio.ops.linear``, and every weight a module fuses in its own
        ``__init__`` -- the TriangleMultiplication in-projection, all of them -- is invisible
        to it. Registration is by the identity of the raw handle, so a walked weight becomes a
        leaf wherever it is passed.
        """
        from .. import autograd as ag
        from ..tenstorrent import walk_device_weights
        walked = {path: t for path, _owner, _key, t in walk_device_weights(self.model)}
        if not self._registered:
            for t in walked.values():
                ag.parameter(t)
            self._registered = True
        return walked

    def __call__(self, batch) -> dict:
        from .. import autograd as ag
        from ..openfold3_fold import build_dm_device_aux, create_noise_schedule
        from ..openfold3_host_prep import (dedup_template_slots, derive_block_aux,
                                           derive_relpos, derive_template_feat,
                                           ref_atom_embed, run_input_atom_encoder)
        from ..openfold3_data import make_openfold3_msa_features
        from ..openfold3_weights import _sub
        import ttnn

        m = self.model
        dev, ckc = self.device, m.ckc
        f = batch["features"]
        self.parameters()

        # ---- host prep. The SHIPPED functions, in the order `worker.py` calls them.
        aux = derive_block_aux(f)
        template_feat, template_slots = dedup_template_slots(derive_template_feat(f))
        relpos = derive_relpos(f)
        msa_feat = make_openfold3_msa_features(f)
        ai = run_input_atom_encoder(dev, ckc, m.sd, f, aux)
        s_input = torch.cat([ai, f["restype"], f["profile"],
                             f["deletion_mean"].unsqueeze(-1)], dim=-1)
        cl0, plm0 = ref_atom_embed(
            _sub(m.sd, "diffusion_module.atom_attn_enc.ref_atom_feature_embedder"), f)
        n_atom, n_token = aux["n_atom"], aux["n_token"]
        ft = lambda x, d=ttnn.bfloat16: ttnn.from_torch(  # noqa: E731
            x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=d)

        tok = f["token_mask"].float()
        pair_mask_d = ft((tok[:, None] * tok[None, :]).reshape(n_token, n_token, 1).unsqueeze(0))
        attn_mask_d = ft(((1.0 - tok) * -1e9).reshape(1, 1, 1, n_token))

        # ---- the taped forward.
        with ag.tape():
            s_input_d = ag.Tensor(ft(s_input.unsqueeze(0)))
            relpos_d = ag.Tensor(ft(relpos.unsqueeze(0)))
            bonds_d = ag.Tensor(ft(f["token_bonds"].unsqueeze(0).unsqueeze(-1)))
            s_init, z_init = m.input_glue(s_input_d, relpos_d, bonds_d)
            s_trunk, z_trunk = m.trunk(
                s_init, z_init, {k: ag.Tensor(ft(v)) for k, v in template_feat.items()},
                ag.Tensor(ft(msa_feat.unsqueeze(0))), s_input_d,
                template_slots=template_slots)

            # ---- the rollout. `no_grad` is upstream's own arrangement, not a shortcut:
            # `model.py:381` wraps `sample_diffusion` in `torch.no_grad()` and trains the
            # confidence heads on what it returns. The shipped sampler is called unchanged.
            with ag.no_grad():
                dm_aux = build_dm_device_aux(
                    dev, ft, cl0=cl0, plm0=plm0, atom_mask=aux["atom_mask"],
                    atom_to_token_index=aux["atom_to_token_index"],
                    npe_q_indices=aux["npe_q_indices"], npe_k_indices=aux["npe_k_indices"],
                    zij_mask=aux["zij_mask"], key_block_idxs=aux["key_block_idxs"],
                    invalid_mask=aux["invalid_mask"], mask_trunked=aux["mask_trunked"],
                    atom_to_token_mean=aux["atom_to_token_mean"],
                    token_mask=tok, n_atom=n_atom, n_token=n_token,
                    nb=aux["nb"], NP=aux["NP"], n_tok_pad=n_token)
                schedule = create_noise_schedule(self.rollout, **m.ns_cfg)
                xl0, rots, trans, noise, ts, ctau = m._gen_rollout(
                    schedule, n_atom, self.seed)
                xl_d = m.sampler(
                    ft(xl0.unsqueeze(0)), _v(s_trunk), _v(s_input_d), _v(z_trunk), _v(relpos_d),
                    ft(tok.reshape(1, n_token)), pair_mask_d,
                    ft(tok.reshape(n_token, 1).unsqueeze(0)),
                    dm_aux["cl0_d"], dm_aux["plm0_d"], dm_aux["amc_d"], dm_aux["amc_na_d"],
                    dm_aux["idx_tt"], dm_aux["flat_tt"], dm_aux["zij_mask_d"],
                    dm_aux["kidx_tt"], dm_aux["valid_d"], dm_aux["mb_d"], dm_aux["pm_d"],
                    dm_aux["mean_d"], dm_aux["tok_pad_tt"], dm_aux["tok_col_pad_tt"],
                    n_atom, dm_aux["NP"] if "NP" in dm_aux else aux["NP"], aux["nb"],
                    n_token, n_token, schedule, rots, trans, noise, ts, ctau, m.step_scale)
                xl = torch.Tensor(ttnn.to_torch(_v(xl_d))).float().reshape(n_atom, 3)

            # ---- the confidence heads, on the rolled-out structure. `forward_device` is
            # their own training entry point and it returns the distogram head with them.
            rep = f["ground_truth"]["start_atom_index"].long() \
                if "ground_truth" in f else f["start_atom_index"].long()
            oh_d = m.confidence_head.distance_onehot(xl[rep])
            out = m.confidence_head.forward_device(
                s_input_d, s_trunk, z_trunk, ag.Tensor(oh_d),
                use_zij_trunk_embedding=True,
                pair_mask_d=pair_mask_d, attn_mask_d=attn_mask_d)

        self.rollout_coords = xl
        return {"distogram_logits": out["distogram_logits"],
                "plddt_logits": out["plddt_logits"],
                "pde_logits": out["pde_logits"],
                "pae_logits": out["pae_logits"],
                "resolved_logits": out["experimentally_resolved_logits"]}


def _v(t):
    """The raw handle under a taped tensor. The rollout is not differentiated."""
    return t.value if hasattr(t, "value") else t


# ------------------------------------------------------------------------------ register

def adapter(path, tokens=None, *, checkpoint=None, rollout: int = 20, num_cycles: int = 1,
            seed: int = 0):
    """``(forward, dataset)`` for OpenFold3. ``path`` is the featurised corpus.

    ``checkpoint`` defaults to ``path``'s sibling ``of3.pt`` so the registration stays a
    one-liner; pass it when the weights live elsewhere.
    """
    path = Path(path)
    ckpt = Path(checkpoint) if checkpoint else (path if path.is_dir() else path.parent) / "of3.pt"
    return (OpenFold3Forward(ckpt, rollout=rollout, num_cycles=num_cycles, seed=seed),
            OpenFold3Dataset(path, tokens=tokens))


catalogue.register(MODEL, adapter)
