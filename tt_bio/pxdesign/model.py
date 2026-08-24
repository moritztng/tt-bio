"""PXDesign-d: the generator, assembled from tt-bio's Protenix modules.

The whole model is `design_condition_embedder` + `diffusion_module`. There is no trunk, no
MSA stack and no confidence head, so this class is `Protenix` with the trunk replaced by
two lookups:

  * `s_trunk` is zeros. Upstream does the same (`ProtenixDesign.get_condition_embedding`),
    and it is the reason a design costs a third of a fold.
  * `z_trunk` is an `Embedding(65, 128)` over the 64-bin distogram of the target, so the
    structural conditioning enters as a pair-tensor lookup rather than 48 Pairformer blocks.

Everything after that -- the diffusion pair conditioning, the atom caches, the EDM sampler,
the 16-block token DiT -- is `tt_bio.protenix` unchanged, running the design checkpoint's
own depths and widths (16 DiT blocks, 4+4 atom transformer, c_z=128), which it reads off
the weights.

What is new arithmetic, and therefore what this file actually adds, is two lines: the
`Embedding(65, 128)` lookup and `input_map`, a `Linear(430 -> 449)` over a per-token
channel set that differs from Protenix's.

Input features come from `tt_bio.pxdesign.featurize` (the design-specific half) on top of a
Protenix-style atom array. tt-bio does not yet build that atom array from a CIF; see
`scripts/pxdesign_port/design_e2e.py` for the captured-input path.
"""
from __future__ import annotations

import torch
import ttnn

from ..protenix import (AtomAttentionEncoder, AtomFeaturization, DiffusionModule, Protenix,
                        edm_sample)
from .featurize import condition_template_index
from ..envflags import env_flag

# Upstream's sampler settings, as an actual `pxdesign` run uses them. 400 steps and a
# constant eta of 2.5 both differ from Protenix's fold defaults (200 steps, eta 1.5).
#
# The eta is the one to be careful about, because `configs_base.py` and the CLI disagree and
# the CLI wins. `configs_base.py` declares `eta_schedule = piecewise_65, 1.0 -> 2.5`, but
# `pxdesign/runner/cli.py common_run_options` defaults `--eta_type const --eta_min 2.5
# --eta_max 2.5` and `pxdesign/utils/infer.py ALIASES` remaps those three flags straight onto
# `sample_diffusion.eta_schedule`. Both `pxdesign infer` and `pxdesign pipeline` go through
# that wrapper on every invocation, and neither preset overrides it, so piecewise_65 never
# reaches a real run. Verified against the upstream reference on the PD-L1 anchor
# (`scripts/pxdesign_port/upstream_ref.py`): const 2.5 reproduces the conditioned target to
# 0.62 / 0.63 A, piecewise_65 to 0.69 / 8.34 A. Pass the `eta_schedule` argument to run the
# config's declared schedule instead.
DESIGN_N_STEP = 400
DESIGN_ETA_SCHEDULE = {"type": "const", "min": 2.5, "max": 2.5}
DESIGN_ETA_SCHEDULE_CONFIG = {"type": "piecewise_65", "min": 1.0, "max": 2.5}
# Churn. These have no CLI flag, so `configs_base.py` is the last word on them and they
# differ from Protenix-v2's 0.8 / 1.0. gamma_min 0.01 keeps the noise re-injection on for
# effectively the whole trajectory instead of stopping once sigma drops below 1.0.
DESIGN_GAMMA0 = 1.0
DESIGN_GAMMA_MIN = 0.01

_EMBEDDER = "design_condition_embedder.condition_template_embedder.embedder.weight"
_INPUT_MAP = "design_condition_embedder.input_embedder.input_map."


class ProtenixDesign(Protenix):
    """PXDesign-d binder generator on Tenstorrent (inference-only).

    design(feats) -> atom coords (n_sample, N_atom, 3). `feats` is a Protenix-style
    input_feature_dict plus the design keys `conditional_templ`, `conditional_templ_mask`,
    `hotspot` and a 36-way `restype`.
    """

    C_S = 384          # s_trunk width; zeros, but the conditioning LN expects the channels

    def __init__(self, model_state_dict, compute_kernel_config, device=None,
                 diffusion_fp32=None):
        # Deliberately not Protenix.__init__: there is no trunk and no confidence head in
        # this checkpoint, and building either from an empty state dict would silently make
        # a wrong module rather than fail.

        import tt_bio.tenstorrent as _TT
        from ..tenstorrent import get_device
        self._w = model_state_dict
        self.compute_kernel_config = compute_kernel_config
        self.dev = device or get_device()
        resolved_fp32 = (env_flag("PROTENIX_DIFFUSION_FP32_DEVICE", True)
                         if diffusion_fp32 is None else diffusion_fp32)
        self._fast = _TT._FAST_MODE
        _TT.set_fast_mode(False)     # --fast is a trunk lever, and there is no trunk here

        def under(pfx):
            return {k[len(pfx):]: v for k, v in self._w.items() if k.startswith(pfx)}

        self.input_aae = AtomAttentionEncoder(
            under("design_condition_embedder.input_embedder.atom_attention_encoder."),
            compute_kernel_config)
        diffusion_dtype = ttnn.float32 if resolved_fp32 else ttnn.bfloat16
        self.diff_feat = AtomFeaturization(under("diffusion_module.atom_attention_encoder."),
                                           compute_kernel_config, dtype=diffusion_dtype)
        self.diffusion = DiffusionModule(under("diffusion_module."), self.dev,
                                         compute_kernel_config, diffusion_fp32=resolved_fp32)
        self.templ_embed = self._w[_EMBEDDER]                        # (65, c_z)
        self.C_Z = int(self.templ_embed.shape[-1])
        self.trunk = None
        self.confidence_head = None

    @classmethod
    def load_from_checkpoint(cls, path, compute_kernel_config=None, device=None):
        """Load the pxdesign generator checkpoint (.pt). Untrusted weights, weights_only."""
        from ..tenstorrent import get_device
        dev = device or get_device()
        ckc = compute_kernel_config or ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
            packer_l1_acc=True)
        ck = torch.load(path, map_location="cpu", weights_only=True)
        return cls(cls.design_state_dict(ck, path), ckc, dev)

    @staticmethod
    def design_state_dict(checkpoint, path="<checkpoint>"):
        """Strip the `module.` prefix and refuse anything that is not the generator.
        Handed protenix-v2 this would otherwise build a design model with no conditioning
        embedding and no input_map, and fail much later with a shape error."""
        ck = checkpoint.get("model", checkpoint)
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
        if _EMBEDDER not in sd:
            raise ValueError(f"{path} is not a PXDesign generator checkpoint: no {_EMBEDDER}")
        return sd

    def _s_inputs(self, feats, fi, Mmat):
        """InputFeatureEmbedderDesign: the shared atom encoder, a design-specific per-token
        concat, then `input_map`.

        `add_feat1`/`add_feat2` are `one_hot(0, 4)` per token -- upstream overwrites whatever
        it was handed with zeros before the one-hot, so they are constant, not features.
        `plddt` and `hotspot` default to zeros when absent, same as upstream."""
        tt, N = self._tt, fi["N"]
        NT = fi["NT"]
        zeros = torch.zeros(NT, 1)
        one_hot0 = torch.zeros(NT, 4)
        one_hot0[:, 0] = 1.0

        def tok(name, width):
            v = feats.get(name)
            if v is None:
                return zeros if width == 1 else one_hot0
            return v.reshape(NT, width).float()

        return self.input_aae(
            tt(feats["ref_pos"]), tt(fi["ref_charge_asinh"]), tt(feats["ref_mask"].reshape(N, 1)),
            tt(fi["f_in"]), tt(fi["d"]), tt(fi["v"]), tt(fi["invd"]), fi["mt"], tt(Mmat),
            tt(feats["restype"].reshape(NT, -1).float()), tt(tok("plddt", 1)),
            tt(tok("hotspot", 1)), tt(one_hot0), tt(one_hot0))

    def _condition_z(self, feats, NT):
        """ConditionTemplateEmbedder: `embedder[mask * (1 + bin)]`, on host.

        A gather of NT*NT rows out of a 65-row table is 4 MB of table lookups at the anchor
        and no arithmetic at all, so it stays on host; the tensor is uploaded once, in the
        diffusion's own dtype, and every op after it is on device."""
        idx = condition_template_index(feats["conditional_templ"],
                                       feats["conditional_templ_mask"])
        if idx.shape != (NT, NT):
            raise ValueError(f"conditional_templ is {tuple(idx.shape)}, expected "
                             f"({NT}, {NT}) -- one row per token")
        hi = int(idx.max())
        if hi >= self.templ_embed.shape[0]:
            raise ValueError(f"condition index {hi} is outside the "
                             f"{self.templ_embed.shape[0]}-row conditioning embedding")
        return self.templ_embed[idx].to(torch.float32)               # (NT, NT, c_z)

    def _trunk_cond(self, feats, *, progress_fn=None, n_cycles=None):
        """Everything before the sampler. Same contract and same `cond` dict as
        `Protenix._trunk_cond`, with the trunk's two outputs replaced by the design's."""
        if n_cycles is not None:
            raise ValueError("PXDesign-d has no trunk, so there is nothing to recycle; "
                             "drop n_cycles")
        fi = self._atom_feat_inputs(feats)
        N, NT, nb, nq, nk = fi["N"], fi["NT"], fi["nb"], fi["nq"], fi["nk"]
        mt, S = fi["mt"], fi["S"]
        Mmat = (S.t() / (S.t().sum(-1, keepdim=True) + 1e-6))
        s_inputs_tt = self._s_inputs(feats, fi, Mmat)
        s_inputs_tt = ttnn.linear(
            s_inputs_tt, self._tt(self._w[_INPUT_MAP + "weight"].t().contiguous()),
            bias=self._tt(self._w[_INPUT_MAP + "bias"].reshape(1, -1)),
            compute_kernel_config=self.compute_kernel_config)
        s_inputs = self._to_host(s_inputs_tt)[:NT]
        s_trunk = torch.zeros(NT, self.C_S)

        dtt = self.diffusion._up
        mt_dev = dtt(mt.reshape(-1, 1).float())
        c_l = self._to_host(self.diff_feat.c_l(dtt(feats["ref_pos"]), dtt(fi["ref_charge_asinh"]),
                                               dtt(feats["ref_mask"].reshape(N, 1)),
                                               dtt(fi["f_in"])), (N, 128))
        p_lm = self._to_host(self.diff_feat.p_lm(dtt(fi["d"]), dtt(fi["v"]), dtt(fi["invd"]),
                                                 mt_dev), (nb, nq, nk, 16))
        relp = feats["relp"] if "relp" in feats else self._generate_relp(feats)
        z_tt = dtt(self._condition_z(feats, NT))
        pair_z = self._diffusion_pair_cond(z_tt, relp).reshape(NT, NT, self.C_Z)
        p_lm = p_lm + self._plm_z_term(pair_z, fi["a2t"], nb, nq, nk)
        cond = {"s_trunk": s_trunk, "s_inputs": s_inputs, "pair_z": pair_z, "c_l": c_l,
                "p_lm": p_lm, "S": S, "mask_trunked": mt.float()}
        if self.diffusion.device_dit:
            cond["dit_z"] = self.diffusion._dit_z_device(pair_z)
        else:
            cond["dit_biases"] = self.diffusion._dit_pair_biases(pair_z)
        return cond, dict(N=N, NT=NT, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=None)

    def design(self, feats, *, n_step=DESIGN_N_STEP, n_sample=1, seed=None,
               eta_schedule=None, progress_fn=None, max_parallel_samples=None):
        """Generate binder coordinates. Returns (n_sample, N_atom, 3) host fp32.

        Defaults are what an upstream run uses: 400 steps, eta constant 2.5, gamma0 1.0,
        gamma_min 0.01. The design_token_mask half of the output is the designed binder;
        the rest reproduces the conditioned target."""
        cond, aux = self._trunk_cond(feats, progress_fn=progress_fn)
        eta = DESIGN_ETA_SCHEDULE if eta_schedule is None else eta_schedule
        M = int(n_sample)
        sampler = dict(n_step=n_step, step_scale=eta, gamma0=DESIGN_GAMMA0,
                       gamma_min=DESIGN_GAMMA_MIN, progress_fn=progress_fn)
        if M > 1 and getattr(self.diffusion, "supports_multiplicity", False):
            return edm_sample(self.diffusion, cond, aux["N"], multiplicity=M,
                              max_parallel_samples=max_parallel_samples or M, seed=seed,
                              **sampler)
        return torch.stack([edm_sample(self.diffusion, cond, aux["N"],
                                       seed=None if seed is None else seed + k, **sampler)[0]
                            for k in range(M)], 0)

    def fold(self, *args, **kwargs):
        raise NotImplementedError(
            "PXDesign-d generates binders rather than folding a sequence, and it has no "
            "confidence head; call design() and score with the Protenix filter stage")
