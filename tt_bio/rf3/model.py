"""RF3 composed end to end on ttnn: trunk recycling, diffusion rollout, heads.

Every block underneath this file was scored against the upstream torch reference
individually before this composition existed. What is new here is only the wiring, and
the wiring has its own traps, all of which are visible in the reference and none of
which a component score can catch:

  - the MSA module is NOT residual-connected. Upstream adopted the Protenix report's
    bugfix, so `Z = msa_module(...)` replaces Z rather than adding to it. Writing the
    residual that every neighbouring line has is the natural mistake.
  - the recycled representations enter through `process_zh` / `process_sh`, a LayerNorm
    and a bias-free linear, added to `Z_init` / `S_init` -- not to the previous cycle's
    output.
  - the featurizer draws one i.i.d. MSA per recycle, so cycle `i` reads
    `msa_stack[i]`. Reusing one sample changes the answer without failing anything.
  - the diffusion module's EDM in/out scaling is around the whole module, and the
    output combines the noisy input with the update at t-dependent weights. The
    encoder sees the SCALED coordinates and the decoder's update is rescaled.

`c_s`, `c_z` and the block counts come off the checkpoint config rather than being
hard-coded, because they are what `load_reference` already reads.
"""

from __future__ import annotations

import torch
import ttnn

from tt_bio.envflags import env_flag
from tt_bio.rf3 import confidence as rf3_confidence
from tt_bio.rf3.confidence_head import ConfidenceHead
from tt_bio.rf3.diffusion_atom_decoder import DiffusionAtomDecoder
from tt_bio.rf3.diffusion_atom_encoder import DiffusionAtomEncoder
from tt_bio.rf3.diffusion_conditioning import DiffusionConditioning
from tt_bio.rf3.distogram_head import DistogramHead
from tt_bio.rf3.feature_init import mlff_constant_from_weights
from tt_bio.rf3.feature_initializer import FeatureInitializer
from tt_bio.rf3.host import HostInputs, distance_onehot
from tt_bio.rf3.msa import MSAModule
from tt_bio.rf3.remap import (PAIRFORMER_DIMS, PAIRFORMER_FLAGS, remap_msa_module,
                              remap_pairformer_stack, remap_template_embedder)
from tt_bio.rf3.sampler import DiffusionSampler, Draws
from tt_bio.rf3.template import TemplateEmbedder
from tt_bio.rf3.token_dit import TokenDiffusionTransformer
from tt_bio.tenstorrent import CORE_GRID_MAIN, Module, Pairformer, WeightScope


def _pairformer_stack(scope: WeightScope, n_blocks: int, cfg, prefix: str
                      ) -> Pairformer:
    """Remap ``n_blocks`` RF3 pairformer blocks onto tt-bio's shared stack."""
    return Pairformer(n_blocks, *PAIRFORMER_DIMS, True,
                      remap_pairformer_stack(scope.as_dict(), n_blocks, prefix),
                      cfg, **PAIRFORMER_FLAGS)


class _NormLinear(Module):
    """LayerNorm followed by a bias-free linear: `process_zh`, `process_sh`, `process_s`."""

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_w = self.torch_to_tt("0.weight", transform=lambda x: x)
        self.norm_b = self.torch_to_tt("0.bias", transform=lambda x: x)
        self.w = self.torch_to_tt("1.weight")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.layer_norm(x, weight=self.norm_w, bias=self.norm_b, epsilon=1e-5,
                            compute_kernel_config=self.compute_kernel_config)
        return ttnn.linear(x, self.w, compute_kernel_config=self.compute_kernel_config,
                           core_grid=CORE_GRID_MAIN)


class Recycler(Module):
    """One trunk pass: recycled Z and S in, updated Z and S out. Weights are shared
    across cycles, so this object is built once and called `n_recycles` times."""

    def __init__(self, state_dict, compute_kernel_config, *,
                 n_pairformer_blocks: int = 48, n_msa_blocks: int = 4):
        super().__init__(state_dict, compute_kernel_config)
        self.process_zh = _NormLinear(self.scope("process_zh"), compute_kernel_config)
        self.process_sh = _NormLinear(self.scope("process_sh"), compute_kernel_config)
        self.template_embedder = TemplateEmbedder(
            remap_template_embedder(self.scope("template_embedder").as_dict()),
            compute_kernel_config)
        self.msa_module = MSAModule(
            n_msa_blocks, remap_msa_module(self.scope("msa_module").as_dict()),
            compute_kernel_config)
        self.pairformer = _pairformer_stack(
            self.weights, n_pairformer_blocks, compute_kernel_config,
            "pairformer_stack.")

    def __call__(self, host: HostInputs, template_channels, msa, s_inputs,
                 s_init, z_init, s, z):
        z = ttnn.add(z_init, self.process_zh(z))
        z = ttnn.add_(z, self.template_embedder(z, template_channels))
        # NOT a residual: upstream takes the Protenix report's bugfix and replaces Z.
        z = self.msa_module(msa, z, s_inputs)
        s = ttnn.add(s_init, self.process_sh(s))
        return self.pairformer(s, z)


#: Hoist the t-independent half of the denoiser out of the diffusion rollout: `z_cond`, the
#: atom-pair track and both windowed bias stacks are the same arithmetic on the same inputs on
#: every step, so a 50-step rollout was rebuilding all of it 49 times. Bit-exact, verified two
#: ways (one denoiser call on fixed inputs, and the full 49-step rollout with the draws replayed)
#: at 128, 256, 512 and 1024 aa: max abs diff 0.0 everywhere. It keeps that state resident for
#: the whole rollout, so it was held off by default until the two large rungs were measured;
#: they are, it is 1.63x to 1.96x on the rollout across the ladder and never negative, and
#: 1024 aa (this port's historical residency wall) folds with room to spare. On by default;
#: set TT_BIO_RF3_HOIST_ROLLOUT=0 to get the per-step path back.
_HOIST_ROLLOUT = env_flag("TT_BIO_RF3_HOIST_ROLLOUT", True)

#: The token DiT's 24 pair biases are the same t- and batch-invariant function of `z_cond`
#: the hoist above already banks, and they were rebuilt on every denoiser call. Bit-exact
#: to build once; set TT_BIO_RF3_DIT_BIAS_HOIST=0 to rebuild them per call.
_HOIST_DIT_BIAS = env_flag("TT_BIO_RF3_DIT_BIAS_HOIST", True)


class DiffusionModule(Module):
    """The denoiser: conditioning, atom encoder, token DiT, atom decoder, EDM scaling."""

    def __init__(self, state_dict, compute_kernel_config, *,
                 sigma_data: float = 16.0, n_dit_blocks: int = 24):
        super().__init__(state_dict, compute_kernel_config)
        self.sigma_data = sigma_data
        self.conditioning = DiffusionConditioning(
            self.scope("diffusion_conditioning"), compute_kernel_config,
            sigma_data=sigma_data)
        enc = self.scope("atom_attention_encoder")
        self.encoder = DiffusionAtomEncoder(
            enc, compute_kernel_config, mlff_constant_from_weights(enc.as_dict()))
        self.process_s = _NormLinear(self.scope("process_s"), compute_kernel_config)
        self.transformer = TokenDiffusionTransformer(
            self.scope("diffusion_transformer"), compute_kernel_config,
            n_block=n_dit_blocks)
        self.ln1_w = self.torch_to_tt("layer_norm_1.weight", transform=lambda x: x)
        self.ln1_b = self.torch_to_tt("layer_norm_1.bias", transform=lambda x: x)
        self.decoder = DiffusionAtomDecoder(
            self.scope("atom_attention_decoder"), compute_kernel_config)

    def prepare(self, host: HostInputs, s_inputs, s_trunk, z_trunk) -> dict:
        """The part of a denoiser call that depends on neither `t` nor the batch index.

        `DiffusionConditioning.pair` takes no `t` at all, and inside the atom encoder and
        decoder the noisy coordinates reach the single track only, so `z_cond`, the whole
        atom-pair track and both stacks of windowed biases are the same arithmetic on the
        same inputs every time. A 50-step rollout at batch D rebuilt them 49 x D times.

        Returned to the caller rather than cached on the module, so the state cannot
        outlive the fold that owns it.
        """
        z_cond = self.conditioning.pair(host.relpos_feat, z_trunk)
        # The token DiT turns `z_cond` into 24 per-head biases with a layer_norm, a
        # projection and a permute each, and did it inside every denoiser call. Same
        # tensor in, same bytes out, 49 x D times per fold. Hoisting them is the same
        # arithmetic on the same operands -- bit-exact by construction -- and costs
        # n_block x n_heads x I^2 x 2 B of residency for the length of the rollout.
        bias = ([self.transformer.bias(z_cond, i)
                 for i in range(self.transformer.n_block)] if _HOIST_DIT_BIAS else None)
        enc = self.encoder.prepare(
            host.single_in, host.pair_in, host.pair_v, host.keys_indexing,
            host.window_mask, host.n_atom_padded, s_trunk, z_cond,
            host.atom_to_token, host.token_to_atom_win)
        return {"z_cond": z_cond, "dit_bias": bias, "enc": enc,
                "dec": self.decoder.prepare(enc["c"], enc["p"], host.window_mask,
                                            host.n_atom_padded)}

    def __call__(self, host: HostInputs, x_noisy: torch.Tensor, t: torch.Tensor,
                 s_inputs, s_trunk, z_trunk, prepared: dict | None = None
                 ) -> torch.Tensor:
        """`x_noisy [D, L, 3]`, `t [D]` on host; returns the denoised `[D, L, 3]`.

        D is the diffusion batch. The conditioning depends on t, which varies over D,
        so a batch is D independent denoiser calls, not one wider one.
        """
        out = []
        for d in range(x_noisy.shape[0]):
            t_d = t[d:d + 1]
            if prepared is None:
                s_cond, z_cond = self.conditioning(
                    host.relpos_feat, z_trunk, s_trunk, s_inputs, t_d)
            else:
                z_cond = prepared["z_cond"]
                s_cond = self.conditioning.single_at(s_trunk, s_inputs, t_d)
            # The conditioning's Fourier term carries the diffusion-batch axis, so its
            # single output comes back rank 4 as [1, D, I, c_s]. Every component harness
            # reshaped to the reference's shape before scoring, so none of them saw it;
            # composed, the extra axis reaches nlp_create_qkv_heads and the op rejects
            # the shape. One `t` per call here, so D is 1 and dropping it is exact.
            s_cond = ttnn.reshape(s_cond, (1, host.n_token, s_cond.shape[-1]))

            # EDM: scale to dimensionless coordinates of roughly unit variance
            scale = float((t_d.double() ** 2 + self.sigma_data ** 2).sqrt())
            r_in, ch_in = host.step_inputs(x_noisy[d:d + 1] / scale, self.device)

            a_i, q_skip, c_skip, p_skip = self.encoder(
                host.single_in, host.pair_in, host.pair_v, host.keys_indexing,
                host.atom_to_token_mean, host.window_mask, host.n_atom_padded,
                s_trunk, z_cond, r_in, ch_in, host.atom_to_token, host.token_to_atom_win,
                prepared=None if prepared is None else prepared["enc"])

            a_i = ttnn.add_(a_i, self.process_s(s_cond))
            a_i = self.transformer(a_i, s_cond, z_cond,
                                   bias=None if prepared is None else prepared["dit_bias"])
            a_i = ttnn.layer_norm(a_i, weight=self.ln1_w, bias=self.ln1_b,
                                  epsilon=1e-5,
                                  compute_kernel_config=self.compute_kernel_config)

            r_update = self.decoder(
                a_i, q_skip, c_skip, p_skip, host.atom_to_token,
                host.keys_indexing, host.window_mask, host.n_atom_padded,
                prepared=None if prepared is None else prepared["dec"])
            r_update = torch.Tensor(ttnn.to_torch(r_update)).float()[
                0, :host.n_atom].reshape(1, host.n_atom, 3)

            # ... and back out of the dimensionless frame
            s2 = self.sigma_data ** 2
            t2 = float(t_d.double() ** 2)
            out.append((s2 / (s2 + t2)) * x_noisy[d:d + 1]
                       + (self.sigma_data * float(t_d) / (s2 + t2) ** 0.5) * r_update)
        return torch.cat(out, dim=0)


class RF3(Module):
    """RF3 on Tenstorrent: `predict(f)` -> coordinates, distogram, confidence."""

    def __init__(self, state_dict, compute_kernel_config, *,
                 n_pairformer_blocks: int = 48, n_msa_blocks: int = 4,
                 n_dit_blocks: int = 24, n_confidence_layers: int = 4,
                 sigma_data: float = 16.0, num_timesteps: int = 200,
                 with_confidence: bool = True):
        super().__init__(state_dict, compute_kernel_config)
        fi = self.scope("feature_initializer")
        self.feature_initializer = FeatureInitializer(
            fi, compute_kernel_config,
            mlff_constant_from_weights(
                fi.child("input_feature_embedder.atom_attention_encoder").as_dict()))
        self.recycler = Recycler(self.scope("recycler"), compute_kernel_config,
                                 n_pairformer_blocks=n_pairformer_blocks,
                                 n_msa_blocks=n_msa_blocks)
        self.diffusion_module = DiffusionModule(
            self.scope("diffusion_module"), compute_kernel_config,
            sigma_data=sigma_data, n_dit_blocks=n_dit_blocks)
        self.distogram_head = DistogramHead(self.scope("distogram_head"),
                                            compute_kernel_config)
        self.confidence_head = (
            ConfidenceHead(self.scope("confidence_head"), compute_kernel_config,
                           n_layers=n_confidence_layers)
            if with_confidence else None)
        self.sampler = DiffusionSampler(num_timesteps=num_timesteps,
                                        sigma_data=sigma_data)

    def trunk(self, host: HostInputs, n_recycles: int, stop_after_first=None,
              progress_fn=None):
        """Feature init followed by `n_recycles` weight-shared trunk passes.

        `stop_after_first(s_inputs, s, z) -> bool` is consulted once, after the first
        recycle, and True abandons the rest of them. That is where upstream puts its
        early-stop decision (`RF3.py:412-439`): it pulls one value out of the recycling
        generator, scores it, and returns without resuming. A predicate that decides
        later would not save the recycles, which is most of what there is to save.
        """
        s_inputs, s_init, z_init = self.feature_initializer(
            host.single_in, host.pair_in, host.pair_v, host.keys_indexing,
            host.atom_to_token_mean, host.window_mask, host.n_atom_padded,
            host.token_feats, host.relpos_feat, host.bond_feat)

        s = ttnn.mul(s_init, 0.0)
        z = ttnn.mul(z_init, 0.0)
        # Recycle-invariant, so projected once rather than per cycle.
        template_channels = self.recycler.template_embedder.embed_template_feats(
            host.template_feats)
        for i in range(n_recycles):
            if progress_fn:
                progress_fn("trunk", step=i, total=n_recycles)
            # One i.i.d. MSA sample per recycle, as the featurizer drew them.
            s, z = self.recycler(host, template_channels,
                                 host.msa_stack[i % len(host.msa_stack)],
                                 s_inputs, s_init, z_init, s, z)
            if i == 0 and stop_after_first is not None \
                    and stop_after_first(s_inputs, s, z):
                break
        return s_inputs, s, z

    def predict(self, f: dict, *, n_recycles: int | None = None,
                diffusion_batch_size: int = 1, rep_atom_idxs: torch.Tensor | None = None,
                coord_to_be_noised: torch.Tensor | None = None,
                partial_t: int = 0, early_stop_plddt: float | None = None,
                is_real_atom: torch.Tensor | None = None,
                draws: Draws | None = None, progress_fn=None) -> dict:
        """One full inference: trunk recycling, diffusion rollout, then the heads.

        `draws` replays a recorded RNG stream, which is how this is scored against the
        reference: sharing draws is the only way an RMSD between two samplers means
        anything (a cross-RNG comparison produces a plausible structure and a number
        that reads as a porting bug).

        `partial_t` starts the diffusion rollout part-way down the noise schedule from
        `coord_to_be_noised` instead of from pure noise. `early_stop_plddt` abandons the
        target before the rollout when the confidence head, run on the trunk output with
        no structure, reports a mean pLDDT below the threshold; the return then carries
        `early_stopped` and no `X_L`.
        """
        host = HostInputs.build(f, self.device)
        if n_recycles is None:
            n_recycles = len(host.msa_stack)

        decision: dict = {}
        stop_after_first = None
        if early_stop_plddt is not None:
            if self.confidence_head is None or is_real_atom is None:
                raise ValueError(
                    "early_stop_plddt needs the confidence head and is_real_atom: the "
                    "decision is the head's mean pLDDT and is_real_atom is what selects "
                    "the real atoms out of the per-slot logits")

            def stop_after_first(s_inputs, s, z):
                decision["mean_plddt"] = self.mean_plddt_no_structure(
                    s_inputs, s, z, is_real_atom)
                return decision["mean_plddt"] < early_stop_plddt

        s_inputs, s, z = self.trunk(host, n_recycles, stop_after_first,
                                    progress_fn=progress_fn)
        if decision.get("mean_plddt") is not None \
                and decision["mean_plddt"] < early_stop_plddt:
            return {"early_stopped": True, "mean_plddt": decision["mean_plddt"],
                    "early_stop_plddt": float(early_stop_plddt)}
        distogram = torch.Tensor(ttnn.to_torch(self.distogram_head(z))).float()

        if coord_to_be_noised is None:
            coord_to_be_noised = torch.zeros(diffusion_batch_size, host.n_atom, 3)
        elif coord_to_be_noised.shape[0] != diffusion_batch_size:
            coord_to_be_noised = coord_to_be_noised[:1].expand(
                diffusion_batch_size, -1, -1).contiguous()

        prepared = (self.diffusion_module.prepare(host, s_inputs, s, z)
                    if _HOIST_ROLLOUT else None)

        def denoise(x_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.diffusion_module(host, x_noisy, t, s_inputs, s, z, prepared)

        x_pred, draws = self.sampler.sample(
            denoise, coord_to_be_noised, diffusion_batch_size, draws=draws,
            partial_t=partial_t, progress_fn=progress_fn)

        out = {"X_L": x_pred, "distogram": distogram, "draws": draws,
               "early_stopped": False}
        if decision.get("mean_plddt") is not None:
            out["mean_plddt"] = decision["mean_plddt"]
        if self.confidence_head is not None and rep_atom_idxs is not None:
            out.update(self.confidence(s_inputs, s, z, x_pred, rep_atom_idxs))
        return out

    def mean_plddt_no_structure(self, s_inputs, s, z,
                                is_real_atom: torch.Tensor) -> float:
        """Mean pLDDT from the confidence head with no structure fed to it.

        The head takes `dist_onehot=None`, which drops the predicted-distance connection
        exactly as the reference does when `X_pred_L is None`. No mini-rollout is needed,
        so this costs one head pass and nothing else.
        """
        logits = self.confidence_head(s_inputs, s, z)["plddt_logits"]
        logits = torch.Tensor(ttnn.to_torch(logits)).float()
        if logits.dim() == 3:
            logits = logits[0]
        return float(rf3_confidence.atomwise_plddt(logits, is_real_atom).mean())

    def confidence(self, s_inputs, s, z, x_pred: torch.Tensor,
                   rep_atom_idxs: torch.Tensor) -> dict:
        """The four confidence heads over the diffusion batch, stacked on dim 0.

        One structure per batch member and one head pass per member: the binned distances
        that drive the head are the member's own coordinates. The head is batch-1 by
        construction -- its Pairformer's triangle attention drops a leading singleton
        axis -- so a D-wide `x_pred` used to reach it as a D-wide `z` and fail a reshape
        volume check instead of quietly folding D into the token axis.
        """
        outs = []
        for d in range(x_pred.shape[0]):
            got = self.confidence_head(
                s_inputs, s, z,
                distance_onehot(x_pred[d:d + 1], rep_atom_idxs, self.device))
            outs.append({k: torch.Tensor(ttnn.to_torch(v)).float()
                         for k, v in got.items()})
        return {k: torch.cat([o[k] for o in outs], dim=0) for k in outs[0]}


def load(ckpt_path, compute_kernel_config, *, use_ema: bool = True, **kw) -> RF3:
    """Build the ttnn RF3 from an upstream checkpoint.

    `shadow.` is the EMA copy, which is what upstream inference uses; `model.` is the
    raw training copy and silently gives a worse network.
    """
    prefix = "shadow." if use_ema else "model."
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    w = {k[len(prefix):]: v.float() for k, v in ck["model"].items()
         if k.startswith(prefix)}
    if not w:
        raise KeyError(f"no weights under {prefix!r}")
    return RF3(w, compute_kernel_config, **kw)
