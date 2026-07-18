# La-Proteina FeatureFactory / PairReprBuilder -- ttnn port (pass 5).
#
# SPDX-License-Identifier: Apache-2.0
#
# Ports the dataset feature-pipeline that rebuilds seqs / c / pair_rep from
# x_t / t / mask (and optional x_sc) at each sampler step. Reference:
# proteinfoundation/nn/feature_factory.py (FeatureFactory + the feature creators
# used by the 160M uncond denoiser) and proteinfoundation/nn/modules/
# pair_rep_initial.py (PairReprBuilder + AdaptiveLayerNorm), Apache-2.0,
# vendored under _vendor/la-proteina-ref.
#
# A FeatureFactory = (list of deterministic feature creators) -> concat ->
# Linear(sum_dims, dim_out, bias=False) -> optional LayerNorm. PairReprBuilder
# adds a pair cond_factory + AdaptiveLayerNorm. The deterministic creators
# (time embeddings, relative sequence separation, pairwise-distance binning)
# are computed on host (tiny, no learned params) and moved to device; the
# linear_out + LayerNorm + AdaLN (the learned-param path) run on device.
# linear_out over a concat is implemented as a SUM of per-feature matmuls
# (split the weight by feature dim) -- same math, no concat op.
#
# Scope: the 160M uncond denoiser feature set (feats_seq / feats_cond_seq /
# feats_pair_repr / feats_pair_cond from configs/nn/
# local_latents_score_nn_160M.yaml). Motif / fold / residue-type / atom37
# features are NOT wired (they are no-ops / zeros for uncond sampling); the
# FeatureFactory is built from an explicit feature list so adding them later
# is local. The pair-distance one-hot binning is the one host round-trip of
# x_t / x_sc per step (perf follow-on: on-device binning).

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import ttnn


_CORE_GRID = ttnn.CoreGrid(y=8, x=8)


# ---------------------------------------------------------------------------
# host-side deterministic feature transforms (mirror the vendored reference
# exactly so the device linear/LN/adaln parity is against bit-identical inputs)
# ---------------------------------------------------------------------------


def get_time_embedding(t: torch.Tensor, edim: int, max_positions: int = 2000) -> torch.Tensor:
    assert len(t.shape) == 1
    t = t * max_positions
    half_dim = edim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
    emb = t.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if edim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1), mode="constant")
    return emb  # [B, edim]


def _bin_and_one_hot(tensor: torch.Tensor, bin_limits: torch.Tensor) -> torch.Tensor:
    bin_indices = torch.bucketize(tensor, bin_limits)
    return torch.nn.functional.one_hot(bin_indices, len(bin_limits) + 1) * 1.0


def _bin_pairwise_distances(x: torch.Tensor, min_dist, max_dist, dim) -> torch.Tensor:
    pair_dists = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)  # [B, N, N]
    bin_limits = torch.linspace(min_dist, max_dist, dim - 1, device=x.device)
    return _bin_and_one_hot(pair_dists, bin_limits)  # [B, N, N, dim]


def _rel_seq_sep(n: int, dim: int, device) -> torch.Tensor:
    inds = torch.Tensor([[i + 1 for i in range(n)] for _ in range(1)]).to(device)  # [1, n]
    seq_sep = inds[:, :, None] - inds[:, None, :]  # [1, n, n]
    low = -(dim / 2.0 - 1)
    high = dim / 2.0 - 1
    bin_limits = torch.linspace(low, high, dim - 1, device=device)
    return _bin_and_one_hot(seq_sep, bin_limits)  # [1, n, n, dim]


# ---------------------------------------------------------------------------
# device helpers
# ---------------------------------------------------------------------------


def _tt(t: torch.Tensor, device, dtype, transform=lambda x: x) -> "ttnn.Tensor":
    return ttnn.from_torch(transform(t), layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)


def _to_host(t: "ttnn.Tensor") -> torch.Tensor:
    return ttnn.to_torch(t).float()


def _pad_tile(host_t: torch.Tensor, tile_in: int) -> torch.Tensor:
    # Pad the last dim to tile_in (a multiple of 32) so the device tensor's
    # logical last dim matches the tile-padded weight's in-dim. F.pad takes the
    # pad tuple last-dim-first, so the (0, tile_in-d) pair leads.
    d = host_t.shape[-1]
    if d == tile_in:
        return host_t
    pad = [0, tile_in - d] + [0] * (2 * (host_t.dim() - 1))
    return torch.nn.functional.pad(host_t, pad, mode="constant", value=0.0)


class _AdaLNPair:
    """ttnn port of AdaptiveLayerNorm (pair mode): cond is [B, N, N, dim_cond]."""

    def __init__(self, device, ck, state_dict: dict, dtype):
        self.device = device
        self.ck = ck
        self.dtype = dtype
        self.norm_cond_w = _tt(state_dict["norm_cond.weight"], device, dtype)
        self.norm_cond_b = _tt(state_dict["norm_cond.bias"], device, dtype)
        self.gamma_w = _tt(state_dict["to_gamma.0.weight"], device, dtype, lambda x: x.t())
        self.gamma_b = _tt(state_dict["to_gamma.0.bias"], device, dtype)
        self.beta_w = _tt(state_dict["to_beta.weight"], device, dtype, lambda x: x.t())

    def __call__(self, x, cond, pair_mask):
        normed = ttnn.layer_norm(x, epsilon=1e-5, compute_kernel_config=self.ck)
        nc = ttnn.layer_norm(cond, weight=self.norm_cond_w, bias=self.norm_cond_b,
                             epsilon=1e-5, compute_kernel_config=self.ck)
        gamma = ttnn.linear(nc, self.gamma_w, bias=self.gamma_b,
                            compute_kernel_config=self.ck, dtype=self.dtype, core_grid=_CORE_GRID)
        gamma = ttnn.sigmoid(gamma)
        beta = ttnn.linear(nc, self.beta_w, compute_kernel_config=self.ck,
                           dtype=self.dtype, core_grid=_CORE_GRID)
        out = ttnn.multiply(normed, gamma)
        out = ttnn.add(out, beta)
        out = ttnn.multiply(out, pair_mask)
        return out


# ---------------------------------------------------------------------------
# feature registry: name -> (dim, kind, host_args)
#   kind in {"xt_bb_ca","xt_local_latents","x_sc_bb_ca","x_sc_local_latents",
#            "zeros_seq","time_emb_bb_ca","time_emb_local_latents",
#            "rel_seq_sep","xt_bb_ca_pair_dists","x_sc_bb_ca_pair_dists",
#            "zeros_pair"}
# dims that depend on config are passed via feat_dims.
# ---------------------------------------------------------------------------


def _feature_dim(name, feat_dims):
    if name == "xt_bb_ca": return 3
    if name == "xt_local_latents": return feat_dims["latent_dim"]
    if name == "x_sc_bb_ca": return 3
    if name == "x_sc_local_latents": return feat_dims["latent_dim"]
    if name == "optional_ca_coors_nm_seq_feat": return 3
    if name == "optional_res_type_seq_feat": return 20
    if name == "time_emb_bb_ca": return feat_dims["t_emb_dim"]
    if name == "time_emb_local_latents": return feat_dims["t_emb_dim"]
    if name == "rel_seq_sep": return feat_dims["seq_sep_dim"]
    if name == "xt_bb_ca_pair_dists": return feat_dims["xt_pair_dist_dim"]
    if name == "x_sc_bb_ca_pair_dists": return feat_dims["x_sc_pair_dist_dim"]
    if name == "optional_ca_pair_dist": return feat_dims["xt_pair_dist_dim"]
    raise KeyError(name)


class TTFeatureFactory:
    """ttnn port of FeatureFactory (seq or pair mode).

    linear_out over concat(features) is computed as the sum of per-feature
    matmuls (weight split by feature dim) -- same math, no concat. Optional
    LayerNorm after. Masking mirrors the reference: linear -> mask -> [LN] ->
    mask (mask-before-linear == mask-after-linear since linear has no bias).
    """

    def __init__(
        self,
        device,
        ck,
        state_dict: dict,
        mode: str,
        feats: List[str],
        dim_out: int,
        feat_dims: dict,
        use_ln_out: bool,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.ck = ck
        self.dtype = dtype
        self.mode = mode
        self.feats = list(feats)
        self.dim_out = dim_out
        self.feat_dims = feat_dims
        # split linear_out.weight [sum_dims, dim_out] by feature dim, padded to
        # the tile-aligned in-dim (ttnn TILE pads the feature's last dim to a
        # multiple of 32; the weight's in-dim must match). Extra lanes are 0 on
        # both sides so the matmul is unchanged.
        W = state_dict["linear_out.weight"]  # [sum_dims, dim_out]
        self.split_w = []
        self.feat_tile_in = []
        offset = 0
        for f in self.feats:
            d = _feature_dim(f, feat_dims)
            tile_in = ((d + 31) // 32) * 32
            sl = W[offset:offset + d].detach().clone().contiguous()
            if tile_in != d:
                pad = torch.zeros(tile_in - d, sl.shape[1], dtype=sl.dtype)
                sl = torch.cat([sl, pad], dim=0)
            self.split_w.append(_tt(sl, device, dtype))
            self.feat_tile_in.append(tile_in)
            offset += d
        assert offset == W.shape[0], (offset, W.shape)
        self.use_ln_out = use_ln_out
        if use_ln_out:
            self.ln_w = _tt(state_dict["ln_out.weight"], device, dtype)
            self.ln_b = _tt(state_dict["ln_out.bias"], device, dtype)
        else:
            self.ln_w = None
            self.ln_b = None

    def _feature_tensor(self, name, batch, mask_pair_tt, n, b):
        # returns a device tensor [B, N, tile_in] (seq) or [B, N, N, tile_in] (pair).
        # Pass-through features (xt/x_sc) are returned as-is (device tensors whose
        # logical last dim is already tile-aligned -- ensured by the sampler).
        # Host-computed features (time emb, rel seq sep, pair dists, zeros) are
        # padded to their tile-aligned last dim before moving to device.
        dev = self.device
        dt = self.dtype
        x_t = batch["x_t"]            # dict dm -> device tensor
        t = batch["t"]                # dict dm -> python float (scalar)
        x_sc = batch.get("x_sc", None)

        def tile_in_for(name):
            d = _feature_dim(name, self.feat_dims)
            return ((d + 31) // 32) * 32

        def host_to_dev(host_t, name):
            return _tt(_pad_tile(host_t, tile_in_for(name)), dev, dt)

        if name == "xt_bb_ca":
            return x_t["bb_ca"]
        if name == "xt_local_latents":
            return x_t["local_latents"]
        if name == "x_sc_bb_ca":
            if x_sc is not None:
                return x_sc["bb_ca"]
            return host_to_dev(torch.zeros(b, n, 3, dtype=torch.float32), name)
        if name == "x_sc_local_latents":
            d = self.feat_dims["latent_dim"]
            if x_sc is not None:
                return x_sc["local_latents"]
            return host_to_dev(torch.zeros(b, n, d, dtype=torch.float32), name)
        if name == "optional_ca_coors_nm_seq_feat":
            return host_to_dev(torch.zeros(b, n, 3, dtype=torch.float32), name)
        if name == "optional_res_type_seq_feat":
            return host_to_dev(torch.zeros(b, n, 20, dtype=torch.float32), name)
        if name in ("time_emb_bb_ca", "time_emb_local_latents"):
            dm = "bb_ca" if name.endswith("bb_ca") else "local_latents"
            edim = self.feat_dims["t_emb_dim"]
            te = get_time_embedding(torch.tensor([t[dm]], dtype=torch.float32), edim)  # [1, edim]
            te = te.to(torch.float32)
            if self.mode == "seq":
                te = te.expand(b, n, edim).contiguous()
            else:
                te = te.expand(b, n, n, edim).contiguous()
            return host_to_dev(te, name)
        if name == "rel_seq_sep":
            d = self.feat_dims["seq_sep_dim"]
            rs = _rel_seq_sep(n, d, torch.device("cpu")).to(torch.float32)  # [1, n, n, d]
            rs = rs.expand(b, n, n, d).contiguous()
            return host_to_dev(rs, name)
        if name == "xt_bb_ca_pair_dists":
            d = self.feat_dims["xt_pair_dist_dim"]
            xh = _to_host(x_t["bb_ca"])[..., :3]  # [B, N, 3]
            pd = _bin_pairwise_distances(xh, self.feat_dims["xt_pair_dist_min"],
                                        self.feat_dims["xt_pair_dist_max"], d)
            return host_to_dev(pd.to(torch.float32), name)
        if name == "x_sc_bb_ca_pair_dists":
            d = self.feat_dims["x_sc_pair_dist_dim"]
            if x_sc is not None:
                xh = _to_host(x_sc["bb_ca"])[..., :3]
                pd = _bin_pairwise_distances(xh, self.feat_dims["x_sc_pair_dist_min"],
                                            self.feat_dims["x_sc_pair_dist_max"], d)
            else:
                pd = torch.zeros(b, n, n, d, dtype=torch.float32)
            return host_to_dev(pd.to(torch.float32), name)
        if name == "optional_ca_pair_dist":
            d = self.feat_dims["xt_pair_dist_dim"]
            return host_to_dev(torch.zeros(b, n, n, d, dtype=torch.float32), name)
        raise KeyError(name)

    def __call__(self, batch, mask_tt, pair_mask_tt, b, n):
        out = None
        for i, name in enumerate(self.feats):
            f = self._feature_tensor(name, batch, pair_mask_tt, n, b)
            contrib = ttnn.linear(f, self.split_w[i], compute_kernel_config=self.ck,
                                  dtype=self.dtype, core_grid=_CORE_GRID)
            out = contrib if out is None else ttnn.add(out, contrib)
        # mask (seq: mask_tt [B,N,1]; pair: pair_mask_tt [B,N,N,1])
        m = mask_tt if self.mode == "seq" else pair_mask_tt
        out = ttnn.multiply(out, m)
        if self.use_ln_out:
            out = ttnn.layer_norm(out, weight=self.ln_w, bias=self.ln_b,
                                 epsilon=1e-5, compute_kernel_config=self.ck)
            out = ttnn.multiply(out, m)
        return out


class TTPairReprBuilder:
    """ttnn port of pair_rep_initial.PairReprBuilder.

    = pair-mode init_repr_factory (Linear + LayerNorm) + optional pair-mode
    cond_factory (Linear + LayerNorm) + AdaptiveLayerNorm(repr, cond, pair_mask).
    """

    def __init__(
        self,
        device,
        ck,
        state_dict: dict,
        feats_repr: List[str],
        feats_cond: List[str],
        dim_feats_out: int,
        dim_cond_pair: int,
        feat_dims: dict,
        dtype=ttnn.bfloat16,
    ):
        self.device = device
        self.ck = ck
        self.dtype = dtype
        self.init_repr_factory = TTFeatureFactory(
            device, ck, state_dict["init_repr_factory"], mode="pair",
            feats=feats_repr, dim_out=dim_feats_out, feat_dims=feat_dims,
            use_ln_out=True, dtype=dtype,
        )
        self.cond_factory = None
        if feats_cond is not None and len(feats_cond) > 0:
            self.cond_factory = TTFeatureFactory(
                device, ck, state_dict["cond_factory"], mode="pair",
                feats=feats_cond, dim_out=dim_cond_pair, feat_dims=feat_dims,
                use_ln_out=True, dtype=dtype,
            )
            self.adaln = _AdaLNPair(device, ck, state_dict["adaln"], dtype)

    def __call__(self, batch, mask_tt, pair_mask_tt, b, n):
        repr = self.init_repr_factory(batch, mask_tt, pair_mask_tt, b, n)
        if self.cond_factory is not None:
            cond = self.cond_factory(batch, mask_tt, pair_mask_tt, b, n)
            repr = self.adaln(repr, cond, pair_mask_tt)
        return repr


class TTLaProteinaDenoiser:
    """Full La-Proteina denoiser NN with the feature pipeline wired in.

    = seq cond_factory (time emb -> c_pre) + seq init_repr_factory (xt/xsc ->
    seqs) + PairReprBuilder (pair_rep) + TTLocalLatentsTransformer (trunk +
    heads). Takes a batch dict {x_t, t, mask, optional x_sc} (x_t/x_sc as device
    tensors, t as python-float scalars per data mode) and returns the nn_out
    dict {bb_ca: {v}, local_latents: {v}} -- the interface the flow-matching
    sampler loop consumes. This is the piece pass 4 bypassed by injecting at
    the post-builder interface; wiring it in is what unblocks the full nsteps
    sampler loop.
    """

    def __init__(
        self,
        device,
        ck,
        state_dict: dict,
        cfg: dict,
        feat_dims: dict,
        dtype=ttnn.bfloat16,
        factory_dtype=None,
    ):
        from tt_bio.la_proteina.denoiser import TTLocalLatentsTransformer
        self.device = device
        self.ck = ck
        self.dtype = dtype
        self.factory_dtype = factory_dtype if factory_dtype is not None else dtype
        self.cfg = cfg
        self.cond_factory = TTFeatureFactory(
            device, ck, state_dict["cond_factory"], mode="seq",
            feats=cfg["feats_cond_seq"], dim_out=cfg["dim_cond"],
            feat_dims=feat_dims, use_ln_out=False, dtype=self.factory_dtype,
        )
        self.init_repr_factory = TTFeatureFactory(
            device, ck, state_dict["init_repr_factory"], mode="seq",
            feats=cfg["feats_seq"], dim_out=cfg["token_dim"],
            feat_dims=feat_dims, use_ln_out=False, dtype=self.factory_dtype,
        )
        self.pair_repr_builder = TTPairReprBuilder(
            device, ck, state_dict["pair_repr_builder"],
            feats_repr=cfg["feats_pair_repr"], feats_cond=cfg["feats_pair_cond"],
            dim_feats_out=cfg["pair_repr_dim"], dim_cond_pair=cfg["dim_cond"],
            feat_dims=feat_dims, dtype=self.factory_dtype,
        )
        self.trunk = TTLocalLatentsTransformer(
            device, ck, state_dict, token_dim=cfg["token_dim"],
            pair_dim=cfg["pair_repr_dim"], nheads=cfg["nheads"],
            dim_cond=cfg["dim_cond"], latent_dim=cfg["latent_dim"],
            nlayers=cfg["nlayers"], use_qkln=cfg["use_qkln"],
            update_pair_repr=cfg["update_pair_repr"],
            update_pair_repr_every_n=cfg["update_pair_repr_every_n"],
            use_tri_mult=cfg["use_tri_mult"], dtype=dtype,
        )

    def __call__(self, batch, mask_tt, pair_mask_tt, pmb_tt, b, n):
        c_pre = self.cond_factory(batch, mask_tt, pair_mask_tt, b, n)
        seqs = self.init_repr_factory(batch, mask_tt, pair_mask_tt, b, n)
        pair_rep = self.pair_repr_builder(batch, mask_tt, pair_mask_tt, b, n)
        if self.factory_dtype != self.dtype:
            c_pre = ttnn.typecast(c_pre, self.dtype)
            seqs = ttnn.typecast(seqs, self.dtype)
            pair_rep = ttnn.typecast(pair_rep, self.dtype)
        local_latents_tt, ca_tt = self.trunk(
            seqs, pair_rep, c_pre, mask_tt, pmb_tt,
        )
        return {"bb_ca": {"v": ca_tt}, "local_latents": {"v": local_latents_tt}}
