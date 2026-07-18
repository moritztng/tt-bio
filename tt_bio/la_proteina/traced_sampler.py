# La-Proteina traced full-step sampler (pass 7 perf pass).
#
# Captures the FULL per-step device compute (cond/init/pair factories + trunk +
# heads + x_1_pred + Euler) as ONE ttnn trace per step, so NO eager ttnn compute
# op runs between replays -- only sanctioned copy_host_to_device_tensor /
# ttnn.copy into pre-allocated input buffers. Pass 6 traced the trunk in
# isolation (1.44x) but the eager Euler+x_1+factory linears running between
# replays corrupted the trace intermediate buffer pool (compounding drift
# step0 1.0 -> step3 -0.04). Folding ALL device compute into the traced region
# removes the interleaved-eager allocations that caused the aliasing.
#
# Per-step-varying host work (time embeddings, pair-distance bucketizing of
# x_t/x_sc, the shared eps draw) stays on host OUTSIDE the trace and is staged
# into the captured input buffers via copy_host_to_device_tensor -- the same
# sanctioned pattern tt_bio/tenstorrent.py uses for the BoltzGen diffusion
# trace. Pair-distance binning stays on host (parity-safe: device fp32 sqrt vs
# host flips the one-hot at bin boundaries -- a quantized feature -- per the
# pass-6 finding), so this is a pure perf pass, bit-identical to the eager loop.
#
# One trace per step (scalars t/dt/gt and (1-t) are baked into the captured
# op stream; they vary per step, so a single trace cannot cover all steps).
# Traces are captured once per (denoiser, nsteps) and reused across seeds.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import ttnn

from tt_bio.la_proteina.sampler import (
    TTEulerStep, _get_schedule, _get_gt, _draws_eps_for_step,
)
from tt_bio.la_proteina.feature_factory import (
    get_time_embedding, _bin_pairwise_distances, _pad_tile, _feature_dim,
)

_CORE_GRID = ttnn.CoreGrid(y=8, x=8)


def _apply_factory(factory, feats_device, mask_tt, pair_mask_tt, ck):
    out = None
    for i, f in enumerate(feats_device):
        contrib = ttnn.linear(f, factory.split_w[i], compute_kernel_config=ck,
                              dtype=factory.dtype, core_grid=_CORE_GRID)
        out = contrib if out is None else ttnn.add(out, contrib)
    m = mask_tt if factory.mode == "seq" else pair_mask_tt
    out = ttnn.multiply(out, m)
    if factory.use_ln_out:
        out = ttnn.layer_norm(out, weight=factory.ln_w, bias=factory.ln_b,
                             epsilon=1e-5, compute_kernel_config=ck)
        out = ttnn.multiply(out, m)
    return out


class TTLaProteinaTracedSampler:
    """Full nsteps sampler loop with the per-step device graph traced.

    Mirrors TTLaProteinaSampler bit-for-bit; only the dispatch path differs
    (captured trace replay vs eager). Inputs that vary per step are staged
    into persistent device buffers via copy_host_to_device_tensor; scalars
    (t/dt/gt, 1-t, euler params) are baked into a per-step trace.
    """

    def __init__(self, device, ck, denoiser, data_modes, sampling_model_args,
                 latent_dims, feat_dims, cfg, dtype=ttnn.bfloat16,
                 math_dtype=None, factory_dtype=None):
        self.device = device
        self.ck = ck
        self.dtype = dtype
        self.math_dtype = math_dtype if math_dtype is not None else dtype
        self.factory_dtype = factory_dtype if factory_dtype is not None else dtype
        self.denoiser = denoiser
        self.data_modes = list(data_modes)
        self.args = sampling_model_args
        self.latent_dims = latent_dims
        self.feat_dims = feat_dims
        self.cfg = cfg
        self.euler = TTEulerStep(device, ck, dtype=dtype, math_dtype=math_dtype)
        self._traces = None  # {nsteps: [per-step trace dicts]}

    def _tile(self, d):
        return ((d + 31) // 32) * 32

    def _host_tt(self, t_host):
        return ttnn.from_torch(t_host.float(), layout=ttnn.TILE_LAYOUT,
                               dtype=self.factory_dtype)

    def _device_body(self, bufs, consts, scal, mask_tt, pair_mask_tt, pmb_tt):
        den = self.denoiser
        # factories (factory_dtype), then typecast to trunk dtype
        c_pre = _apply_factory(den.cond_factory,
                               [bufs["time_emb_bb_ca_seq"], bufs["time_emb_local_latents_seq"]],
                               mask_tt, pair_mask_tt, self.ck)
        seqs = _apply_factory(den.init_repr_factory, [
            bufs["xt_bb_ca"], bufs["xt_local_latents"],
            bufs["x_sc_bb_ca"], bufs["x_sc_local_latents"],
            consts["ca_coors"], consts["res_type"],
        ], mask_tt, pair_mask_tt, self.ck)
        repr_ = _apply_factory(den.pair_repr_builder.init_repr_factory, [
            consts["rel_seq_sep"], bufs["xt_pair_dists"],
            bufs["x_sc_pair_dists"], consts["ca_pair_dist"],
        ], mask_tt, pair_mask_tt, self.ck)
        cond = _apply_factory(den.pair_repr_builder.cond_factory,
                              [bufs["time_emb_bb_ca_pair"], bufs["time_emb_local_latents_pair"]],
                              mask_tt, pair_mask_tt, self.ck)
        pair_rep = den.pair_repr_builder.adaln(repr_, cond, pair_mask_tt)
        if self.factory_dtype != self.dtype:
            c_pre = ttnn.typecast(c_pre, self.dtype)
            seqs = ttnn.typecast(seqs, self.dtype)
            pair_rep = ttnn.typecast(pair_rep, self.dtype)
        v_local_latents, v_bb_ca = den.trunk(seqs, pair_rep, c_pre, mask_tt, pmb_tt)
        v = {"bb_ca": v_bb_ca, "local_latents": v_local_latents}
        x = {"bb_ca": bufs["xt_bb_ca"], "local_latents": bufs["xt_local_latents"]}
        x1 = {}
        for dm in self.data_modes:
            x1[dm] = ttnn.multiply(
                ttnn.add(x[dm], ttnn.multiply(v[dm], 1.0 - scal["t"][dm])), mask_tt)
        x_next = {}
        for dm in self.data_modes:
            p = self.args[dm]["simulation_step_params"]
            x_next[dm] = self.euler(
                x[dm], v[dm], bufs["eps_" + dm], mask_tt,
                t=scal["t"][dm], dt=scal["dt"][dm], gt=scal["gt"][dm],
                sampling_mode=p["sampling_mode"],
                sc_scale_noise=p["sc_scale_noise"],
                sc_scale_score=p["sc_scale_score"],
                t_lim_ode=p["t_lim_ode"], t_lim_ode_below=p["t_lim_ode_below"],
                center_every_step=p["center_every_step"],
            )
        return x_next, x1

    def _feat_tile(self, name):
        return self._tile(_feature_dim(name, self.feat_dims))

    def _alloc_buf(self, shape):
        z = torch.zeros(shape, dtype=torch.float32)
        return ttnn.from_torch(z, layout=ttnn.TILE_LAYOUT, device=self.device,
                              dtype=self.dtype)

    def _alloc_feat_buf(self, name, pair=False):
        d = _feature_dim(name, self.feat_dims)
        tile = self._tile(d)
        b, n = 1, self._n
        shape = (b, n, n, tile) if pair else (b, n, tile)
        z = torch.zeros(shape, dtype=torch.float32)
        dt = self.factory_dtype
        return ttnn.from_torch(z, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=dt)

    def _get_consts(self):
        den = self.denoiser
        n, b = self._n, 1
        ff_seq = den.init_repr_factory
        ff_pair = den.pair_repr_builder.init_repr_factory
        return {
            "ca_coors": ff_seq._const_feature("optional_ca_coors_nm_seq_feat", n, b),
            "res_type": ff_seq._const_feature("optional_res_type_seq_feat", n, b),
            "rel_seq_sep": ff_pair._const_feature("rel_seq_sep", n, b),
            "ca_pair_dist": ff_pair._const_feature("optional_ca_pair_dist", n, b),
        }

    def _build_buffers(self):
        n = self._n
        bufs = {
            "xt_bb_ca": self._alloc_buf((1, n, self._tile(3))),
            "xt_local_latents": self._alloc_buf((1, n, self._tile(self.latent_dims["local_latents"]))),
            "x_sc_bb_ca": self._alloc_buf((1, n, self._tile(3))),
            "x_sc_local_latents": self._alloc_buf((1, n, self._tile(self.latent_dims["local_latents"]))),
            "time_emb_bb_ca_seq": self._alloc_feat_buf("time_emb_bb_ca"),
            "time_emb_local_latents_seq": self._alloc_feat_buf("time_emb_local_latents"),
            "time_emb_bb_ca_pair": self._alloc_feat_buf("time_emb_bb_ca", pair=True),
            "time_emb_local_latents_pair": self._alloc_feat_buf("time_emb_local_latents", pair=True),
            "xt_pair_dists": self._alloc_feat_buf("xt_bb_ca_pair_dists", pair=True),
            "x_sc_pair_dists": self._alloc_feat_buf("x_sc_bb_ca_pair_dists", pair=True),
            "eps_bb_ca": self._alloc_buf((1, n, self._tile(3))),
            "eps_local_latents": self._alloc_buf((1, n, self._tile(self.latent_dims["local_latents"]))),
        }
        return bufs

    def _device_body_wrap(self, bufs, consts, scal, mask_tt, pair_mask_tt, pmb_tt):
        # adapter so capture can pass the buffer dict directly
        return self._device_body(bufs, consts, scal, mask_tt, pair_mask_tt, pmb_tt)

    def _capture(self, nsteps, mask_tt, pair_mask_tt, pmb_tt):
        dev = self.device
        ts = {dm: _get_schedule(self.args[dm]["schedule"]["mode"], nsteps,
                                self.args[dm]["schedule"]["p"]) for dm in self.data_modes}
        gt = {dm: _get_gt(ts[dm][:-1], self.args[dm]["gt"]["mode"],
                          self.args[dm]["gt"]["p"],
                          self.args[dm]["gt"]["clamp_val"]) for dm in self.data_modes}
        bufs = self._build_buffers()
        consts = self._get_consts()
        # warmup (build const cache + prime program cache) with step-0 scalars
        scal0 = {"t": {dm: float(ts[dm][0]) for dm in self.data_modes},
                 "dt": {dm: float(ts[dm][1] - ts[dm][0]) for dm in self.data_modes},
                 "gt": {dm: float(gt[dm][0]) for dm in self.data_modes}}
        self._device_body(bufs, consts, scal0, mask_tt, pair_mask_tt, pmb_tt)
        ttnn.synchronize_device(dev)
        self._device_body(bufs, consts, scal0, mask_tt, pair_mask_tt, pmb_tt)
        ttnn.synchronize_device(dev)
        traces = []
        for k in range(nsteps):
            scal_k = {"t": {dm: float(ts[dm][k]) for dm in self.data_modes},
                      "dt": {dm: float(ts[dm][k + 1] - ts[dm][k]) for dm in self.data_modes},
                      "gt": {dm: float(gt[dm][k]) for dm in self.data_modes}}
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            out_next, out_x1 = self._device_body(bufs, consts, scal_k,
                                                 mask_tt, pair_mask_tt, pmb_tt)
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            ttnn.synchronize_device(dev)
            traces.append({"tid": tid, "out_next": out_next, "out_x1": out_x1,
                           "scalars": scal_k})
        self._traces = {"nsteps": nsteps, "bufs": bufs, "consts": consts,
                        "traces": traces, "ts": ts, "gt": gt}

    def _host_tiled(self, t_host, dtype):
        return ttnn.from_torch(t_host.float(), layout=ttnn.TILE_LAYOUT, dtype=dtype)

    def _stage(self, host_t, buf, dtype):
        ttnn.copy_host_to_device_tensor(self._host_tiled(host_t, dtype), buf)

    def __call__(self, x0, mask_tt, pair_mask_tt, pmb_tt, nsteps, n, self_cond=True):
        dev = self.device
        self._n = n
        if self._traces is None or self._traces["nsteps"] != nsteps:
            if self._traces is not None:
                self._release()
            self._capture(nsteps, mask_tt, pair_mask_tt, pmb_tt)
        tr = self._traces
        bufs = tr["bufs"]
        b = 1
        for dm in self.data_modes:
            self._stage(ttnn.to_torch(x0[dm]).float(), bufs["xt_" + dm], self.dtype)
            zsc = torch.zeros(b, n, self._tile(self.latent_dims[dm]))
            self._stage(zsc, bufs["x_sc_" + dm], self.dtype)
            self._stage(zsc, bufs["eps_" + dm], self.dtype)
        d = self.feat_dims["x_sc_pair_dist_dim"]
        self._stage(torch.zeros(b, n, n, self._tile(d)), bufs["x_sc_pair_dists"], self.factory_dtype)
        cur_xt = {dm: ttnn.to_torch(x0[dm]).float() for dm in self.data_modes}
        cur_xsc = {dm: torch.zeros(b, n, self._tile(self.latent_dims[dm])) for dm in self.data_modes}
        x_last = None
        for k in range(nsteps):
            scal = tr["traces"][k]["scalars"]
            self._build_and_stage_features(k, scal, bufs, cur_xt, cur_xsc, n, b)
            ttnn.execute_trace(dev, tr["traces"][k]["tid"], cq_id=0, blocking=True)
            cur_xt = {dm: ttnn.to_torch(tr["traces"][k]["out_next"][dm]).float() for dm in self.data_modes}
            cur_xsc = {dm: ttnn.to_torch(tr["traces"][k]["out_x1"][dm]).float() for dm in self.data_modes}
            x_last = tr["traces"][k]["out_next"]
        return x_last

    def _build_and_stage_features(self, k, scal, bufs, cur_xt, cur_xsc, n, b):
        for dm in self.data_modes:
            t = scal["t"][dm]
            edim = self.feat_dims["t_emb_dim"]
            te = get_time_embedding(torch.tensor([t], dtype=torch.float32), edim)
            te_seq = te.expand(b, n, edim).contiguous().to(torch.float32)
            te_pair = te.expand(b, n, n, edim).contiguous().to(torch.float32)
            name = "time_emb_bb_ca" if dm == "bb_ca" else "time_emb_local_latents"
            self._stage(_pad_tile(te_seq, self._tile(edim)), bufs[name + "_seq"], self.factory_dtype)
            self._stage(_pad_tile(te_pair, self._tile(edim)), bufs[name + "_pair"], self.factory_dtype)
        xh_t = cur_xt["bb_ca"][..., :3]
        pd_t = _bin_pairwise_distances(xh_t, self.feat_dims["xt_pair_dist_min"],
                                       self.feat_dims["xt_pair_dist_max"],
                                       self.feat_dims["xt_pair_dist_dim"])
        self._stage(_pad_tile(pd_t.to(torch.float32), self._tile(self.feat_dims["xt_pair_dist_dim"])),
                    bufs["xt_pair_dists"], self.factory_dtype)
        xh_sc = cur_xsc["bb_ca"][..., :3]
        if xh_sc.abs().sum() > 0:
            pd_sc = _bin_pairwise_distances(xh_sc, self.feat_dims["x_sc_pair_dist_min"],
                                            self.feat_dims["x_sc_pair_dist_max"],
                                            self.feat_dims["x_sc_pair_dist_dim"])
        else:
            pd_sc = torch.zeros(b, n, n, self.feat_dims["x_sc_pair_dist_dim"], dtype=torch.float32)
        self._stage(_pad_tile(pd_sc.to(torch.float32), self._tile(self.feat_dims["x_sc_pair_dist_dim"])),
                    bufs["x_sc_pair_dists"], self.factory_dtype)
        for dm in self.data_modes:
            self._stage(cur_xt[dm], bufs["xt_" + dm], self.dtype)
            self._stage(cur_xsc[dm], bufs["x_sc_" + dm], self.dtype)
        for dm in self.data_modes:
            p = self.args[dm]["simulation_step_params"]
            do_draw = _draws_eps_for_step(scal["t"][dm], p["sampling_mode"],
                                          p["t_lim_ode"], p["t_lim_ode_below"])
            d = self.latent_dims[dm]
            if do_draw:
                eps_host = torch.randn(b, n, d, dtype=torch.float32)
            else:
                eps_host = torch.zeros(b, n, d, dtype=torch.float32)
            self._stage(_pad_tile(eps_host, self._tile(d)), bufs["eps_" + dm], self.dtype)

    def _release(self):
        if self._traces is None:
            return
        for tr in self._traces["traces"]:
            try:
                ttnn.release_trace(self.device, tr["tid"])
            except Exception:
                pass
        self._traces = None
