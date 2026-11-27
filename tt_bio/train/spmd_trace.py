"""Device-resident SPMD MeshTrace & host-prep offload for multi-chip Blackhole Galaxy RFD4 training.

Why this exists (diagnosed on anx-glx 8x4, 32 chips):
  1. ``FDMeshCommandQueue::write_program_cmds_to_subgrid`` loops serially over all ``world`` chips
     on every ``ttnn`` op (~0.20 ms/op x ~10,000 ops/step = ~2.0 s pure serial host dispatch).
     ``FDMeshCommandQueue::enqueue_trace`` collapses an entire phase to 1 trace issue per chip
     (measured 22.28x speedup on a 100-matmul chain on 8x4, 20.31 ms -> 0.91 ms).
  2. Single-process SPMD runs ``glue.conditioning(32*K)`` and ``glue.init_repr_factory(32*K)``
     for all 32 ranks on the host CPU (~564 ms/step at depth=2). Moving ``cond_factory.linear_out``,
     ``transition_c_1``, ``transition_c_2``, and ``init_repr_factory.linear_out`` onto the 32 chips
     in parallel in exact fp32 HiFi4 eliminates ~520 ms/step with zero numerical regression.

All trace captures share a single fenced ``TraceScratch`` DRAM hole (default 512 MiB/bank =
4 GiB per 32 GiB BH chip), and all folded weights / masters are refreshed in-place so buffer
addresses remain invariant across steps. Opt-out via ``--no-spmd-trace`` for eager A/B parity.
"""

from __future__ import annotations

import gc
import math
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import torch


def _safe_dealloc(t, *, keep=None) -> None:
    if t is None or t is keep:
        return
    import ttnn
    try:
        if t.is_allocated():
            ttnn.deallocate(t)
    except Exception:
        pass


def _pad_last_32(x: torch.Tensor) -> torch.Tensor:
    rem = int(x.shape[-1]) % 32
    if rem == 0:
        return x
    pad = 32 - rem
    return torch.nn.functional.pad(x, (0, pad))


def _put_replicated_f32(t: torch.Tensor, device):
    import ttnn
    from .mesh_dp import mesh_world

    kw: dict[str, Any] = {
        "dtype": ttnn.float32,
        "layout": ttnn.TILE_LAYOUT,
        "device": device,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
    }
    if mesh_world(device) > 1:
        kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    return ttnn.from_torch(t.detach().to(torch.float32).contiguous(), **kw)


def can_offload_cond_init(glue) -> bool:
    """Return True if ``glue`` matches the standard RFD4 cond/init_repr structure for device offload."""
    try:
        cf = glue.cond_factory
        ir = glue.init_repr_factory
        t1 = glue.transition_c_1
        t2 = glue.transition_c_2
        if getattr(cf, "ret_zero", False) or getattr(ir, "ret_zero", False):
            return False
        if not isinstance(getattr(cf, "ln_out", None), torch.nn.Identity):
            return False
        if not isinstance(getattr(ir, "ln_out", None), torch.nn.Identity):
            return False
        if cf.linear_out.bias is not None or ir.linear_out.bias is not None:
            return False
        if len(cf.feat_creators) < 2:
            return False
        if type(cf.feat_creators[-1]).__name__ != "RecycleRemainingBatchFeat":
            return False
        for tr in (t1, t2):
            if getattr(tr, "use_layer_norm", False):
                return False
            if tr.swish_linear[0].bias is not None or tr.linear_out.bias is not None:
                return False
        return True
    except Exception:
        return False


def _eval_factory_feats(factory, batch: dict, creators: Sequence[Any]) -> torch.Tensor:
    """Evaluate a subset of ``FeatureFactory.feat_creators`` and concatenate + padding-mask."""
    mask = batch["padding_mask"]
    n = int(mask.shape[-1])
    parts = []
    for fc in creators:
        feat = fc(batch)
        if feat.ndim == 2:
            feat = feat.reshape(feat.shape[0], 1, -1).expand(-1, n, -1)
        parts.append(feat)
    feats = torch.cat(parts, dim=-1)
    return factory.apply_padding_mask(feats, mask)


def cand_conds_and_s0_device(
    glue,
    rank_bts: Sequence[Mapping[str, Any]],
    mk_tok,
    step_depth: int,
    trunk_dtype,
    device,
    *,
    pool=None,
    dev_masters: Optional[Mapping[str, Any]] = None,
) -> Tuple[Any, list[Any]]:
    """Compute grad-off candidate ``(s0_tt, ck_tt)`` directly on ``device`` in exact fp32 HiFi4.

    Runs only the lightweight recycle-invariant feature extractors on CPU across ``pool``
    (zero full-batch ``torch.cat`` and zero CPU ``transition_c_1/2`` or ``linear_out`` matmuls),
    uploads compact features via parallel on-card tilization, and runs ``init_repr_factory.linear_out``,
    ``cond_factory.linear_out``, ``transition_c_1``, and ``transition_c_2`` across all ``world``
    chips in parallel.
    """
    import ttnn
    from .. import autograd as ag
    from .mesh_dp import put_sharded

    cf = glue.cond_factory
    ir = glue.init_repr_factory
    t1 = glue.transition_c_1
    t2 = glue.transition_c_2
    base_creators = list(cf.feat_creators[:-1])
    rem_creator = cf.feat_creators[-1]
    ir_creators = list(ir.feat_creators)

    def _extract_rank(b: Mapping[str, Any]):
        with torch.no_grad():
            cf_b = _eval_factory_feats(cf, b, base_creators).to(torch.float32)
            ir_b = _eval_factory_feats(ir, b, ir_creators).to(torch.float32)
            return cf_b, ir_b

    if pool is not None and len(rank_bts) > 1:
        extracted = list(pool.map(_extract_rank, rank_bts))
    else:
        extracted = [_extract_rank(b) for b in rank_bts]

    cf_base_h = torch.cat([x[0] for x in extracted], dim=0)  # [world*K, n, base_dim]
    ir_feat_h = torch.cat([x[1] for x in extracted], dim=0)  # [world*K, n, ir_dim]
    base_dim = int(cf_base_h.shape[-1])
    ir_dim = int(ir_feat_h.shape[-1])
    k_local = int(rank_bts[0]["padding_mask"].shape[0])
    n_tok = int(rank_bts[0]["padding_mask"].shape[1])

    cf_base_pad = _pad_last_32(cf_base_h)
    ir_feat_pad = _pad_last_32(ir_feat_h)

    W_cf = cf.linear_out.weight.detach().to(torch.float32)  # [256, 609]
    W_cf_base = _pad_last_32(W_cf[:, :base_dim])
    W_cf_rem = W_cf[:, base_dim:]  # [256, rem_dim]
    W_ir = _pad_last_32(ir.linear_out.weight.detach().to(torch.float32))  # [768, ir_pad]

    cfg = ag.precise_config()
    m_f32 = None
    m_f32_owned = False
    if mk_tok is not None:
        if mk_tok.dtype == ttnn.float32:
            m_f32 = mk_tok
        else:
            m_f32 = ttnn.typecast(mk_tok, ttnn.float32)
            m_f32_owned = True

    with ag.no_grad():
        # 1. s0_tt = (ir_feat @ W_ir^T) * mask
        ir_tt = put_sharded(ir_feat_pad, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        w_ir_tt = _put_replicated_f32(W_ir, device)
        s0_tt = ttnn.matmul(ir_tt, w_ir_tt, transpose_b=True, compute_kernel_config=cfg)
        ttnn.deallocate(ir_tt)
        ttnn.deallocate(w_ir_tt)
        if m_f32 is not None:
            s0_m = ttnn.multiply(s0_tt, m_f32)
            ttnn.deallocate(s0_tt)
            s0_tt = s0_m

        # 2. c_base = cf_base @ W_cf_base^T (recycle-invariant, computed once for all cycles)
        cf_tt = put_sharded(cf_base_pad, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        w_cf_tt = _put_replicated_f32(W_cf_base, device)
        c_base = ttnn.matmul(cf_tt, w_cf_tt, transpose_b=True, compute_kernel_config=cfg)
        ttnn.deallocate(cf_tt)
        ttnn.deallocate(w_cf_tt)

        # Transition weights: reuse device-resident replicated fp32 masters when available (zero upload).
        def _w_or_put(master_key: str, host_t: torch.Tensor):
            if dev_masters is not None and master_key in dev_masters:
                return dev_masters[master_key], False
            return _put_replicated_f32(host_t, device), True

        w_t1_1, own_t1_1 = _w_or_put("glue.transition_c_1.swish_linear.0.weight", t1.swish_linear[0].weight)
        w_t1_2, own_t1_2 = _w_or_put("glue.transition_c_1.linear_out.weight", t1.linear_out.weight)
        w_t2_1, own_t2_1 = _w_or_put("glue.transition_c_2.swish_linear.0.weight", t2.swish_linear[0].weight)
        w_t2_2, own_t2_2 = _w_or_put("glue.transition_c_2.linear_out.weight", t2.linear_out.weight)
        half = int(w_t1_1.shape[0]) // 2

        def _transition(x_in, w1, w2):
            h = ttnn.matmul(x_in, w1, transpose_b=True, compute_kernel_config=cfg)
            a = ttnn.slice(h, [0, 0, 0], [k_local, n_tok, half])
            g = ttnn.slice(h, [0, 0, half], [k_local, n_tok, 2 * half])
            ttnn.deallocate(h)
            sg = ttnn.silu(g)
            ttnn.deallocate(g)
            sw = ttnn.multiply(sg, a)
            ttnn.deallocate(sg)
            ttnn.deallocate(a)
            out = ttnn.matmul(sw, w2, transpose_b=True, compute_kernel_config=cfg)
            ttnn.deallocate(sw)
            if m_f32 is not None:
                om = ttnn.multiply(out, m_f32)
                ttnn.deallocate(out)
                out = om
            return out

        rem_seq = [None]
        for cyc in range(1, int(step_depth)):
            rem_seq.append(int(step_depth) - cyc - 1 if glue.condition_on_recycle_remaining else None)

        # Deduplicate unique rem values (e.g. if condition_on_recycle_remaining=False or rem >= dim-1)
        dummy_b = {
            "padding_mask": rank_bts[0]["padding_mask"][:1, :1],
            "x_t": {"bb_ca": rank_bts[0]["x_t"]["bb_ca"][:1, :1, :]},
        }
        cache_by_rem: dict[Optional[int], Any] = {}
        ck_tt: list[Any] = []
        for rem in rem_seq:
            key = None if rem is None else min(int(rem), int(rem_creator.dim) - 1)
            if key in cache_by_rem:
                ck_tt.append(ttnn.clone(cache_by_rem[key]))
                continue
            if key is None:
                c_in = ttnn.multiply(c_base, m_f32) if m_f32 is not None else ttnn.clone(c_base)
            else:
                with torch.no_grad():
                    rv = rem_creator({**dummy_b, "recycle_remaining": key}).to(torch.float32)  # [1, rem_dim]
                    bias_vec = (rv @ W_cf_rem.T).reshape(1, 1, -1)  # [1, 1, 256]
                bias_tt = _put_replicated_f32(bias_vec, device)
                c_sum = ttnn.add(c_base, bias_tt)
                ttnn.deallocate(bias_tt)
                if m_f32 is not None:
                    c_in = ttnn.multiply(c_sum, m_f32)
                    ttnn.deallocate(c_sum)
                else:
                    c_in = c_sum
            c1 = _transition(c_in, w_t1_1, w_t1_2)
            ttnn.deallocate(c_in)
            c2 = _transition(c1, w_t2_1, w_t2_2)
            ttnn.deallocate(c1)
            if trunk_dtype != ttnn.float32:
                c_out = ttnn.typecast(c2, trunk_dtype)
                ttnn.deallocate(c2)
            else:
                c_out = c2
            cache_by_rem[key] = c_out
            ck_tt.append(c_out)

        ttnn.deallocate(c_base)
        for wt, owned in ((w_t1_1, own_t1_1), (w_t1_2, own_t1_2), (w_t2_1, own_t2_1), (w_t2_2, own_t2_2)):
            if owned:
                ttnn.deallocate(wt)
        if m_f32_owned and m_f32 is not None:
            ttnn.deallocate(m_f32)

    return s0_tt, ck_tt


class SPMDMeshTracer:
    """Manages fenced ``MeshTrace`` captures across all static phases of a multi-chip SPMD step.

    All captures share a single ``TraceScratch`` DRAM hole so intermediates written during replay
    can never alias live eager tensors (or saved activations between paired ``win_fwd`` and
    ``win_bwd_fold``). Any capture refusal gracefully marks only that key eager.
    """

    def __init__(
        self,
        device,
        *,
        enabled: bool = True,
        hole_per_bank: int = 1600 << 20,
        trace_win_pair: bool = True,
        trace_muon_ns: bool = True,
    ):
        self.device = device
        self.enabled = bool(enabled)
        self.trace_win_pair = bool(trace_win_pair)
        self.trace_muon_ns = bool(trace_muon_ns)
        self.hole_per_bank = int(hole_per_bank)
        self._hole_ready = not self.trace_win_pair
        self._scratch = None
        self._tids: list[Any] = []
        self._owned_slots: list[Any] = []
        self._refold_state: Any = 0  # 0=unseen, "eager", or tid
        self._untaped_seen: dict[tuple, int] = {}
        self._untaped_traces: dict[tuple, Any] = {}  # key -> "eager" or dict(tid, ...)
        self._win_seen: dict[tuple, int] = {}
        self._win_traces: dict[tuple, Any] = {}  # key -> "eager" or dict(tid_fwd, tid_bwd, ...)
        self._ns_seen: dict[tuple, int] = {}
        self._ns_traces: dict[tuple, Any] = {}
        self.stats = {
            "refold_replay": 0,
            "untaped_replay": 0,
            "win_pair_replay": 0,
            "ns_replay": 0,
            "captures": 0,
            "refused": 0,
        }
        self.refusal_reasons: list[str] = []

    def _get_scratch(self):
        if self._scratch is None:
            from ..tenstorrent import TraceScratch
            self._scratch = TraceScratch(self.device)
        return self._scratch

    def _capture_runs(self, runs: Sequence[Callable[[], Any]], settle: Optional[Callable[[], None]] = None):
        sc = self._get_scratch()
        need = self.hole_per_bank // 2
        size = self.hole_per_bank
        tids = sc.capture(need, size, list(runs), settle=settle)
        if tids is None:
            raise RuntimeError(f"TraceScratch could not reserve {size >> 20} MiB/bank hole")
        self._tids.extend(tids)
        self.stats["captures"] += len(tids)
        return tids

    def _track(self, *tensors):
        for t in tensors:
            if t is not None:
                self._owned_slots.append(t)

    # ---- T1: In-place refold + _t1_w refresh ----
    def run_refold(self, dfolds, dev_masters: Mapping[str, Any], dev_hook: Optional[Callable] = None) -> None:
        import ttnn

        if not self.enabled or self._refold_state == "eager":
            dfolds.refold(dev_masters, on_refold=dev_hook, in_place=self.enabled)
            return

        if self._refold_state == 0:
            # Step 0 warm eager pass (populates lazy _t1_w and compiles every refold kernel).
            dfolds.refold(dev_masters, on_refold=dev_hook, in_place=True)
            self._refold_state = 1
            return

        if self._refold_state == 1:
            if not self._hole_ready:
                dfolds.refold(dev_masters, on_refold=dev_hook, in_place=True)
                return
            ttnn.synchronize_device(self.device)
            try:
                tids = self._capture_runs([lambda: dfolds.refold(dev_masters, on_refold=dev_hook, in_place=True)])
                self._refold_state = tids[0]
            except Exception as exc:
                self._refold_state = "eager"
                self.stats["refused"] += 1
                self.refusal_reasons.append(f"refold:{type(exc).__name__}:{exc}")
                dfolds.refold(dev_masters, on_refold=dev_hook, in_place=True)
                return
            ttnn.execute_trace(self.device, self._refold_state, cq_id=0, blocking=False)
            self.stats["refold_replay"] += 1
            return

        ttnn.execute_trace(self.device, self._refold_state, cq_id=0, blocking=False)
        self.stats["refold_replay"] += 1

    # ---- T2 & T3: Untaped trunk forward (cand_trunk & win_pre), factored per recycle ----
    def _run_untaped_pass(
        self,
        tag: str,
        trunk,
        s_init,
        z_init,
        c_one,
        masks: Sequence[Any],
        zd,
        low,
        prev=None,
    ):
        """Run one recycle cycle matching ``Trunk.__call__`` bit-for-bit under MeshTrace."""
        import ttnn
        from .. import tenstorrent as T

        has_prev = prev is not None
        key = (
            tag,
            "pN" if has_prev else "p0",
            tuple(s_init.shape), str(s_init.dtype),
            tuple(z_init.shape), str(z_init.dtype),
            tuple(c_one.shape), str(c_one.dtype),
            str(zd), str(low),
            tuple(m is not None for m in masks),
        )

        def _call(si, zi, ci, mk_a, pr_a):
            with T.lever_scope(trunk._levers):
                zb = None
                if pr_a is None:
                    s_in, z_in = ttnn.clone(si), ttnn.clone(zi)
                else:
                    s_in = trunk._recycle(si, pr_a[0], trunk.s_norm, trunk.s_proj, trunk.residual_dtype)
                    z_rec = trunk._recycle(zi, pr_a[1], trunk.z_norm, trunk.z_proj, zd, low=low)
                    if low is not None:
                        z_in, zb = z_rec
                    else:
                        z_in = z_rec
                return trunk.run_pass(s_in, z_in, ci, mk_a, zb=zb)

        tr = self._untaped_traces.get(key)
        if tr == "eager":
            return _call(s_init, z_init, c_one, masks, prev)

        if isinstance(tr, dict):
            ttnn.copy(s_init, tr["s_in"])
            ttnn.copy(z_init, tr["z_in"])
            ttnn.copy(c_one, tr["c_in"])
            if has_prev:
                ttnn.copy(prev[0], tr["prev_s"])
                ttnn.copy(prev[1], tr["prev_z"])
            for m, slot in zip(masks, tr["mk_in"]):
                if slot is not None:
                    ttnn.copy(m, slot)
            ttnn.execute_trace(self.device, tr["tid"], cq_id=0, blocking=False)
            self.stats["untaped_replay"] += 1
            return ttnn.clone(tr["s_out"]), ttnn.clone(tr["z_out"])

        seen = self._untaped_seen.get(key, 0)
        if seen == 0 or not self._hole_ready:
            self._untaped_seen[key] = 1
            return _call(s_init, z_init, c_one, masks, prev)

        # Call 2: allocate fixed slots outside the hole, run once to size/populate outputs, capture.
        s_in = ttnn.clone(s_init)
        z_in = ttnn.clone(z_init)
        c_in = ttnn.clone(c_one)
        prev_s = ttnn.clone(prev[0]) if has_prev else None
        prev_z = ttnn.clone(prev[1]) if has_prev else None
        prev_in = (prev_s, prev_z) if has_prev else None
        mk_in = [None if m is None else ttnn.clone(m) for m in masks]
        s_w, z_w = _call(s_in, z_in, c_in, mk_in, prev_in)
        s_out = ttnn.clone(s_w)
        z_out = ttnn.clone(z_w)
        # Warm ttnn.copy kernels eagerly for every output shape/dtype before entering trace capture.
        ttnn.copy(s_w, s_out)
        ttnn.copy(z_w, z_out)
        ttnn.synchronize_device(self.device)

        def _cap():
            sc, zc = _call(s_in, z_in, c_in, mk_in, prev_in)
            ttnn.copy(sc, s_out)
            ttnn.copy(zc, z_out)
            ttnn.deallocate(sc)
            ttnn.deallocate(zc)

        try:
            tids = self._capture_runs([_cap])
            self._track(s_in, z_in, c_in, prev_s, prev_z, *mk_in, s_out, z_out)
            self._untaped_traces[key] = dict(
                tid=tids[0], s_in=s_in, z_in=z_in, c_in=c_in,
                prev_s=prev_s, prev_z=prev_z, mk_in=mk_in,
                s_out=s_out, z_out=z_out,
            )
        except Exception as exc:
            for t in (s_in, z_in, c_in, prev_s, prev_z, *mk_in, s_out, z_out):
                _safe_dealloc(t)
            self._untaped_traces[key] = "eager"
            self.stats["refused"] += 1
            self.refusal_reasons.append(f"untaped:{tag}:{key[1]}:{type(exc).__name__}:{exc}")

        return s_w, z_w

    def run_untaped_trunk(
        self,
        tag: str,
        trunk,
        s,
        z,
        cs: Sequence[Any],
        masks: Sequence[Any],
        *,
        return_penultimate: bool = False,
    ):
        import ttnn
        from .. import tenstorrent as T
        from ..rfd4.trunk import _masked_copy

        if not self.enabled:
            if return_penultimate:
                return trunk(s, z, cs, masks, return_penultimate=True)
            return trunk(s, z, cs, masks)

        with T.lever_scope(trunk._levers):
            res_mask = masks[4]
            zd = trunk.pair_dtype(masks)
            res_pair_mask_c = masks[5] if zd == trunk.residual_dtype else masks[3]
            s_init = _masked_copy(s, res_mask, trunk.residual_dtype, alias=True)
            z_init = _masked_copy(z, res_pair_mask_c, zd, alias=True)
            low = trunk.dtype if zd == ttnn.float32 and trunk.dtype == ttnn.bfloat16 else None
            do_final_mask = res_mask is not None and T.lever("mask_at_mix")

        cur_s, cur_z = self._run_untaped_pass(
            tag, trunk, s_init, z_init, cs[0], masks, zd, low, prev=None,
        )
        penult = (None, None)
        last = len(cs) - 1
        for i in range(1, len(cs)):
            nxt_s, nxt_z = self._run_untaped_pass(
                tag, trunk, s_init, z_init, cs[i], masks, zd, low, prev=(cur_s, cur_z),
            )
            if return_penultimate and i == last:
                penult = (cur_s, cur_z)
            else:
                ttnn.deallocate(cur_s)
                ttnn.deallocate(cur_z)
            cur_s, cur_z = nxt_s, nxt_z

        if do_final_mask:
            with T.lever_scope(trunk._levers):
                ms = _masked_copy(cur_s, res_mask, trunk.residual_dtype)
                mz = _masked_copy(cur_z, res_pair_mask_c, zd)
                ttnn.deallocate(cur_s)
                ttnn.deallocate(cur_z)
                cur_s, cur_z = ms, mz
                if penult[0] is not None:
                    pm = masks[5] if penult[1].dtype == trunk.residual_dtype else masks[3]
                    ps = _masked_copy(penult[0], res_mask, penult[0].dtype)
                    pz = _masked_copy(penult[1], pm, penult[1].dtype)
                    ttnn.deallocate(penult[0])
                    ttnn.deallocate(penult[1])
                    penult = (ps, pz)

        for t, given in ((s_init, s), (z_init, z)):
            if t is not given:
                ttnn.deallocate(t)

        if return_penultimate:
            return cur_s, cur_z, penult
        return cur_s, cur_z

    # ---- T4: Paired win_last forward + (tape_bwd + dfolds.tape_backward) ----
    def win_pair_key(self, s_i_val, z_in_val, c_vals: Sequence[Any], m1: Sequence[Any], prev) -> tuple:
        return (
            prev is not None,
            tuple(m is not None for m in m1),
            tuple(s_i_val.shape), str(s_i_val.dtype),
            tuple(z_in_val.shape), str(z_in_val.dtype),
            len(c_vals),
        )

    def has_win_pair(self, key: tuple) -> bool:
        return self.enabled and self.trace_win_pair and isinstance(self._win_traces.get(key), dict)

    def win_pair_should_capture(self, key: tuple) -> bool:
        if not self.enabled or not self.trace_win_pair or self._win_traces.get(key) == "eager":
            return False
        if key in self._win_traces:
            return False
        seen = self._win_seen.get(key, 0)
        if seen == 0:
            self._win_seen[key] = 1
            return False
        return True

    def replay_win_pair_fwd(self, key: tuple, s_i_val, z_in_val, c_vals: Sequence[Any], m1: Sequence[Any], prev):
        import ttnn

        tr = self._win_traces[key]
        ttnn.copy(s_i_val, tr["s_i"])
        ttnn.copy(z_in_val, tr["z_in"])
        for cv, slot in zip(c_vals, tr["c_in"]):
            ttnn.copy(cv, slot)
        if prev is not None and tr["prev_in"] is not None:
            ttnn.copy(prev[0], tr["prev_in"][0])
            ttnn.copy(prev[1], tr["prev_in"][1])
        for mv, slot in zip(m1, tr["m1_in"]):
            if slot is not None:
                ttnn.copy(mv, slot)
        ttnn.execute_trace(self.device, tr["tid_fwd"], cq_id=0, blocking=False)
        return tr["s_out"]

    def replay_win_pair_bwd_fold(self, key: tuple, ds_out):
        import ttnn

        tr = self._win_traces[key]
        ttnn.copy(ds_out, tr["ds_out"])
        ttnn.deallocate(ds_out)
        ttnn.execute_trace(self.device, tr["tid_bwd"], cq_id=0, blocking=False)
        self.stats["win_pair_replay"] += 1
        return tr["d_s_i"], tr["d_z_in"], tr["d_c"], tr["dev_grads"], tr["idle"]

    def capture_win_pair(
        self,
        key: tuple,
        *,
        levers,
        winner,
        dfolds,
        dev_masters: Mapping[str, Any],
        s_i_val,
        z_in_val,
        c_vals: Sequence[Any],
        m1: Sequence[Any],
        prev,
        s_out_ref,
        ds_out_ref,
        d_s_i_ref,
        d_z_in_ref,
        d_c_refs: Sequence[Any],
        trunk_dev_grads_ref: Mapping[str, Any],
        idle_ref: Sequence[Any],
    ) -> bool:
        """Capture paired ``(tid_fwd, tid_bwd)`` sharing one fenced hole with live saved activations."""
        import ttnn
        from .. import autograd as ag
        from .. import tenstorrent as T

        if not self.enabled or key in self._win_traces:
            return False

        # Allocate persistent input/output slots OUTSIDE the TraceScratch hole.
        s_i_slot = ttnn.clone(s_i_val)
        z_in_slot = ttnn.clone(z_in_val)
        c_slots = [ttnn.clone(cv) for cv in c_vals]
        prev_slots = None if prev is None else (ttnn.clone(prev[0]), ttnn.clone(prev[1]))
        m1_slots = [None if mv is None else ttnn.clone(mv) for mv in m1]
        s_out_slot = ttnn.clone(s_out_ref)
        ds_out_slot = ttnn.clone(ds_out_ref)
        d_s_i_slot = ttnn.clone(d_s_i_ref)
        d_z_in_slot = ttnn.clone(d_z_in_ref)
        d_c_slots = [ttnn.clone(dc) for dc in d_c_refs]
        grad_names = sorted(trunk_dev_grads_ref.keys())
        dev_grad_slots = {nm: ttnn.clone(trunk_dev_grads_ref[nm]) for nm in grad_names}
        # Warm ttnn.copy kernels eagerly for every output shape/dtype before trace capture.
        ttnn.copy(s_out_ref, s_out_slot)
        ttnn.copy(d_s_i_ref, d_s_i_slot)
        ttnn.copy(d_z_in_ref, d_z_in_slot)
        for dc, dcs in zip(d_c_refs, d_c_slots):
            ttnn.copy(dc, dcs)
        for nm in grad_names:
            ttnn.copy(trunk_dev_grads_ref[nm], dev_grad_slots[nm])
        ttnn.synchronize_device(self.device)

        held: dict[str, Any] = {}

        def _run_fwd():
            with T.lever_scope(levers), ag.tape():
                s_leaf = ag.Tensor(s_i_slot, requires_grad=True)
                z_leaf = ag.Tensor(z_in_slot, requires_grad=True)
                c_leaves = [ag.Tensor(cs, requires_grad=True) for cs in c_slots]
                s_c, z_c = winner(s_leaf, z_leaf, c_leaves, m1_slots, prev=prev_slots)
            ttnn.copy(s_c.value, s_out_slot)
            held.update(
                s_c=s_c, z_c=z_c,
                s_leaf=s_leaf, z_leaf=z_leaf, c_leaves=c_leaves,
            )

        def _run_bwd_fold():
            s_c = held["s_c"]
            z_c = held["z_c"]
            s_leaf = held["s_leaf"]
            z_leaf = held["z_leaf"]
            c_leaves = held["c_leaves"]
            seed = ttnn.clone(ds_out_slot)
            ag.backward([s_c], [seed])
            ttnn.deallocate(seed)
            _safe_dealloc(getattr(s_c, "value", s_c))
            _safe_dealloc(getattr(z_c, "value", z_c))
            gs = getattr(s_leaf.grad, "value", s_leaf.grad)
            ttnn.copy(gs, d_s_i_slot)
            ttnn.deallocate(gs)
            s_leaf.grad = None
            gz = getattr(z_leaf.grad, "value", z_leaf.grad)
            ttnn.copy(gz, d_z_in_slot)
            ttnn.deallocate(gz)
            z_leaf.grad = None
            for cl, dcs in zip(c_leaves, d_c_slots):
                gc_ = getattr(cl.grad, "value", cl.grad)
                ttnn.copy(gc_, dcs)
                ttnn.deallocate(gc_)
                cl.grad = None
            dg_c, _idle_c = dfolds.tape_backward(dev_masters)
            for nm in grad_names:
                g = dg_c.pop(nm)
                ttnn.copy(g, dev_grad_slots[nm])
                ttnn.deallocate(g)
            for g in dg_c.values():
                _safe_dealloc(g)

        def _settle():
            tape_cm = held.pop("tape_cm", None)
            lev_cm = held.pop("lev_cm", None)
            held.clear()
            if tape_cm is not None:
                try:
                    tape_cm.__exit__(None, None, None)
                except Exception:
                    pass
            if lev_cm is not None:
                try:
                    lev_cm.__exit__(None, None, None)
                except Exception:
                    pass
            ag.release_pins()
            gc.collect()

        try:
            tids = self._capture_runs([_run_fwd, _run_bwd_fold], settle=_settle)
            self._hole_ready = True
            self._track(
                s_i_slot, z_in_slot, *c_slots, *(prev_slots or ()), *m1_slots,
                s_out_slot, ds_out_slot, d_s_i_slot, d_z_in_slot, *d_c_slots,
                *dev_grad_slots.values(),
            )
            self._win_traces[key] = dict(
                tid_fwd=tids[0],
                tid_bwd=tids[1],
                s_i=s_i_slot,
                z_in=z_in_slot,
                c_in=c_slots,
                prev_in=prev_slots,
                m1_in=m1_slots,
                s_out=s_out_slot,
                ds_out=ds_out_slot,
                d_s_i=d_s_i_slot,
                d_z_in=d_z_in_slot,
                d_c=d_c_slots,
                dev_grads=dev_grad_slots,
                idle=list(idle_ref),
            )
            return True
        except Exception as exc:
            _settle()
            for t in (
                s_i_slot, z_in_slot, *c_slots, *(prev_slots or ()), *m1_slots,
                s_out_slot, ds_out_slot, d_s_i_slot, d_z_in_slot, *d_c_slots,
                *dev_grad_slots.values(),
            ):
                _safe_dealloc(t)
            self._hole_ready = True
            self._win_traces[key] = "eager"
            self.stats["refused"] += 1
            self.refusal_reasons.append(f"win_pair:{type(exc).__name__}:{exc}")
            return False

    # ---- T5a: Traced batch of sharded Muon Newton-Schulz loops (zero dynamic scalars) ----
    def run_muon_ns_batch(
        self,
        key: tuple,
        u_list: Sequence[Any],
        run_fn: Callable[[Sequence[Any]], list[Any]],
    ) -> list[Any]:
        import ttnn

        if not self.enabled or not self.trace_muon_ns:
            return run_fn(u_list)

        tr = self._ns_traces.get(key)
        if tr == "eager":
            return run_fn(u_list)

        if isinstance(tr, dict):
            for u, slot in zip(u_list, tr["u_in"]):
                ttnn.copy(u, slot)
            ttnn.execute_trace(self.device, tr["tid"], cq_id=0, blocking=False)
            self.stats["ns_replay"] += 1
            return [ttnn.clone(o) for o in tr["o_out"]]

        seen = self._ns_seen.get(key, 0)
        if seen == 0 or not self._hole_ready:
            self._ns_seen[key] = 1
            return run_fn(u_list)

        u_in = [ttnn.clone(u) for u in u_list]
        o_warm = run_fn(u_in)
        o_out = [ttnn.clone(o) for o in o_warm]
        # Warm ttnn.copy kernels for every bucket shape/dtype before trace capture.
        for ow, oo in zip(o_warm, o_out):
            ttnn.copy(ow, oo)
        ttnn.synchronize_device(self.device)

        def _cap():
            os_c = run_fn(u_in)
            for oc, slot in zip(os_c, o_out):
                ttnn.copy(oc, slot)
                ttnn.deallocate(oc)

        try:
            tids = self._capture_runs([_cap])
            self._track(*u_in, *o_out)
            self._ns_traces[key] = dict(tid=tids[0], u_in=u_in, o_out=o_out)
        except Exception as exc:
            for t in (*u_in, *o_out):
                _safe_dealloc(t)
            self._ns_traces[key] = "eager"
            self.stats["refused"] += 1
            self.refusal_reasons.append(f"muon_ns:{type(exc).__name__}:{exc}")
        return o_warm

    def release(self) -> None:
        import ttnn

        for tid in self._tids:
            try:
                ttnn.release_trace(self.device, tid)
            except Exception:
                pass
        self._tids.clear()
        for t in self._owned_slots:
            _safe_dealloc(t)
        self._owned_slots.clear()
        if self._scratch is not None:
            try:
                self._scratch.release()
            except Exception:
                pass
            self._scratch = None
        self._refold_state = 0
        self._untaped_traces.clear()
        self._win_traces.clear()
        self._ns_traces.clear()
