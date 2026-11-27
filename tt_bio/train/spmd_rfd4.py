"""SPMD MeshDevice data-parallel orchestration for device-resident RFD4 training (D89 / D99).

Provides:
  * Exact B200 ``final-mat-nocg`` ``n_recycle`` depth-sequence loading (:func:`load_depths_jsonl`)
    and G4 whole-Galaxy per-dollar metrics (:func:`compute_spmd_g4_metrics`).
  * Per-rank batch selection & heterogeneous per-rank exploration winner gathering
    (:func:`select_rank_batches`, :func:`gather_winner_batches`).
  * :class:`SPMDZero1Step`: runs the Axis-0 K-sharded Muon ZeRO-1 Sandwich
    (``RS_2D -> AG_ax1 -> 1/rm Muon NS -> AG_ax0``) alongside ``DeviceCCLReducer`` for the
    remaining non-sharded parameters, keeping full replicated ``opt.master`` bit-identical across
    every chip in the Galaxy mesh.

Zero ``ttnn`` import at module load (preserves ``tests/test_training_opt_in.py``).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

from .ccl import (
    DEFAULT_FULL_MESH,
    DeviceCCLReducer,
    SPMDMuonPlan,
    SPMDMuonSharder,
    plan_spmd_muon_shards,
)
from .mesh_dp import (
    assert_mesh_consensus,
    mesh_shape,
    mesh_world,
    read_axis0_vectors,
    read_rep0,
    stack_batches,
)

B200_REF_EX_S_PER_GPU = 5.2440
DGX_B200_GPUS = 8
DGX_B200_USD = 562_154.0
BH_GALAXY_CHIPS = 32
BH_GALAXY_USD = 160_000.0

__all__ = [
    "B200_REF_EX_S_PER_GPU",
    "DGX_B200_GPUS",
    "DGX_B200_USD",
    "BH_GALAXY_CHIPS",
    "BH_GALAXY_USD",
    "parse_mesh_arg",
    "load_depths_jsonl",
    "select_rank_batches",
    "take_example",
    "gather_winner_batches",
    "compute_spmd_g4_metrics",
    "master_consensus_digest",
    "SPMDZero1Step",
]


def parse_mesh_arg(spec: str) -> Tuple[int, int]:
    """Parse ``'8x4'``, ``'4x4'``, ``'2x4'``, ``'1x4'``, or ``'1x1'`` into ``(rm, cm)``."""
    parts = spec.lower().strip().split("x")
    if len(parts) != 2:
        raise ValueError(f"mesh spec must be 'RxC' (e.g. '8x4'), got {spec!r}")
    rm, cm = int(parts[0]), int(parts[1])
    if rm <= 0 or cm <= 0:
        raise ValueError(f"mesh dimensions must be positive, got {(rm, cm)}")
    return (rm, cm)


def load_depths_jsonl(path: str | Path) -> List[int]:
    """Load per-step ``n_recycle`` depth sequence from a B200/TT ``steps.jsonl`` file."""
    p = Path(path)
    depths: List[int] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "n_recycle" in row:
                depths.append(int(row["n_recycle"]))
            elif "depth" in row:
                depths.append(int(row["depth"]))
            else:
                raise ValueError(f"row in {p} missing 'n_recycle'/'depth': {list(row)}")
    if not depths:
        raise ValueError(f"no depth entries found in {p}")
    return depths


def take_example(batch: Mapping[str, Any], idx: int) -> dict:
    """Slice candidate ``idx`` (preserving leading dim 1) from a single-rank ``[K, ...]`` batch."""
    import torch

    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, Mapping):
            out[k] = take_example(v, idx)
        elif torch.is_tensor(v):
            out[k] = v[idx:idx + 1]
        else:
            out[k] = v
    return out


def select_rank_batches(
    batch_pool: Sequence[Mapping[str, Any]],
    step_idx: int,
    world: int,
) -> Tuple[List[int], List[Mapping[str, Any]], dict]:
    """Select ``world`` batches for ``step_idx`` using the canonical ``(step*world + r) % len`` order."""
    if not batch_pool:
        raise ValueError("batch_pool must be non-empty")
    if world <= 0:
        raise ValueError(f"world must be > 0, got {world}")
    m = len(batch_pool)
    indices = [(step_idx * world + r) % m for r in range(world)]
    per_rank = [batch_pool[i] for i in indices]
    stacked = per_rank[0] if world == 1 else stack_batches(per_rank)
    return indices, per_rank, stacked


def gather_winner_batches(
    per_rank_batches: Sequence[Mapping[str, Any]],
    win_indices: Sequence[int],
) -> Tuple[List[dict], dict]:
    """Slice each rank's chosen exploration winner and stack into ``[world, ...]``."""
    if len(per_rank_batches) != len(win_indices):
        raise ValueError(
            f"per_rank_batches ({len(per_rank_batches)}) != win_indices ({len(win_indices)})"
        )
    per_rank_w = [take_example(bt, int(w)) for bt, w in zip(per_rank_batches, win_indices)]
    stacked = per_rank_w[0] if len(per_rank_w) == 1 else stack_batches(per_rank_w)
    return per_rank_w, stacked


def compute_spmd_g4_metrics(
    step_times_s: Sequence[float],
    depths: Sequence[int],
    world: int,
    *,
    warmup: int = 5,
    single_chip_step_s: Optional[float] = None,
    b200_ex_s_per_gpu: float = B200_REF_EX_S_PER_GPU,
    dgx_b200_gpus: int = DGX_B200_GPUS,
    dgx_b200_usd: float = DGX_B200_USD,
    bh_galaxy_chips: int = BH_GALAXY_CHIPS,
    bh_galaxy_usd: float = BH_GALAXY_USD,
) -> dict:
    """Compute exact G4 throughput, per-$ ratio vs DGX B200, and DP scaling efficiency."""
    if len(step_times_s) <= warmup:
        raise ValueError(f"need > {warmup} steps, got {len(step_times_s)}")
    timed_s = [float(x) for x in step_times_s[warmup:]]
    timed_d = [int(d) for d in depths[warmup:len(step_times_s)]]
    n = len(timed_s)
    total_s = sum(timed_s)
    mean_s = total_s / n
    sorted_s = sorted(timed_s)
    med_s = sorted_s[n // 2] if (n % 2 == 1) else 0.5 * (sorted_s[n // 2 - 1] + sorted_s[n // 2])
    mean_depth = sum(timed_d) / len(timed_d) if timed_d else float("nan")

    # Throughput in examples/s across all ``world`` chips (1 example/chip/step).
    ex_s_total = float(world) / mean_s
    ex_s_per_chip = 1.0 / mean_s
    # Per-chip $ basis: bh_galaxy_usd / bh_galaxy_chips; whole Galaxy at world=32 is identical.
    bh_usd_for_world = bh_galaxy_usd * (float(world) / float(bh_galaxy_chips))
    b200_ex_s_dgx = float(dgx_b200_gpus) * float(b200_ex_s_per_gpu)
    ratio_per_dollar = (ex_s_total / bh_usd_for_world) / (b200_ex_s_dgx / float(dgx_b200_usd))

    out = {
        "world": int(world),
        "n_timed": n,
        "warmup": int(warmup),
        "mean_depth": mean_depth,
        "mean_step_s": mean_s,
        "median_step_s": med_s,
        "min_step_s": min(timed_s),
        "max_step_s": max(timed_s),
        "ex_s_total": ex_s_total,
        "ex_s_per_chip": ex_s_per_chip,
        "ratio_per_dollar_vs_b200": ratio_per_dollar,
        "clears_g4_1p50x": ratio_per_dollar >= 1.50,
    }
    if single_chip_step_s is not None and single_chip_step_s > 0.0:
        out["single_chip_step_s"] = float(single_chip_step_s)
        out["dp_efficiency"] = float(single_chip_step_s) / mean_s
    return out


def master_consensus_digest(
    masters: Mapping[str, object],
    device,
    *,
    check_names: Optional[Collection[str]] = None,
) -> Tuple[str, int]:
    """Verify bit-identical consensus across all mesh ranks for ``check_names`` (or all) masters.

    Returns ``(combined_sha256_16, n_checked)``. Raises ``AssertionError`` if any chip diverges.
    """
    names = sorted(masters) if check_names is None else sorted(nm for nm in check_names if nm in masters)
    h = hashlib.sha256()
    for nm in names:
        d = assert_mesh_consensus(masters[nm], device, name=nm)
        h.update(nm.encode("utf-8"))
        h.update(d.encode("ascii"))
    return h.hexdigest()[:16], len(names)


class SPMDZero1Step:
    """Orchestrates Axis-0 K-sharded Muon ZeRO-1 + ``DeviceCCLReducer`` with ``DeviceOptimizer``.

    When ``use_zero1_muon=True`` and ``mesh_shape[0] > 1`` (e.g. ``8x4``, ``4x4``, ``2x4``):
      * The 10 large tile-aligned RFD4 Muon buckets (265 matrices, 99.66% of Muon FLOPs) keep
        their master shard, momentum buffer, and EMA shadow sharded ``1/rm`` along mesh axis 0
        via :class:`SPMDMuonSharder` (``RS_2D -> AG_ax1 -> 1/rm Muon NS -> AG_ax0``).
      * All remaining parameters (AdamW + tiny Muon buckets) reduce through :class:`DeviceCCLReducer`
        and step identically via ``base_opt`` (constructed with ``shard_mask = non_sharded_names``).
      * ``self.master`` always holds the complete, bit-identical replicated master dict on every
        chip after every step, so ``DeviceFolds.refold()`` and ``assert_mesh_consensus`` work
        unchanged.

    When ``use_zero1_muon=False`` or ``mesh_shape[0] == 1`` (e.g. ``1x4``):
      * Runs ``base_opt.step(dev_grads, reduce_fn=self.reducer, ...)`` directly with zero sharding.
    """

    def __init__(
        self,
        base_opt,
        initial_weights: Mapping[str, object],
        device,
        *,
        full_shape: Tuple[int, int] = DEFAULT_FULL_MESH,
        replicated_names: Collection[str] = (),
        use_zero1_muon: bool = True,
        num_links: int = 2,
        ag_ring: bool = True,
        force_linear: bool = False,
        scale: float = 1.0,
    ) -> None:
        self.base_opt = base_opt
        self.device = device
        self.mshape = mesh_shape(device)
        self.world = mesh_world(device)
        self.full_shape = tuple(int(v) for v in full_shape)
        self.replicated_names = frozenset(replicated_names)
        self.use_zero1_muon = bool(use_zero1_muon and self.mshape[0] > 1 and base_opt.muon is not None)

        self.muon_plan: Optional[SPMDMuonPlan] = None
        self.muon_sharder: Optional[SPMDMuonSharder] = None
        self.sh_master: Dict[Tuple[int, int], object] = {}
        self.sh_mom: Dict[Tuple[int, int], object] = {}
        self.sh_ema: Dict[Tuple[int, int], object] = {}

        if self.use_zero1_muon:
            self.muon_plan = plan_spmd_muon_shards(
                base_opt.shapes, base_opt.muon_names - self.replicated_names, self.mshape
            )
            if not self.muon_plan.sharded_names:
                self.use_zero1_muon = False

        if self.use_zero1_muon:
            assert self.muon_plan is not None
            self.muon_sharder = SPMDMuonSharder(
                self.muon_plan,
                full_shape=self.full_shape,
                num_links=num_links,
                ag_ring=ag_ring,
                force_linear=force_linear,
                scale=scale,
            )
            self.sh_master = self.muon_sharder.shard_initial_masters(initial_weights, device)
            # Detach sharded Muon buckets from base_opt so DRAM is not duplicated (~1.3 GB/chip
            # saved) and base_opt.host_master() / master_digest() / ema.update() never read stale
            # unsharded buckets.
            import ttnn

            sh_names = self.muon_plan.sharded_names
            kept_muon = []
            for bk in base_opt._muon_buckets:
                if all(nm in sh_names for nm in bk.names):
                    for attr in ("master_b", "state1_b", "state2_b", "ema_b", "_zero1"):
                        t = getattr(bk, attr, None)
                        if t is not None and t.is_allocated():
                            ttnn.deallocate(t)
                        setattr(bk, attr, None)
                else:
                    kept_muon.append(bk)
            base_opt._muon_buckets = kept_muon
            base_opt._all_buckets = base_opt._muon_buckets + base_opt._adam_buckets
            if base_opt.ema is not None and getattr(base_opt.ema, "_buckets", None) is not None:
                base_opt.ema._buckets = base_opt._all_buckets
            # Remaining non-sharded params reduce via DeviceCCLReducer (skipping sharded Muon + host glue).
            self.reducer = DeviceCCLReducer(
                self.mshape,
                full_shape=self.full_shape,
                num_links=num_links,
                ag_ring=ag_ring,
                force_linear=force_linear,
                scale=scale,
                replicated_names=self.replicated_names | self.muon_plan.sharded_names,
            )
        else:
            self.reducer = DeviceCCLReducer(
                self.mshape,
                full_shape=self.full_shape,
                num_links=num_links,
                ag_ring=ag_ring,
                force_linear=force_linear,
                scale=scale,
                replicated_names=self.replicated_names,
            )

    @property
    def master(self) -> Dict[str, object]:
        return self.base_opt.master

    def sharded_names(self) -> frozenset[str]:
        if self.use_zero1_muon and self.muon_plan is not None:
            return self.muon_plan.sharded_names
        return frozenset()

    def step(
        self,
        dev_grads: Mapping[str, object],
        *,
        disabled: Collection[str] = (),
        free_grads: bool = False,
        skip_reduce_for_broken_arm: bool = False,
        profile_sync: bool = False,
    ) -> dict:
        """Run one synchronized SPMD optimizer step on ``MeshDevice``."""
        import time as _time
        import ttnn

        if not self.use_zero1_muon:
            sync_s = 0.0
            if skip_reduce_for_broken_arm:
                rfn = None
            elif profile_sync:
                def rfn(g):
                    nonlocal sync_s
                    ttnn.synchronize_device(self.device)
                    _ts0 = _time.time()
                    out = self.reducer(g)
                    ttnn.synchronize_device(self.device)
                    sync_s += _time.time() - _ts0
                    return out
            else:
                rfn = self.reducer
            info = self.base_opt.step(
                dev_grads, reduce_fn=rfn, disabled=disabled, free_grads=free_grads
            )
            if profile_sync:
                info["sync_s"] = round(sync_s, 5)
            return info

        import numpy as np
        from .device_optim import (
            _clone_exact_f32,
            _concat_dim0_exact_f32,
            _safe_dealloc,
            _sum_sq_last2_exact_f32,
            newton_schulz_device,
            transpose_exact_f32,
        )

        assert self.muon_sharder is not None and self.muon_plan is not None
        opt = self.base_opt
        dis = set(disabled)
        sh_names = self.muon_plan.sharded_names
        ns_has_tall = "tall" in inspect.signature(newton_schulz_device).parameters
        sync_s = 0.0

        # 1. Phase A: RS_2D -> AG_ax1 for sharded Muon + exact 2D all-reduce for remaining params.
        if profile_sync:
            ttnn.synchronize_device(self.device)
            _ts0 = _time.time()
        if skip_reduce_for_broken_arm:
            # Deliberate broken arm for G5 verification: drop cross-rank reduction on sharded Muon.
            grad_shards = {}
            for wg in self.muon_plan.width_groups:
                for slot in wg.buckets:
                    pieces = []
                    owned = []
                    for j in range(slot.k_shard):
                        nm = slot.names[j] if j < slot.k else None
                        g = dev_grads.get(nm) if (nm and nm not in dis) else None
                        if g is not None:
                            p = ttnn.reshape(g, [1, slot.m, slot.cols])
                            pieces.append(p)
                            owned.append((p, g))
                        else:
                            z = ttnn.zeros(
                                [1, slot.m, slot.cols], dtype=ttnn.float32,
                                layout=ttnn.TILE_LAYOUT, device=self.device,
                            )
                            pieces.append(z)
                            owned.append((z, None))
                    grad_shards[slot.shape] = _concat_dim0_exact_f32(pieces)
                    for p, orig in owned:
                        _safe_dealloc(p, keep=orig)
            rem_grads = {nm: g for nm, g in dev_grads.items() if nm not in sh_names}
        else:
            grad_shards = self.muon_sharder.reduce_scatter_grads(
                dev_grads, self.device, disabled=dis
            )
            rem_in = {nm: g for nm, g in dev_grads.items() if nm not in sh_names}
            rem_grads = self.reducer(rem_in)
        if profile_sync:
            ttnn.synchronize_device(self.device)
            sync_s += _time.time() - _ts0

        # 2. Schedule & exact global norms + dead_grads across all 558 params (MeshDevice-safe).
        lr_now = opt.lr if opt.schedule is None else float(opt.schedule(opt.steps))
        lr_muon = (
            lr_now if (opt.muon is None or opt.muon.schedule is None)
            else float(opt.muon.schedule(opt.steps))
        )
        opt.steps += 1
        opt.beta1_pow *= opt.beta1
        opt.beta2_pow *= opt.beta2
        bc1 = 1.0 - opt.beta1_pow
        bc2 = 1.0 - opt.beta2_pow

        dead: set[str] = set()
        ordered_slots = [slot for wg in self.muon_plan.width_groups for slot in wg.buckets]
        sh_sq_slices = [_sum_sq_last2_exact_f32(grad_shards[slot.shape]) for slot in ordered_slots]
        if len(sh_sq_slices) == 1:
            sh_sq_joined = sh_sq_slices[0]
        else:
            sh_sq_joined = _concat_dim0_exact_f32(sh_sq_slices)
            for s in sh_sq_slices:
                ttnn.deallocate(s)
        row_vecs = read_axis0_vectors(sh_sq_joined, self.mshape, self.device)
        ttnn.deallocate(sh_sq_joined)

        rm = self.mshape[0]
        sh_sq = 0.0
        slot_off = 0
        for slot in ordered_slots:
            for r0 in range(rm):
                vec = row_vecs[r0]
                for j in range(slot.k_shard):
                    idx = r0 * slot.k_shard + j
                    if idx >= slot.k:
                        continue
                    nm = slot.names[idx]
                    if nm in dis or dev_grads.get(nm) is None:
                        dead.add(nm)
                    else:
                        p_sq = float(vec[slot_off + j])
                        if p_sq == 0.0:
                            dead.add(nm)
                        sh_sq += p_sq
            slot_off += slot.k_shard

        packed_rem_muon = []
        for bk in opt._muon_buckets:
            if any(nm in sh_names for nm in bk.names):
                continue
            g_b, flags = opt._pack_grads(bk, rem_grads, dis)
            packed_rem_muon.append((bk, g_b, flags))
        packed_adam = []
        for bk in opt._adam_buckets:
            g_b, flags = opt._pack_grads(bk, rem_grads, dis)
            packed_adam.append((bk, g_b, flags))

        rem_parts = []
        rem_meta = []
        for group_kind, packed in (("muon", packed_rem_muon), ("adam", packed_adam)):
            for bk, g_b, flags in packed:
                if g_b is None:
                    dead.update(bk.names)
                    continue
                rem_parts.append(_sum_sq_last2_exact_f32(g_b))
                rem_meta.append((group_kind, bk, flags))

        rem_muon_sq = 0.0
        adam_sq = 0.0
        if rem_parts:
            if len(rem_parts) == 1:
                rem_joined = rem_parts[0]
            else:
                rem_joined = _concat_dim0_exact_f32(rem_parts)
                for p in rem_parts:
                    ttnn.deallocate(p)
            rem_vals = read_rep0(rem_joined, self.device).float().numpy().reshape(-1)
            ttnn.deallocate(rem_joined)
            off = 0
            for group_kind, bk, flags in rem_meta:
                s0 = bk.canon[0]
                for i, (nm, ok) in enumerate(zip(bk.names, flags)):
                    sl = rem_vals[off + i * s0:off + (i + 1) * s0]
                    if not ok:
                        dead.add(nm)
                    else:
                        p_sq = float(np.sum(sl, dtype=np.float64))
                        if p_sq == 0.0:
                            dead.add(nm)
                        if group_kind == "muon":
                            rem_muon_sq += p_sq
                        else:
                            adam_sq += p_sq
                off += bk.k * s0

        muon_gnorm = math.sqrt(max(sh_sq + rem_muon_sq, 0.0))
        adam_gnorm = math.sqrt(max(adam_sq, 0.0))
        clip_muon = opt.clip_coef(muon_gnorm)
        clip_adam = opt.clip_coef(adam_gnorm)
        opt.last_grad_norm = adam_gnorm
        opt.last_grad_norm_muon = muon_gnorm
        opt.last_dead_grads = frozenset(dead)

        # 3. Phase B: 1/rm Muon step matching DeviceOptimizer._step_muon_bucket verbatim.
        mu = opt.muon
        one_minus_mu = float(1.0 - mu.momentum)
        masks = self.muon_sharder.active_masks_for_step(dev_grads, dis, self.device)
        for wg in self.muon_plan.width_groups:
            for slot in wg.buckets:
                shp = slot.shape
                g_sh = grad_shards[shp]
                g_eff = ttnn.multiply(g_sh, float(clip_muon)) if clip_muon != 1.0 else g_sh
                m_prev = self.sh_mom.get(shp)
                m_pair = masks[shp]
                if m_prev is None:
                    buf_new = ttnn.multiply(g_eff, one_minus_mu)
                else:
                    diff = ttnn.subtract(g_eff, m_prev)
                    scaled = ttnn.multiply(diff, one_minus_mu)
                    ttnn.deallocate(diff)
                    buf_cand = ttnn.add(m_prev, scaled)
                    ttnn.deallocate(scaled)
                    if m_pair is None:
                        ttnn.deallocate(m_prev)
                        buf_new = buf_cand
                    else:
                        ma, mi = m_pair
                        p1 = ttnn.multiply(buf_cand, ma)
                        p2 = ttnn.multiply(m_prev, mi)
                        ttnn.deallocate(buf_cand)
                        ttnn.deallocate(m_prev)
                        buf_new = ttnn.add(p1, p2)
                        ttnn.deallocate(p1)
                        ttnn.deallocate(p2)
                self.sh_mom[shp] = buf_new

                if mu.nesterov:
                    d2 = ttnn.subtract(buf_new, g_eff)
                    s2 = ttnn.multiply(d2, one_minus_mu)
                    ttnn.deallocate(d2)
                    u_b = ttnn.subtract(buf_new, s2)
                    ttnn.deallocate(s2)
                else:
                    u_b = _clone_exact_f32(buf_new)

                if g_eff is not g_sh:
                    ttnn.deallocate(g_eff)
                ttnn.deallocate(g_sh)

                flipped = slot.m > slot.cols
                tall_native = (
                    ns_has_tall
                    and flipped
                    and opt.ns_mode in ("split2_fast", "split2_hyb1", "split2_hyb2", "fp32")
                )
                if flipped and not tall_native:
                    u_ns = transpose_exact_f32(u_b)
                    ttnn.deallocate(u_b)
                else:
                    u_ns = u_b

                ns_kw = {"tall": tall_native} if ns_has_tall else {}
                o_ns = newton_schulz_device(
                    u_ns, steps=mu.ns_steps, coeffs=mu.ns_coefficients, eps=mu.eps,
                    mode=opt.ns_mode, **ns_kw,
                )
                ttnn.deallocate(u_ns)

                if flipped and not tall_native:
                    o_b = transpose_exact_f32(o_ns)
                    ttnn.deallocate(o_ns)
                else:
                    o_b = o_ns

                w_prev = self.sh_master[shp]
                decay_scale = float(1.0 - lr_muon * mu.weight_decay)
                step_scale = float(-lr_muon * mu.ratio(shp))
                t2 = ttnn.multiply(o_b, step_scale)
                ttnn.deallocate(o_b)
                if decay_scale == 1.0:
                    w_cand = ttnn.add(w_prev, t2)
                    ttnn.deallocate(t2)
                else:
                    t1 = ttnn.multiply(w_prev, decay_scale)
                    w_cand = ttnn.add(t1, t2)
                    ttnn.deallocate(t1)
                    ttnn.deallocate(t2)

                if m_pair is None:
                    ttnn.deallocate(w_prev)
                    w_new = w_cand
                else:
                    ma, mi = m_pair
                    p1 = ttnn.multiply(w_cand, ma)
                    p2 = ttnn.multiply(w_prev, mi)
                    ttnn.deallocate(w_cand)
                    ttnn.deallocate(w_prev)
                    w_new = ttnn.add(p1, p2)
                    ttnn.deallocate(p1)
                    ttnn.deallocate(p2)
                    ttnn.deallocate(ma)
                    ttnn.deallocate(mi)
                self.sh_master[shp] = w_new

                if opt.ema is not None:
                    if shp not in self.sh_ema:
                        self.sh_ema[shp] = _clone_exact_f32(w_new)
                    else:
                        e_prev = self.sh_ema[shp]
                        diff = ttnn.subtract(w_new, e_prev)
                        sc = ttnn.multiply(diff, float(1.0 - opt.ema.decay))
                        ttnn.deallocate(diff)
                        e_nxt = ttnn.add(e_prev, sc)
                        ttnn.deallocate(sc)
                        ttnn.deallocate(e_prev)
                        self.sh_ema[shp] = e_nxt

        # 4. Step remaining unsharded Muon + AdamW buckets identically on every chip.
        for bk, g_b, flags in packed_rem_muon:
            opt._step_muon_bucket(bk, g_b, flags, clip_muon, lr_muon)
            if g_b is not None and g_b.is_allocated():
                ttnn.deallocate(g_b)
        for bk, g_b, _flags in packed_adam:
            opt._step_adam_bucket(bk, g_b, dis, clip_adam, lr_now, bc1, bc2)
            if g_b is not None and g_b.is_allocated():
                ttnn.deallocate(g_b)
        if opt.ema is not None:
            opt.ema.update()

        # 5. Phase C: AG_axis0 -> bit-identical replicated masters for all sharded Muon params.
        if profile_sync:
            ttnn.synchronize_device(self.device)
            _ts1 = _time.time()
        gathered = self.muon_sharder.all_gather_masters(self.sh_master, self.device)
        if profile_sync:
            ttnn.synchronize_device(self.device)
            sync_s += _time.time() - _ts1
        for nm, new_m in gathered.items():
            old_m = opt.master.get(nm)
            opt.master[nm] = new_m
            _safe_dealloc(old_m, keep=new_m)

        if not skip_reduce_for_broken_arm:
            for nm, g in rem_grads.items():
                if g is not None and nm not in self.replicated_names and g is not dev_grads.get(nm):
                    _safe_dealloc(g)
        if free_grads:
            for g in dev_grads.values():
                if g is not None and g.is_allocated():
                    ttnn.deallocate(g)

        out = {
            "step": opt.steps,
            "lr": lr_now,
            "lr_muon": lr_muon,
            "grad_norm": adam_gnorm,
            "grad_norm_muon": muon_gnorm,
            "clip": clip_adam,
            "clip_adam": clip_adam,
            "clip_muon": clip_muon,
            "dead_grads": dead,
        }
        if profile_sync:
            out["sync_s"] = round(sync_s, 5)
        return out

    def host_master(self) -> Dict[str, "np.ndarray"]:
        """Return full host fp32 master dict across both unsharded and axis-0 sharded Muon buckets."""
        out = self.base_opt.host_master()
        if self.use_zero1_muon and self.muon_sharder is not None and self.sh_master:
            out.update(self.muon_sharder.gather_host_shards(self.sh_master))
        return out

    def master_digest(self) -> str:
        """SHA-256 prefix over sorted fp32 master bytes across all parameters."""
        import hashlib

        hm = self.host_master()
        h = hashlib.sha256()
        for k in sorted(hm):
            h.update(hm[k].tobytes())
        return h.hexdigest()[:16]

    def host_ema_shadow(self) -> Dict[str, "np.ndarray"]:
        """Return full host EMA shadow combining replicated and axis-0 sharded Muon buckets."""
        out: Dict[str, Any] = {}
        if self.base_opt.ema is not None:
            out.update(self.base_opt.ema.host_shadow())
        if self.use_zero1_muon and self.muon_sharder is not None and self.sh_ema:
            out.update(self.muon_sharder.gather_host_shards(self.sh_ema))
        return out

    def deallocate(self) -> None:
        import ttnn

        self.reducer.deallocate_caches()
        if self.muon_sharder is not None:
            self.muon_sharder.deallocate_caches()
        for d in (self.sh_master, self.sh_mom, self.sh_ema):
            for v in d.values():
                if v is not None and v.is_allocated():
                    ttnn.deallocate(v)
            d.clear()
