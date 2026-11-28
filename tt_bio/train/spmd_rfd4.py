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


WINNER_SMALL_KEYS = frozenset({
    "padding_mask",
    "is_in_context_conditioning",
    "supports_local_latent",
    "is_resolved_mask",
    "loss_weight",
    "x_1",
    "x_t",
    "t",
    "residue_type",
    "condition_sequence_1_residue",
    "tetrahedral_vertices",
    "insertion_source_idx",
    "condition_cath-label_1_residue",
    "condition_secondary-structure_1_residue",
    "token_type",
    "charge",
    "element",
})


def take_example(
    batch: Mapping[str, Any],
    idx: int,
    *,
    keys: Optional[Collection[str]] = None,
) -> dict:
    """Slice candidate ``idx`` (preserving leading dim 1) from a single-rank ``[K, ...]`` batch."""
    import torch

    out: dict[str, Any] = {}
    for k, v in batch.items():
        if keys is not None and k not in keys:
            continue
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
    *,
    stack: bool = True,
) -> Tuple[List[int], List[Mapping[str, Any]], Optional[dict]]:
    """Select ``world`` batches for ``step_idx`` using the canonical ``(step*world + r) % len`` order."""
    if not batch_pool:
        raise ValueError("batch_pool must be non-empty")
    if world <= 0:
        raise ValueError(f"world must be > 0, got {world}")
    m = len(batch_pool)
    indices = [(step_idx * world + r) % m for r in range(world)]
    per_rank = [batch_pool[i] for i in indices]
    if not stack:
        return indices, per_rank, None
    stacked = per_rank[0] if world == 1 else stack_batches(per_rank)
    return indices, per_rank, stacked


def gather_winner_batches(
    per_rank_batches: Sequence[Mapping[str, Any]],
    win_indices: Sequence[int],
    *,
    keys: Optional[Collection[str]] = None,
) -> Tuple[List[dict], dict]:
    """Slice each rank's chosen exploration winner and stack into ``[world, ...]``."""
    if len(per_rank_batches) != len(win_indices):
        raise ValueError(
            f"per_rank_batches ({len(per_rank_batches)}) != win_indices ({len(win_indices)})"
        )
    per_rank_w = [take_example(bt, int(w), keys=keys) for bt, w in zip(per_rank_batches, win_indices)]
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
        self._state_epoch: int = 0
        self._mask_cache: Dict[tuple, Tuple[object, object]] = {}

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

    @property
    def muon_names(self) -> frozenset[str]:
        return self.base_opt.muon_names

    def sharded_names(self) -> frozenset[str]:
        if self.use_zero1_muon and self.muon_plan is not None:
            return self.muon_plan.sharded_names
        return frozenset()

    def _get_cached_masks(self, kind: str, bk_idx: int, flags: Tuple[bool, ...], s0: int):
        k = (kind, bk_idx, flags, s0)
        pair = self._mask_cache.get(k)
        if pair is None:
            pair = self.base_opt._mask_tensor(list(flags), s0)
            self._mask_cache[k] = pair
        return pair

    def _get_cached_sh_masks(self, slot, flags: Tuple[bool, ...]):
        if all(flags):
            return None
        k = ("sh_muon", slot.shape, flags)
        pair = self._mask_cache.get(k)
        if pair is None:
            import torch
            import ttnn
            from .ccl import _safe_dealloc

            rm, cm = self.muon_sharder.mesh_shape
            mapper = ttnn.ShardTensor2dMesh(self.device, (rm, cm), dims=(0, None))
            act_t = torch.tensor(
                [1.0 if f else 0.0 for f in flags], dtype=torch.float32
            ).reshape(rm, slot.k_shard, 1, 1)
            inact_t = 1.0 - act_t
            ma4 = ttnn.from_torch(
                act_t.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                device=self.device, mesh_mapper=mapper,
            )
            mi4 = ttnn.from_torch(
                inact_t.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                device=self.device, mesh_mapper=mapper,
            )
            ma = ttnn.reshape(ma4, [slot.k_shard, 1, 1])
            mi = ttnn.reshape(mi4, [slot.k_shard, 1, 1])
            _safe_dealloc(ma4, keep=ma)
            _safe_dealloc(mi4, keep=mi)
            pair = (ma, mi)
            self._mask_cache[k] = pair
        return pair

    @staticmethod
    def _buf_addr(t) -> int:
        if t is None:
            return 0
        fn = getattr(t, "buffer_address", None)
        return int(fn()) if fn is not None else id(t)

    def _try_traced_opt_body(
        self,
        tracer,
        sh_slots: list,
        grad_shards: Mapping[Tuple[int, int], object],
        dev_grads: Mapping[str, object],
        packed_rem_muon: list,
        packed_adam: list,
        dis: set,
        *,
        clip_muon: float,
        lr_muon: float,
        clip_adam: float,
        lr_now: float,
        bc1: float,
        bc2: float,
        in_place_masters: bool,
    ) -> bool:
        """Execute Phase B + unsharded Muon + AdamW + EMA inside a single MeshTrace (Trace A)."""
        opt = self.base_opt
        mu = opt.muon
        has_ema = opt.ema is not None
        if not (
            tracer is not None
            and getattr(tracer, "enabled", False)
            and in_place_masters
            and bool(sh_slots)
            and all(s.shape in self.sh_mom for s in sh_slots)
            and (
                not has_ema
                or (
                    all(s.shape in self.sh_ema for s in sh_slots)
                    and opt.ema.n_averaged >= 1
                    and getattr(opt.ema, "_buckets", None) is not None
                    and all(bk.ema_b is not None for bk in opt.ema._buckets)
                )
            )
            and all((g_b is None) or (bk.state1_b is not None) for bk, g_b, _ in packed_rem_muon)
            and all(
                (not any(nm not in dis for nm in bk.names))
                or (g_b is None)
                or (bk.state1_b is not None and bk.state2_b is not None)
                for bk, g_b, _ in packed_adam
            )
        ):
            return False

        import torch
        import ttnn
        from .device_optim import _clone_exact_f32, newton_schulz_device
        from .mesh_dp import mesh_world

        # Pack dynamic per-step scalars into ONE [S, 1, 1] replicated fp32 TILE tensor.
        # Layout: [clip_muon, decay_muon, clip_adam, inv_bc1, inv_bc2, lr_adam, decay_adam_only,
        #          step_scale(ratio_0), ..., step_scale(ratio_{R-1})]
        decay_muon = float(1.0 - lr_muon * mu.weight_decay)
        inv_bc1 = float(1.0 / bc1)
        inv_bc2 = float(1.0 / bc2)
        decay_adam_only = float(1.0 - lr_now * opt.weight_decay)
        ratio_idx: Dict[float, int] = {}
        scalar_vals: List[float] = [
            float(clip_muon),
            decay_muon,
            float(clip_adam),
            inv_bc1,
            inv_bc2,
            float(lr_now),
            decay_adam_only,
        ]
        for slot in sh_slots:
            r = float(mu.ratio(slot.shape))
            if r not in ratio_idx:
                ratio_idx[r] = len(scalar_vals)
                scalar_vals.append(float(-lr_muon * r))
        for bk, g_b, _ in packed_rem_muon:
            if g_b is not None:
                r = float(mu.ratio(bk.shape))
                if r not in ratio_idx:
                    ratio_idx[r] = len(scalar_vals)
                    scalar_vals.append(float(-lr_muon * r))

        rep_cls = getattr(ttnn, "ReplicateTensorToMesh", None)
        sc_kw = (
            {"mesh_mapper": rep_cls(self.device)}
            if (rep_cls is not None and mesh_world(self.device) > 1) else {}
        )
        scalars_dev = ttnn.from_torch(
            torch.tensor(scalar_vals, dtype=torch.float32).reshape(-1, 1, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            **sc_kw,
        )

        inputs: List[object] = [scalars_dev]
        sh_spec = []
        for slot in sh_slots:
            inputs.append(grad_shards[slot.shape])
            fl_t = tuple(
                (i < slot.k) and (slot.names[i] not in dis) and (dev_grads.get(slot.names[i]) is not None)
                for i in range(slot.k_pad)
            )
            m_pair = self._get_cached_sh_masks(slot, fl_t)
            sh_spec.append((slot, fl_t, m_pair))

        rem_muon_spec = []
        for idx, (bk, g_b, flags) in enumerate(packed_rem_muon):
            fl_t = tuple(bool(f) for f in flags)
            has_g = g_b is not None
            g_pos = len(inputs) if has_g else -1
            if has_g:
                inputs.append(g_b)
            m_pair = None
            if has_g and not all(fl_t):
                m_pair = self._get_cached_masks("rem_muon", idx, fl_t, bk.canon[0])
            rem_muon_spec.append((bk, g_pos, fl_t, m_pair))

        adam_spec = []
        for idx, (bk, g_b, _flags) in enumerate(packed_adam):
            en_t = tuple(nm not in dis for nm in bk.names)
            has_g = g_b is not None
            has_s1 = bk.state1_b is not None
            g_pos = len(inputs) if has_g else -1
            if has_g:
                inputs.append(g_b)
            m_pair = None
            if any(en_t) and not all(en_t):
                m_pair = self._get_cached_masks("adam", idx, en_t, bk.canon[0])
            adam_spec.append((bk, g_pos, en_t, has_s1, m_pair))

        # Key on epoch, active/enabled flags, and exact persistent buffer addresses.
        addr = self._buf_addr
        sh_addrs = tuple(
            (s.shape, fl_t,
             addr(m_pair[0]) if m_pair is not None else 0,
             addr(m_pair[1]) if m_pair is not None else 0,
             addr(self.sh_master[s.shape]), addr(self.sh_mom[s.shape]),
             addr(self.sh_ema[s.shape]) if has_ema else 0)
            for s, fl_t, m_pair in sh_spec
        )
        rm_key = tuple(
            (bk.shape, g_pos >= 0, fl_t, addr(bk.master_b), addr(bk.state1_b),
             addr(bk.ema_b) if has_ema else 0,
             tuple(addr(opt.master[nm]) for nm in bk.names))
            for bk, g_pos, fl_t, _ in rem_muon_spec
        )
        ad_key = tuple(
            (bk.shape, g_pos >= 0, en_t, has_s1, addr(bk.master_b),
             addr(bk.state1_b), addr(bk.state2_b),
             addr(bk.ema_b) if has_ema else 0,
             tuple(addr(opt.master[nm]) for nm in bk.names))
            for bk, g_pos, en_t, has_s1, _ in adam_spec
        )
        trace_key = (
            self._state_epoch,
            has_ema,
            len(scalar_vals),
            opt.ns_mode,
            mu.ns_steps,
            tuple(mu.ns_coefficients),
            sh_addrs,
            rm_key,
            ad_key,
        )

        one_minus_mu = float(1.0 - mu.momentum)
        b1, b2 = float(opt.beta1), float(opt.beta2)
        om1, om2 = float(1.0 - b1), float(1.0 - b2)
        wd_adam = float(opt.weight_decay)
        eps_adam = float(opt.eps)
        ema_w = float(1.0 - opt.ema.decay) if has_ema else 0.0

        def _body(staged: Sequence[object]) -> None:
            sc_vec = staged[0]
            sc_cache: Dict[int, object] = {}

            def _sc(i: int):
                t = sc_cache.get(i)
                if t is None:
                    t = ttnn.slice(sc_vec, [i, 0, 0], [i + 1, 1, 1])
                    sc_cache[i] = t
                return t

            clip_m_t = _sc(0)
            decay_m_t = _sc(1)
            clip_a_t = _sc(2)
            inv_bc1_t = _sc(3)
            inv_bc2_t = _sc(4)
            lr_a_t = _sc(5)
            decay_a_only_t = _sc(6)

            # 1. Sharded Muon buckets (1/rm): momentum -> NS -> master -> EMA, all in-place.
            for si, (slot, _fl_t, m_pair) in enumerate(sh_spec):
                shp = slot.shape
                g_sh = staged[1 + si]
                m_prev = self.sh_mom[shp]
                w_prev = self.sh_master[shp]

                g_eff = ttnn.multiply(g_sh, clip_m_t)
                diff = ttnn.subtract(g_eff, m_prev)
                scaled = ttnn.multiply(diff, one_minus_mu)
                ttnn.deallocate(diff)
                buf_cand = ttnn.add(m_prev, scaled)
                ttnn.deallocate(scaled)
                if m_pair is None:
                    ttnn.copy(buf_cand, m_prev)
                    ttnn.deallocate(buf_cand)
                else:
                    ma, mi = m_pair
                    p1 = ttnn.multiply(buf_cand, ma)
                    p2 = ttnn.multiply(m_prev, mi)
                    ttnn.deallocate(buf_cand)
                    buf_new = ttnn.add(p1, p2)
                    ttnn.deallocate(p1)
                    ttnn.deallocate(p2)
                    ttnn.copy(buf_new, m_prev)
                    ttnn.deallocate(buf_new)

                if mu.nesterov:
                    d2 = ttnn.subtract(m_prev, g_eff)
                    s2 = ttnn.multiply(d2, one_minus_mu)
                    ttnn.deallocate(d2)
                    u_b = ttnn.subtract(m_prev, s2)
                    ttnn.deallocate(s2)
                else:
                    u_b = _clone_exact_f32(m_prev)
                ttnn.deallocate(g_eff)

                o_b = newton_schulz_device(
                    u_b,
                    steps=mu.ns_steps,
                    coeffs=mu.ns_coefficients,
                    eps=mu.eps,
                    mode=opt.ns_mode,
                )
                ttnn.deallocate(u_b)

                step_sc_t = _sc(ratio_idx[float(mu.ratio(shp))])
                t2 = ttnn.multiply(o_b, step_sc_t)
                ttnn.deallocate(o_b)
                t1 = ttnn.multiply(w_prev, decay_m_t)
                w_cand = ttnn.add(t1, t2)
                ttnn.deallocate(t1)
                ttnn.deallocate(t2)
                if m_pair is None:
                    ttnn.copy(w_cand, w_prev)
                    ttnn.deallocate(w_cand)
                else:
                    ma, mi = m_pair
                    p1 = ttnn.multiply(w_cand, ma)
                    p2 = ttnn.multiply(w_prev, mi)
                    ttnn.deallocate(w_cand)
                    w_new = ttnn.add(p1, p2)
                    ttnn.deallocate(p1)
                    ttnn.deallocate(p2)
                    ttnn.copy(w_new, w_prev)
                    ttnn.deallocate(w_new)

                if has_ema:
                    e_prev = self.sh_ema[shp]
                    ediff = ttnn.subtract(w_prev, e_prev)
                    esc = ttnn.multiply(ediff, ema_w)
                    ttnn.deallocate(ediff)
                    e_nxt = ttnn.add(e_prev, esc)
                    ttnn.deallocate(esc)
                    ttnn.copy(e_nxt, e_prev)
                    ttnn.deallocate(e_nxt)

            # 2. Remaining unsharded Muon buckets (in-place).
            for bk, g_pos, fl_t, m_pair in rem_muon_spec:
                if g_pos < 0:
                    continue
                g_b = staged[g_pos]
                all_active = all(fl_t)

                g_eff = ttnn.multiply(g_b, clip_m_t)
                diff = ttnn.subtract(g_eff, bk.state1_b)
                scaled = ttnn.multiply(diff, one_minus_mu)
                ttnn.deallocate(diff)
                buf_cand = ttnn.add(bk.state1_b, scaled)
                ttnn.deallocate(scaled)
                if all_active:
                    ttnn.copy(buf_cand, bk.state1_b)
                    ttnn.deallocate(buf_cand)
                else:
                    ma, mi = m_pair
                    p1 = ttnn.multiply(buf_cand, ma)
                    p2 = ttnn.multiply(bk.state1_b, mi)
                    ttnn.deallocate(buf_cand)
                    buf_new = ttnn.add(p1, p2)
                    ttnn.deallocate(p1)
                    ttnn.deallocate(p2)
                    ttnn.copy(buf_new, bk.state1_b)
                    ttnn.deallocate(buf_new)

                if mu.nesterov:
                    d2 = ttnn.subtract(bk.state1_b, g_eff)
                    s2 = ttnn.multiply(d2, one_minus_mu)
                    ttnn.deallocate(d2)
                    u_b = ttnn.subtract(bk.state1_b, s2)
                    ttnn.deallocate(s2)
                else:
                    u_b = _clone_exact_f32(bk.state1_b)
                ttnn.deallocate(g_eff)

                o_b = newton_schulz_device(
                    u_b,
                    steps=mu.ns_steps,
                    coeffs=mu.ns_coefficients,
                    eps=mu.eps,
                    mode=opt.ns_mode,
                )
                ttnn.deallocate(u_b)

                step_sc_t = _sc(ratio_idx[float(mu.ratio(bk.shape))])
                t1 = ttnn.multiply(bk.master_b, decay_m_t)
                t2 = ttnn.multiply(o_b, step_sc_t)
                ttnn.deallocate(o_b)
                cand = ttnn.add(t1, t2)
                ttnn.deallocate(t1)
                ttnn.deallocate(t2)
                if all_active:
                    ttnn.copy(cand, bk.master_b)
                    ttnn.deallocate(cand)
                else:
                    ma, mi = m_pair
                    p1 = ttnn.multiply(cand, ma)
                    p2 = ttnn.multiply(bk.master_b, mi)
                    ttnn.deallocate(cand)
                    w_new = ttnn.add(p1, p2)
                    ttnn.deallocate(p1)
                    ttnn.deallocate(p2)
                    ttnn.copy(w_new, bk.master_b)
                    ttnn.deallocate(w_new)

                opt._unpack_master(bk, in_place=True)

            # 3. AdamW buckets (in-place).
            for bk, g_pos, en_t, has_s1, m_pair in adam_spec:
                if not any(en_t):
                    continue
                all_enabled = all(en_t)
                if g_pos < 0 and not has_s1 and wd_adam == 0.0:
                    continue

                if g_pos < 0:
                    if has_s1:
                        m_cand = ttnn.multiply(bk.state1_b, b1)
                        v_cand = ttnn.multiply(bk.state2_b, b2)
                    else:
                        m_cand = v_cand = None
                else:
                    g_b = staged[g_pos]
                    g_eff = ttnn.multiply(g_b, clip_a_t)
                    g2 = ttnn.square(g_eff)
                    mb = ttnn.multiply(bk.state1_b, b1)
                    mg = ttnn.multiply(g_eff, om1)
                    m_cand = ttnn.add(mb, mg)
                    ttnn.deallocate(mb)
                    ttnn.deallocate(mg)
                    vb = ttnn.multiply(bk.state2_b, b2)
                    vg = ttnn.multiply(g2, om2)
                    v_cand = ttnn.add(vb, vg)
                    ttnn.deallocate(vb)
                    ttnn.deallocate(vg)
                    ttnn.deallocate(g2)
                    ttnn.deallocate(g_eff)

                if m_cand is not None:
                    if all_enabled:
                        ttnn.copy(m_cand, bk.state1_b)
                        ttnn.copy(v_cand, bk.state2_b)
                        ttnn.deallocate(m_cand)
                        ttnn.deallocate(v_cand)
                    else:
                        ma, mi = m_pair
                        m1 = ttnn.multiply(m_cand, ma)
                        m2 = ttnn.multiply(bk.state1_b, mi)
                        v1 = ttnn.multiply(v_cand, ma)
                        v2 = ttnn.multiply(bk.state2_b, mi)
                        ttnn.deallocate(m_cand)
                        ttnn.deallocate(v_cand)
                        m_new = ttnn.add(m1, m2)
                        v_new = ttnn.add(v1, v2)
                        ttnn.deallocate(m1)
                        ttnn.deallocate(m2)
                        ttnn.deallocate(v1)
                        ttnn.deallocate(v2)
                        ttnn.copy(m_new, bk.state1_b)
                        ttnn.copy(v_new, bk.state2_b)
                        ttnn.deallocate(m_new)
                        ttnn.deallocate(v_new)

                if not has_s1:
                    cand = ttnn.multiply(bk.master_b, decay_a_only_t)
                else:
                    m_hat = ttnn.multiply(bk.state1_b, inv_bc1_t)
                    v_hat = ttnn.multiply(bk.state2_b, inv_bc2_t)
                    sqv = ttnn.sqrt(v_hat)
                    ttnn.deallocate(v_hat)
                    den = ttnn.add(sqv, eps_adam)
                    ttnn.deallocate(sqv)
                    ratio = ttnn.divide(m_hat, den)
                    ttnn.deallocate(m_hat)
                    ttnn.deallocate(den)
                    if wd_adam != 0.0:
                        wd_term = ttnn.multiply(bk.master_b, wd_adam)
                        comb = ttnn.add(ratio, wd_term)
                        ttnn.deallocate(ratio)
                        ttnn.deallocate(wd_term)
                        ratio = comb
                    upd = ttnn.multiply(ratio, lr_a_t)
                    ttnn.deallocate(ratio)
                    cand = ttnn.subtract(bk.master_b, upd)
                    ttnn.deallocate(upd)

                if all_enabled:
                    ttnn.copy(cand, bk.master_b)
                    ttnn.deallocate(cand)
                else:
                    ma, mi = m_pair
                    p1 = ttnn.multiply(cand, ma)
                    p2 = ttnn.multiply(bk.master_b, mi)
                    ttnn.deallocate(cand)
                    w_new = ttnn.add(p1, p2)
                    ttnn.deallocate(p1)
                    ttnn.deallocate(p2)
                    ttnn.copy(w_new, bk.master_b)
                    ttnn.deallocate(w_new)

                opt._unpack_master(bk, in_place=True)

            # 4. Non-sharded EMA buckets (in-place; n_averaged incremented on host after trace).
            if has_ema:
                for bk in opt.ema._buckets:
                    e = bk.ema_b
                    ediff = ttnn.subtract(bk.master_b, e)
                    esc = ttnn.multiply(ediff, ema_w)
                    ttnn.deallocate(ediff)
                    e_nxt = ttnn.add(e, esc)
                    ttnn.deallocate(esc)
                    ttnn.copy(e_nxt, e)
                    ttnn.deallocate(e_nxt)

            for t in sc_cache.values():
                ttnn.deallocate(t)

        input_sig = tuple((tuple(x.shape), x.dtype) for x in inputs)
        ran = tracer.run_in_place_trace(
            "opt_body", trace_key, inputs, _body,
            warm_key="opt_body", share_slots_key=input_sig,
        )
        ttnn.deallocate(scalars_dev)
        if not ran:
            return False

        if has_ema:
            opt.ema.n_averaged += 1
        for slot in sh_slots:
            g_sh = grad_shards[slot.shape]
            if g_sh.is_allocated():
                ttnn.deallocate(g_sh)
        for _bk, g_b, _flags in packed_rem_muon:
            if g_b is not None and g_b.is_allocated():
                ttnn.deallocate(g_b)
        for _bk, g_b, _flags in packed_adam:
            if g_b is not None and g_b.is_allocated():
                ttnn.deallocate(g_b)
        return True

    def step(
        self,
        dev_grads: Mapping[str, object],
        *,
        disabled: Collection[str] = (),
        free_grads: bool = False,
        skip_reduce_for_broken_arm: bool = False,
        profile_sync: bool = False,
        in_place_masters: bool = False,
        free_skip: Optional[set] = None,
        tracer=None,
    ) -> dict:
        """Run one synchronized SPMD optimizer step on ``MeshDevice``."""
        import time as _time
        import ttnn

        skip = set(free_skip) if free_skip is not None else ()

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
            step_kw = dict(reduce_fn=rfn, disabled=disabled, free_grads=free_grads)
            if in_place_masters:
                step_kw["in_place_masters"] = True
            if skip:
                step_kw["free_skip"] = skip
            info = self.base_opt.step(dev_grads, **step_kw)
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
        )

        assert self.muon_sharder is not None and self.muon_plan is not None
        opt = self.base_opt
        dis = set(disabled)
        sh_names = self.muon_plan.sharded_names
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

        # 3. Phase B + unsharded Muon + AdamW + EMA: single MeshTrace (Trace A) when steady-state.
        mu = opt.muon
        one_minus_mu = float(1.0 - mu.momentum)
        sh_slots = [slot for wg in self.muon_plan.width_groups for slot in wg.buckets]

        traced_body = self._try_traced_opt_body(
            tracer,
            sh_slots,
            grad_shards,
            dev_grads,
            packed_rem_muon,
            packed_adam,
            dis,
            clip_muon=clip_muon,
            lr_muon=lr_muon,
            clip_adam=clip_adam,
            lr_now=lr_now,
            bc1=bc1,
            bc2=bc2,
            in_place_masters=in_place_masters,
        )

        if not traced_body:
            masks = self.muon_sharder.active_masks_for_step(dev_grads, dis, self.device)
            u_list = []
            for slot in sh_slots:
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

                u_list.append(u_b)

            def _run_ns_all(us):
                return [
                    newton_schulz_device(
                        u_cur, steps=mu.ns_steps, coeffs=mu.ns_coefficients, eps=mu.eps,
                        mode=opt.ns_mode,
                    )
                    for u_cur in us
                ]

            if tracer is not None and sh_slots:
                ns_key = (
                    "muon_ns",
                    tuple((s.k_shard, s.m, s.cols) for s in sh_slots),
                    opt.ns_mode, mu.ns_steps, tuple(mu.ns_coefficients),
                )
                o_list = tracer.run_muon_ns_batch(ns_key, u_list, _run_ns_all)
            else:
                o_list = _run_ns_all(u_list)
            for u_cur in u_list:
                ttnn.deallocate(u_cur)

            for slot, o_b in zip(sh_slots, o_list):
                shp = slot.shape
                m_pair = masks[shp]
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
            ip_kw = {"in_place_masters": True} if in_place_masters else {}
            for bk, g_b, flags in packed_rem_muon:
                opt._step_muon_bucket(bk, g_b, flags, clip_muon, lr_muon, **ip_kw)
                if g_b is not None and g_b.is_allocated():
                    ttnn.deallocate(g_b)
            for bk, g_b, _flags in packed_adam:
                opt._step_adam_bucket(bk, g_b, dis, clip_adam, lr_now, bc1, bc2, **ip_kw)
                if g_b is not None and g_b.is_allocated():
                    ttnn.deallocate(g_b)
            if opt.ema is not None:
                opt.ema.update()

        # 5. Phase C: AG_axis0 -> bit-identical replicated masters for all sharded Muon params.
        if profile_sync:
            ttnn.synchronize_device(self.device)
            _ts1 = _time.time()
        full_wgs = self.muon_sharder.all_gather_width_groups(self.sh_master, self.device)
        if profile_sync:
            ttnn.synchronize_device(self.device)
            sync_s += _time.time() - _ts1

        can_in_place_ag = in_place_masters and all(
            opt.master.get(nm) is not None for nm in self.muon_plan.sharded_names
        )
        ag_traced = False
        if can_in_place_ag and tracer is not None and getattr(tracer, "enabled", False):
            ag_key = (
                self._state_epoch,
                tuple(tuple(int(d) for d in f.shape) for f in full_wgs),
                tuple(self._buf_addr(opt.master[nm]) for nm in sorted(self.muon_plan.sharded_names)),
            )
            ag_traced = tracer.run_in_place_trace(
                "ag_unpack",
                ag_key,
                full_wgs,
                lambda staged: self.muon_sharder.unpack_width_groups_in_place(staged, opt.master),
            )
        if not ag_traced:
            if can_in_place_ag:
                self.muon_sharder.unpack_width_groups_in_place(full_wgs, opt.master)
            else:
                for wg, full in zip(self.muon_plan.width_groups, full_wgs):
                    for slot in wg.buckets:
                        for i, nm in enumerate(slot.names):
                            r = i // slot.k_shard
                            j = i % slot.k_shard
                            off = r * wg.row_block_h + slot.slot_off + j * slot.m
                            sl = ttnn.slice(full, [0, 0, off, 0], [1, 1, off + slot.m, wg.cols])
                            new_m = ttnn.reshape(sl, [slot.m, wg.cols])
                            _safe_dealloc(sl, keep=new_m)
                            old_m = opt.master.get(nm)
                            opt.master[nm] = new_m
                            _safe_dealloc(old_m, keep=new_m)
        for full in full_wgs:
            if full.is_allocated():
                ttnn.deallocate(full)

        if not skip_reduce_for_broken_arm:
            for nm, g in rem_grads.items():
                if g is not None and nm not in self.replicated_names and g is not dev_grads.get(nm):
                    _safe_dealloc(g)
        if free_grads:
            for nm, g in dev_grads.items():
                if g is not None and g.is_allocated() and nm not in skip:
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

    def state_dict(self) -> dict:
        return self.base_opt.state_dict()

    def load_state_dict(self, d: dict) -> None:
        self.base_opt.load_state_dict(d)

    def drop_unowned(self) -> None:
        self.base_opt.drop_unowned()

    def state_families(self) -> Dict[str, Dict[str, "np.ndarray"]]:
        """Per-parameter host fp32 optimizer and EMA state dicts across replicated and sharded buckets."""
        import numpy as np

        fams = self.base_opt.state_families()
        if self.use_zero1_muon and self.muon_sharder is not None and self.muon_plan is not None:
            mom = dict(fams.get("momentum_buffer", {}))
            if self.sh_mom:
                mom.update(self.muon_sharder.gather_host_shards(self.sh_mom))
            else:
                for wg in self.muon_plan.width_groups:
                    for slot in wg.buckets:
                        for nm in slot.names:
                            mom[nm] = np.zeros(slot.shape, dtype=np.float32)
            fams["momentum_buffer"] = mom
            if self.base_opt.ema is not None and self.base_opt.ema.n_averaged > 0:
                ema = dict(fams.get("ema_shadow", {}))
                if self.sh_ema:
                    ema.update(self.muon_sharder.gather_host_shards(self.sh_ema))
                else:
                    ema.update(self.muon_sharder.gather_host_shards(self.sh_master))
                fams["ema_shadow"] = ema
        return fams

    def load_host_state(
        self,
        masters: Mapping[str, object],
        families: Mapping[str, Mapping[str, object]] = (),
    ) -> None:
        """Reload checkpoint masters, moments, and EMA across replicated and axis-0 sharded buckets."""
        import numpy as np
        from .device_optim import _safe_dealloc

        fams = dict(families) if families else {}
        self._state_epoch += 1
        self.base_opt.load_host_state(masters, fams)
        if not (self.use_zero1_muon and self.muon_sharder is not None and self.muon_plan is not None):
            return

        sh_names = self.muon_plan.sharded_names

        def _replace_shards(
            dst: Dict[Tuple[int, int], object],
            src_map: Optional[Mapping[str, object]],
        ) -> None:
            new_map = (
                self.muon_sharder.shard_initial_masters(src_map, self.device)
                if src_map is not None else {}
            )
            for shp, old_t in list(dst.items()):
                _safe_dealloc(old_t, keep=new_map.get(shp))
            dst.clear()
            dst.update(new_map)

        _replace_shards(self.sh_master, masters)

        mom = fams.get("momentum_buffer")
        if mom and any(nm in mom for nm in sh_names):
            mom_full = {
                nm: (mom[nm] if nm in mom else np.zeros(self.base_opt.shapes[nm], dtype=np.float32))
                for nm in sh_names
            }
            _replace_shards(self.sh_mom, mom_full)
        else:
            _replace_shards(self.sh_mom, None)

        if self.base_opt.ema is not None:
            es = fams.get("ema_shadow")
            if es and any(nm in es for nm in sh_names):
                ema_full = {nm: (es[nm] if nm in es else masters[nm]) for nm in sh_names}
                _replace_shards(self.sh_ema, ema_full)
            else:
                _replace_shards(self.sh_ema, None)
        else:
            _replace_shards(self.sh_ema, None)

        gathered = self.muon_sharder.all_gather_masters(self.sh_master, self.device)
        for nm, new_m in gathered.items():
            old_m = self.base_opt.master.get(nm)
            self.base_opt.master[nm] = new_m
            _safe_dealloc(old_m, keep=new_m)

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
        for m_act, m_inact in self._mask_cache.values():
            for m in (m_act, m_inact):
                if m is not None and m.is_allocated():
                    ttnn.deallocate(m)
        self._mask_cache.clear()
