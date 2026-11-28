"""Optimizer steps on RFD4-Proteina's real trunk (release weights, 14 layers + 7 pair updates), one chip.

Per step, as upstream's recipe runs it: `k` noise candidates through the production trunk grad-off at recycle depth
`d` (one batched inference call), then the winner's LAST pass re-run taped on the production methods
by the production `Trunk.__call__` under the tape, earlier passes no_grad as upstream's `recycle_grad: last`. Not yet bridged: the host glue (conditioning, init_repr, pair builder) and heads; their inputs
are seeded tensors and the loss seed lands on s (`<s, C>/numel`). Prints one JSON line per step.

  --save-at k / --resume PATH: checkpoint after step k / continue from it in a fresh process (bit-identical check)
  --freeze SUB: the G2 fail arm. The first trained upstream leaf whose name contains SUB stops requiring grad
     before the build, so its fold uploads as a constant: one parameter detached from the graph, as a load-path
     `.detach()` bug would. (Cutting the token stream at a layer detaches nothing: the pair biases and z carry
     every earlier layer forward.)
  --batch P1,P2,..: the real step, step i on batch P[i % len], each a batch upstream itself trained on (`perf/r4t/loss_parity_reference.py` dumps:
     every feature, x_1/x_t/t, k candidates, depth, upstream winners and losses). Host glue (conditioning,
     initial representations, pair builder) and heads run with grad on the host and join the tape through
     `train.bridge.Bridge`; the loss is `train.rfd4_objective.rfd4_loss`. The candidates' winner is chosen
     from our own heads, as upstream does, and printed beside upstream's.
  --exempt FILE: G2 exemptions, one `name<TAB>upstream's reason` a line
  --dp-cards C0,C1,..: data parallel, one rank per card (processes), gradients summed by the launcher's
     collective (`train.launcher.ProcessAxis`); --shard steps a ZeRO-1 slice per rank. Rank r's example at
     step i is index i*world + r, so a world-w step consumes w distinct examples; each rank's JSON lines go
     to OUT/rank<r>.log and its checkpoints under OUT/rank<r>/. Master digests must agree across ranks.

Moved here from `perf/r4t_step/trunk_step.py` (r4t D56), which is now a thin caller; `main(argv)`
takes the same flags and prints the same JSON lines.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import resource
import threading
import time

from tt_bio.envflags import env_flag

#: Parameters whose upstream gradient is identically zero on unconditional batches (D29).
#: Selected by ``--exempt default`` (used by Tier 0, which has no ``perf/`` tree in a wheel).
UPSTREAM_EXEMPT = {
    "glue.init_repr_factory.feat_creators.7.ss_embedding.weight":
        "upstream Embedding(4, 16, padding_idx=0) and every unconditional token is NONE "
        "(index 0): upstream gradient is an exact zero (r4t-accuracy D29)",
}


def _checkpoint_due(step: int, total_steps: int, save_every: int, save_at: int) -> tuple[bool, bool]:
    """Return ``(should_save, force)`` for the periodic checkpointer at 1-based ``step``.

    When ``--save-every`` is set, save on every multiple and on the final step (matching
    ``tt-bio train --checkpoint-every``'s contract), skipping if ``--save-at`` already wrote it.
    """
    if not save_every or (save_at and step == save_at):
        return False, False
    last = step == total_steps
    return (last or step % save_every == 0), last


def _load_depths(depths_csv: str, depths_from: str) -> list[int]:
    """Parse an explicit per-step recycle-depth schedule from ``--depths`` or ``--depths-from``."""
    if depths_csv:
        return [max(1, int(x.strip())) for x in depths_csv.split(",") if x.strip()]
    if not depths_from:
        return []
    out: list[int] = []
    with open(os.path.expanduser(depths_from)) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("{"):
                rec = json.loads(line)
                val = rec.get("n_recycle", rec.get("depth"))
                if val is None:
                    raise ValueError(f"{depths_from}: JSON line missing n_recycle/depth: {line[:80]}")
                out.append(max(1, int(val)))
            else:
                for tok in line.replace(",", " ").split():
                    out.append(max(1, int(tok)))
    if not out:
        raise ValueError(f"{depths_from}: no depths parsed")
    return out


def _dealloc_unique(tensors) -> None:
    """Deallocate each distinct allocated ttnn.Tensor in ``tensors`` once."""
    import ttnn
    seen: set[int] = set()
    for t in tensors:
        if t is None:
            continue
        tid = id(t)
        if tid in seen:
            continue
        seen.add(tid)
        if t.is_allocated():
            ttnn.deallocate(t)


def _device_trunk_masks(mask_cpu, dev, dtype, residual_dtype=None, elide=None):
    """Build the 6-tuple of trunk device masks on card from a tiny ``[b, n]`` mask upload.

    Bit-exact (0.0 max abs diff) with :func:`tt_bio.rfd4.trunk.trunk_masks`, while avoiding
    building and uploading ``[b, n, n]`` pair masks on host CPU (~60 ms & ~16 MB/step saved).
    """
    import torch
    import ttnn
    from tt_bio.rfd4 import trunk as TR

    m = mask_cpu.to(torch.float32)
    all_one = bool((m == 1).all())
    if (TR.ELIDE_FULL_MASKS if elide is None else elide) and all_one:
        return None, None, None, None, None, None
    b, n = int(m.shape[0]), int(m.shape[1])

    def _up(t):
        return ttnn.from_torch(
            t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,
            device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    tok = _up(m[..., None])                        # [b, n, 1]
    m_row = _up(m[:, None, :])                     # [b, 1, n]
    pair_3d = ttnn.multiply(tok, m_row)            # [b, n, n]
    ttnn.deallocate(m_row)
    pair_1nn = ttnn.reshape(pair_3d, (b, 1, n, n))
    inv = ttnn.subtract(pair_1nn, 1.0)
    if pair_1nn is not pair_3d and pair_1nn.is_allocated():
        try:
            if pair_1nn.buffer_address() != pair_3d.buffer_address():
                ttnn.deallocate(pair_1nn)
        except Exception:
            pass
    attn_bias = ttnn.multiply(inv, -TR.NEG_INF)    # [b, 1, n, n]: 0 where valid, NEG_INF where masked
    ttnn.deallocate(inv)
    if bool(m.all()):
        ttnn.deallocate(pair_3d)
        pair_out = None
    else:
        pair_out = pair_3d
    m_col4 = _up(m[:, :, None, None])              # [b, n, 1, 1]
    m_row4 = _up(m[:, None, :, None])              # [b, 1, n, 1]
    pair_c = ttnn.multiply(m_col4, m_row4)         # [b, n, n, 1]
    ttnn.deallocate(m_col4)
    ttnn.deallocate(m_row4)
    res = residual_dtype or dtype
    if res == dtype:
        res_tok, res_pair_c = tok, pair_c
    else:
        res_tok, res_pair_c = ttnn.typecast(tok, res), ttnn.typecast(pair_c, res)
    return (tok, attn_bias, pair_out, pair_c, res_tok, res_pair_c)


def _device_transition_f32(x_dev, w_sw_f32, w_out_f32, mask_dev, ckc, cg):
    """Grad-off SwiGLU Transition on device in fp32 (matches upstream ``Transition.forward``)."""
    import ttnn
    from .device_optim import _split2_bf16

    def _mm2(a_f32, b_f32):
        # 2-part split-bf16 exact fp32 matmul (BH single-pass fp32 matmul truncates inputs).
        a_hi, a_lo = _split2_bf16(a_f32)
        b_hi, b_lo = _split2_bf16(b_f32)
        y = ttnn.matmul(a_hi, b_hi, transpose_b=True, dtype=ttnn.float32,
                        compute_kernel_config=ckc, core_grid=cg)
        p1 = ttnn.matmul(a_hi, b_lo, transpose_b=True, dtype=ttnn.float32,
                         compute_kernel_config=ckc, core_grid=cg)
        p2 = ttnn.matmul(a_lo, b_hi, transpose_b=True, dtype=ttnn.float32,
                         compute_kernel_config=ckc, core_grid=cg)
        ttnn.deallocate(a_hi)
        ttnn.deallocate(a_lo)
        ttnn.deallocate(b_hi)
        ttnn.deallocate(b_lo)
        y1 = ttnn.add(y, p1)
        ttnn.deallocate(y)
        ttnn.deallocate(p1)
        y2 = ttnn.add(y1, p2)
        ttnn.deallocate(y1)
        ttnn.deallocate(p2)
        return y2

    sh = [int(d) for d in x_dev.shape]
    h = _mm2(x_dev, w_sw_f32)
    d_inner = int(h.shape[-1]) // 2
    a = ttnn.slice(h, [0] * len(sh), sh[:-1] + [d_inner])
    g = ttnn.slice(h, [0] * (len(sh) - 1) + [d_inner], sh[:-1] + [2 * d_inner])
    ttnn.deallocate(h)
    # Exact SiLU: BH ttnn.silu LUT (~2e-3) is amplified ~13x by transition_c_2.
    sig = ttnn.sigmoid(g, mode=ttnn.SigmoidMode.Accurate)
    sg = ttnn.multiply(g, sig)
    ttnn.deallocate(g)
    ttnn.deallocate(sig)
    ag = ttnn.multiply(a, sg)
    ttnn.deallocate(a)
    ttnn.deallocate(sg)
    out = _mm2(ag, w_out_f32)
    ttnn.deallocate(ag)
    if mask_dev is not None:
        out2 = ttnn.multiply(out, mask_dev)
        ttnn.deallocate(out)
        out = out2
    return out


def _device_cand_conds(glue, bt, depth: int, master: dict, mk, dev, out_dtype):
    """Build grad-off candidate conditioning ``[depth]`` of ``[K, N, dim_cond]`` on device.

    Hoists the recycle-invariant ``cond_factory`` creators once across cycles, then runs
    ``transition_c_1`` + ``transition_c_2`` either on ``[K, 1, dim]`` host (0.0 bit-exact when all
    conditioning features are 2D per-example) or batched ``[U, K, N, dim]`` in fp32 on device.
    """
    import torch
    import ttnn

    cf = glue.cond_factory
    mask = bt["padding_mask"]
    K, n = int(mask.shape[0]), int(mask.shape[1])
    if getattr(cf, "ret_zero", False):
        return [
            ttnn.from_torch(
                glue.conditioning(bt, None if cyc == 0 else (depth - cyc - 1 if glue.condition_on_recycle_remaining else None))
                .detach().contiguous(),
                dtype=out_dtype, layout=ttnn.TILE_LAYOUT, device=dev,
            )
            for cyc in range(depth)
        ]

    # Precompute recycle-invariant features once across all `depth` cycles.
    inv_feats = []
    rec_idx = None
    for idx, fc in enumerate(cf.feat_creators):
        if type(fc).__name__ == "RecycleRemainingBatchFeat":
            rec_idx = idx
            inv_feats.append(None)
        else:
            inv_feats.append(fc(bt))

    rems: list[int | None] = [None]
    for cyc in range(1, depth):
        rems.append(depth - cyc - 1 if glue.condition_on_recycle_remaining else None)
    # Deduplicate identical `rem` values (e.g. recycle_remaining saturates at dim - 1).
    uniq_rems: list[int | None] = []
    cyc_to_u: list[int] = []
    rec_dim = cf.feat_creators[rec_idx].get_dim() if rec_idx is not None else 1
    seen_key: dict[int | None, int] = {}
    for r in rems:
        key = None if (r is None or rec_idx is None) else min(r, rec_dim - 1)
        if key not in seen_key:
            seen_key[key] = len(uniq_rems)
            uniq_rems.append(r)
        cyc_to_u.append(seen_key[key])

    # Check whether every 3D feature is constant across valid tokens per example (e.g. FoldEmbeddingSeqFeat).
    m_bool = mask > 0.5
    first_idx = m_bool.to(torch.int64).argmax(dim=-1)
    k_idx = torch.arange(K, device=mask.device)
    inv_2d = []
    row_const = True
    for f in inv_feats:
        if f is None or f.ndim == 2:
            inv_2d.append(f)
        elif f.ndim == 3:
            f0 = f[k_idx, first_idx]
            if bool(((f == f0.unsqueeze(1)) | (~m_bool.unsqueeze(-1))).all()):
                inv_2d.append(f0)
            else:
                row_const = False
                break
        else:
            row_const = False
            break
    if (row_const and isinstance(cf.ln_out, torch.nn.Identity)
            and not glue.transition_c_1.use_layer_norm
            and not glue.transition_c_2.use_layer_norm):
        # Every feature is constant across valid tokens per example: evaluate transitions on [K, 1, d]
        # in exact host fp32 (~0.5 ms, bit-identical to glue.conditioning), then tile-broadcast on card.
        ones_m = mask.any(dim=-1, keepdim=True).to(mask.dtype)  # [K, 1]
        u_devs = []
        tok_m = mk[0]
        for r in uniq_rems:
            parts = []
            for idx, f in enumerate(inv_2d):
                if idx == rec_idx:
                    f = cf.feat_creators[idx]({**bt, "recycle_remaining": r})
                parts.append(f)
            f2 = torch.cat(parts, dim=-1) * ones_m              # [K, D_in]
            c2 = cf.linear_out(f2).unsqueeze(1)                 # [K, 1, dim_cond]
            c2 = glue.transition_c_2(glue.transition_c_1(c2, ones_m), ones_m)
            if n % 32 == 0:
                c_t = ttnn.from_torch(
                    c2.expand(K, 32, -1).to(torch.float32).contiguous(),
                    dtype=out_dtype, layout=ttnn.TILE_LAYOUT,
                    device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                c_rep = ttnn.repeat(c_t, (1, n // 32, 1)) if n > 32 else c_t
                if c_rep is not c_t:
                    ttnn.deallocate(c_t)
                if tok_m is not None:
                    c_full = ttnn.multiply(c_rep, tok_m)
                    ttnn.deallocate(c_rep)
                else:
                    c_full = c_rep
            else:
                c_host = (c2.expand(K, n, -1) * mask.unsqueeze(-1)).to(torch.float32).contiguous()
                c_full = ttnn.from_torch(
                    c_host, dtype=out_dtype, layout=ttnn.TILE_LAYOUT,
                    device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            u_devs.append(c_full)
        return [u_devs[ui] for ui in cyc_to_u]

    # General 3D feature path: batched fp32 transition_c_1 + transition_c_2 on device.
    k_sw1 = "glue.transition_c_1.swish_linear.0.weight"
    k_lo1 = "glue.transition_c_1.linear_out.weight"
    k_sw2 = "glue.transition_c_2.swish_linear.0.weight"
    k_lo2 = "glue.transition_c_2.linear_out.weight"
    can_dev = (
        not glue.transition_c_1.use_layer_norm
        and not glue.transition_c_2.use_layer_norm
        and glue.transition_c_1.swish_linear[0].bias is None
        and glue.transition_c_1.linear_out.bias is None
        and glue.transition_c_2.swish_linear[0].bias is None
        and glue.transition_c_2.linear_out.bias is None
        and all(k in master for k in (k_sw1, k_lo1, k_sw2, k_lo2))
    )
    raw_u = []
    for r in uniq_rems:
        parts = []
        for idx, f in enumerate(inv_feats):
            if idx == rec_idx:
                f = cf.feat_creators[idx]({**bt, "recycle_remaining": r})
            if f.ndim == 2:
                f = f.reshape(K, 1, -1).expand(K, n, -1)
            parts.append(f)
        feats = cf.apply_padding_mask(torch.cat(parts, dim=-1), mask)
        proj = cf.linear_out(feats)
        c_raw = cf.apply_padding_mask(cf.ln_out(proj).to(proj.dtype), mask)
        raw_u.append(c_raw)
    if not can_dev:
        u_devs = [
            ttnn.from_torch(
                glue.transition_c_2(glue.transition_c_1(c, mask), mask).contiguous(),
                dtype=out_dtype, layout=ttnn.TILE_LAYOUT, device=dev,
            )
            for c in raw_u
        ]
        return [u_devs[ui] for ui in cyc_to_u]

    U = len(raw_u)
    x_stack = torch.cat(raw_u, dim=0).to(torch.float32).contiguous()  # [U*K, N, 256]
    x_dev = ttnn.from_torch(
        x_stack, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    g = dev.compute_with_storage_grid_size()
    cg = ttnn.CoreGrid(y=g.y, x=g.x)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    m3_owned = False
    if mk[4] is not None:
        m_base = mk[4] if mk[4].dtype == ttnn.float32 else ttnn.typecast(mk[4], ttnn.float32)
        m_base_owned = (m_base is not mk[4])
        if U > 1:
            m3 = ttnn.repeat(m_base, (U, 1, 1))
            if m_base_owned:
                ttnn.deallocate(m_base)
            m3_owned = True
        else:
            m3 = m_base
            m3_owned = m_base_owned
    else:
        m3 = None
    y1 = _device_transition_f32(x_dev, master[k_sw1], master[k_lo1], m3, ckc, cg)
    ttnn.deallocate(x_dev)
    y2 = _device_transition_f32(y1, master[k_sw2], master[k_lo2], m3, ckc, cg)
    ttnn.deallocate(y1)
    if m3_owned:
        ttnn.deallocate(m3)
    if out_dtype != ttnn.float32:
        y_out = ttnn.typecast(y2, out_dtype)
        ttnn.deallocate(y2)
    else:
        y_out = y2
    d_c = int(y_out.shape[-1])
    if U == 1:
        u_devs = [y_out]
    else:
        u_devs = [ttnn.slice(y_out, [ui * K, 0, 0], [(ui + 1) * K, n, d_c]) for ui in range(U)]
        ttnn.deallocate(y_out)
    return [u_devs[ui] for ui in cyc_to_u]


def _device_cand_init_repr(glue, bt, master: dict, mk, dev):
    """Build grad-off candidate ``s0`` (``[K, N, token_dim]`` fp32) on device via ``opt.master``."""
    import torch
    import ttnn

    irf = glue.init_repr_factory
    w_key = "glue.init_repr_factory.linear_out.weight"
    if (getattr(irf, "ret_zero", False)
            or not isinstance(irf.ln_out, torch.nn.Identity)
            or irf.linear_out.bias is not None
            or w_key not in master):
        s0 = irf(bt) * bt["padding_mask"][..., None]
        return ttnn.from_torch(
            s0.detach().to(torch.float32).contiguous(),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    mask = bt["padding_mask"]
    K, n = int(mask.shape[0]), int(mask.shape[1])
    parts = []
    for fc in irf.feat_creators:
        f = fc(bt)
        if f.ndim == 2:
            f = f.reshape(K, 1, -1).expand(K, n, -1)
        parts.append(f)
    feats = torch.cat(parts, dim=-1).to(torch.float32).contiguous()
    f_dev = ttnn.from_torch(
        feats, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if mk[4] is not None:
        m4 = mk[4] if mk[4].dtype == ttnn.float32 else ttnn.typecast(mk[4], ttnn.float32)
        fm = ttnn.multiply(f_dev, m4)
        if m4 is not mk[4]:
            ttnn.deallocate(m4)
        ttnn.deallocate(f_dev)
        f_dev = fm
    g = dev.compute_with_storage_grid_size()
    cg = ttnn.CoreGrid(y=g.y, x=g.x)
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    s0_dev = ttnn.matmul(f_dev, master[w_key], transpose_b=True, compute_kernel_config=ckc, core_grid=cg)
    ttnn.deallocate(f_dev)
    return s0_dev


def main(argv=None, on_step=None):
    """``on_step(row)`` is called with each step's row, the same dict that is printed. Tier 0
    (`rfd4_tier0`) hands it the status writer; a bare run hands nothing and only prints."""
    ap = argparse.ArgumentParser(prog="trunk_step")
    ap.add_argument("--n", type=int, default=256, help="token length; taken from --batch when one is given")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--depths", default="",
                    help="comma-separated per-step recycle depths (overrides batch metadata depth)")
    ap.add_argument("--depths-from", default="",
                    help="path to JSONL (n_recycle/depth per line, e.g. final-mat-nocg/steps.jsonl) or integer list")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--save-at", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--freeze", default="")
    ap.add_argument("--batch", default="")
    ap.add_argument("--exempt", default="")
    ap.add_argument("--out", default=os.path.expanduser("~/r4t/step/tck"))
    ap.add_argument("--curve-dump", default="", help="the G3 curve leg: a directory of plan-step "
                    "inputs from `curve_upstream.py --dump-dir` (plan.json beside them). Step i "
                    "trains on step<i>.npz at ITS drawn depth, in order, no cycling.")
    ap.add_argument("--save-every", type=int, default=0, help="checkpoint every N steps (newest "
                    "two kept), so a long leg resumes with --resume instead of starting over")
    ap.add_argument("--keep", type=int, default=0, help="with --save-every: keep the newest N "
                    "checkpoints by step, not by loss (a resume ring)")
    ap.add_argument("--pin-spike", type=float, default=0.0, help="pin, never prune, the "
                    "checkpoint a step started from when its loss exceeds F x the median of the "
                    "previous 100 (>= 20 seen), so the spike replays from it")
    ap.add_argument("--curve-out", default="", help="with --curve-dump: the CurveLog to append to, "
                    "the file grade_parity.py --ours reads")
    ap.add_argument("--dp-cards", default="")
    ap.add_argument("--shard", action="store_true")
    ap.add_argument("--dp-rank", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--dp-run", default="", help=argparse.SUPPRESS)
    ap.add_argument("--cand-pair", choices=("host", "card"), default="card",
                    help="where the grad-off K-candidate pass builds z_init: host (upstream's builder) or the serving "
                         "card path (rfd4.pair.PairOnCard: host constant + bins, card assembly).")
    ap.add_argument("--win-pair", choices=("host", "card"), default="card",
                    help="where the taped winner builds z_init and runs its backward: host (upstream's builder, torch "
                         "autograd, the pair and its gradient over PCIe) or the card (PairOnCard.taped; needs --cand-pair card)")
    ap.add_argument("--cand-levers", choices=("training", "serving", "inference", "serve", "serve_train"), default="serve",
                    help="lever set of the grad-off K-candidate pass: the training set, or that set plus the dropped levers that "
                         "switch at run time (pair_bf16z, dit_sigf32, tok_recycle_mm; r4t-perf K2). The pass only picks the "
                         "winner, so the gradient is unchanged whenever the winner is; inference also lets the fused kernels without a tape entry "
                         "serve that pass (ops.NOGRAD_IS_INFERENCE, r4t-perf K4); serve (r4t-perf K14) runs it under ag.suspended() "
                         "with the serving fused kernels; serve_train keeps training levers under ag.suspended()")
    ap.add_argument("--win-pre", choices=("tape", "suspend", "reuse"), default="suspend",
                    help="how win_trunk runs pre-final recycle cycles 0..depth-2 (r4t-bwd B1): tape (inside ag.tape() no_grad, "
                         "baseline), suspend (outside ag.tape() under ag.suspended() with serving fused kernels; passes D37 "
                         "loss parity on all 9 batches), or reuse (slice cand_trunk's penultimate (s_{d-1}[win], z_{d-1}[win]))")
    ap.add_argument("--break-win-pre", action="store_true",
                    help="B1 broken arm: negate the pre-final cycle-0 state s_prev entering cycle d-1 (must FAIL loss parity)")
    ap.add_argument("--phase-timing", action="store_true",
                    help="record synced per-phase wall seconds (cand_trunk_s, win_trunk_s, tape_bwd_s, win_plus_bwd_s) in JSON")
    ap.add_argument("--lnbw-weights", choices=("on", "off"), default="on",
                    help="LayerNorm backward with trainable gamma/beta in one fused kernel (tt_bio.lnbw weights form, "
                         "r4t-perf L2) or the composed ttnn path")
    ap.add_argument("--grad-dump", default="", help="write every trained parameter's gradient of the first "
                    "step (host glue + upstream trunk leaves, after the fold VJPs) to PATH.npz with PATH.json "
                    "(winner, batch, loss, clock) and stop before the optimizer: the device side of "
                    "perf/r4t_tape/net_gradcheck.py")
    ap.add_argument("--arm", choices=("prod", "fp32-exact"), default="prod",
                    help="fp32-exact: the D49 control, float32 trunk with no levers and every contraction "
                         "under tt_bio.train.exact_fp32 (host float64, rounded once)")
    ap.add_argument("--break", dest="brk", action="store_true",
                    help="with --grad-dump: the gross broken arm, block 0's Q slice dropped from its fold VJP")
    ap.add_argument("--attn-fp32", choices=("on", "off"), default="on",
                    help="q/k/v projections, q/k norms and attention in float32 (load_trunk attn_fp32). On by default: "
                         "with it off (bf16 attention) the RFD4 curve spikes at step 823 and fails G3; off is the opt-out")
    ap.add_argument("--gate-bw", choices=("on", "off"), default="on",
                    help="backward of every o * sigmoid(g) gate as one kernel (tt_bio.gate_bw, r4t-perf G1) or the "
                         "seven-op composed path")
    ap.add_argument("--cover-steps", type=int, default=0,
                    help="steps the no-gradient guard intersects over (default: the guard's own minimum). A run cycling "
                         "logged batches sets it to the cycle length: a head no early batch drives is not dead")
    ap.add_argument("--pcie-meter", choices=("off", "on", "sites"), default="off",
                    help="count host<->device bytes and transfers per step phase (perf/r4t_perf/pcie_meter.py); "
                         "sites adds the top call sites by bytes")
    ap.add_argument("--optimizer", choices=("auto", "muon", "adamw"), default="auto",
                    help="update rule: muon (upstream Muon + Adam, rfd4_objective.update_rule), "
                         "adamw (plain AdamW at 1e-4), or auto (muon for --curve-dump, adamw otherwise)")
    ap.add_argument("--resident", action="store_true",
                    help="device-resident training (D89): DeviceOptimizer (Muon+Adam+clip+EMA), "
                         "DeviceFolds (on-card fold_bwd+refold), and DeviceHeads + on-card insertion loss")
    ap.add_argument("--ns-mode", choices=("fp32", "hyb_e1", "split2_fast", "split2", "split2_hifi4"), default="fp32",
                    help="Newton-Schulz precision mode for DeviceOptimizer under --resident")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="EMA decay for DeviceOptimizer under --resident (<0 disables EMA)")
    ap.add_argument("--mesh", default="",
                    help="SPMD MeshDevice data-parallel grid over Galaxy Ethernet fabric (e.g. 1x4, 2x4, 4x4, 8x4); "
                         "implies --resident")
    ap.add_argument("--full-mesh", default="8x4",
                    help="physical Galaxy mesh shape for --mesh submesh opening (default: 8x4)")
    ap.add_argument("--spmd-zero1", dest="spmd_zero1", action="store_true", default=True,
                    help="enable Axis-0 SPMD K-sharded Muon ZeRO-1 under --mesh when mesh rows > 1 (default: on)")
    ap.add_argument("--no-spmd-zero1", dest="spmd_zero1", action="store_false",
                    help="disable Axis-0 SPMD Muon ZeRO-1 under --mesh (run full 2D all-reduce + replicated Muon)")
    ap.add_argument("--spmd-trace", dest="spmd_trace", action="store_true", default=True,
                    help="enable fenced MeshTrace replay for static SPMD phases under --mesh (default: on)")
    ap.add_argument("--no-spmd-trace", dest="spmd_trace", action="store_false",
                    help="disable MeshTrace replay under --mesh (eager per-op dispatch A/B reference)")
    ap.add_argument("--spmd-dev-prep", dest="spmd_dev_prep", action="store_true", default=True,
                    help="offload candidate cond_factory/transition_c/init_repr matmuls onto MeshDevice (default: on)")
    ap.add_argument("--no-spmd-dev-prep", dest="spmd_dev_prep", action="store_false",
                    help="run candidate cond_factory/transition_c/init_repr on host CPU under --mesh")
    ap.add_argument("--eth-dispatch", choices=("auto", "on", "off"), default="auto",
                    help="ETH dispatch cores under --mesh (auto=ETH on BH Galaxy for 12x10=120 Tensix cores)")
    ap.add_argument("--depth-override", type=int, default=0,
                    help="override per-step recycle depth to a fixed positive integer on real batches")
    ap.add_argument("--break-spmd-reduce", action="store_true",
                    help="G5 broken arm under --mesh: skip cross-rank CCL gradient reduction (must FAIL consensus/parity)")
    ap.add_argument("--consensus-every", type=int, default=0,
                    help="check bit-identical all-rank master consensus every N steps under --mesh (0 = final step only)")
    ap.add_argument("--dump-master-np", default="",
                    help="write final fp32 host master weights dict to PATH.npz for serial-sum vs SPMD parity checks")
    ap.add_argument("--serial-accum", type=int, default=1,
                    help="accumulate gradients over M consecutive batches on 1 chip before stepping (exact serial-sum reference)")
    a = ap.parse_args(argv)
    spmd_mesh_shape = None
    full_mesh_shape = None
    if a.mesh:
        from tt_bio.train.spmd_rfd4 import parse_mesh_arg
        spmd_mesh_shape = parse_mesh_arg(a.mesh)
        full_mesh_shape = parse_mesh_arg(a.full_mesh)
        if a.dp_cards:
            raise SystemExit("--mesh cannot be combined with --dp-cards")
        if a.curve_dump:
            raise SystemExit("--mesh cannot be combined with --curve-dump")
        if a.serial_accum != 1:
            raise SystemExit("--mesh cannot be combined with --serial-accum > 1")
        a.resident = True
    if a.serial_accum < 1:
        raise SystemExit("--serial-accum must be >= 1")
    if a.serial_accum > 1 and not a.resident:
        raise SystemExit("--serial-accum > 1 requires --resident")
    depths_seq = _load_depths(a.depths, a.depths_from)
    if a.arm == "fp32-exact":
        os.environ["TT_BIO_LEVERS"] = "none"   # Trunk builds on mode_levers(), which reads this, not lever_scope
        a.lnbw_weights = "off"   # the fused LN-backward kernel is a perf lever; the control arm verified by r4t-tape has none
        a.gate_bw = "off"        # likewise the fused gate backward (r4t-perf G1)
        a.cand_levers = "training"   # and the serving levers in the candidate pass (r4t-perf K2/K14)
        a.win_pair = "host"      # and the card pair-builder backward (r4t-perf P2)
        a.win_pre = "tape"       # exact control runs unmodified pre-final cycles
    if a.curve_dump and a.dp_cards:
        raise SystemExit("--curve-dump is a single-card leg: every rank would append to one CurveLog")
    if a.dp_cards and a.dp_rank < 0:
        return launch(a, argv)
    world = len(a.dp_cards.split(",")) if a.dp_cards else 1
    rank = max(a.dp_rank, 0)
    if a.curve_dump:
        from tt_bio.train.curve import CurveLog, read_dump_dir
        cdir = os.path.expanduser(a.curve_dump)
        cplan, cfiles = read_dump_dir(cdir)   # refuses a dump that is not this plan's
        if not a.curve_out:
            raise SystemExit("--curve-out is required with --curve-dump")
        a.batch = ",".join(str(f) for f in cfiles)
        a.steps = min(a.steps, len(cfiles)) if a.steps != 6 else len(cfiles)
    if a.batch:
        # A real batch fixes the token length. The trimul prewarm below fills the in-projection entry at
        # a.n; at any other width the winner uploads it inside its taped verb and those weights get no
        # gradient (a 64-token crop under the 256 default lost 56 in-projection grads).
        import numpy as _np
        lens = {int(_np.load(f)["in/padding_mask"].shape[1]) for f in a.batch.split(",")}
        if len(lens) != 1:
            raise SystemExit(f"--batch files disagree on token length: {sorted(lens)}")
        a.n = lens.pop()
    if a.pcie_meter != "off":
        import pcie_meter as PM   # perf/r4t_perf; the perf/r4t_step caller puts it on sys.path
        PM.install(sites=a.pcie_meter == "sites")
    else:
        PM = None
    mark = (lambda name: PM.phase(name)) if PM else (lambda name: None)
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1)   # kill -USR1 <pid> prints every thread stack to the log
    if spmd_mesh_shape is None:
        os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch
    import ttnn
    from tt_bio import tenstorrent as T, autograd as ag
    from tt_bio import lnbw
    lnbw.WEIGHTS = a.lnbw_weights == "on"
    from tt_bio import gate_bw
    gate_bw.GATE_BW_FUSED = a.gate_bw == "on"
    from tt_bio.rfd4 import network as N, engine as E, trunk as TR
    from tt_bio.train.folds import Folds
    from tt_bio.train.checkpoint import Checkpointer, load_adapter
    from tt_bio.train.optim import AdamW
    from tt_bio.train.parity import GradientCoverage, zero_or_missing
    import contextlib
    from tt_bio.train import exact_fp32 as EX
    TD = ttnn.float32 if a.arm == "fp32-exact" else None
    arm = contextlib.ExitStack()
    if TD is not None:   # the trimul follows the override; contractions go exact on the host
        arm.enter_context(T.device_dtype_override(ttnn.float32))
        arm.enter_context(EX.exact_fp32())

    use_spmd_trace = bool(spmd_mesh_shape is not None and a.spmd_trace and not a.grad_dump and not a.brk)
    if spmd_mesh_shape is not None:
        from tt_bio.train.mesh_dp import mesh_world, open_galaxy_submesh
        eth_opt = None if a.eth_dispatch == "auto" else (a.eth_dispatch == "on")
        dev = arm.enter_context(
            open_galaxy_submesh(
                spmd_mesh_shape,
                full_shape=full_mesh_shape,
                eth_dispatch=eth_opt,
                trace_region_size=(512 << 20) if use_spmd_trace else 0,
            )
        )
        world = mesh_world(dev)
        rank = 0
        card = 0
    else:
        dev = T.get_device()
        card = int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0])
    from tt_bio.train.provenance import open_nodes
    nodes = open_nodes()
    clk_nodes = list(nodes) if (spmd_mesh_shape is not None and nodes) else ([nodes[0]] if nodes else [card])
    clk, stop = [], threading.Event()

    def sample():
        paths = [f"/sys/class/tenstorrent/tenstorrent!{clk_node}/tt_aiclk" for clk_node in clk_nodes]
        while not stop.is_set():
            for path in paths:
                try:
                    clk.append(int(open(path).read().split()[0]))
                except Exception:
                    pass
            time.sleep(0.05)

    def used():
        v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        return int(v.total_bytes_allocated_per_bank) * int(v.num_banks)

    def put(t, dtype=ttnn.float32):
        return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
    cfg = config["nn"]
    glue = N.FlowGlue(state, config)
    folds = Folds(state)
    card_pair = []   # r4t-perf P1 (--cand-pair card): the candidate pass builds its pair as serving does

    def pair_card():
        """This step's PairOnCard, or None. Under --resident, on_refold refreshes card_pair[0]
        after each step; otherwise build a fresh PairOnCard each step."""
        if a.cand_pair != "card":
            return None
        if a.resident and card_pair:
            return card_pair[0]
        from tt_bio.rfd4.pair import PairOnCard, PairSplit
        if card_pair:
            card_pair.pop().release()
        glue.pair_split = PairSplit(glue.pair_repr_builder)
        card_pair.append(PairOnCard(glue.pair_split, dev))
        return card_pair[0]
    trunk = E.load_trunk(glue, state, cfg, training=True, dtype=TD, attn_fp32=a.attn_fp32 == "on")   # an unrecorded build for the lever set and dtypes; build() replaces it

    RB = {}

    # R4T_REBUILD_INLINE=1 (r4t-perf R1): a per-step rebuild skips the grad-off forward whose only job is to make
    # the lazy uploads happen under the recording. The recording stays open into the step's own forward instead,
    # and each lazy upload of a leaf-fed tensor is declared a parameter the moment it is made, before the taped
    # winner reads it. The first build and a resume build still run the forward: the unreached/orphan checks
    # after them need every weight to exist.
    inline = env_flag("R4T_REBUILD_INLINE", True)   # default ON since ca9fa7ac4: digest-identical to the lazy-forward rebuild (rb/I4.jsonl)
    pack = env_flag("R4T_DEVICE_PACK", False)   # r4t-perf R2: bfp8 weights packed on the card in the rebuild (Folds.recording device_pack)
    # The trimul in-projection is cached per chunk width on first use, and caches filled in the grad-off
    # candidate pass reach the taped winner as untracked constants (R1 first form: step-2 digest moved, every
    # linear_{a,b}_{p,g} grad dead). `train_in_proj` holds the two sources as registered device tensors and
    # cuts chunks per call, so nothing is cached. Implied by R4T_REBUILD_INLINE.
    in_proj = env_flag("R4T_TRAIN_IN_PROJ", False)   # train_in_proj rounds dW to bf16 (gcmp.txt); not implied

    def trimuls(root):
        seen, out, todo = set(), [], [root]
        while todo:
            o = todo.pop()
            if id(o) in seen:
                continue
            seen.add(id(o))
            if hasattr(o, "prewarm") and hasattr(o, "width_variants"):
                out.append(o)
            if isinstance(o, (list, tuple)):
                todo.extend(o)
            elif isinstance(o, dict):
                todo.extend(o.values())
            elif hasattr(o, "__dict__") and not isinstance(o, (torch.Tensor, ttnn.Tensor, type)):
                todo.extend(v for v in vars(o).values() if not isinstance(v, (torch.Tensor, ttnn.Tensor)))
        return out

    open_rec = []

    def walk_register():
        """Inline mode, between the grad-off candidates and the taped winner: register what the candidate pass
        cached (per-width uploads AND the device concats derived from them), as the stock rebuild's walk after
        its lazy forward does, before any taped read."""
        if open_rec:
            for _p, _o, _k, t in T.walk_device_weights(trunk):
                ag.parameter(t)

    def close_rec():
        while open_rec:
            open_rec.pop()()

    def build(lazy=True):
        """The serving trunk from the current upstream leaves, every upload recorded with its fold. Called
        again after each optimizer step: the device weights are derived, never stepped themselves."""
        nonlocal trunk
        _rb0 = time.time()
        if trunk is not None:
            for _p, _o, _k, t in T.walk_device_weights(trunk):
                ttnn.deallocate(getattr(t, "value", t))
            ag.release_pins()
            ag.forget_parameters()   # the old leaves point at freed handles; build() registers the new ones
        folds.sources.clear()
        folds.consts.clear()
        ttnn.synchronize_device(dev); _rb1 = time.time()
        close_rec()
        if lazy or not inline:
            with folds.recording(device_pack=pack) as sd:
                trunk = E.load_trunk(glue, sd, cfg, training=True, dtype=TD, attn_fp32=a.attn_fp32 == "on")
                if in_proj:
                    RB["trimuls"] = len([m.train_in_proj() for m in trimuls(trunk)])
                ttnn.synchronize_device(dev); _rb2 = time.time()
                one_example(0, False)   # lazy uploads (trimul in-projections at this chunk width), on the taped route
                ttnn.synchronize_device(dev); _rb3 = time.time()
        else:
            rec = folds.recording(device_pack=pack)
            sd = rec.__enter__()
            trunk = E.load_trunk(glue, sd, cfg, training=True, dtype=TD, attn_fp32=a.attn_fp32 == "on")
            if in_proj:
                RB["trimuls"] = len([m.train_in_proj() for m in trimuls(trunk)])
            # The trimul chunk width depends on the batch: the K-candidate pass fills its own width, and the
            # batch-1 winner would upload its in-projection INSIDE its taped verb, after the verb has resolved
            # its parents (R1 forms 1 and 3: 14 sources recorded, read, never on the tape). `prewarm` fills
            # exactly the entry `__call__` will use, so the winner width exists and is walked below.
            tmods = [m for m in trimuls(trunk) if hasattr(m, "prewarm")]
            RB["prewarmed"] = len([m.prewarm(a.n, 1) for m in tmods])
            if a.cand_levers != "serve":
                k_cand = cur["K"] if real else a.k
                if k_cand != 1:
                    [m.prewarm(a.n, k_cand) for m in tmods]
            ttnn.synchronize_device(dev); _rb2 = _rb3 = time.time()
            recorded = ttnn.from_torch

            def declare(x, *a_, **k_):
                t_ = recorded(x, *a_, **k_)
                if folds.sources.get(id(t_), (None,))[0] is t_:
                    ag.parameter(t_)
                return t_

            ttnn.from_torch = declare

            def _close():
                ttnn.from_torch = recorded
                rec.__exit__(None, None, None)

            open_rec.append(_close)
        for _p, _o, _k, t in T.walk_device_weights(trunk):
            ag.parameter(t)
        RB.update(free_s=round(_rb1 - _rb0, 4), load_s=round(_rb2 - _rb1, 4), lazy_fwd_s=round(_rb3 - _rb2, 4),
                  register_s=round(time.time() - _rb3, 4))

    levers = trunk._levers
    n, D, P, C_ = a.n, cfg["token_dim"], cfg["pair_repr_dim"], cfg["dim_cond"]
    g = torch.Generator().manual_seed(3)
    cond = torch.randn(1, n, C_, generator=g)
    Cs = torch.randn(1, n, D, generator=g)
    numel = Cs.numel()
    seed_s = put(Cs / numel)
    masks1 = TR.trunk_masks(torch.ones(1, n), dev, dtype=trunk.dtype, residual_dtype=trunk.residual_dtype)
    masksk = TR.trunk_masks(torch.ones(a.k, n), dev, dtype=trunk.dtype, residual_dtype=trunk.residual_dtype)
    cond1 = put(cond, trunk.dtype)
    condk = put(cond.expand(a.k, -1, -1), trunk.dtype)
    zd = trunk.pair_dtype(masks1)

    def inputs(step):
        gs = torch.Generator().manual_seed(1000 + step)
        s0 = torch.randn(a.k, n, D, generator=gs)
        z0 = torch.randn(a.k, n, n, P, generator=gs) * 0.5
        return s0, z0

    def winner(s_init, z_init, conds=None, masks=None, *, prev=None):
        """The production `Trunk.__call__` at depth d: under a tape every cycle but the last runs no_grad
        (`ops.recycle_region`, upstream's `recycle_grad: last`), so only the last pass is differentiated."""
        conds = [cond1] * a.depth if conds is None else conds
        masks = masks1 if masks is None else masks
        return trunk(s_init, z_init, conds, masks, prev=prev)

    def one_example(step, taped):
        step_d = int(depths_seq[step % len(depths_seq)]) if depths_seq else (
            int(a.depth_override) if a.depth_override > 0 else a.depth
        )
        if spmd_mesh_shape is not None:
            from tt_bio.train.mesh_dp import put_sharded
            s_parts, z_parts = [], []
            for r in range(world):
                sr, zr = inputs(step * world + r)
                s_parts.append(sr)
                z_parts.append(zr)
            s0_all = torch.cat(s_parts, dim=0)
            z0_all = torch.cat(z_parts, dim=0)
            with T.lever_scope(levers), ag.tape():
                with ag.no_grad():
                    sk, zk = trunk(put_sharded(s0_all, dev, dtype=ttnn.float32),
                                   put_sharded(z0_all, dev, dtype=zd),
                                   [condk] * step_d, masksk)
                for t in (sk, zk):
                    ttnn.deallocate(t.value if isinstance(t, ag.Tensor) else t)
                s_win = torch.cat([sr[0:1] for sr in s_parts], dim=0)
                z_win = torch.cat([zr[0:1] for zr in z_parts], dim=0)
                s_init = put_sharded(s_win, dev, dtype=ttnn.float32)
                z_init = put_sharded(z_win, dev, dtype=zd)
                if not taped:
                    with ag.no_grad():
                        return winner(s_init, z_init, [cond1] * step_d)
                return winner(s_init, z_init, [cond1] * step_d)
        s0, z0 = inputs(step)
        # Everything runs inside one tape so the walked (taped) weights meet taped verbs everywhere; the
        # k candidates are the sampler's batched inference call and stay grad-off.
        with T.lever_scope(levers), ag.tape():
            with ag.no_grad():
                sk, zk = trunk(put(s0), put(z0, zd), [condk] * step_d, masksk)
            win = 0   # the objective picks the winner from the heads' loss; heads are not bridged yet (see doc)
            for t in (sk, zk):
                ttnn.deallocate(t.value if isinstance(t, ag.Tensor) else t)
            s_init, z_init = put(s0[win:win + 1]), put(z0[win:win + 1], zd)
            if not taped:
                with ag.no_grad():
                    return winner(s_init, z_init, [cond1] * step_d)
            return winner(s_init, z_init, [cond1] * step_d)

    opt = None
    real = None
    if a.batch:
        import numpy as np
        from tt_bio.train.rfd4_data import load_batch
        from tt_bio.train import rfd4_objective as RO
        from tt_bio.train.bridge import Bridge, host_params
        batches = [(os.path.basename(f),) + load_batch(f) for f in a.batch.split(",")]
        cur = {}

        def use(step, *, depth_step=None):
            nm, b, m, _rec, rl = batches[step if a.curve_dump else step % len(batches)]
            ds = step if depth_step is None else depth_step
            if depths_seq:
                d_use = int(depths_seq[ds % len(depths_seq)])
            elif a.depth_override > 0:
                d_use = int(a.depth_override)
            else:
                d_use = max(1, int(m["depth"]))
            cur.update(name=nm, bt=b, meta=m, ref=rl, K=int(b["padding_mask"].shape[0]), depth=d_use)
        use(0)
        glue.requires_grad_(True)

        def take(d, i):
            if isinstance(d, dict):
                return {k: take(v, i) for k, v in d.items()}
            return d[i:i + 1] if torch.is_tensor(d) and d.dim() and d.shape[0] == cur["K"] else d

        def npy(d):
            return {k: npy(v) for k, v in d.items()} if isinstance(d, dict) else (
                d.detach().numpy() if torch.is_tensor(d) else d)

        def conds_for(b, depth_val=None):
            depth = cur["depth"] if depth_val is None else int(depth_val)
            cs = [glue.conditioning(b, None)]
            for cyc in range(1, depth):
                cs.append(glue.conditioning(b, depth - cyc - 1) if glue.condition_on_recycle_remaining else cs[0])
            return cs

        def flat(o):
            v = lambda m: next(iter(o[m].values()))
            out = {"v_bb_ca": v("bb_ca"), "v_local_latents": v("local_latents")}
            if "sequence" in o:
                out["sequence_logits"] = o["sequence"]["logits"]
            if "insertion" in o:
                out["insertion_logits"] = o["insertion"]["logits"]
            return out

        def tmask(b):
            if a.arm != "fp32-exact":
                return _device_trunk_masks(b["padding_mask"].float(), dev, dtype=trunk.dtype, residual_dtype=trunk.residual_dtype)
            return TR.trunk_masks(b["padding_mask"].float(), dev, dtype=trunk.dtype, residual_dtype=trunk.residual_dtype)

        @contextlib.contextmanager
        def cand_scope():
            """--cand-levers serving: the candidate pass adds the run-time levers training drops; pair_bf16z also
            carries the pair stream in bf16, a dtype the trunk and its pair updates read from attributes.
            --cand-levers serve (r4t-perf K14): also swaps `trunk._levers` (so `Trunk.__call__`'s `@under_lever_scope`
            does not overwrite the candidate lever set) and runs outside `ag.tape()` under `ag.suspended()`, unlocking
            the shipped serving fused kernels (`trimul_t1`, `pair_tail_cm`, `pair_relu`, `outer_residual`,
            `rms_pair_bias`, `rne_add`) on the grad-off candidate pass.
            --cand-levers serve_train: runs outside `ag.tape()` under `ag.suspended()` with `levers` (fp32 pair stream)."""
            if a.cand_levers in ("training", "serve_train"):
                yield
                return
            from tt_bio import ops as _ops
            nogi = _ops.NOGRAD_IS_INFERENCE
            _ops.NOGRAD_IS_INFERENCE = a.cand_levers == "inference" or nogi
            mods = [trunk] + list(trunk.pair_updates.values())
            saved = [(m, m.pair_residual_dtype if m is trunk else m.residual_dtype) for m in mods]
            saved_levers = trunk._levers
            cand_levs = levers | {"pair_bf16z", "dit_sigf32", "tok_recycle_mm"}
            trunk.pair_residual_dtype = ttnn.bfloat16
            for m in mods[1:]:
                m.residual_dtype = ttnn.bfloat16
            if a.cand_levers == "serve":
                trunk._levers = cand_levs
            try:
                with T.lever_scope(cand_levs):
                    yield
            finally:
                trunk._levers = saved_levers
                for m, d in saved:
                    setattr(m, "pair_residual_dtype" if m is trunk else "residual_dtype", d)
                _ops.NOGRAD_IS_INFERENCE = nogi

        def _slice_win(t, win):
            sh = [int(d) for d in t.shape]
            return ttnn.slice(t, [win] + [0] * (len(sh) - 1), [win + 1] + sh[1:])

        def real_example(taped):
            """Upstream's step on the dumped batch: k candidates grad-off through glue + trunk + heads, the
            winner by our own exploration loss, then the winner re-run with grad (host glue, taped trunk)."""
            bt, K = cur["bt"], cur["K"]
            mk = tmask(bt)
            use_res_pair = (res_pair is not None)
            use_dev_glue = (a.resident and a.arm != "fp32-exact" and opt is not None)
            pc = None if use_res_pair else pair_card()
            pair_cache = None
            phase = {}
            want_penult = a.win_pre == "reuse" and cur["depth"] > 1
            s0 = None
            z0 = None
            ck = None
            s0_in = None
            ck_in = None
            with torch.no_grad():
                if use_dev_glue:
                    ck_in = _device_cand_conds(glue, bt, cur["depth"], opt.master, mk, dev, trunk.dtype)
                    if use_res_pair:
                        enc_p = res_pair.encode_step(bt)
                        s0_in = _device_cand_init_repr(glue, bt, opt.master, mk, dev)
                    else:
                        s0, z0 = glue.initial_states(bt, pair_constant=glue.pair_split.constant(bt) if pc else None)
                else:
                    ck = conds_for(bt)
                    if use_res_pair:
                        enc_p = res_pair.encode_step(bt)
                        s0 = glue.init_repr_factory(bt) * bt["padding_mask"][..., None]
                    else:
                        s0, z0 = glue.initial_states(bt, pair_constant=glue.pair_split.constant(bt) if pc else None)
            win = None
            penult_k = (None, None)
            cand_outside = a.cand_levers in ("serve", "serve_train")
            if s0_in is None:
                s0_in = put(s0)
            if ck_in is None:
                ck_in = [put(c, trunk.dtype) for c in ck]
            if cand_outside:
                if a.phase_timing:
                    ttnn.synchronize_device(dev)
                    _tc0 = time.time()
                with ag.suspended(), cand_scope():
                    if use_res_pair:
                        zk0, pair_cache = res_pair.candidates(enc_p, trunk.pair_dtype(mk))
                    else:
                        zk0 = pc(z0, 0) if pc else put(z0, trunk.pair_dtype(mk))
                    if want_penult:
                        sk, zk, penult_k = trunk(s0_in, zk0, ck_in, mk, return_penultimate=True)
                    else:
                        sk, zk = trunk(s0_in, zk0, ck_in, mk)
                    if a.resident:
                        from tt_bio.train.rfd4_heads import device_candidate_scores
                        win = device_candidate_scores(glue, sk, bt, K)
                    else:
                        sh = ttnn.to_torch(sk).float().reshape(s0.shape)
                for t in (sk, zk):
                    ttnn.deallocate(t)
                if a.phase_timing:
                    ttnn.synchronize_device(dev)
                    phase["cand_trunk_s"] = round(time.time() - _tc0, 5)
            else:
                if a.phase_timing:
                    ttnn.synchronize_device(dev)
                    _tc0 = time.time()
                with T.lever_scope(levers), ag.tape(), ag.no_grad(), cand_scope():
                    if use_res_pair:
                        zk0, pair_cache = res_pair.candidates(enc_p, trunk.pair_dtype(mk))
                    else:
                        zk0 = pc(z0, 0) if pc else put(z0, trunk.pair_dtype(mk))
                    if want_penult:
                        sk, zk, penult_k = trunk(s0_in, zk0, ck_in, mk, return_penultimate=True)
                    else:
                        sk, zk = trunk(s0_in, zk0, ck_in, mk)
                    if a.resident:
                        from tt_bio.train.rfd4_heads import device_candidate_scores
                        win = device_candidate_scores(glue, getattr(sk, "value", sk), bt, K)
                    else:
                        sh = ttnn.to_torch(getattr(sk, "value", sk)).float().reshape(s0.shape)
                    for t in (sk, zk):
                        ttnn.deallocate(getattr(t, "value", t))
                penult_k = tuple(getattr(t, "value", t) if t is not None else None for t in penult_k)
                if a.phase_timing:
                    ttnn.synchronize_device(dev)
                    phase["cand_trunk_s"] = round(time.time() - _tc0, 5)
            _dealloc_unique([s0_in, *mk])
            keep_ck_for_pre = bool(taped and use_dev_glue and a.win_pre == "suspend" and cur["depth"] > 1)
            if not keep_ck_for_pre:
                _dealloc_unique(ck_in)
                ck_in = None
            if win is None:
                with torch.no_grad():
                    ok = {k: v.double().numpy() for k, v in flat(glue.heads(sh, bt)).items()}
                _t, bd, _s = RO.rfd4_loss({**npy(bt), "exploration_k": K}, ok, RO.WEIGHTS)
                win = int(bd["winners"][0])
            bw = take(bt, win)
            br = Bridge(dev, trunk.dtype)
            m1 = tmask(bw)
            if not taped:
                for t in penult_k:
                    if t is not None:
                        ttnn.deallocate(t)
                if pair_cache is not None:
                    pair_cache.release()
                if ck_in is not None:
                    _dealloc_unique(ck_in)
                with T.lever_scope(levers), ag.tape(), torch.no_grad(), ag.no_grad():
                    walk_register()
                    s_i, z_i = glue.initial_states(bw)
                    return winner(put(s_i), put(z_i, trunk.pair_dtype(m1)),
                                  [put(c, trunk.dtype) for c in conds_for(bw)], m1)
            # When pre-final cycles are reused or run grad-off via ck_in, only the final taped cycle
            # needs host autograd conditioning (earlier cycles were detached and carried zero grad).
            if keep_ck_for_pre or (want_penult and penult_k[0] is not None):
                last_rem = 0 if (cur["depth"] > 1 and glue.condition_on_recycle_remaining) else None
                cs = [glue.conditioning(bw, last_rem)]
            else:
                cs = conds_for(bw)
            use_card_pair = bool(pc and a.win_pair == "card")
            z_in_pre = None
            if use_res_pair:
                s_i = glue.init_repr_factory(bw) * bw["padding_mask"][..., None]
                gamma_w, beta_w = res_pair.modulation(bw)
                with T.lever_scope(levers):
                    win_pair_dt = trunk.pair_dtype(m1)
                z_in_pre = res_pair.winner_taped(pair_cache, win, gamma_w, beta_w, br, win_pair_dt)
                z_i = None
                zw = None
            elif use_card_pair:
                from tt_bio.rfd4.pair import PairInputs
                from tt_bio.train.rfd4_pair import pair_taped
                s_i = glue.init_repr_factory(bw) * bw["padding_mask"][..., None]
                gamma, beta = glue.pair_split.modulation(bw)
                zw = PairInputs(z0.constant, z0.bins[win:win + 1], z0.scale[win:win + 1], gamma, beta)
                z_i = None
            else:
                s_i, z_i = glue.initial_states(bw)
                zw = None
            prev = None
            keep = ()
            if a.phase_timing:
                ttnn.synchronize_device(dev)
                _tw0 = time.time()
            if want_penult and penult_k[0] is not None:
                s_prev = _slice_win(penult_k[0], win)
                z_prev = _slice_win(penult_k[1], win)
                ttnn.deallocate(penult_k[0])
                ttnn.deallocate(penult_k[1])
                if a.break_win_pre:
                    bad = ttnn.multiply(s_prev, -1.0)
                    ttnn.deallocate(s_prev)
                    s_prev = bad
                prev = (s_prev, z_prev)
                keep = prev
            elif a.win_pre == "suspend" and cur["depth"] > 1:
                with ag.suspended(), cand_scope():
                    sp_in = put(s_i.detach())
                    if use_res_pair:
                        want_dt = trunk.pair_dtype(m1)
                        zp_in = z_in_pre.value if z_in_pre.value.dtype == want_dt else ttnn.typecast(z_in_pre.value, want_dt)
                        zp_owned = (zp_in is not z_in_pre.value)
                    elif use_card_pair:
                        zw_ng = PairInputs(z0.constant, z0.bins[win:win + 1], z0.scale[win:win + 1],
                                           gamma.detach(), beta.detach())
                        zp_in = pc(zw_ng, 0)
                        zp_owned = True
                    else:
                        zp_in = put(z_i.detach(), trunk.pair_dtype(m1))
                        zp_owned = True
                    if keep_ck_for_pre and ck_in is not None:
                        cp_in = [_slice_win(c, win) for c in ck_in[:-1]]
                        _dealloc_unique(ck_in)
                        ck_in = None
                    else:
                        cp_in = [put(c.detach(), trunk.dtype) for c in cs[:-1]]
                    s_prev, z_prev = trunk(sp_in, zp_in, cp_in, m1)
                    _dealloc_unique([sp_in, *cp_in])
                    if zp_owned:
                        ttnn.deallocate(zp_in)
                if a.break_win_pre:
                    bad = ttnn.multiply(s_prev, -1.0)
                    ttnn.deallocate(s_prev)
                    s_prev = bad
                prev = (s_prev, z_prev)
                keep = prev
            if ck_in is not None:
                _dealloc_unique(ck_in)
                ck_in = None
            with T.lever_scope(levers), ag.tape():
                walk_register()
                if use_res_pair:
                    z_in = z_in_pre
                elif use_card_pair:
                    z_in = pair_taped(pc, zw, br, bw, trunk.pair_dtype(m1))
                else:
                    z_in = br.up(z_i, trunk.pair_dtype(m1))
                c_arg = [br.up(cs[-1])] if prev is not None else [br.up(c) for c in cs]
                s, z = winner(br.up(s_i, ttnn.float32), z_in, c_arg, m1, prev=prev)
            if a.phase_timing:
                ttnn.synchronize_device(dev)
                phase["win_trunk_s"] = round(time.time() - _tw0, 5)
            return s, z, dict(br=br, bw=bw, win=win, shape=s_i.shape, keep=(*keep, *m1), phase=phase)

        def real_loss(s, ctx):
            """rfd4_loss on our heads; under --resident runs DeviceHeads + on-card insertion loss."""
            br, bw = ctx["br"], ctx["bw"]
            if a.resident:
                from tt_bio.train.rfd4_heads import device_winner_loss
                out = device_winner_loss(
                    glue, s if isinstance(s, ag.Tensor) else ag.Tensor(s, requires_grad=True),
                    bw, bridge=br, ref_terms=cur["ref"], win_idx=ctx["win"],
                )
                _dealloc_unique(ctx.get("keep", ()))
                return out
            sh = br.down(s if isinstance(s, ag.Tensor) else ag.Tensor(s, requires_grad=True)).reshape(ctx["shape"])
            o = flat(glue.heads(sh, bw))
            total, bd, seeds = RO.rfd4_loss({**npy(bw), "exploration_k": 1},
                                            {k: v.detach().double().numpy() for k, v in o.items()}, RO.WEIGHTS)
            sur = sum((o[k] * torch.from_numpy(np.asarray(g)).to(o[k].dtype)).sum() for k, g in seeds.items())
            if a.phase_timing:
                _orig_bwd = ag.backward
                _tape_t = [0.0]

                def _timed_bwd(*ba, **bk):
                    ttnn.synchronize_device(dev)
                    _t0 = time.time()
                    try:
                        return _orig_bwd(*ba, **bk)
                    finally:
                        ttnn.synchronize_device(dev)
                        _tape_t[0] += time.time() - _t0

                ag.backward = _timed_bwd
                try:
                    br.backward(sur)
                finally:
                    ag.backward = _orig_bwd
                tb = round(_tape_t[0], 5)
                ctx["phase"]["tape_bwd_s"] = tb
                ctx["phase"]["win_plus_bwd_s"] = round(ctx["phase"].get("win_trunk_s", 0.0) + tb, 5)
            else:
                br.backward(sur)
            _dealloc_unique(ctx.get("keep", ()))
            terms = {k: v.get("value") for k, v in bd.items() if isinstance(v, dict)}
            ref = {k: float(cur["ref"][k][ctx["win"]]) for k in terms if k in cur["ref"]
                   and ctx["win"] < len(cur["ref"][k])}
            return total, terms, ref

        _spmd_pool = None

        def _get_spmd_pool():
            nonlocal _spmd_pool
            if _spmd_pool is None:
                from concurrent.futures import ThreadPoolExecutor
                _spmd_pool = ThreadPoolExecutor(max_workers=min(32, max(1, world)))
            return _spmd_pool

        def _trunk_masks_spmd(mask_2d: torch.Tensor):
            """Build exact sharded trunk masks on card from one [world*B, 1, n] upload (~0.5 ms)."""
            from tt_bio.train.device_optim import _safe_dealloc
            from tt_bio.train.mesh_dp import put_sharded
            m = mask_2d.to(torch.float32)
            if TR.ELIDE_FULL_MASKS and bool((m == 1).all()):
                return None, None, None, None, None, None
            b_local = int(m.shape[0]) // world
            n_tok = int(m.shape[1])
            dt = trunk.dtype
            # Single tiny [world*B, 1, n] upload (n % 32 == 0 -> card-tilized in parallel);
            # avoids serial host CPU tilizing of [world*B, n, 1, 1] (~99 ms/step on 8x4).
            tok_row = put_sharded(m[:, None, :], dev, dtype=dt, layout=ttnn.TILE_LAYOUT,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
            tok = ttnn.transpose(tok_row, -2, -1)
            pair_3d = ttnn.multiply(tok, tok_row)
            ttnn.deallocate(tok_row)
            pair_c = ttnn.unsqueeze(pair_3d, -1)
            one_minus = ttnn.rsub(pair_3d, 1.0)
            neg = ttnn.multiply(one_minus, TR.NEG_INF)
            ttnn.deallocate(one_minus)
            attn = ttnn.reshape(neg, (b_local, 1, n_tok, n_tok))
            _safe_dealloc(neg, keep=attn)
            res = trunk.residual_dtype or dt
            if res == dt:
                res_tok, res_pair_c = tok, pair_c
            else:
                res_tok, res_pair_c = ttnn.typecast(tok, res), ttnn.typecast(pair_c, res)
            if bool(m.all()):
                _safe_dealloc(pair_3d, keep=pair_c)
                pair_3d = None
            return tok, attn, pair_3d, pair_c, res_tok, res_pair_c

        def real_example_spmd(step):
            """SPMD data-parallel real step across all ``world`` chips of a MeshDevice."""
            from tt_bio.train.mesh_dp import (
                MeshBridge, put_sharded, select_per_rank_candidate,
            )
            from tt_bio.train.rfd4_heads import device_candidate_scores_spmd
            from tt_bio.train.spmd_rfd4 import WINNER_SMALL_KEYS, gather_winner_batches, select_rank_batches
            from tt_bio.train.spmd_trace import can_offload_cond_init, cand_conds_and_s0_device

            if res_pair is None:
                raise RuntimeError("--mesh real-batch path requires ResidentPairBuilder (--cand-pair card --win-pair card)")
            use_dev_prep = bool(a.spmd_dev_prep and can_offload_cond_init(glue))
            use_dev_win = bool(dev_win_glue is not None)
            batch_pool = [it[1] for it in batches]
            batch_indices, rank_bts, stacked_bt = select_rank_batches(
                batch_pool, step, world, stack=not use_dev_prep,
            )
            pref = spmd_prefetcher.get(step, rank_bts) if spmd_prefetcher is not None else None
            # Defer submitting step+1 prefetch until replay_win_pair_bwd_fold launches (~337 ms pure C++
            # trace + ~170 ms opt/refold), preventing 4-thread GIL contention against cand_pair / cand_prep.
            _prefetch_submitted = [False]

            def _submit_next_prefetch():
                if not _prefetch_submitted[0] and spmd_prefetcher is not None and (step + 1) < a.steps:
                    _prefetch_submitted[0] = True
                    _, next_rank_bts, _ = select_rank_batches(batch_pool, step + 1, world, stack=False)
                    spmd_prefetcher.submit(step + 1, next_rank_bts)
            rank_items = [batches[i] for i in batch_indices]
            rank_names = [it[0] for it in rank_items]
            rank_metas = [it[2] for it in rank_items]
            rank_refs = [it[4] for it in rank_items]
            K = int(rank_bts[0]["padding_mask"].shape[0])
            if depths_seq:
                step_depth = int(depths_seq[step % len(depths_seq)])
            elif a.depth_override > 0:
                step_depth = int(a.depth_override)
            else:
                step_depth = max(1, int(rank_metas[0]["depth"]))
            cur.update(name=rank_names[0], bt=rank_bts[0], meta=rank_metas[0], ref=rank_refs[0], K=K, depth=step_depth)

            mask_cand = (
                torch.cat([b["padding_mask"] for b in rank_bts], dim=0)
                if use_dev_prep else stacked_bt["padding_mask"]
            )
            mk = _trunk_masks_spmd(mask_cand)
            phase = {}
            want_penult = a.win_pre == "reuse" and step_depth > 1
            pool = _get_spmd_pool()
            if a.phase_timing:
                _te0 = time.time()
            if use_dev_prep:
                with torch.no_grad():
                    if pref is not None and pref.static_encs is not None:
                        encs = pref.static_encs
                        if pref.pair_cond_feats is not None and all(x is not None for x in pref.pair_cond_feats):
                            pcf_all = torch.cat(pref.pair_cond_feats, dim=0)
                            g_all, b_all = res_pair.modulation_from_cond_feats(pcf_all, mask_cand)
                            for r_i, enc in enumerate(encs):
                                enc.gamma = g_all[r_i * K:(r_i + 1) * K]
                                enc.beta = b_all[r_i * K:(r_i + 1) * K]
                        else:
                            for r_i, enc in enumerate(encs):
                                enc.gamma, enc.beta = res_pair.modulation(rank_bts[r_i])
                    else:
                        encs = list(pool.map(res_pair.encode_step, rank_bts))
                ck = None
                s0 = None
            else:
                with torch.no_grad():
                    ck = conds_for(stacked_bt, step_depth)
                    encs = list(pool.map(res_pair.encode_step, rank_bts))
                    s0 = glue.init_repr_factory(stacked_bt) * stacked_bt["padding_mask"][..., None]
            if a.phase_timing:
                phase["enc_s"] = round(time.time() - _te0, 5)

            penult_k = (None, None)
            cand_outside = a.cand_levers in ("serve", "serve_train")
            cand_meta = pref.cand_meta if pref is not None else None
            if a.phase_timing:
                ttnn.synchronize_device(dev)
                _tc0 = time.time()
            pair_st = pref.pair_stacked if pref is not None else None
            if cand_outside:
                with ag.suspended(), cand_scope():
                    zk0, pair_cache = res_pair.candidates_spmd(encs, trunk.pair_dtype(mk), pair_stacked=pair_st)
                    if a.phase_timing:
                        ttnn.synchronize_device(dev)
                        phase["cand_pair_s"] = round(time.time() - _tc0, 5)
                        _tp0 = time.time()
                    if use_dev_prep:
                        s0_tt, ck_tt = cand_conds_and_s0_device(
                            glue, rank_bts, mk[0], step_depth, trunk.dtype, dev,
                            pool=pool, dev_masters=opt.master,
                            prefetched=pref, factory_split=factory_split,
                        )
                    else:
                        s0_tt = put_sharded(s0, dev, dtype=ttnn.float32)
                        ck_tt = [put_sharded(c, dev, dtype=trunk.dtype) for c in ck]
                    if a.phase_timing:
                        ttnn.synchronize_device(dev)
                        phase["cand_prep_s"] = round(time.time() - _tp0, 5)
                        _tk0 = time.time()
                    if spmd_tracer is not None:
                        res_cand = spmd_tracer.run_untaped_trunk(
                            "cand", trunk, s0_tt, zk0, ck_tt, mk, return_penultimate=want_penult,
                        )
                    elif want_penult:
                        res_cand = trunk(s0_tt, zk0, ck_tt, mk, return_penultimate=True)
                    else:
                        res_cand = trunk(s0_tt, zk0, ck_tt, mk)
                    if want_penult:
                        sk, zk, penult_k = res_cand
                    else:
                        sk, zk = res_cand
                    if a.phase_timing:
                        ttnn.synchronize_device(dev)
                        phase["cand_run_s"] = round(time.time() - _tk0, 5)
                        _ts0 = time.time()
                    win_indices = device_candidate_scores_spmd(
                        glue, sk, rank_bts, K, pool=pool, cand_meta=cand_meta,
                    )
                    if a.phase_timing:
                        phase["cand_score_s"] = round(time.time() - _ts0, 5)
                for t in (sk, zk, s0_tt, zk0):
                    if t is not None and t.is_allocated():
                        ttnn.deallocate(t)
                _dealloc_unique(mk)
            else:
                with T.lever_scope(levers), ag.tape(), ag.no_grad(), cand_scope():
                    zk0, pair_cache = res_pair.candidates_spmd(encs, trunk.pair_dtype(mk), pair_stacked=pair_st)
                    if use_dev_prep:
                        s0_tt, ck_tt = cand_conds_and_s0_device(
                            glue, rank_bts, mk[0], step_depth, trunk.dtype, dev,
                            pool=pool, dev_masters=opt.master,
                            prefetched=pref, factory_split=factory_split,
                        )
                    else:
                        s0_tt = put_sharded(s0, dev, dtype=ttnn.float32)
                        ck_tt = [put_sharded(c, dev, dtype=trunk.dtype) for c in ck]
                    if want_penult:
                        sk, zk, penult_k = trunk(s0_tt, zk0, ck_tt, mk, return_penultimate=True)
                    else:
                        sk, zk = trunk(s0_tt, zk0, ck_tt, mk)
                    win_indices = device_candidate_scores_spmd(
                        glue, getattr(sk, "value", sk), rank_bts, K, pool=pool, cand_meta=cand_meta,
                    )
                    for t in (sk, zk, s0_tt, zk0):
                        tv = getattr(t, "value", t)
                        if tv is not None and tv.is_allocated():
                            ttnn.deallocate(tv)
                    _dealloc_unique(mk)
                penult_k = tuple(getattr(t, "value", t) if t is not None else None for t in penult_k)
            if a.phase_timing:
                ttnn.synchronize_device(dev)
                phase["cand_trunk_s"] = round(time.time() - _tc0, 5)

            if a.phase_timing:
                _twp0 = time.time()
            win_keys = WINNER_SMALL_KEYS if (use_dev_win and pref is not None) else None
            rank_bws, stacked_bw = gather_winner_batches(rank_bts, win_indices, keys=win_keys)
            br = MeshBridge(dev, trunk.dtype)
            m1 = _trunk_masks_spmd(stacked_bw["padding_mask"])
            slice_pre_conds = bool(use_dev_prep and a.win_pre == "suspend" and step_depth > 1)
            if slice_pre_conds:
                cp_in_sliced = [select_per_rank_candidate(c, win_indices, dev) for c in ck_tt[:-1]]
                _dealloc_unique(ck_tt)
            else:
                _dealloc_unique(ck_tt)
                cp_in_sliced = None

            glue_saved = None
            if use_dev_win:
                win_rem = 0 if (step_depth > 1 and glue.condition_on_recycle_remaining) else None
                s_i_dev, c_last_dev, glue_saved = dev_win_glue.forward(
                    stacked_bw, win_indices, K,
                    prefetched=pref, m1_tok=m1[0], recycle_remaining=win_rem,
                    trunk_dtype=trunk.dtype, dev_masters=opt.master,
                )
                s_i = None
                cs = None
            else:
                if slice_pre_conds:
                    cs = [glue.conditioning(stacked_bw, 0 if glue.condition_on_recycle_remaining else None)]
                else:
                    cs = conds_for(stacked_bw, step_depth)
                s_i = glue.init_repr_factory(stacked_bw) * stacked_bw["padding_mask"][..., None]
                s_i_dev = None
                c_last_dev = None

            if (
                pref is not None
                and pref.pair_cond_feats is not None
                and all(x is not None for x in pref.pair_cond_feats)
            ):
                w_pcf = torch.cat(
                    [pref.pair_cond_feats[r_i][int(w_r):int(w_r) + 1] for r_i, w_r in enumerate(win_indices)],
                    dim=0,
                )
                gamma_w, beta_w = res_pair.modulation_from_cond_feats(w_pcf, stacked_bw["padding_mask"])
            else:
                gamma_w, beta_w = res_pair.modulation(stacked_bw)
            if a.phase_timing:
                ttnn.synchronize_device(dev)
                phase["win_prep_s"] = round(time.time() - _twp0, 5)
                _twz0 = time.time()
            with T.lever_scope(levers):
                win_pair_dt = trunk.pair_dtype(m1)
            z_in_pre = res_pair.winner_taped_spmd(pair_cache, encs, win_indices, gamma_w, beta_w, br, win_pair_dt)
            if a.phase_timing:
                ttnn.synchronize_device(dev)
                phase["win_pair_s"] = round(time.time() - _twz0, 5)

            prev = None
            keep = ()
            if a.phase_timing:
                ttnn.synchronize_device(dev)
                _tw0 = time.time()
            if want_penult and penult_k[0] is not None:
                s_prev = select_per_rank_candidate(penult_k[0], win_indices, dev)
                z_prev = select_per_rank_candidate(penult_k[1], win_indices, dev)
                ttnn.deallocate(penult_k[0])
                ttnn.deallocate(penult_k[1])
                if a.break_win_pre:
                    bad = ttnn.multiply(s_prev, -1.0)
                    ttnn.deallocate(s_prev)
                    s_prev = bad
                prev = (s_prev, z_prev)
                keep = prev
            elif a.win_pre == "suspend" and step_depth > 1:
                with ag.suspended(), cand_scope():
                    sp_in = s_i_dev if use_dev_win else put_sharded(s_i.detach(), dev, dtype=ttnn.float32)
                    sp_in_owned = not use_dev_win
                    want_dt = trunk.pair_dtype(m1)
                    zp_in = (z_in_pre.value if z_in_pre.value.dtype == want_dt
                             else ttnn.typecast(z_in_pre.value, want_dt))
                    zp_owned = (zp_in is not z_in_pre.value)
                    cp_in = (
                        cp_in_sliced if cp_in_sliced is not None
                        else [put_sharded(c.detach(), dev, dtype=trunk.dtype) for c in cs[:-1]]
                    )
                    if spmd_tracer is not None:
                        s_prev, z_prev = spmd_tracer.run_untaped_trunk("win_pre", trunk, sp_in, zp_in, cp_in, m1)
                    else:
                        s_prev, z_prev = trunk(sp_in, zp_in, cp_in, m1)
                    for t in cp_in:
                        ttnn.deallocate(t)
                    if sp_in_owned:
                        ttnn.deallocate(sp_in)
                    if zp_owned:
                        ttnn.deallocate(zp_in)
                if a.break_win_pre:
                    bad = ttnn.multiply(s_prev, -1.0)
                    ttnn.deallocate(s_prev)
                    s_prev = bad
                prev = (s_prev, z_prev)
                keep = prev

            if use_dev_win:
                s_i_br = ag.Tensor(s_i_dev, requires_grad=True)
                c_br = [ag.Tensor(c_last_dev, requires_grad=True)]
            else:
                s_i_br = br.up(s_i, ttnn.float32)
                c_br = [br.up(cs[-1])] if prev is not None else [br.up(c) for c in cs]
            c_vals = [cb.value for cb in c_br]
            win_key = (
                spmd_tracer.win_pair_key(s_i_br.value, z_in_pre.value, c_vals, m1, prev)
                if spmd_tracer is not None else None
            )
            win_extra = {}
            if spmd_tracer is not None and not spmd_tracer.has_win_pair(win_key):
                donor = spmd_tracer.find_win_pair_donor(win_key)
                if donor is not None:
                    spmd_tracer.capture_win_pair_direct(
                        win_key,
                        donor=donor,
                        levers=levers,
                        winner=winner,
                        dfolds=dfolds,
                        dev_masters=opt.master,
                        s_i_val=s_i_br.value,
                        z_in_val=z_in_pre.value,
                        c_vals=c_vals,
                        m1=m1,
                        prev=prev,
                    )
            if spmd_tracer is not None and spmd_tracer.has_win_pair(win_key):
                s = spmd_tracer.replay_win_pair_fwd(
                    win_key, s_i_br.value, z_in_pre.value, c_vals, m1, prev,
                )
                z = None
                win_extra = dict(
                    win_mode="replay", win_key=win_key, s_i_br=s_i_br, z_in_pre=z_in_pre,
                    c_br=c_br, m1=m1, prev=prev, glue_saved=glue_saved,
                )
            elif spmd_tracer is not None and spmd_tracer.win_pair_should_capture(win_key):
                s_leaf = ag.Tensor(s_i_br.value, requires_grad=True)
                z_leaf = ag.Tensor(z_in_pre.value, requires_grad=True)
                c_leaves = [ag.Tensor(cb.value, requires_grad=True) for cb in c_br]
                with T.lever_scope(levers), ag.tape():
                    s, z = winner(s_leaf, z_leaf, c_leaves, m1, prev=prev)
                win_extra = dict(
                    win_mode="capture", win_key=win_key, s_leaf=s_leaf, z_leaf=z_leaf,
                    c_leaves=c_leaves, s_i_br=s_i_br, z_in_pre=z_in_pre, c_br=c_br,
                    m1=m1, prev=prev, z_out=z, glue_saved=glue_saved,
                )
            else:
                with T.lever_scope(levers), ag.tape():
                    s, z = winner(s_i_br, z_in_pre, c_br, m1, prev=prev)
                win_extra = dict(
                    win_mode="eager", m1=m1, s_i_br=s_i_br, c_br=c_br, glue_saved=glue_saved,
                )
            if a.phase_timing:
                ttnn.synchronize_device(dev)
                phase["win_trunk_s"] = round(time.time() - _tw0, 5)
            return s, z, dict(
                br=br, rank_bws=rank_bws, stacked_bw=stacked_bw, rank_names=rank_names,
                rank_metas=rank_metas, rank_refs=rank_refs, win_indices=win_indices,
                step_depth=step_depth, keep=keep, phase=phase,
                submit_next_prefetch=_submit_next_prefetch, **win_extra,
            )

        def real_loss_spmd(s, ctx):
            import gc as _gc
            from tt_bio.train.rfd4_heads import device_winner_loss_spmd

            if a.phase_timing:
                ttnn.synchronize_device(dev)
                _tb0 = time.time()
            win_mode = ctx.get("win_mode", "eager")
            glue_saved = ctx.get("glue_saved")

            def _merge_trunk_head_grads(trunk_g, head_g, skip_init, head_skip=()):
                merged = dict(trunk_g)
                skip_out = set(skip_init)
                for kh, gh in head_g.items():
                    if kh in merged:
                        sg = ttnn.add(merged[kh], gh)
                        if kh not in skip_out:
                            ttnn.deallocate(merged[kh])
                        if kh not in head_skip:
                            ttnn.deallocate(gh)
                        merged[kh] = sg
                        skip_out.discard(kh)
                    else:
                        merged[kh] = gh
                        if kh in head_skip:
                            skip_out.add(kh)
                return merged, skip_out

            if win_mode == "replay":
                head_grads_box: dict = {}
                head_skip_box: set = set()
                mean_total, per_rank_totals, per_rank_terms, per_rank_refs, ds_out = device_winner_loss_spmd(
                    glue, s, ctx["rank_bws"], bridge=ctx["br"], ref_terms_list=ctx["rank_refs"],
                    win_indices=ctx["win_indices"], stacked_bw=ctx["stacked_bw"], stop_at_s=True,
                    pool=_get_spmd_pool(),
                    head_tracer=spmd_tracer, dfolds=dfolds, dev_masters=opt.master,
                    head_grads_out=head_grads_box, head_skip_out=head_skip_box,
                )
                if head_grads_box:
                    head_grads = head_grads_box
                else:
                    head_grads, _ = dfolds.tape_backward(opt.master)
                    ag.release_pins()
                d_s_i, d_z_in, d_c, trunk_grads, trunk_idle = spmd_tracer.replay_win_pair_bwd_fold(
                    ctx["win_key"], ds_out,
                )
                ctx["submit_next_prefetch"]()
                glue_grads = {}
                if glue_saved is not None and dev_win_glue is not None:
                    glue_grads = dev_win_glue.backward(d_s_i, d_c[0], glue_saved, opt.master)
                else:
                    ctx["s_i_br"].add_grad(ttnn.clone(d_s_i))
                    for cb, dcb in zip(ctx["c_br"], d_c):
                        cb.add_grad(ttnn.clone(dcb))
                dz_c = ttnn.clone(d_z_in)
                ag.backward([ctx["z_in_pre"]], [dz_c])
                ttnn.deallocate(dz_c)
                ctx["br"].backward_ups()
                merged_g, merged_skip = _merge_trunk_head_grads(
                    trunk_grads, head_grads, set(trunk_grads), head_skip=head_skip_box,
                )
                if glue_grads:
                    merged_g, merged_skip = _merge_trunk_head_grads(merged_g, glue_grads, merged_skip)
                ctx["traced_dev_grads"] = merged_g
                ctx["traced_free_skip"] = merged_skip
                ctx["traced_idle"] = trunk_idle
                _dealloc_unique([
                    *ctx.get("keep", ()), *ctx.get("m1", ()),
                    ctx["s_i_br"].value, ctx["z_in_pre"].value,
                    *[cb.value for cb in ctx["c_br"]],
                ])
            elif win_mode == "capture":
                mean_total, per_rank_totals, per_rank_terms, per_rank_refs, ds_out = device_winner_loss_spmd(
                    glue, s, ctx["rank_bws"], bridge=ctx["br"], ref_terms_list=ctx["rank_refs"],
                    win_indices=ctx["win_indices"], stacked_bw=ctx["stacked_bw"], stop_at_s=True,
                    pool=_get_spmd_pool(),
                )
                head_grads, _ = dfolds.tape_backward(opt.master)
                s_out_ref = ttnn.clone(s.value)
                ag.backward([s], [ds_out])
                d_s_i = ctx["s_leaf"].grad
                ctx["s_leaf"].grad = None
                d_z_in = ctx["z_leaf"].grad
                ctx["z_leaf"].grad = None
                d_c = []
                for cl in ctx["c_leaves"]:
                    d_c.append(cl.grad)
                    cl.grad = None
                trunk_grads, trunk_idle = dfolds.tape_backward(opt.master)
                # Free eager trunk outputs & tape pins before fenced TraceScratch capture.
                _dealloc_unique([getattr(s, "value", s), getattr(ctx.get("z_out"), "value", ctx.get("z_out"))])
                ctx["z_out"] = None
                ag.release_pins()
                _gc.collect()
                spmd_tracer.capture_win_pair(
                    ctx["win_key"],
                    levers=levers,
                    winner=winner,
                    dfolds=dfolds,
                    dev_masters=opt.master,
                    s_i_val=ctx["s_i_br"].value,
                    z_in_val=ctx["z_in_pre"].value,
                    c_vals=[cb.value for cb in ctx["c_br"]],
                    m1=ctx["m1"],
                    prev=ctx["prev"],
                    s_out_ref=s_out_ref,
                    ds_out_ref=ds_out,
                    d_s_i_ref=d_s_i,
                    d_z_in_ref=d_z_in,
                    d_c_refs=d_c,
                    trunk_dev_grads_ref=trunk_grads,
                    idle_ref=trunk_idle,
                )
                ttnn.deallocate(s_out_ref)
                ttnn.deallocate(ds_out)
                ctx["submit_next_prefetch"]()
                glue_grads = {}
                if glue_saved is not None and dev_win_glue is not None:
                    glue_grads = dev_win_glue.backward(d_s_i, d_c[0], glue_saved, opt.master)
                    ttnn.deallocate(d_s_i)
                    for dcb in d_c:
                        ttnn.deallocate(dcb)
                else:
                    ctx["s_i_br"].add_grad(d_s_i)
                    for cb, dcb in zip(ctx["c_br"], d_c):
                        cb.add_grad(dcb)
                ag.backward([ctx["z_in_pre"]], [d_z_in])
                ttnn.deallocate(d_z_in)
                ctx["br"].backward_ups()
                merged_g, merged_skip = _merge_trunk_head_grads(trunk_grads, head_grads, ())
                if glue_grads:
                    merged_g, merged_skip = _merge_trunk_head_grads(merged_g, glue_grads, merged_skip)
                ctx["traced_dev_grads"] = merged_g
                ctx["traced_free_skip"] = merged_skip
                ctx["traced_idle"] = trunk_idle
                _dealloc_unique([
                    *ctx.get("keep", ()), *ctx.get("m1", ()),
                    ctx["s_i_br"].value, ctx["z_in_pre"].value,
                    *[cb.value for cb in ctx["c_br"]],
                ])
            else:
                mean_total, per_rank_totals, per_rank_terms, per_rank_refs = device_winner_loss_spmd(
                    glue,
                    s if isinstance(s, ag.Tensor) else ag.Tensor(s, requires_grad=True),
                    ctx["rank_bws"],
                    bridge=ctx["br"],
                    ref_terms_list=ctx["rank_refs"],
                    win_indices=ctx["win_indices"],
                    stacked_bw=ctx["stacked_bw"],
                    pool=_get_spmd_pool(),
                )
                if glue_saved is not None and dev_win_glue is not None:
                    d_s_i = ctx["s_i_br"].grad
                    ctx["s_i_br"].grad = None
                    d_c0 = ctx["c_br"][0].grad
                    ctx["c_br"][0].grad = None
                    ctx["extra_dev_grads"] = dev_win_glue.backward(d_s_i, d_c0, glue_saved, opt.master)
                    _dealloc_unique([d_s_i, d_c0, ctx["s_i_br"].value, *[cb.value for cb in ctx["c_br"]]])
                _dealloc_unique([*ctx.get("keep", ()), *ctx.get("m1", ())])
                ctx["submit_next_prefetch"]()
            if a.phase_timing:
                ttnn.synchronize_device(dev)
                tb = round(time.time() - _tb0, 5)
                ctx["phase"]["tape_bwd_s"] = tb
                ctx["phase"]["win_plus_bwd_s"] = round(ctx["phase"].get("win_trunk_s", 0.0) + tb, 5)
            return mean_total, per_rank_terms[0], per_rank_refs[0], per_rank_totals

        real = (real_example, real_loss)

    ag.install()
    dfolds = None
    dheads = None
    res_pair = None
    host_glue_params = {}
    refold_hook = None
    refold_dev_hook = None
    spmd_opt = None
    spmd_tracer = None
    factory_split = None
    dev_win_glue = None
    spmd_prefetcher = None
    if a.resident:
        from tt_bio.rfd4.heads import DeviceHeads
        from tt_bio.train.device_optim import DeviceOptimizer, rfd4_device_optimizer
        from tt_bio.train.folds import DeviceFolds, make_rfd4_refold_hook
        from tt_bio.train.spmd_trace import (
            DEVICE_GLUE_WEIGHTS, DeviceWinnerGlue, FactoryStaticSplit,
            SPMDPrefetcher, can_offload_cond_init,
        )

        # Free unrecorded dummy trunk from line 229 before building the single persistent trunk.
        if trunk is not None:
            for _p, _o, _k, t in T.walk_device_weights(trunk):
                ttnn.deallocate(getattr(t, "value", t))
            ag.release_pins()
            ag.forget_parameters()
        state_res = dict(state)
        if real:
            for gn, gp in glue.named_parameters():
                if gp.requires_grad:
                    state_res[f"glue.{gn}"] = gp.detach()
        dfolds = DeviceFolds(state_res, dev)
        with dfolds.recording(device_pack=pack) as sd:
            trunk = E.load_trunk(glue, sd, cfg, training=True, dtype=TD, attn_fp32=a.attn_fp32 == "on")
            RB["prewarmed"] = len([m.prewarm(a.n, 1) for m in trimuls(trunk) if hasattr(m, "prewarm")])
            if real:
                dheads = DeviceHeads(glue, dev, state=sd)
                glue._device_heads = dheads
        dfolds.bind(trunk, *([dheads] if dheads is not None else []))
        for spec in dfolds.specs:
            ag.parameter(spec.handle)
        unreached = set(dfolds.unreached())
        folded_names = [nm for nm in dfolds.leaves if nm not in unreached]
        folded_set = set(folded_names)
        if real:
            host_glue_params = {
                f"glue.{gn}": gp for gn, gp in glue.named_parameters()
                if gp.requires_grad and f"glue.{gn}" not in folded_set
            }
        init_w = {nm: dfolds.leaves[nm].detach() for nm in folded_names}
        init_w.update({nm: p.detach() for nm, p in host_glue_params.items()})
        params = init_w
        if a.freeze:
            fz = next(nm for nm in sorted(params) if a.freeze in nm)
            if fz in dfolds.leaves:
                dfolds.leaves[fz].requires_grad_(False)
            if fz in host_glue_params:
                host_glue_params[fz].requires_grad_(False)
            print(json.dumps(dict(frozen=fz)), flush=True)
        # Under SPMD --spmd-dev-prep with suspend/reuse win_pre (1 taped conditioning tensor),
        # move the 6 dense winner glue projections onto device so their grads are sharded per-chip
        # (CCL-reduced across all chips) and never synced back to host CPU.
        use_dev_win_glue = bool(
            real and spmd_mesh_shape is not None and a.spmd_dev_prep
            and not a.freeze and a.win_pre in ("suspend", "reuse")
            and can_offload_cond_init(glue)
            and all(w in host_glue_params for w in DEVICE_GLUE_WEIGHTS)
        )
        if use_dev_win_glue:
            for w_nm in DEVICE_GLUE_WEIGHTS:
                host_glue_params.pop(w_nm, None)
        names = list(params)
        max_depth = (max(depths_seq) if depths_seq else
                     (a.depth_override if a.depth_override > 0 else
                      (max(int(b[2]["depth"]) for b in batches) if a.batch else a.depth)))
        cyc = {nm: "depth-1 draws feed a zero state into the cycle projection; trains at depth >= 2"
               for nm in names if "cycle" in nm and max_depth < 2}
        ax = None
        if world > 1 and spmd_mesh_shape is None:
            from tt_bio.train.launcher import ProcessAxis
            ax = ProcessAxis(name="dp", device_ids=tuple(range(world)), dp_rank=rank, run=a.dp_run)
        ema_d = None if a.ema_decay < 0 else a.ema_decay
        if a.optimizer == "adamw":
            opt = DeviceOptimizer(init_w, dev, lr=1e-4, clip_norm=1.0, weight_decay=0.0,
                                  muon=None, ema_decay=ema_d, ns_mode=a.ns_mode)
        else:
            opt = rfd4_device_optimizer(init_w, dev, lr=1e-4, ema_decay=ema_d, ns_mode=a.ns_mode)
        spmd_opt = None
        if spmd_mesh_shape is not None:
            from tt_bio.train.spmd_rfd4 import SPMDZero1Step
            spmd_opt = SPMDZero1Step(
                opt, init_w, dev,
                full_shape=full_mesh_shape,
                replicated_names=set(host_glue_params),
                use_zero1_muon=a.spmd_zero1,
            )
        if real and a.cand_pair == "card" and a.win_pair == "card":
            from tt_bio.train.rfd4_pair import ResidentPairBuilder
            res_pair = ResidentPairBuilder(glue.pair_repr_builder, dev)
        if real and spmd_mesh_shape is not None and a.spmd_dev_prep and can_offload_cond_init(glue):
            factory_split = FactoryStaticSplit(glue)
            if use_dev_win_glue:
                dev_win_glue = DeviceWinnerGlue(glue, factory_split, dev, tracer=spmd_tracer)
            spmd_prefetcher = SPMDPrefetcher(factory_split, res_pair=res_pair)
        refold_hook = make_rfd4_refold_hook(
            trunk, glue if real else None,
            host_glue_params=host_glue_params,
            pair_card_ref=(card_pair if (real and a.cand_pair == "card" and res_pair is None) else None),
            device=dev,
        )
        if use_spmd_trace:
            from tt_bio.train.spmd_trace import SPMDMeshTracer
            spmd_tracer = SPMDMeshTracer(dev, enabled=True)
            refold_dev_hook = make_rfd4_refold_hook(
                trunk, glue if real else None,
                host_glue_params=None,
                pair_card_ref=None,
                device=dev,
                in_place=True,
                sync_host=False,
            )

        def _sync_host_glue_params():
            if not host_glue_params:
                return
            from tt_bio.train.mesh_dp import read_rep0
            with torch.no_grad():
                for name, p in host_glue_params.items():
                    dm = opt.master.get(name)
                    if dm is None:
                        continue
                    arr = read_rep0(dm).float().reshape(p.shape).to(p.dtype)
                    p.copy_(arr)

        if real and a.cand_pair == "card" and res_pair is None:
            pair_card()
    else:
        build()
        unreached = set(folds.unreached())   # one autograd pass over every fold; never inside a per-leaf loop
        every = folds.params()
        params = {nm: p for nm, p in every.items() if nm not in unreached}
        if real:
            params.update(host_params(glue, "glue."))
        if a.freeze:
            fz = next(nm for nm in sorted(params) if a.freeze in nm)
            folds.leaves[fz].requires_grad_(False)
            build()
            print(json.dumps(dict(frozen=fz)), flush=True)
        names = list(params)
        max_depth = (max(depths_seq) if depths_seq else
                     (a.depth_override if a.depth_override > 0 else
                      (max(int(b[2]["depth"]) for b in batches) if a.batch else a.depth)))
        cyc = {nm: "depth-1 draws feed a zero state into the cycle projection; trains at depth >= 2"
               for nm in names if "cycle" in nm and max_depth < 2}
        ax = None
        if world > 1:
            from tt_bio.train.launcher import ProcessAxis
            ax = ProcessAxis(name="dp", device_ids=tuple(range(world)), dp_rank=rank, run=a.dp_run)
        use_muon = a.optimizer == "muon" or (a.optimizer == "auto" and bool(a.curve_dump))
        if use_muon:
            # Upstream's step: Muon + Adam, per-group clip 1.0 (rfd4_objective.update_rule). Plain
            # AdamW read 11 % above upstream from step 7 on the first graded leg; that gap was the optimizer.
            from tt_bio.train.rfd4_objective import update_rule
            opt = update_rule(dict(params), data_parallel=ax, shard=a.shard)
        else:
            opt = AdamW(dict(params), lr=1e-4, clip_norm=1.0, weight_decay=0.0, data_parallel=ax, shard=a.shard)
    ckpt_opt = spmd_opt if (a.resident and spmd_opt is not None) else opt
    start = 0
    if a.resume:
        start = int(load_adapter(a.resume, dict(params), dev, opt=ckpt_opt)["step"])
        if a.resident:
            dfolds.refold(opt.master, on_refold=refold_hook)
        else:
            build()
    ck = Checkpointer(a.out, every=a.save_every or 10**9, keep=a.keep or (2 if a.save_every else 10))
    seen = []   # losses this process logged, for --pin-spike
    if a.exempt == "default":
        for nm, why in UPSTREAM_EXEMPT.items():
            if nm in params:
                cyc[nm] = why
    elif a.exempt:
        for ln in open(a.exempt):
            if ln.strip() and not ln.startswith("#"):
                nm, why = ln.rstrip("\n").split("\t", 1)
                cyc[nm] = why
    cov = GradientCoverage(names, exempt=cyc, where="r4t-step trunk: ",
                           **({"steps": a.cover_steps} if a.cover_steps else {}))
    threading.Thread(target=sample, daemon=True).start()
    clog = None
    if a.curve_dump:
        from tt_bio.train.rfd4_objective import UPDATE_RULE as RO_RULE
        clog = CurveLog(os.path.expanduser(a.curve_out), cplan, "ours",
                        meta={"sha": os.environ.get("TT_BIO_SHA", ""), "card": card,
                              "dump_dir": cdir, "dtype": str(trunk.dtype), **RO_RULE,
                              "levers": sorted(levers) if isinstance(levers, (set, list, tuple))
                              else str(levers), "device": "tenstorrent", "k": a.k, "n": a.n},
                        start=start)
    exit_code = 0
    try:
        for step in range(start, a.steps):
            ttnn.synchronize_device(dev)
            base = used()
            c0 = len(clk)
            t0 = time.time()
            mark("fwd")
            extra = {}
            ctx = None
            acc_dev_grads = None
            acc_idle = set()
            if a.serial_accum > 1 and a.resident and spmd_mesh_shape is None:
                # Exact 1-chip serial-sum reference over M consecutive batches for G5 parity grading.
                losses_acc = []
                terms_acc, ref_acc, wins_acc, names_acc = None, None, [], []
                for m_idx in range(a.serial_accum):
                    use(step * a.serial_accum + m_idx, depth_step=step)
                    s_m, z_m, ctx_m = real[0](True) if real else (*one_example(step * a.serial_accum + m_idx, True), None)
                    close_rec()
                    ttnn.synchronize_device(dev)
                    if m_idx == 0:
                        retained = used() - base
                    if real:
                        l_m, tm_m, rf_m = real[1](s_m, ctx_m)
                        losses_acc.append(float(l_m))
                        wins_acc.append(ctx_m["win"])
                        names_acc.append(cur["name"])
                        if m_idx == 0:
                            terms_acc, ref_acc = tm_m, rf_m
                    else:
                        st_m = s_m if isinstance(s_m, ag.Tensor) else ag.Tensor(s_m)
                        losses_acc.append(float((ttnn.to_torch(st_m.value).float().reshape(Cs.shape) * Cs).sum() / numel))
                        ag.backward([st_m], [seed_s])
                    ttnn.synchronize_device(dev)
                    dg_m, id_m = dfolds.tape_backward(opt.master)
                    ag.release_pins()
                    acc_idle.update(id_m)
                    if acc_dev_grads is None:
                        acc_dev_grads = dg_m
                    else:
                        for k_g, v_g in dg_m.items():
                            if k_g in acc_dev_grads and acc_dev_grads[k_g] is not None:
                                prev_g = acc_dev_grads[k_g]
                                acc_dev_grads[k_g] = ttnn.add(prev_g, v_g)
                                ttnn.deallocate(prev_g)
                                ttnn.deallocate(v_g)
                            else:
                                acc_dev_grads[k_g] = v_g
                    del s_m, z_m
                t1 = t2 = time.time()
                after_bwd = used()
                loss = float(sum(losses_acc) / len(losses_acc))
                s = z = st = None
                if real:
                    extra = dict(
                        serial_accum=a.serial_accum, batch=names_acc[0], batches=names_acc,
                        winner=wins_acc[0], winners=wins_acc, depth_upstream=cur["depth"],
                        depth_used=cur["depth"], terms=terms_acc, upstream_terms=ref_acc,
                    )
            elif spmd_mesh_shape is not None and real:
                s, z, ctx = real_example_spmd(step)
                close_rec()
                ttnn.synchronize_device(dev)
                t1 = time.time()
                mark("heads_bwd")
                retained = used() - base
                loss, terms, ref, per_rank_losses = real_loss_spmd(s, ctx)
                ttnn.synchronize_device(dev)
                extra_mem = used()
                st = s
                extra = dict(
                    dev_after_heads_B=extra_mem, batch=ctx["rank_names"][0],
                    winner=ctx["win_indices"][0], winners=ctx["win_indices"],
                    upstream_winner=int(ctx["rank_metas"][0]["winners"][0]),
                    depth_upstream=ctx["step_depth"], depth_used=ctx["step_depth"],
                    per_rank_losses=[round(x, 6) for x in per_rank_losses],
                    terms=terms, upstream_terms=ref,
                    **({"phase": ctx["phase"]} if a.phase_timing else {}),
                )
            elif real:
                use(step * world + rank, depth_step=step)
                s, z, ctx = real[0](True)
                close_rec()
                ttnn.synchronize_device(dev)
                t1 = time.time()
                mark("heads_bwd")
                retained = used() - base
                loss, terms, ref = real[1](s, ctx)
                ttnn.synchronize_device(dev)
                extra_mem = used()
                st = s
                extra = dict(dev_after_heads_B=extra_mem, batch=cur["name"], winner=ctx["win"],
                             upstream_winner=(None if a.curve_dump else int(cur["meta"]["winners"][0])),
                             depth_upstream=cur["depth"], depth_used=cur["depth"],
                             terms=terms, upstream_terms=ref, **({"phase": ctx["phase"]} if a.phase_timing else {}))
            else:
                s, z = one_example(step if spmd_mesh_shape is not None else (step * world + rank), True)
                close_rec()
                ttnn.synchronize_device(dev)
                t1 = time.time()
                mark("heads_bwd")
                retained = used() - base
                st = s if isinstance(s, ag.Tensor) else ag.Tensor(s)
                if spmd_mesh_shape is not None:
                    from tt_bio.train.mesh_dp import read_sharded
                    s_all = read_sharded(st.value, dev).float().reshape(world, *Cs.shape)
                    loss = float(((s_all * Cs.unsqueeze(0)).sum(dim=(1, 2, 3)) / numel).mean())
                else:
                    loss = float((ttnn.to_torch(st.value).float().reshape(Cs.shape) * Cs).sum() / numel)
                ag.backward([st], [seed_s])
            if acc_dev_grads is None:
                ttnn.synchronize_device(dev)
                t2 = time.time()
                after_bwd = used()
            if a.grad_dump and a.brk:
                walked = {p: t for p, _o, _k, t in T.walk_device_weights(trunk)}
                cut = next(p for p in walked if p.endswith("q_w"))
                folds.sources.pop(id(walked[cut]))
                extra["dropped"] = cut
            if a.resident:
                mark("fold_bwd")
                free_skip = ()
                if acc_dev_grads is not None:
                    dev_grads, idle = acc_dev_grads, list(acc_idle)
                elif isinstance(ctx, dict) and "traced_dev_grads" in ctx:
                    dev_grads = ctx.pop("traced_dev_grads")
                    free_skip = ctx.pop("traced_free_skip", ())
                    idle = ctx.pop("traced_idle", [])
                else:
                    dev_grads, idle = dfolds.tape_backward(opt.master)
                    ag.release_pins()
                    extra_g = ctx.pop("extra_dev_grads", None) if isinstance(ctx, dict) else None
                    if extra_g:
                        for kg, gg in extra_g.items():
                            if kg in dev_grads:
                                sg = ttnn.add(dev_grads[kg], gg)
                                ttnn.deallocate(dev_grads[kg])
                                ttnn.deallocate(gg)
                                dev_grads[kg] = sg
                            else:
                                dev_grads[kg] = gg
                # Upload only the tiny non-pair host-glue grads (~1.87 M elems = 7.5 MB).
                # Under --mesh, MeshBridge.backward_ups() already summed host-glue grads across
                # all `world` chips, and SPMDZero1Step passes `replicated_names=host_glue_params`
                # so CCL does not re-sum them.
                for gn, gp in host_glue_params.items():
                    if gp.grad is not None:
                        dev_grads[gn] = ttnn.from_torch(
                            gp.grad.detach().to(torch.float32).contiguous(),
                            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                        )
                        gp.grad = None
                ttnn.synchronize_device(dev)
                t2b = time.time()
                mark("opt")
                if spmd_opt is not None:
                    opt_info = spmd_opt.step(
                        dev_grads, free_grads=True,
                        skip_reduce_for_broken_arm=a.break_spmd_reduce,
                        profile_sync=a.phase_timing,
                        in_place_masters=use_spmd_trace,
                        free_skip=free_skip,
                        tracer=spmd_tracer,
                    )
                else:
                    opt_info = opt.step(dev_grads, free_grads=True)
                dead = set(opt_info["dead_grads"]) | {nm for nm in names if nm not in opt.master}
                if ax:
                    order = sorted(names)
                    hits = ax.sum_scalars([float(nm not in dead) for nm in order])
                    dead = {nm for nm, h in zip(order, hits) if h == 0.0}
                ttnn.synchronize_device(dev)
                t2c = time.time()
                mark("rebuild")
                if spmd_tracer is not None:
                    spmd_tracer.run_refold(dfolds, opt.master, refold_dev_hook)
                    _sync_host_glue_params()
                else:
                    dfolds.refold(opt.master, on_refold=refold_hook)
                ttnn.synchronize_device(dev)
                t3 = time.time()
                del s, z, st
                pcie_snap = PM.take() if PM else None
                mark("digest")
                want_cons = (
                    spmd_mesh_shape is not None
                    and ((step + 1 == a.steps) or (a.consensus_every > 0 and (step + 1) % a.consensus_every == 0))
                )
                cons_ok = None
                cons_n = 0
                cons_err = ""
                if want_cons:
                    from tt_bio.train.spmd_rfd4 import master_consensus_digest
                    try:
                        md, cons_n = master_consensus_digest(opt.master, dev)
                        cons_ok = True
                    except AssertionError as exc:
                        cons_ok = False
                        cons_err = str(exc)
                        md = "DIVERGED"
                elif step + 1 == a.steps:
                    md = (spmd_opt.master_digest() if spmd_opt is not None else opt.master_digest())
                else:
                    md = "resident"
                sm = clk[c0:] or clk[-1:]
                gnorm = math.hypot(opt.last_grad_norm, getattr(opt, "last_grad_norm_muon", 0.0))
                n_elems = int(sum(int(math.prod(shp)) for shp in opt.shapes.values()))
                spmd_fields = {}
                if spmd_mesh_shape is not None:
                    spmd_fields = dict(
                        mesh=f"{spmd_mesh_shape[0]}x{spmd_mesh_shape[1]}",
                        spmd_zero1=bool(spmd_opt.use_zero1_muon),
                        spmd_trace=bool(use_spmd_trace),
                        spmd_dev_prep=bool(a.spmd_dev_prep),
                        eth_dispatch=a.eth_dispatch,
                        break_spmd_reduce=bool(a.break_spmd_reduce),
                        grad_norm_adam=float(opt.last_grad_norm),
                        grad_norm_muon=float(getattr(opt, "last_grad_norm_muon", 0.0)),
                    )
                    if spmd_tracer is not None:
                        spmd_fields["trace_stats"] = dict(spmd_tracer.stats)
                        if spmd_tracer.refusal_reasons:
                            spmd_fields["trace_refusals"] = list(spmd_tracer.refusal_reasons)
                    if "sync_s" in opt_info:
                        spmd_fields["sync_s"] = opt_info["sync_s"]
                    if cons_ok is not None:
                        spmd_fields["consensus_ok"] = bool(cons_ok)
                        spmd_fields["consensus_checked"] = int(cons_n)
                        if cons_err:
                            spmd_fields["consensus_err"] = cons_err
                else:
                    spmd_fields = dict(
                        grad_norm_adam=float(opt.last_grad_norm),
                        grad_norm_muon=float(getattr(opt, "last_grad_norm_muon", 0.0)),
                    )
                step_d_log = (cur["depth"] if real else
                              (int(depths_seq[step % len(depths_seq)]) if depths_seq else
                               (int(a.depth_override) if a.depth_override > 0 else a.depth)))
                row = dict(step=step + 1, rank=rank, world=world, resident=True, ns_mode=a.ns_mode,
                           loss=loss, lr=opt_info["lr"], grad_norm=gnorm,
                           fwd_s=round(t1 - t0, 4), bwd_s=round(t2 - t1, 4),
                           fold_bwd_s=round(t2b - t2, 4), opt_s=round(t2c - t2b, 4),
                           rebuild_s=round(t3 - t2c, 4), rebuild_split=dict(RB), idle_dev_w=len(idle),
                           step_s=round(t3 - t0, 4), retained_B=retained,
                           dead=len(dead), dead_names=sorted(dead)[:20],
                           dev_base_B=base, dev_after_bwd_B=after_bwd,
                           params=len(params), n_elems=n_elems,
                           master_digest=md, aiclk=[min(sm or [0]), max(sm or [0])],
                           aiclk_med=sorted(sm)[len(sm) // 2] if sm else 0,
                           k=a.k, depth=step_d_log, n=n,
                           lnbw_reach=dict(lnbw.REACH), gate_bw=list(gate_bw.STATS),
                           mmvjp=dict(ag.MMVJP_STATS), dw_2d=dict(ag._mm_layout.DW_2D_REACH),
                           split_k=dict(ag.SPLIT_K_STATS), sum_leading=dict(ag.SUM_LEADING_STATS),
                           widen_reach=dict(ag._rne_add.WIDEN_REACH),
                           kernels={k: list(v) for k, v in __import__("tt_bio.taped_ttnn", fromlist=["x"]).KERNEL_STATS.items()},
                           **spmd_fields, **extra, **({"pcie": pcie_snap} if pcie_snap is not None else {}),
                           host_maxrss_GiB=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3))
            else:
                mark("fold_bwd")
                idle = folds.tape_backward()   # device-weight gradients through each fold's VJP into the upstream leaves
                grads = {nm: (None if p.grad is None else p.grad.float().numpy()) for nm, p in params.items()}
                ag.release_pins()   # the backward is done; build() released them too, one rebuild later
                dead = zero_or_missing(grads)
                if ax:
                    # G2 judges the gradient the optimizer steps on, the rank sum: a leaf that only another
                    # rank's example reaches (a feature embedding its batch lacks) is alive. Dead = dead on every rank.
                    order = sorted(grads)
                    hits = ax.sum_scalars([float(nm not in dead) for nm in order])
                    dead = {nm for nm, h in zip(order, hits) if h == 0.0}
                if os.environ.get("R4T_DUMP_G") and step == start:
                    import numpy as _np
                    _np.savez(os.environ["R4T_DUMP_G"], **{nm.replace(".", "__"): g for nm, g in grads.items()
                                                            if g is not None and "pair_update_layers" in nm})
                if a.grad_dump:
                    import numpy as _np
                    sm = clk[c0:] or clk[-1:]
                    _np.savez(a.grad_dump + ".npz", **{nm: g for nm, g in grads.items() if g is not None})
                    meta = dict(extra, loss=loss, arm=a.arm, attn_fp32=a.attn_fp32, levers=sorted(levers) if not isinstance(levers, str) else levers,
                                dtype=str(trunk.dtype), n=n, card=card, aiclk=[min(sm or [0]), max(sm or [0])],
                                exact={"%s/%s" % k: v for k, v in sorted(EX.EXACT_STATS.items())},
                                no_grad=sorted(nm for nm, g in grads.items() if g is None), sha=os.environ.get("TT_BIO_SHA", ""))
                    open(a.grad_dump + ".json", "w").write(json.dumps(meta, indent=1, default=str))
                    print(json.dumps(dict(grad_dump=a.grad_dump, n_grads=len(grads), loss=loss, aiclk=meta["aiclk"],
                                          exact=meta["exact"])), flush=True)
                    return 0
                t2b = time.time()
                mark("opt")
                opt.step(replicas={nm: [g] for nm, g in grads.items() if g is not None}) if ax else opt.step()
                for p in params.values():
                    p.grad = None
                t2c = time.time()
                mark("rebuild")
                build(lazy=False)
                ttnn.synchronize_device(dev)
                t3 = time.time()
                mark("digest")
                del s, z, st
                md = hashlib.sha256(b"".join(opt.master[k].tobytes() for k in sorted(opt.master))).hexdigest()[:16]
                sm = clk[c0:] or clk[-1:]
                gnorm = math.hypot(opt.last_grad_norm, getattr(opt, "last_grad_norm_muon", 0.0))
                step_d_log = (cur["depth"] if real else
                              (int(depths_seq[step % len(depths_seq)]) if depths_seq else
                               (int(a.depth_override) if a.depth_override > 0 else a.depth)))
                row = dict(step=step + 1, rank=rank, world=world, loss=loss, lr=opt.last_lr, grad_norm=gnorm,
                           fwd_s=round(t1 - t0, 4), bwd_s=round(t2 - t1, 4),
                           fold_bwd_s=round(t2b - t2, 4), opt_s=round(t2c - t2b, 4), rebuild_s=round(t3 - t2c, 4), rebuild_split=dict(RB), idle_dev_w=len(idle), step_s=round(t3 - t0, 4), retained_B=retained,
                           dead=len(dead), dead_names=sorted(dead)[:20], dev_base_B=base, dev_after_bwd_B=after_bwd, params=len(params), n_elems=int(sum(v.size for v in opt.master.values())),
                           master_digest=md, aiclk=[min(sm or [0]), max(sm or [0])], aiclk_med=sorted(sm)[len(sm) // 2] if sm else 0,
                           k=a.k, depth=step_d_log, n=n, lnbw_reach=dict(lnbw.REACH), gate_bw=list(gate_bw.STATS), mmvjp=dict(ag.MMVJP_STATS), dw_2d=dict(ag._mm_layout.DW_2D_REACH), split_k=dict(ag.SPLIT_K_STATS), sum_leading=dict(ag.SUM_LEADING_STATS), widen_reach=dict(ag._rne_add.WIDEN_REACH), kernels={k: list(v) for k, v in __import__("tt_bio.taped_ttnn", fromlist=["x"]).KERNEL_STATS.items()}, **extra,
                           **({"pcie": PM.take()} if PM else {}), host_maxrss_GiB=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3))
            print(json.dumps(row), flush=True)
            if on_step is not None:
                on_step({**row, "s": row["step_s"]})
            if a.resident and spmd_mesh_shape is not None and cons_ok is False:
                exit_code = 2
                if not a.break_spmd_reduce:
                    raise RuntimeError(f"SPMD master consensus mismatch at step {step + 1}: {cons_err}")
                break
            if a.save_at and step + 1 == a.save_at:   # before the guard, so a refusal still leaves the resume test its input
                print(json.dumps(dict(saved=str(ck.save(step + 1, ckpt_opt, metrics={"loss": loss}, force=True)))), flush=True)
            if a.pin_spike:
                ref = sorted(seen[-100:])[len(seen[-100:]) // 2] if len(seen) >= 20 else None
                if ref is not None and loss > a.pin_spike * ref:
                    print(json.dumps(dict(spike=step, loss=loss, median100=ref, example=cur["name"] if real else None,
                                          pinned=str(ck.pin(step)))), flush=True)
                seen.append(loss)
            due_save, force_save = _checkpoint_due(step + 1, a.steps, a.save_every, a.save_at)
            if due_save:   # --keep: a ring by step, so no metric to rank by
                print(json.dumps(dict(saved=str(ck.save(step + 1, ckpt_opt, metrics=None if a.keep else {"loss": loss},
                                                        force=force_save)))), flush=True)
            if clog is not None:
                clog.record(step, loss, terms=extra.get("terms"), n_recycle=cur["depth"],
                            seconds=round(t3 - t0, 2), dead=len(dead), example=cur["name"],
                            aiclk_med=sorted(sm)[len(sm) // 2] if sm else 0, master_digest=md)
            # G2 rides along, as on the upstream side: a depth-1 step trains neither cycle projection
            # (state/r4t/rows/accuracy.md), so counting it would refuse ~1 healthy run in 150.
            if not a.resume and not a.break_spmd_reduce and (not real or cur["depth"] >= 2):
                cov.observe(dead)
        if a.dump_master_np:
            import numpy as _np
            hm = (spmd_opt.host_master() if (a.resident and spmd_opt is not None)
                  else (opt.host_master() if a.resident else {k: v for k, v in opt.master.items()}))
            os.makedirs(os.path.dirname(os.path.abspath(a.dump_master_np)), exist_ok=True)
            _np.savez(a.dump_master_np, **hm)
    finally:
        stop.set()
        if clog is not None:
            clog.close()
        if spmd_mesh_shape is not None:
            if locals().get("spmd_prefetcher") is not None:
                try:
                    spmd_prefetcher.close()
                except Exception:
                    pass
            if locals().get("dev_win_glue") is not None:
                try:
                    dev_win_glue.release()
                except Exception:
                    pass
            if spmd_tracer is not None:
                try:
                    spmd_tracer.release()
                except Exception:
                    pass
            for obj in (locals().get("spmd_opt"), locals().get("opt"), locals().get("res_pair")):
                if obj is not None and hasattr(obj, "deallocate"):
                    try:
                        obj.deallocate()
                    except Exception:
                        pass
            arm.close()
        else:
            arm.close()
    return exit_code


def launch(a, argv=None):
    """One process per card, each pinned to its own card and lease, re-running the same entry point; waits for all."""
    import subprocess
    import sys
    from pathlib import Path
    cards = a.dp_cards.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = Path("/dev/shm") / f"r4t-trunk-dp-{os.getpid()}"
    run.mkdir()
    holder = os.environ.get("TT_BIO_LEASE_HOLDER", "worker:r4t-scale")
    skip = {"--dp-cards", "--out", "--dp-rank", "--dp-run"}
    fwd, it = [], iter(sys.argv[1:] if argv is None else argv)
    for t in it:   # forward every flag but the launcher's own
        if t.split("=")[0] in skip:
            if "=" not in t:
                next(it, None)
            continue
        fwd.append(t)
    entry = [os.path.abspath(sys.argv[0])] if sys.argv[0].endswith(".py") else ["-m", __name__]
    procs = []
    for r, c in enumerate(cards):
        env = dict(os.environ, TT_VISIBLE_DEVICES=c, TT_BIO_LEASE_CARDS=c, TT_BIO_LEASE_HOLDER=holder)
        cmd = [sys.executable, *entry, *fwd, "--dp-cards", a.dp_cards, "--dp-rank", str(r),
               "--dp-run", str(run), "--out", str(out / f"rank{r}")]
        procs.append(subprocess.Popen(cmd, env=env, stdout=open(out / f"rank{r}.log", "w"), stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in procs]
    subprocess.run(["rm", "-rf", str(run)])
    print(json.dumps(dict(world=len(cards), cards=cards, rcs=rcs)), flush=True)
    return 0 if not any(rcs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
