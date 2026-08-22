"""Where inside OpenDDE's 4-block MSA module does z lose its 0.999337?

Pass 1 of this investigation priced the 48-block Pairformer stack and found it innocent:
78% of the trunk-level z drop is that stack amplifying (7.86x) an error that already
exists at its input, `z_post_msa` at PCC 0.999337. The 7.86x gain is what makes this
script worth 8x a same-sized fix inside the Pairformer: to land z_post_pairformer at
0.999 the MSA module has to hand over 0.99998, not 0.9993.

Same discipline as pass 1, one module upstream:

  stage 1  cache the reference's own MSA chain, sub-step by sub-step, and score the
           shipped device path's per-block z and m against it. Also caches the device's
           own z entering the MSA module and its m_feat, so stage 2 can replay from
           either input. Includes the instrument check: the hand-stepped reference chain
           must reproduce `msa_module.forward`'s z.
  stage 2  teacher-forced sub-op taps. Feed device block b the REFERENCE's (z, m), tap
           every sub-update, score it against the reference's own update at that same
           input. No accumulation, so this is per-block injection, and the check that
           makes it trustworthy is that the taps reassemble the block's own measured error.
  stage 3  the Pairformer tri_att_end tap pass 1 could not read: the device's ending
           variant transposes its update back before the tap sees it (tenstorrent.py
           :4214) while the reference adds it in the transposed frame, so pass 1's column
           compared X against X^T and was excluded. Scores both frames explicitly.
  stage 4  A/A control on the shipped fold path: same input twice, then the same device
           blocks driven in isolation, to settle whether the 0.947762 (fold) vs 0.957846
           (harness) gap is L1 pressure or measurement.

  OPENDDE_SRC=/tmp/opendde-src PYTHONPATH=<worktree> TT_VISIBLE_DEVICES=0 \
    python3 scripts/opendde_msa_subop_probe.py --stage 1
"""
import argparse
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")

import torch

from scripts.opendde_real_seam_parity import SRC, _full_construct_features, pcc_ratio
from scripts.opendde_pairformer_block_trace import (
    N_BLOCKS, SUB_OPS, _load_reference, dev_block_z, dev_sub, ref_block_z, ref_sub,
)

torch.set_grad_enabled(False)

N_MSA_BLOCKS = 4
MSA_STEPS = ("pwa", "transition_m", "opm")


def _stat(tag, got, ref):
    """One line per tap. rel = error rms over reference rms, the quantity that composes."""
    g, r = got.float().reshape(ref.shape), ref.float()
    pcc, ratio = pcc_ratio(g, r)
    rms_r = float(r.pow(2).mean().sqrt())
    rms_e = float((g - r).pow(2).mean().sqrt())
    print(f"{tag:44s} PCC={pcc:.6f} norm_ratio={ratio:.6f} "
          f"rms_ref={rms_r:.4e} rms_err={rms_e:.4e} rel={rms_e / max(rms_r, 1e-30):.4e}",
          flush=True)
    return pcc, rms_e, rms_r


# --------------------------------------------------------------------------- reference

def ref_msa_block(blk, m, z, taps=None):
    """One reference MSABlock, z and m, verbatim from MSABlock.forward's mesh-None branch.

    MSAStack mutates m in place through a slice; this steps the same arithmetic
    out-of-place so a tap survives the next sub-op.
    """
    u = blk.msa_stack.msa_pair_weighted_averaging(m, z)
    _tap(taps, "pwa", u)
    m = m + u
    u = blk.msa_stack.transition_m(m)
    _tap(taps, "transition_m", u)
    m = m + u
    u = blk.outer_product_mean_msa(m, inplace_safe=False, chunk_size=None)
    _tap(taps, "opm", u)
    z = z + u
    pair_taps = {} if taps is not None else None
    z = ref_block_z(blk.pair_stack, z, taps=pair_taps)
    if taps is not None:
        for k, v in pair_taps.items():
            taps[f"pair.{k}"] = v
    return m, z


def _tap(taps, name, value):
    if taps is not None:
        taps[name] = value
    return value


def _ref_front_half(state_dict, feats, fi, p):
    """Everything the reference computes before the MSA module, plus the MSA input m."""
    reference, cfg = _load_reference(state_dict)
    ref_feats = dict(feats)
    ref_feats["relp"] = p._generate_relp(feats)
    ref_feats["d_lm"] = fi["d"].reshape(fi["nb"], fi["nq"], fi["nk"], 3)
    ref_feats["v_lm"] = fi["v"].reshape(fi["nb"], fi["nq"], fi["nk"], 1)
    ref_feats["pad_info"] = {"mask_trunked": fi["mt"].bool()}

    s_inputs = reference.input_embedder(ref_feats, inplace_safe=False, chunk_size=None)
    s_init = reference.linear_no_bias_sinit(s_inputs)
    z_init = (reference.linear_no_bias_zinit1(s_init)[..., None, :]
              + reference.linear_no_bias_zinit2(s_init)[..., None, :, :])
    z_init = z_init + reference.relative_position_encoding(ref_feats["relp"])
    z_init = z_init + reference.linear_no_bias_token_bond(
        ref_feats["token_bonds"].unsqueeze(-1))
    z = z_init + reference.linear_no_bias_z_cycle(
        reference.layernorm_z_cycle(torch.zeros_like(z_init)))
    z = z + reference.template_embedder(
        ref_feats, z, triangle_multiplicative=cfg.triangle_multiplicative,
        triangle_attention=cfg.triangle_attention, inplace_safe=False, chunk_size=None)
    m0 = reference.msa_module._prepare_msa_sample(
        input_feature_dict=ref_feats, s_inputs=s_inputs, z_token_dim=z.shape[-2])
    return reference, cfg, ref_feats, s_inputs, z, m0


# --------------------------------------------------------------------------- device

def _dev_setup():
    import ttnn
    from tt_bio.opendde import OpenDDE, load_opendde_checkpoint
    from tt_bio.tenstorrent import get_device
    state_dict = load_opendde_checkpoint()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    model = OpenDDE(state_dict, ckc, dev)
    return model, model._protenix, state_dict


def dev_msa_block(p, blk, m, z, taps=None):
    """One device MSA block, z and m, mirroring Trunk._msa's whole-tensor branch.

    (opm, pwa, tm, pl) is the tuple Trunk.__init__ builds. The whole-tensor branch is
    what a 136 aa / 84-deep target takes: _msa_take_whole_path sees ~1.5 MB.
    """
    import ttnn
    opm, pwa, tm, pl = blk
    to_host = p._to_host
    u = ttnn.reshape(pwa(m, ttnn.clone(z)), tuple(m.shape))
    if taps is not None:
        taps["pwa"] = to_host(u)
    m = ttnn.add(m, u)
    u = ttnn.reshape(tm(m), tuple(m.shape))
    if taps is not None:
        taps["transition_m"] = to_host(u)
    m = ttnn.add(m, u)
    u = opm(m, None, None)
    if taps is not None:
        taps["opm"] = to_host(u)
    z = ttnn.add(z, u)
    pair_taps = {} if taps is not None else None
    # pl(None, z)[1] would run the pair stack in one call; stepping it is what makes a
    # per-sub-op tap possible, and dev_block_z is the sequence pass 1 validated against
    # the layer's own output.
    z = dev_block_z(pl, z, taps=pair_taps, to_host=to_host if taps is not None else None)
    if taps is not None:
        for k, v in pair_taps.items():
            taps[f"pair.{k}"] = v
    return m, z


# --------------------------------------------------------------------------- stages

def stage1(args):
    import ttnn
    from scripts.opendde_real_seam_parity import _residue_trunk

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    feats = _full_construct_features()
    model, p, state_dict = _dev_setup()

    # Tap the shipped path: z and m_feat entering the MSA module, and z/m after each block.
    dev_taps = {}
    trunk = p.trunk
    original = type(trunk)._msa

    def wrapped(self, z3, m_feat):
        if self is not trunk or "z_in" in dev_taps:
            return original(self, z3, m_feat)
        dev_taps["z_in"] = p._to_host(z3)
        dev_taps["m_in"] = (m_feat if torch.is_tensor(m_feat)
                            else p._to_host(m_feat))
        # step the blocks here, so a per-block tap sees the shipped allocation order
        for bi, blk in enumerate(self.MSA):
            m_feat, z3 = _shipped_block(self, p, blk, m_feat, z3)
            dev_taps[f"m_{bi}"] = p._to_host(m_feat)
            dev_taps[f"z_{bi}"] = p._to_host(z3)
        return z3

    type(trunk)._msa = wrapped
    try:
        _, _, got_z, fi, _, _ = _residue_trunk(model, feats, n_cycles=1)
    finally:
        type(trunk)._msa = original
    torch.save(fi, cache / "fi.pt")
    for k, v in dev_taps.items():
        torch.save(v, cache / f"dev_{k}.pt")
    print(f"cached device taps: {sorted(dev_taps)}", flush=True)

    reference, cfg, ref_feats, s_inputs, z_pre, m0 = _ref_front_half(
        state_dict, feats, fi, p)
    torch.save(z_pre, cache / "ref_z_pre.pt")
    torch.save(m0, cache / "ref_m0.pt")
    torch.save(s_inputs, cache / "ref_s_inputs.pt")
    print(f"ref z_pre {tuple(z_pre.shape)} m0 {tuple(m0.shape)}", flush=True)

    m, z = m0, z_pre
    for bi in range(N_MSA_BLOCKS):
        m, z = ref_msa_block(reference.msa_module.blocks[bi], m, z)
        torch.save(m, cache / f"ref_m_{bi}.pt")
        torch.save(z, cache / f"ref_z_{bi}.pt")
        print(f"ref MSA block {bi} done |z|={float(z.pow(2).mean().sqrt()):.4f} "
              f"|m|={float(m.pow(2).mean().sqrt()):.4f}", flush=True)

    # INSTRUMENT CHECK: the hand-stepped chain must be msa_module.forward's own z.
    z_full = reference.msa_module(
        ref_feats, z_pre, s_inputs, pair_mask=None,
        triangle_multiplicative=cfg.triangle_multiplicative,
        triangle_attention=cfg.triangle_attention, inplace_safe=False, chunk_size=None)
    same = bool(torch.equal(z, z_full))
    print(f"INSTRUMENT hand-stepped MSA chain == msa_module.forward: {same} "
          f"maxabs={float((z - z_full).abs().max()):.3e}", flush=True)
    torch.save(z_full, cache / "ref_z_msa_out.pt")

    print("\n--- shipped device path vs reference, MSA module ---", flush=True)
    # This loop is re-stepped here rather than called, so it has to reproduce the number
    # the trunk report published for the un-instrumented path: z_post_msa PCC 0.999337.
    _stat("ANCHOR z_post_msa (expect 0.999337)", dev_taps[f"z_{N_MSA_BLOCKS - 1}"], z_full)
    _stat("z entering MSA", dev_taps["z_in"], z_pre)
    _stat("m_feat entering MSA", dev_taps["m_in"], m0)
    for bi in range(N_MSA_BLOCKS):
        _stat(f"z after MSA block {bi}", dev_taps[f"z_{bi}"],
              torch.load(cache / f"ref_z_{bi}.pt"))
        _stat(f"m after MSA block {bi}", dev_taps[f"m_{bi}"],
              torch.load(cache / f"ref_m_{bi}.pt"))


def _shipped_block(trunk, p, blk, m_feat, z3):
    """Trunk._msa's per-block body for the whole-tensor, msa_update_first path."""
    import ttnn
    opm, pwa, tm, pl = blk
    m_up = trunk._up(m_feat) if torch.is_tensor(m_feat) else m_feat
    m2 = ttnn.add(m_up, ttnn.reshape(pwa(m_up, ttnn.clone(z3)), tuple(m_up.shape)))
    m_new = ttnn.add(m2, ttnn.reshape(tm(m2), tuple(m2.shape)))
    if m_up is not m_feat:
        ttnn.deallocate(m_up)
    z3 = ttnn.add(z3, opm(m_new, None, None))
    z3 = pl(None, z3)[1]
    return m_new, z3


def stage2(args):
    """Teacher-forced per-sub-op injection inside the MSA blocks."""
    import ttnn
    cache = Path(args.cache)
    model, p, state_dict = _dev_setup()
    reference, _, _, _, z_pre, m0 = _ref_front_half(
        state_dict, _full_construct_features(), torch.load(cache / "fi.pt"), p)
    for bi in [int(x) for x in args.blocks.split(",")]:
        m_src = m0 if bi == 0 else torch.load(cache / f"ref_m_{bi - 1}.pt")
        z_src = z_pre if bi == 0 else torch.load(cache / f"ref_z_{bi - 1}.pt")
        ref_taps = {}
        m_ref, z_ref = ref_msa_block(reference.msa_module.blocks[bi], m_src, z_src,
                                     taps=ref_taps)
        dev_taps = {}
        n = z_src.shape[-2]
        zt = p.trunk._up(z_src.reshape(1, n, n, -1))
        mt = p.trunk._up(m_src.reshape(1, *m_src.shape))
        m_dev, z_dev = dev_msa_block(p, p.trunk.MSA[bi], mt, zt, taps=dev_taps)
        print(f"\n--- MSA block {bi}, teacher-forced ---", flush=True)
        _stat(f"b{bi} BLOCK z out", p._to_host(z_dev, tuple(z_ref.shape)), z_ref)
        _stat(f"b{bi} BLOCK m out", p._to_host(m_dev, tuple(m_ref.shape)), m_ref)
        for name in MSA_STEPS:
            _stat(f"b{bi} {name}", dev_taps[name], ref_taps[name])
        for name in SUB_OPS:
            ref_u = ref_taps[f"pair.{name}"]
            got = dev_taps[f"pair.{name}"]
            if name == "tri_att_end":
                # The device's ending variant transposes its update back before the tap
                # sees it (tenstorrent.py:4214); the reference adds it in the transposed
                # frame. Score both so the frame is named, not assumed.
                _stat(f"b{bi} pair.{name} [refT]", got, ref_u.transpose(-2, -3))
                _stat(f"b{bi} pair.{name} [direct]", got, ref_u)
            else:
                _stat(f"b{bi} pair.{name}", got, ref_u)
        ttnn.deallocate(z_dev)


def stage3(args):
    """Pairformer tri_att_end, both frames. Pass 1 left this column unmeasured."""
    import ttnn
    pf = Path(args.pf_cache)
    model, p, state_dict = _dev_setup()
    ref_model, _ = _load_reference(state_dict)
    ref_blocks = ref_model.pairformer_stack.blocks
    blocks = p.trunk.PF.blocks
    z_msa = torch.load(pf / "ref_z_msa.pt")
    for i in [int(x) for x in args.blocks.split(",")]:
        src = z_msa if i == 0 else torch.load(pf / f"ref_z_{i - 1:02d}.pt")
        n = src.shape[-2]
        ref_taps = {}
        z_ref = ref_block_z(ref_blocks[i], src, taps=ref_taps)
        dev_taps = {}
        zt = p.trunk._up(src.reshape(1, n, n, -1))
        zt = dev_block_z(blocks[i], zt, taps=dev_taps, to_host=lambda t: p._to_host(t))
        print(f"\n--- Pairformer block {i} ---", flush=True)
        _stat(f"pf{i} BLOCK z out", p._to_host(zt, tuple(z_ref.shape)), z_ref)
        for name in SUB_OPS:
            ref_u = ref_taps[name]
            got = dev_taps[name]
            if name == "tri_att_end":
                _stat(f"pf{i} {name} [refT]", got, ref_u.transpose(-2, -3))
                _stat(f"pf{i} {name} [direct]", got, ref_u)
            else:
                _stat(f"pf{i} {name}", got, ref_u)
        ttnn.deallocate(zt)



def stage5(args):
    """Device-off amplification arm for the MSA module, the pass-1 technique one level up.

    Replays the fp32 reference MSA chain from the DEVICE's own inputs. No device compute,
    so whatever it loses is the module amplifying an error it inherited. Four arms isolate
    which input carries it.

    The m arms need one correction first: the reference's row sampler calls
    `torch.randperm` (opendde/model/msa_sampling.py:56), so the reference and the device
    hold the SAME 76 rows in a different order. z is invariant to that (every MSA sub-op is
    row-wise and OuterProductMean averages over rows), but m scored row-for-row is not, and
    the raw comparison reads 0.8235 where the real device error is 2.9e-03. The permutation
    is recovered by nearest-row matching, unambiguous here: every device row matches one
    reference row at cos >= 0.999998 with a >= 0.0014 gap to the runner-up.
    """
    from tt_bio.opendde import load_opendde_checkpoint
    cache = Path(args.cache)
    reference, _ = _load_reference(load_opendde_checkpoint())
    blocks = reference.msa_module.blocks
    ref_z = [torch.load(cache / f"ref_z_{b}.pt") for b in range(N_MSA_BLOCKS)]
    ref_m0 = torch.load(cache / "ref_m0.pt").float()
    ref_z_pre = torch.load(cache / "ref_z_pre.pt").float()
    dev_z_in = torch.load(cache / "dev_z_in.pt").float().reshape(ref_z_pre.shape)
    dev_m_in = torch.load(cache / "dev_m_in.pt").float().reshape(ref_m0.shape)

    perm = _match_msa_rows(dev_m_in, ref_m0)
    dev_m_aligned = dev_m_in[_invert(perm)]
    _stat("m_in raw (row order differs)", dev_m_in, ref_m0)
    _stat("m_in row-matched", dev_m_in, ref_m0[perm])

    arms = (
        ("ref-z + ref-m (identity check)", ref_z_pre, ref_m0),
        ("dev-z + ref-m", dev_z_in, ref_m0),
        ("ref-z + dev-m", ref_z_pre, dev_m_aligned),
        ("dev-z + dev-m", dev_z_in, dev_m_aligned),
    )
    for tag, z0, m0 in arms:
        print(f"\n--- {tag} ---", flush=True)
        m, z = m0, z0
        for b in range(N_MSA_BLOCKS):
            m, z = ref_msa_block(blocks[b], m, z)
            _stat(f"  z after block {b}", z, ref_z[b])


def _match_msa_rows(dev, ref):
    """perm[i] = the reference row that device row i holds."""
    d = dev.reshape(dev.shape[0], -1)
    r = ref.reshape(ref.shape[0], -1)
    d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
    r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sim = d @ r.T
    perm = sim.argmax(dim=1)
    top2 = sim.topk(2, dim=1).values
    assert len(set(perm.tolist())) == dev.shape[0], "row matching is not a permutation"
    assert float(top2[:, 0].min()) > 0.999, "row matching is not unambiguous"
    assert float((top2[:, 0] - top2[:, 1]).min()) > 1e-4, "row matching is degenerate"
    return perm


def _invert(perm):
    out = torch.empty_like(perm)
    out[perm] = torch.arange(perm.numel())
    return out



def stage6(args):
    """In-chain substitution screen on the triangle-attention core.

    Stage 2 puts `tri_att_start` at PCC 0.212 with the device update 2.83x the reference
    update's norm, which is a wrong magnitude rather than a rounding wobble. This splits the
    op in two: everything before the attention (layer_norm, the qkv and bias projections) and
    the attention core itself (`_tri_att_sdpa`). It swaps the core for an exact fp32 host
    softmax over the device's OWN q, k, v and bias and lets the device finish the op, so the
    difference between the arms is the core and nothing else.

      dev        untouched, the baseline stage 2 measured
      host       host fp32 attention over every key the device holds
      host-nopad host fp32 attention with the tile-padded keys masked out

    Reuses the in-chain substitution technique from `af2ig-onesidedness-predicts-bf16-widening-payoff`
    rather than a standalone op screen: an isolated screen bounds the op's error, it does not
    say what the fold would gain (`rfd3-isolated-screen-underprices-residency-lever`).
    """
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device
    cache = Path(args.cache)
    model, p, state_dict = _dev_setup()
    reference, _, _, _, z_pre, m0 = _ref_front_half(
        state_dict, _full_construct_features(), torch.load(cache / "fi.pt"), p)
    dev = get_device()
    seen = []
    original = T._tri_att_sdpa

    def host_core(q, k, v, bias, scale):
        qh = ttnn.to_torch(q).float()
        kh = ttnn.to_torch(k).float()
        vh = ttnn.to_torch(v).float()
        bh = ttnn.to_torch(bias).float()
        if not seen:
            print(f"  [core] q={tuple(qh.shape)} k={tuple(kh.shape)} v={tuple(vh.shape)} "
                  f"bias={tuple(bh.shape)} scale={scale:.6f} qdtype={q.dtype}", flush=True)
            print(f"  [core] padded q={tuple(q.padded_shape)} bias={tuple(bias.padded_shape)}",
                  flush=True)
        seen.append(1)
        s_k = kh.shape[-2]
        keep = args.n if args.arm == "host-nopad" and args.n and args.n < s_k else s_k
        out = torch.empty_like(qh)
        step = 8
        for b0 in range(0, qh.shape[0], step):
            b1 = min(b0 + step, qh.shape[0])
            logit = qh[b0:b1] @ kh[b0:b1].transpose(-1, -2)
            logit = (logit + bh[:, :, :logit.shape[-2], :logit.shape[-1]]) * scale
            if keep < s_k:
                logit[..., keep:] = float("-inf")
            out[b0:b1] = logit.softmax(dim=-1) @ vh[b0:b1]
        return ttnn.from_torch(out, dtype=q.dtype, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    if args.arm != "dev":
        T._tri_att_sdpa = host_core
    try:
        for bi in [int(x) for x in args.blocks.split(",")]:
            m_src = m0 if bi == 0 else torch.load(cache / f"ref_m_{bi - 1}.pt")
            z_src = z_pre if bi == 0 else torch.load(cache / f"ref_z_{bi - 1}.pt")
            ref_taps = {}
            m_ref, z_ref = ref_msa_block(reference.msa_module.blocks[bi], m_src, z_src,
                                         taps=ref_taps)
            dev_taps = {}
            n = z_src.shape[-2]
            zt = p.trunk._up(z_src.reshape(1, n, n, -1))
            mt = p.trunk._up(m_src.reshape(1, *m_src.shape))
            m_dev, z_dev = dev_msa_block(p, p.trunk.MSA[bi], mt, zt, taps=dev_taps)
            print(f"\n--- MSA block {bi}, arm {args.arm} ---", flush=True)
            _stat(f"b{bi} BLOCK z out", p._to_host(z_dev, tuple(z_ref.shape)), z_ref)
            for name in ("tri_att_start", "tri_att_end"):
                ref_u = ref_taps[f"pair.{name}"]
                got = dev_taps[f"pair.{name}"]
                if name == "tri_att_end":
                    ref_u = ref_u.transpose(-2, -3)
                _stat(f"b{bi} pair.{name}", got, ref_u)
            ttnn.deallocate(z_dev)
    finally:
        T._tri_att_sdpa = original



def stage7(args):
    """Crop ladder on one triangle-attention op: is it the kernel's exp, or the tile padding?

    Stage 6 puts the whole disagreement inside `_tri_att_sdpa`. Two candidates survive, and
    a crop ladder separates them without needing to read a padded buffer. ttnn pads the
    sequence dim up to a 32 multiple, so at 136 tokens the kernel holds 160 key columns and
    the 24 it invents carry `layer_norm(0)` rather than -inf; at a crop that is already a
    multiple of 32 there is no padding to invent. A precision effect does not care about
    the crop, a padding leak dies on it.
    """
    import ttnn
    from tt_bio.opendde import load_opendde_checkpoint
    cache = Path(args.cache)
    model, p, state_dict = _dev_setup()
    reference, _ = _load_reference(state_dict)
    z_full = torch.load(cache / f"ref_z_{args.src_block}.pt").float()
    ref_pair = reference.msa_module.blocks[args.src_block + 1].pair_stack
    dev_pair = p.trunk.MSA[args.src_block + 1][3]
    for crop in [int(x) for x in args.crops.split(",")]:
        z = z_full[..., :crop, :crop, :].contiguous()
        print(f"\n--- crop {crop} (padded to {-(-crop // 32) * 32}) ---", flush=True)
        names = SUB_OPS if args.all_subops else ("tri_att_start", "tri_att_end")
        for name in names:
            ref_u = ref_sub(ref_pair, name, z.transpose(-2, -3).contiguous()
                            if name == "tri_att_end" else z)
            zt = p.trunk._up(z.reshape(1, crop, crop, -1))
            got = dev_sub(dev_pair, name, zt)
            got_h = p._to_host(got)
            if name == "tri_att_end":
                ref_u = ref_u.transpose(-2, -3)
            _stat(f"crop{crop} {name}", got_h, ref_u)
            ttnn.deallocate(got)
            ttnn.deallocate(zt)



def _chain_ref(reference, z0, m0, n_pf):
    m, z = m0, z0
    for b in range(N_MSA_BLOCKS):
        m, z = ref_msa_block(reference.msa_module.blocks[b], m, z)
    z_msa = z
    for i in range(n_pf):
        z = ref_block_z(reference.pairformer_stack.blocks[i], z)
        if (i + 1) % 12 == 0:
            print(f"  ref pf block {i} done", flush=True)
    return z_msa, z


def stage8(args):
    """End-to-end: MSA module + Pairformer, one arm per call, at a chosen crop.

    Stage 7 isolates the padding leak at op level. This prices it where it matters, at the
    output of the whole trunk z path, so the 0.947 headline can be compared against the same
    chain with the leak removed. Three arms, and the crop is the controlled variable:

      --crop 128 --arm dev   sequence length already a multiple of 32, no padding to invent
      --crop 136 --arm dev   the real length, the leak live
      --crop 136 --arm host  the real length, exact attention over the real keys only

    `--arm ref` builds and caches the fp32 reference chain for a crop; the device arms score
    against it. Cropping z and m rather than folding a shorter target is deliberate: it holds
    the weights, the MSA and the input error fixed and moves only the length.
    """
    import ttnn
    cache = Path(args.cache)
    out = cache / f"e2e_{args.crop}"
    out.mkdir(parents=True, exist_ok=True)
    crop = args.crop
    z_pre = torch.load(cache / "ref_z_pre.pt").float()[..., :crop, :crop, :].contiguous()
    m0 = torch.load(cache / "ref_m0.pt").float()[..., :crop, :].contiguous()

    if args.arm == "ref":
        from tt_bio.opendde import load_opendde_checkpoint
        reference, _ = _load_reference(load_opendde_checkpoint())
        z_msa, z_out = _chain_ref(reference, z_pre, m0, args.n_pf)
        torch.save(z_msa, out / "ref_z_msa.pt")
        torch.save(z_out, out / f"ref_z_out_{args.n_pf}.pt")
        print(f"cached reference chain at crop {crop}, {args.n_pf} pf blocks", flush=True)
        return

    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device
    model, p, _ = _dev_setup()
    dev = get_device()
    original = T._tri_att_sdpa
    reported = []

    def host_core(q, k, v, bias, scale):
        qh, kh = ttnn.to_torch(q).float(), ttnn.to_torch(k).float()
        vh, bh = ttnn.to_torch(v).float(), ttnn.to_torch(bias).float()
        if not reported:
            print(f"  [core] logical k={tuple(kh.shape)} padded k={tuple(k.padded_shape)}",
                  flush=True)
            reported.append(1)
        o = torch.empty_like(qh)
        for b0 in range(0, qh.shape[0], 8):
            b1 = min(b0 + 8, qh.shape[0])
            logit = qh[b0:b1] @ kh[b0:b1].transpose(-1, -2)
            logit = (logit + bh[:, :, :logit.shape[-2], :logit.shape[-1]]) * scale
            o[b0:b1] = logit.softmax(dim=-1) @ vh[b0:b1]
        return ttnn.from_torch(o, dtype=q.dtype, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    if args.arm == "host":
        T._tri_att_sdpa = host_core
    try:
        zt = p.trunk._up(z_pre.reshape(1, crop, crop, -1))
        mt = p.trunk._up(m0.reshape(1, *m0.shape))
        for b in range(N_MSA_BLOCKS):
            mt, zt = dev_msa_block(p, p.trunk.MSA[b], mt, zt)
        ref_msa = torch.load(out / "ref_z_msa.pt")
        _stat(f"crop{crop} {args.arm} z_post_msa", p._to_host(zt, tuple(ref_msa.shape)),
              ref_msa)
        for i in range(args.n_pf):
            zt = dev_block_z(p.trunk.PF.blocks[i], zt)
        ref_out = torch.load(out / f"ref_z_out_{args.n_pf}.pt")
        _stat(f"crop{crop} {args.arm} z_post_pairformer",
              p._to_host(zt, tuple(ref_out.shape)), ref_out)
        ttnn.deallocate(zt)
    finally:
        T._tri_att_sdpa = original


def _dev_msa_block_sub(p, blk, ref_blk, m, z, sub, m_shape, crop):
    """One device MSA block with ONE op class swapped for an exact fp32 reference.

    In-chain substitution (the af2ig technique): the reference op is fed the device's own
    input and everything else stays on device, so the arm prices that op class where it
    actually sits rather than in isolation.
    """
    import ttnn
    opm, pwa, tm, pl = blk
    u = ttnn.reshape(pwa(m, ttnn.clone(z)), tuple(m.shape))
    m = ttnn.add(m, u)
    u = ttnn.reshape(tm(m), tuple(m.shape))
    m = ttnn.add(m, u)
    if sub in ("opm", "both"):
        mh = p._to_host(m, m_shape).float()
        uh = ref_blk.outer_product_mean_msa(mh, inplace_safe=False, chunk_size=None)
        u = p.trunk._up(uh.reshape(1, crop, crop, -1))
    else:
        u = opm(m, None, None)
    z = ttnn.add(z, u)
    return m, dev_block_z(pl, z)


def stage9(args):
    """What does each candidate actually cost at the trunk z output?

    Stage 8 measured the padding leak end to end by removing it (crop 128, and an exact
    host attention core at crop 136) and found z_post_msa barely moves. This asks the same
    question of the other candidate the sub-op table flagged, OuterProductMean, with the
    same in-chain substitution and the same two anchors, so the terms are comparable:

      --sub none    the shipped device chain
      --sub triatt  exact fp32 attention core, over the device's own q/k/v/bias, real keys only
      --sub opm     exact fp32 OuterProductMean, over the device's own m
      --sub both    neither on device
    """
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device
    cache = Path(args.cache)
    out = cache / f"e2e_{args.crop}"
    crop = args.crop
    z_pre = torch.load(cache / "ref_z_pre.pt").float()[..., :crop, :crop, :].contiguous()
    m0 = torch.load(cache / "ref_m0.pt").float()[..., :crop, :].contiguous()
    model, p, state_dict = _dev_setup()
    dev = get_device()
    ref_blocks = None
    if args.sub in ("opm", "both"):
        reference, _ = _load_reference(state_dict)
        ref_blocks = reference.msa_module.blocks

    original = T._tri_att_sdpa
    reported = []

    def host_core(q, k, v, bias, scale):
        qh, kh = ttnn.to_torch(q).float(), ttnn.to_torch(k).float()
        vh, bh = ttnn.to_torch(v).float(), ttnn.to_torch(bias).float()
        if not reported:
            print(f"  [core] logical k={tuple(kh.shape)} padded k={tuple(k.padded_shape)}",
                  flush=True)
            reported.append(1)
        o = torch.empty_like(qh)
        for b0 in range(0, qh.shape[0], 8):
            b1 = min(b0 + 8, qh.shape[0])
            logit = qh[b0:b1] @ kh[b0:b1].transpose(-1, -2)
            logit = (logit + bh[:, :, :logit.shape[-2], :logit.shape[-1]]) * scale
            o[b0:b1] = logit.softmax(dim=-1) @ vh[b0:b1]
        return ttnn.from_torch(o, dtype=q.dtype, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    if args.sub in ("triatt", "both"):
        T._tri_att_sdpa = host_core
    try:
        zt = p.trunk._up(z_pre.reshape(1, crop, crop, -1))
        mt = p.trunk._up(m0.reshape(1, *m0.shape))
        for b in range(N_MSA_BLOCKS):
            mt, zt = _dev_msa_block_sub(
                p, p.trunk.MSA[b], None if ref_blocks is None else ref_blocks[b],
                mt, zt, args.sub, (1, *m0.shape), crop)
        ref_msa = torch.load(out / "ref_z_msa.pt")
        _stat(f"crop{crop} sub={args.sub} z_post_msa",
              p._to_host(zt, tuple(ref_msa.shape)), ref_msa)
        for i in range(args.n_pf):
            zt = dev_block_z(p.trunk.PF.blocks[i], zt)
        ref_out = torch.load(out / f"ref_z_out_{args.n_pf}.pt")
        _stat(f"crop{crop} sub={args.sub} z_post_pairformer",
              p._to_host(zt, tuple(ref_out.shape)), ref_out)
        ttnn.deallocate(zt)
    finally:
        T._tri_att_sdpa = original


def stage4(args):
    """STEP 2: why does the shipped fold read z at 0.947762 where the harness reads 0.957846?

    Everything runs in ONE process off ONE tapped input, so run-to-run and process-to-process
    variance cannot be the answer by construction. The ladder walks from the harness path to
    the shipped path one difference at a time:

      A/A   the shipped fold twice, same input, z_post_pairformer against itself
      L1    dev_block_z x 48 from the fold's own z_in            (the harness path)
      L2    block(None, z) x 48                                  (+ the layer's own call)
      L3    block(s, z) x 48 with the fold's own s               (+ the s branch and its L1 cost)

    L1 and L2 run the same five sub-ops in the same order, so a gap between them is the
    allocation pattern, not the arithmetic. A gap that only appears at L3 is the s branch.
    """
    import ttnn
    from scripts.opendde_real_seam_parity import _residue_trunk

    cache = Path(args.cache)
    ref_out = torch.load(cache / f"e2e_{args.crop}" / f"ref_z_out_{args.n_pf}.pt")
    feats = _full_construct_features()
    model, p, _ = _dev_setup()

    taps = {}
    original = type(p.trunk.PF).__call__

    def wrapped(self, s, z, *a, **kw):
        if self is p.trunk.PF and "z_in" not in taps:
            taps["z_in"] = p._to_host(z)
            taps["s_in"] = p._to_host(s)
        out = original(self, s, z, *a, **kw)
        if self is p.trunk.PF:
            taps.setdefault("z_out_%d" % len([k for k in taps if k.startswith("z_out")]),
                            p._to_host(out[1]))
        return out

    type(p.trunk.PF).__call__ = wrapped
    try:
        for rep in range(2):
            _residue_trunk(model, feats, n_cycles=1)
            print(f"  shipped fold rep {rep} done", flush=True)
    finally:
        type(p.trunk.PF).__call__ = original

    a, b = taps["z_out_0"].float(), taps["z_out_1"].float()
    print(f"\nA/A shipped fold twice: bit_exact={bool(torch.equal(a, b))} "
          f"maxabs={float((a - b).abs().max()):.3e} "
          f"rel={float((a - b).pow(2).mean().sqrt() / a.pow(2).mean().sqrt()):.3e}",
          flush=True)
    _stat("SHIPPED z_post_pairformer (expect 0.947762)", a, ref_out)

    z_in = taps["z_in"].float().reshape(1, *ref_out.shape[-3:])
    s_in = taps["s_in"].float()
    n = ref_out.shape[-2]
    blocks = p.trunk.PF.blocks

    for arm in args.arms.split(","):
        zt = p.trunk._up(z_in.reshape(1, n, n, -1))
        st = p.trunk._up(s_in.reshape(1, n, -1)) if arm == "L3" else None
        for i in range(args.n_pf):
            if arm == "L1":
                zt = dev_block_z(blocks[i], zt)
            elif arm == "L2":
                _, zt = blocks[i](None, zt)
            else:
                st, zt = blocks[i](st, zt)
        _stat(f"{arm} z_post_pairformer", p._to_host(zt, tuple(ref_out.shape)), ref_out)
        ttnn.deallocate(zt)
        if st is not None:
            ttnn.deallocate(st)


def stage10(args):
    """Is OuterProductMean the same bug class as the triangle attention, on the row axis?

    Stage 9 showed opm carries an error of the same size as the padding leak. opm reduces
    over the MSA row dim, which ttnn pads to a multiple of 32 exactly as it pads the token
    dim. Same discriminator as stage 7, moved to that axis: a row ladder. If the error dies
    when the row count is already a multiple of 32, it is invented rows, not arithmetic.
    """
    import ttnn
    cache = Path(args.cache)
    crop = args.crop
    m0 = torch.load(cache / "ref_m0.pt").float()[..., :crop, :].contiguous()
    model, p, state_dict = _dev_setup()
    reference, _ = _load_reference(state_dict)
    ref_blk = reference.msa_module.blocks[0]
    dev_opm = p.trunk.MSA[0][0]
    print(f"m0 {tuple(m0.shape)} tokens={crop}", flush=True)
    for rows in [int(x) for x in args.rows.split(",")]:
        mh = m0[..., :rows, :, :].contiguous() if m0.dim() == 3 else m0[:rows].contiguous()
        ref_u = ref_blk.outer_product_mean_msa(mh, inplace_safe=False, chunk_size=None)
        mt = p.trunk._up(mh.reshape(1, *mh.shape))
        got = dev_opm(mt, None, None)
        pad = tuple(mt.padded_shape)
        _stat(f"rows {rows:3d} (padded {pad}) opm",
              p._to_host(got, tuple(ref_u.shape)), ref_u)
        ttnn.deallocate(got)
        ttnn.deallocate(mt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--cache", default="_run/msa_probe")
    ap.add_argument("--pf-cache", default="_run/pf_trace")
    ap.add_argument("--blocks", default="0,1,2,3")
    ap.add_argument("--arm", default="dev",
                    choices=("dev", "host", "host-nopad", "ref"))
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--crops", default="96,128,132,136")
    ap.add_argument("--src-block", type=int, default=0)
    ap.add_argument("--crop", type=int, default=136)
    ap.add_argument("--n-pf", type=int, default=48)
    ap.add_argument("--all-subops", action="store_true")
    ap.add_argument("--arms", default="L1,L2,L3")
    ap.add_argument("--rows", default="32,64,76,84,96")
    ap.add_argument("--sub", default="none",
                    choices=("none", "triatt", "opm", "both"))
    args = ap.parse_args()
    {1: stage1, 2: stage2, 3: stage3, 4: stage4, 5: stage5, 6: stage6, 7: stage7,
     8: stage8, 9: stage9, 10: stage10}[args.stage](args)


if __name__ == "__main__":
    main()
