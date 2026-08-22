"""Per-block PCC trace of OpenDDE's 48-block residue Pairformer, z track.

The trunk-level report in scripts/opendde_trunk_real_parity.py puts z at PCC 0.999337
entering the Pairformer and 0.947509 leaving it, one cycle, while s holds at 0.999620.
This script localizes that: it steps the reference and the device stack block by block
from the same z and reports four arms.

  ref-fp32      the upstream torch reference in fp32, the anchor for every other arm
  bf16-torch    the same reference with z and every weight cast to bfloat16, still on
                host torch. This is the precision floor of a 48-block bf16 chain and it
                is the control that says whether the device is losing anything the
                dtype does not already lose.
  device-free   the ttnn stack running its own z forward, scored per block against
                ref-fp32. Its slope is the compounding rate.
  device-forced the ttnn stack fed ref-fp32's z at every block, one block at a time.
                No accumulation, so this is per-block injection.

z does not depend on s inside a block (PairformerBlock.forward only reads z for the s
update), so all four arms run a z-only path and s is never materialized.

Stage 1 runs the reference front half once and caches z entering the Pairformer plus the
per-block ref-fp32 chain under --cache. Stage 2 reads that cache, so the arms are cheap
to re-run.

  OPENDDE_SRC=/tmp/opendde-src PYTHONPATH=<worktree> \
    python3 scripts/opendde_pairformer_block_trace.py --stage 1
  ... --stage 2 --arms bf16-torch,device-free,device-forced
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

torch.set_grad_enabled(False)

ROOT = Path(__file__).resolve().parent.parent
N_BLOCKS = 48
SUB_OPS = ("tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end", "pair_transition")


# --------------------------------------------------------------------------- reference

def _load_reference(state_dict):
    sys.path.insert(0, SRC)
    sys.modules.setdefault("optree", types.ModuleType("optree"))
    from opendde.config.inference import build_inference_config
    from opendde.model.opendde import OpenDDE as ReferenceOpenDDE

    cfg = build_inference_config(fill_required_with_null=True)
    cfg.triangle_multiplicative = "torch"
    cfg.triangle_attention = "torch"
    cfg.enable_efficient_fusion = False
    reference = ReferenceOpenDDE(cfg).eval()
    reference.load_state_dict(state_dict, strict=True)
    return reference, cfg


def ref_sub(block, name, z):
    """One sub-op update, exactly as PairformerBlock.forward's non-inplace branch calls it."""
    if name == "tri_mul_out":
        return block.tri_mul_out(z, mask=None, inplace_safe=False, _add_with_inplace=False,
                                 triangle_multiplicative="torch")
    if name == "tri_mul_in":
        return block.tri_mul_in(z, mask=None, inplace_safe=False, _add_with_inplace=False,
                                triangle_multiplicative="torch")
    if name == "tri_att_start":
        return block.tri_att_start(z, mask=None, triangle_attention="torch",
                                   inplace_safe=False, chunk_size=None)
    if name == "tri_att_end":
        return block.tri_att_end(z, mask=None, triangle_attention="torch",
                                 inplace_safe=False, chunk_size=None)
    return block.pair_transition(z)


def ref_block_z(block, z, taps=None):
    """The z half of PairformerBlock.forward, non-inplace branch, verbatim.

    tri_att_end runs on the transposed pair, which is what the reference does and what
    the device's transpose_bias path mirrors; the transpose is part of the op, not of
    this harness.
    """
    z = z + _tap(taps, "tri_mul_out", ref_sub(block, "tri_mul_out", z))
    z = z + _tap(taps, "tri_mul_in", ref_sub(block, "tri_mul_in", z))
    z = z + _tap(taps, "tri_att_start", ref_sub(block, "tri_att_start", z))
    z = z.transpose(-2, -3).contiguous()
    z = z + _tap(taps, "tri_att_end", ref_sub(block, "tri_att_end", z))
    z = z.transpose(-2, -3).contiguous()
    z = z + _tap(taps, "pair_transition", ref_sub(block, "pair_transition", z))
    return z


def _tap(taps, name, value):
    if taps is not None:
        taps[name] = value
    return value


# --------------------------------------------------------------------------- device

def dev_sub(block, name, z):
    if name == "tri_mul_out":
        return block.triangle_multiplication_start(z, None)
    if name == "tri_mul_in":
        return block.triangle_multiplication_end(z, None)
    if name == "tri_att_start":
        return block.triangle_attention_start(z, None)
    if name == "tri_att_end":
        return block.triangle_attention_end(z, None)
    return block.transition_z(z)


def dev_block_z(block, z, taps=None, to_host=None):
    """The z half of PairformerLayer.__call__, minus the s branch.

    ttnn.add_ writes into z, so a tap has to be downloaded before the next sub-op runs.
    """
    import ttnn
    for name in SUB_OPS:
        update = dev_sub(block, name, z)
        if taps is not None:
            taps[name] = to_host(update)
        z = ttnn.add_(z, update)
        ttnn.deallocate(update)
    return z


# --------------------------------------------------------------------------- stages

def stage1(args):
    """Run the reference front half + the fp32 block chain once and cache both."""
    import ttnn
    from scripts.opendde_real_seam_parity import _residue_trunk
    from tt_bio.opendde import OpenDDE, load_opendde_checkpoint
    from tt_bio.tenstorrent import get_device
    import torch.nn.functional as F

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    feats = _full_construct_features()
    state_dict = load_opendde_checkpoint()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    model = OpenDDE(state_dict, ckc, dev)
    p = model._protenix
    # the fold path itself, for fi (the trunked distance/vector features the reference
    # needs) and for the end-to-end z the trunk report already published
    _, _, got_z, fi, _, _ = _residue_trunk(model, feats, n_cycles=1)

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
    z_msa = reference.msa_module(
        ref_feats, z, s_inputs, pair_mask=None,
        triangle_multiplicative=cfg.triangle_multiplicative,
        triangle_attention=cfg.triangle_attention, inplace_safe=False, chunk_size=None)
    torch.save(z_msa, cache / "ref_z_msa.pt")
    print(f"cached ref_z_msa {tuple(z_msa.shape)}", flush=True)

    blocks = reference.pairformer_stack.blocks
    zi = z_msa
    for i in range(N_BLOCKS):
        zi = ref_block_z(blocks[i], zi)
        torch.save(zi, cache / f"ref_z_{i:02d}.pt")
        print(f"ref-fp32 block {i} done", flush=True)

    # instrument check: the z-only chain must reproduce the full stack's z bit for bit,
    # since s never feeds z. Anything else means this harness is not the reference.
    _, ref_s_full, ref_z_full = reference.get_pairformer_output(
        ref_feats, N_cycle=1, inplace_safe=False, chunk_size=None)
    torch.save(ref_z_full, cache / "ref_z_full.pt")
    same = bool(torch.equal(zi, ref_z_full))
    print(f"INSTRUMENT z-only chain == full stack z: {same} "
          f"maxabs={float((zi - ref_z_full).abs().max()):.3e}", flush=True)
    pcc, ratio = pcc_ratio(got_z, ref_z_full)
    print(f"fold-path z vs ref PCC={pcc:.6f} norm_ratio={ratio:.6f}", flush=True)


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


def stage2(args):
    cache = Path(args.cache)
    arms = args.arms.split(",")
    ref = [torch.load(cache / f"ref_z_{i:02d}.pt") for i in range(N_BLOCKS)]
    z_msa = torch.load(cache / "ref_z_msa.pt")
    n = z_msa.shape[-2]

    if "bf16-torch" in arms:
        state_dict = None
        from tt_bio.opendde import load_opendde_checkpoint
        state_dict = load_opendde_checkpoint()
        reference, _ = _load_reference(state_dict)
        blocks = reference.pairformer_stack.blocks
        for b in blocks:
            b.to(torch.bfloat16)
        zi = z_msa.to(torch.bfloat16)
        for i in range(N_BLOCKS):
            zi = ref_block_z(blocks[i], zi)
            pcc, ratio = pcc_ratio(zi.float(), ref[i])
            print(f"bf16-torch   block {i:2d} PCC={pcc:.6f} norm_ratio={ratio:.6f}",
                  flush=True)
        for b in blocks:
            b.to(torch.float32)

    if "device-free" in arms or "device-forced" in arms or args.subops:
        import ttnn
        model, p, state_dict = _dev_setup()
        blocks = p.trunk.PF.blocks
        to_host = lambda t, shape=None: p._to_host(t, shape)

        if "device-free" in arms:
            zt = p.trunk._up(z_msa.reshape(1, n, n, -1))
            for i in range(N_BLOCKS):
                zt = dev_block_z(blocks[i], zt)
                pcc, ratio = pcc_ratio(to_host(zt, tuple(ref[i].shape)), ref[i])
                print(f"device-free  block {i:2d} PCC={pcc:.6f} norm_ratio={ratio:.6f}",
                      flush=True)
            ttnn.deallocate(zt)

        if "device-forced" in arms:
            for i in range(N_BLOCKS):
                src = z_msa if i == 0 else ref[i - 1]
                zt = p.trunk._up(src.reshape(1, n, n, -1))
                zt = dev_block_z(blocks[i], zt)
                pcc, ratio = pcc_ratio(to_host(zt, tuple(ref[i].shape)), ref[i])
                print(f"device-forced block {i:2d} PCC={pcc:.6f} norm_ratio={ratio:.6f}",
                      flush=True)
                ttnn.deallocate(zt)

        if args.subops:
            ref_model, _ = _load_reference(state_dict)
            ref_blocks = ref_model.pairformer_stack.blocks
            for i in [int(x) for x in args.subops.split(",")]:
                src = z_msa if i == 0 else ref[i - 1]
                ref_taps = {}
                ref_block_z(ref_blocks[i], src, taps=ref_taps)
                dev_taps = {}
                zt = p.trunk._up(src.reshape(1, n, n, -1))
                zt = dev_block_z(blocks[i], zt, taps=dev_taps, to_host=to_host)
                ttnn.deallocate(zt)
                for name in SUB_OPS:
                    r = ref_taps[name]
                    g = dev_taps[name].reshape(r.shape)
                    pcc, ratio = pcc_ratio(g, r)
                    rms_r = float(r.float().pow(2).mean().sqrt())
                    rms_e = float((g.float() - r.float()).pow(2).mean().sqrt())
                    print(f"subop block {i:2d} {name:16s} PCC={pcc:.6f} "
                          f"norm_ratio={ratio:.6f} rms_ref={rms_r:.4e} "
                          f"rms_err={rms_e:.4e} rel={rms_e / max(rms_r, 1e-30):.4e}",
                          flush=True)



def stage3(args):
    """Capture the fold path's own z entering and leaving the Pairformer.

    Wraps Pairformer.__call__ instead of touching model code, so the tap sees exactly the
    tensor the shipped path passes, with the shipped s update and allocation order intact.
    """
    import ttnn
    from scripts.opendde_real_seam_parity import _residue_trunk
    from tt_bio.opendde import load_opendde_checkpoint

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    feats = _full_construct_features()
    model, p, _ = _dev_setup()
    taps = {}
    original = type(p.trunk.PF).__call__

    def wrapped(self, s, z, *a, **kw):
        if self is p.trunk.PF:
            taps.setdefault("z_in", p._to_host(z))
        out = original(self, s, z, *a, **kw)
        if self is p.trunk.PF:
            taps.setdefault("z_out", p._to_host(out[1]))
        return out

    type(p.trunk.PF).__call__ = wrapped
    try:
        _residue_trunk(model, feats, n_cycles=1)
    finally:
        type(p.trunk.PF).__call__ = original
    for key in ("z_in", "z_out"):
        torch.save(taps[key], cache / f"dev_{key}.pt")
        print(f"cached dev_{key} {tuple(taps[key].shape)}", flush=True)
    ref = torch.load(cache / "ref_z_msa.pt")
    pcc, ratio = pcc_ratio(taps["z_in"].reshape(ref.shape), ref)
    print(f"dev z_in vs ref_z_msa PCC={pcc:.6f} norm_ratio={ratio:.6f}", flush=True)
    ref_out = torch.load(cache / "ref_z_47.pt")
    pcc, ratio = pcc_ratio(taps["z_out"].reshape(ref_out.shape), ref_out)
    print(f"dev z_out vs ref_z_47 PCC={pcc:.6f} norm_ratio={ratio:.6f}", flush=True)


def stage4(args):
    """Amplification arms: run each stack from the fold path's own z, not the reference's.

    ref-from-devz is the decisive one. It replays the fp32 reference stack, no device at
    all, on the device's z. Whatever it loses is amplification of an error the Pairformer
    inherited, so anything it reproduces is not a Pairformer defect.
    """
    cache = Path(args.cache)
    arms = args.arms.split(",")
    ref = [torch.load(cache / f"ref_z_{i:02d}.pt") for i in range(N_BLOCKS)]
    dev_z_in = torch.load(cache / "dev_z_in.pt").reshape(ref[0].shape).float()
    n = ref[0].shape[-2]

    if "ref-from-devz" in arms or "bf16-from-devz" in arms:
        from tt_bio.opendde import load_opendde_checkpoint
        reference, _ = _load_reference(load_opendde_checkpoint())
        blocks = reference.pairformer_stack.blocks
        if "ref-from-devz" in arms:
            zi = dev_z_in.clone()
            for i in range(N_BLOCKS):
                zi = ref_block_z(blocks[i], zi)
                pcc, ratio = pcc_ratio(zi, ref[i])
                print(f"ref-from-devz  block {i:2d} PCC={pcc:.6f} "
                      f"norm_ratio={ratio:.6f}", flush=True)
        if "bf16-from-devz" in arms:
            for b in blocks:
                b.to(torch.bfloat16)
            zi = dev_z_in.to(torch.bfloat16)
            for i in range(N_BLOCKS):
                zi = ref_block_z(blocks[i], zi)
                pcc, ratio = pcc_ratio(zi.float(), ref[i])
                print(f"bf16-from-devz block {i:2d} PCC={pcc:.6f} "
                      f"norm_ratio={ratio:.6f}", flush=True)
            for b in blocks:
                b.to(torch.float32)

    if "device-from-devz" in arms:
        import ttnn
        model, p, _ = _dev_setup()
        blocks = p.trunk.PF.blocks
        zt = p.trunk._up(dev_z_in.reshape(1, n, n, -1))
        for i in range(N_BLOCKS):
            zt = dev_block_z(blocks[i], zt)
            pcc, ratio = pcc_ratio(p._to_host(zt, tuple(ref[i].shape)), ref[i])
            print(f"device-from-devz block {i:2d} PCC={pcc:.6f} "
                  f"norm_ratio={ratio:.6f}", flush=True)
        ttnn.deallocate(zt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--cache", default="_run/pf_trace")
    ap.add_argument("--arms", default="bf16-torch,device-free,device-forced")
    ap.add_argument("--subops", default="")
    args = ap.parse_args()
    {1: stage1, 2: stage2, 3: stage3, 4: stage4}[args.stage](args)


if __name__ == "__main__":
    main()
