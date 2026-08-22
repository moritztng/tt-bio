"""tri_att_end's real localized error share in OpenDDE's 48-block Pairformer.

Pass 1 of `opendde-pairformer-z-parity-drop` localized the port's own 13% term to
TriangleAttention starting-node and had to exclude the ending-node column: the device's
ending variant transposes its update back to the original frame before the tap reads it
(`tenstorrent.py`, `if self.ending: x = _pair_transpose(...)` on the module output), while
the reference adds its update in the transposed frame. The column compared X against X^T,
which leaves the ending variant's real error unmeasured, not zero.

Two corrections, both in this harness. No forward path is touched, nothing in the model
changes, so no model output can move.

  1. Frame. Every tap is expressed in the ORIGINAL frame: the reference's ending-node
     input and update are transposed back before scoring. Both frames are still printed so
     the choice is named rather than assumed.
  2. Conditioning. The device block accumulates its own updates, so tri_att_end sees a z
     that tri_att_start has already perturbed, and pass 1 measured tri_att_start's update
     norm inflated up to 6.89x. Scoring the device update against the reference chain's
     update therefore charges tri_att_end for tri_att_start's error. Pass 1's device-off
     amplification arm, one level down. Four arms per sub-op:

       total  device update from the device's own input, scored against the reference
              chain's update. What the naive tap reads: implementation plus whatever the
              earlier sub-ops of this block already did to z.
       impl   same device update, scored against the reference sub-op replayed in host
              fp32 on the DEVICE's own input. Implementation only, at that input.
       iso    the same device sub-op called ON ITS OWN on the device's OWN tapped input,
              scored the same way `impl` is. Same input values as `impl`, different device
              state around the call, so iso != impl means the op's result depends on the
              calling context and not on its input.
       tf     device sub-op fed the REFERENCE's input, called on its own, scored against
              the reference's own update. No conditioning on either side.
       tf^T   the same, scored against the transposed reference update. A frame check on
              every sub-op, not just the ending variant.

Also checks, per sub-op, whether the call MUTATED the caller's z. `tenstorrent.py` notes the
starting variant can alias the caller's pair tensor; if the alias is ever written, the
shipped `PairformerLayer` residual adds into a corrupted z and that is a forward-path bug,
not a measurement artifact.

The default `--pad 0` IS the shipped OpenDDE configuration: the Protenix-family trunk calls
`pl(None, z3)` (`protenix.py:2243`, `:2353`) and `self.PF(s, z3)` (`:2438`), so `mask`,
`attn_mask_start` and `attn_mask_end` are all None, and the trunk does not pad the token dim.
`--pad 64` is therefore not what OpenDDE runs, it is the CURE arm: z zero-padded to a multiple
of `PAIRFORMER_PAD_MULTIPLE` on the host plus the additive -1e9 `attn_mask`, built by the same
recipe `PairformerModule` uses on its own non-affinity path (`tenstorrent.py:7042-7046`: one
`attn_mask` shared by start and end, the 1-D `mask_1d` as `mask_tt`), with every tap cropped
back before scoring. Models that reach the Pairformer through `PairformerModule` do take it.

`--maskmode` varies ONLY the TriangleMultiplication mask (`mask_tt`), never the additive
attention mask, and never the padded values. `TriangleMultiplication.__call__` does
`mask_u = unsqueeze(mask, -1)` and multiplies it into `a_chunk` of shape [1,S,S,C], so a 1-D
[1,S] mask broadcasts along the SECOND token axis only. The outgoing variant contracts that
axis and the incoming variant contracts the first, so the four modes below are an axis
discriminator: if which variant is clean follows which axis is zeroed, the leak is the mask's
axis and not the padded content.

  1d    mask_1d, [1,S]              what `PairformerModule` builds with no pair_mask
  j     ones_i * mask_1d_j, [1,S,S] second token axis only, the 1-D mask's actual effect
  i     mask_1d_i * ones_j, [1,S,S] first token axis only
  pair  outer product, [1,S,S]      both axes, what `Fp32PairformerModule` already builds

`--padtap` answers a separate question with no new instrument: is `tri_att_end` immune to the
tile-padding leak because its internal `_pair_transpose` rebuilds the tensor? It uses
`tri_att_start`, the one op already known to read the leaked padding, as the DETECTOR. Same
logical z, dirty tile padding (in-loop) vs zero-filled (host round trip); if the transpose
scrubs the padding then `tri_att_start` on the transposed pair is bit-identical across the two
while `tri_att_start` on the untransposed pair is not.

Needs pass 1's reference cache (`--stage 1` of `opendde_pairformer_block_trace.py`).

  OPENDDE_SRC=/tmp/opendde-src PYTHONPATH=<worktree> TT_VISIBLE_DEVICES=1 \
    TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:<slug> \
    python3 scripts/opendde_pf_triend_probe.py --blocks 0,8,24,36,47
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")

import torch

from scripts.opendde_real_seam_parity import pcc_ratio
from scripts.opendde_pairformer_block_trace import (
    SUB_OPS, _dev_setup, _load_reference, dev_sub, ref_block_z, ref_sub,
)

torch.set_grad_enabled(False)

# The one sub-op whose device update and reference update live in different frames.
TRANSPOSED = "tri_att_end"


def _T(t):
    return t.transpose(-2, -3).contiguous()


def _stat(tag, got, ref):
    """rel = error rms over reference rms, the quantity that composes across sub-ops."""
    g, r = got.float().reshape(ref.shape), ref.float()
    pcc, ratio = pcc_ratio(g, r)
    rms_r = float(r.pow(2).mean().sqrt())
    rms_e = float((g - r).pow(2).mean().sqrt())
    print(f"{tag:38s} PCC={pcc:9.6f} norm_ratio={ratio:8.4f} rms_ref={rms_r:.4e} "
          f"rms_err={rms_e:.4e} rel={rms_e / max(rms_r, 1e-30):.4e}", flush=True)
    return rms_e, rms_r


def ref_block_io(block, z, ins, ups):
    """`ref_block_z` with per-sub-op input and update taps, all in the ORIGINAL frame.

    The ending variant runs on z^T in the reference; its input and update are carried back
    here so every tap sits in the frame the device's taps do. Transposing an add is the
    same floats in a different order, so the returned z is bit-identical to `ref_block_z`'s
    and that is asserted below.
    """
    for name in ("tri_mul_out", "tri_mul_in", "tri_att_start"):
        ins[name] = z
        ups[name] = ref_sub(block, name, z)
        z = z + ups[name]
    ins[TRANSPOSED] = z
    update_t = ref_sub(block, TRANSPOSED, _T(z))
    ups[TRANSPOSED] = _T(update_t)
    z = z + ups[TRANSPOSED]
    ins["pair_transition"] = z
    ups["pair_transition"] = ref_sub(block, "pair_transition", z)
    return z + ups["pair_transition"]


def ref_on(block, name, z_in):
    """The reference sub-op on an arbitrary input, answer in the ORIGINAL frame."""
    if name == TRANSPOSED:
        return _T(ref_sub(block, name, _T(z_in)))
    return ref_sub(block, name, z_in)


def dev_sub_masked(block, name, z, masks):
    """`dev_sub` with the masks `PairformerLayer.__call__` passes at this position."""
    mask, attn = masks
    if name == "tri_mul_out":
        return block.triangle_multiplication_start(z, mask)
    if name == "tri_mul_in":
        return block.triangle_multiplication_end(z, mask)
    if name == "tri_att_start":
        return block.triangle_attention_start(z, attn)
    if name == "tri_att_end":
        return block.triangle_attention_end(z, attn)
    return block.transition_z(z)


def _sub(block, name, z, masks):
    return dev_sub(block, name, z) if masks is None else dev_sub_masked(block, name, z, masks)


def dev_on(p, block, name, z_host, shape, masks=None, crop=None):
    """One device sub-op on an arbitrary host input, answer in the ORIGINAL frame.

    The ending variant takes the original frame and transposes internally, so no frame
    handling is needed on this side.
    """
    import ttnn
    m = z_host.shape[-2]
    zt = p.trunk._up(z_host.reshape(1, m, m, -1))
    update = _sub(block, name, zt, masks)
    out = (crop or (lambda t: t.reshape(shape)))(p._to_host(update))
    ttnn.deallocate(update)
    try:
        ttnn.deallocate(zt)
    except RuntimeError:
        pass                      # some paths alias and free the caller's pair tensor
    return out


def dev_block_io(block, z, to_host, masks=None):
    """The z half of `PairformerLayer.__call__`, tapping each sub-op's input and update.

    `ttnn.add_` writes into z, so the input has to come down before the add and the update
    before the next sub-op runs.
    """
    import ttnn
    ins, ups, cfgs = {}, {}, {}
    for name in SUB_OPS:
        ins[name] = to_host(z)
        cfgs[name] = str(z.memory_config())
        update = _sub(block, name, z, masks)
        after = to_host(z)
        if not torch.equal(after, ins[name]):
            d = float((after - ins[name]).abs().max())
            print(f"  MUTATION {name} wrote the caller's z, maxabs {d:.3e}", flush=True)
        ups[name] = to_host(update)
        z = ttnn.add_(z, update)
        ttnn.deallocate(update)
    return z, ins, ups, cfgs


def padtap(p, block, src):
    """Does `tri_att_end`'s internal transpose scrub the tile padding it was handed?

    `tri_att_start` is the detector: it is the one sub-op whose result is known to depend on
    the padded columns (in-loop vs re-uploaded differ, VERDICT-TRIEND). Feed it a z with
    DIRTY tile padding and the same z with ZERO-FILLED tile padding.

      raw        no transpose. Must DIFFER, otherwise the detector is not live and nothing
                 below it means anything.
      reshaped   through the 4D->3D->4D relabel `tri_att.__call__` does first, no transpose.
                 A control: if the relabel alone scrubs, the transposed arm proves nothing.
      transposed through `_pair_transpose`, the call the ending variant makes. EQUAL here
                 with `raw` DIFFERENT is the transpose scrubbing the padding.
      end        `tri_att_end` itself on both, the claim being explained.
    """
    import ttnn
    from tt_bio.tenstorrent import _pair_transpose, _transpose_memory_config

    n = src.shape[-2]
    # dirty padding: three in-loop sub-ops accumulated with ttnn.add_, exactly as the block
    # does, so z arrives at the ending variant's position the way the model hands it over.
    z = p.trunk._up(src.reshape(1, n, n, -1))
    for name in ("tri_mul_out", "tri_mul_in", "tri_att_start"):
        u = dev_sub(block, name, z)
        z = ttnn.add_(z, u)
        ttnn.deallocate(u)
    z_dirty, host = z, p._to_host(z)
    z_clean = p.trunk._up(host.reshape(1, n, n, -1))
    print(f"padtap: logical values identical across the two z: "
          f"{bool(torch.equal(host, p._to_host(z_clean)))}  "
          f"padded_shape={tuple(z_dirty.padded_shape)} logical={tuple(z_dirty.shape)}",
          flush=True)

    def relabel(t):
        return ttnn.reshape(ttnn.reshape(t, tuple(t.shape)[1:]), (1,) + tuple(t.shape)[1:])

    def xpose(t):
        t3 = ttnn.reshape(t, tuple(t.shape)[1:])
        return ttnn.reshape(_pair_transpose(t3, _transpose_memory_config(t3)),
                            (1,) + tuple(t.shape)[1:])

    for tag, pre, op in (("raw", lambda t: t, "tri_att_start"),
                         ("reshaped", relabel, "tri_att_start"),
                         ("transposed", xpose, "tri_att_start"),
                         ("end", lambda t: t, "tri_att_end")):
        a = p._to_host(dev_sub(block, op, pre(z_dirty)))
        b = p._to_host(dev_sub(block, op, pre(z_clean)))
        same = bool(torch.equal(a, b))
        d = float((a.float() - b.float()).abs().max())
        print(f"padtap {tag:11s} -> {op:14s} dirty==clean: {str(same):5s} maxabs={d:.3e}",
              flush=True)


def main() -> None:
    import ttnn
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="_run/pf_trace")
    ap.add_argument("--blocks", default="0,8,24,36,47")
    ap.add_argument("--pad", type=int, default=0,
                    help="pad the token dim to this multiple and pass the shipped masks "
                         "(64 = PAIRFORMER_PAD_MULTIPLE, what the model runs)")
    ap.add_argument("--maskmode", default="1d", choices=("1d", "j", "i", "pair"),
                    help="how the TriangleMultiplication mask is built in the --pad arm; "
                         "an axis discriminator, see the module docstring")
    ap.add_argument("--crop", type=int, default=0,
                    help="crop the token dim of the cached z to this before anything else; "
                         "with --pad it varies HOW MANY columns are invented (0 at a "
                         "multiple of the pad), which is the ladder that says whether a "
                         "mask fix holds or just happens to work at 136")
    ap.add_argument("--padtap", action="store_true",
                    help="tri_att_end's immunity: does its internal transpose scrub the "
                         "tile padding? Uses tri_att_start as the detector, then exits")
    args = ap.parse_args()
    cache = Path(args.cache)

    model, p, state_dict = _dev_setup()
    ref_model, _ = _load_reference(state_dict)
    ref_blocks = ref_model.pairformer_stack.blocks
    dev_blocks = p.trunk.PF.blocks
    z_msa = torch.load(cache / "ref_z_msa.pt")
    to_host = lambda t: p._to_host(t)

    for i in [int(x) for x in args.blocks.split(",")]:
        src = z_msa if i == 0 else torch.load(cache / f"ref_z_{i - 1:02d}.pt")
        if args.crop:
            c = args.crop
            src = src[..., :c, :c, :].contiguous()
        shape = tuple(src.shape)
        n = shape[-2]
        rb, db = ref_blocks[i], dev_blocks[i]

        if args.padtap:
            print(f"\n=== padtap, Pairformer block {i} ===", flush=True)
            padtap(p, db, src)
            continue

        ins_r, ups_r = {}, {}
        z_ref = ref_block_io(rb, src, ins_r, ups_r)
        z_ref_z = ref_block_z(rb, src)
        cached = torch.load(cache / f"ref_z_{i:02d}.pt")
        print(f"\n=== Pairformer block {i}  |z_in| rms="
              f"{float(src.pow(2).mean().sqrt()):.4f}  |z_out| rms="
              f"{float(z_ref.pow(2).mean().sqrt()):.4f}", flush=True)
        print(f"INSTRUMENT ref_block_io == ref_block_z: {bool(torch.equal(z_ref, z_ref_z))}"
              f" | == cached chain: {bool(torch.equal(z_ref, cached))}", flush=True)
        # the frame plumbing itself: replaying the reference op on the reference's own
        # tapped input has to return the reference's own tapped update, exactly.
        for name in SUB_OPS:
            again = ref_on(rb, name, ins_r[name])
            if not torch.equal(again, ups_r[name]):
                d = float((again - ups_r[name]).abs().max())
                print(f"INSTRUMENT replay {name} maxabs diff {d:.3e}", flush=True)

        if args.pad:
            S = n + (-n) % args.pad
            z_in = torch.nn.functional.pad(src, (0, 0, 0, S - n, 0, S - n))
            mask_1d = torch.nn.functional.pad(torch.ones(1, n), (0, S - n))
            ones = torch.ones_like(mask_1d)
            mask_mul = {
                "1d": mask_1d,
                "j": ones[:, :, None] * mask_1d[:, None, :],
                "i": mask_1d[:, :, None] * ones[:, None, :],
                "pair": mask_1d[:, :, None] * mask_1d[:, None, :],
            }[args.maskmode]
            masks = (p.trunk._up(mask_mul),
                     p.trunk._up((1 - mask_1d).unsqueeze(1).unsqueeze(1) * -1e9))
            crop = lambda t: t.reshape(1, S, S, -1)[0, :n, :n, :].contiguous()
            print(f"shipped config: token dim {n} -> {S}, additive -1e9 mask on the pad, "
                  f"trimul mask_tt mode={args.maskmode} shape={tuple(mask_mul.shape)}",
                  flush=True)
        else:
            S, z_in, masks = n, src, None
            crop = lambda t: t.reshape(shape)

        z_dev, ins_d, ups_d, cfgs = dev_block_io(
            db, p.trunk._up(z_in.reshape(1, S, S, -1)), to_host, masks)
        host_dev_out = crop(p._to_host(z_dev))
        ttnn.deallocate(z_dev)
        blk_e, blk_r = _stat(f"b{i} BLOCK z out", host_dev_out, z_ref)
        blk_pcc = pcc_ratio(host_dev_out.reshape(-1), z_ref.reshape(-1))[0]

        # reassembly: the device taps must rebuild the device block output, otherwise a tap
        # is not the sub-op's real result and nothing below it means anything.
        z_re = src.clone()
        for name in SUB_OPS:
            z_re = z_re + crop(ups_d[name]).float()
        print(f"reassembled dev taps vs dev block out PCC="
              f"{pcc_ratio(z_re, host_dev_out)[0]:.8f}", flush=True)

        errs = {}
        for name in SUB_OPS:
            got = crop(ups_d[name]).float()
            e_tot, _ = _stat(f"b{i} {name} total", got, ups_r[name])
            dev_in = crop(ins_d[name]).float()
            impl = ref_on(rb, name, dev_in)
            e_impl, _ = _stat(f"b{i} {name} impl", got, impl)
            iso_in = (torch.nn.functional.pad(dev_in, (0, 0, 0, S - n, 0, S - n))
                      if args.pad else dev_in)
            e_iso, _ = _stat(f"b{i} {name} iso",
                             dev_on(p, db, name, iso_in, shape, masks, crop), impl)
            tf_in = (torch.nn.functional.pad(ins_r[name], (0, 0, 0, S - n, 0, S - n))
                     if args.pad else ins_r[name])
            tf = dev_on(p, db, name, tf_in, shape, masks, crop).float()
            e_tf, _ = _stat(f"b{i} {name} tf", tf, ups_r[name])
            _stat(f"b{i} {name} tf [frame-swap]", tf, _T(ups_r[name]))
            errs[name] = (e_tf, e_iso, e_impl, e_tot)
            print(f"  in-loop z memory_config for {name}: {cfgs[name]}", flush=True)

        quad = {k: sum(v[j] ** 2 for v in errs.values()) ** 0.5
                for j, k in enumerate(("tf", "iso", "impl", "total"))}
        s_tf, e_tf = errs["tri_att_start"][0], errs[TRANSPOSED][0]
        share = e_tf ** 2 / max(s_tf ** 2 + e_tf ** 2, 1e-30)
        print(f"b{i} SUMMARY block 1-PCC={1 - blk_pcc:.4e} block err rms={blk_e:.4f} | "
              f"quadrature tf={quad['tf']:.4f} iso={quad['iso']:.4f} "
              f"impl={quad['impl']:.4f} total={quad['total']:.4f} | "
              f"tf start={s_tf:.4f} end={e_tf:.4f} "
              f"end/start={e_tf / max(s_tf, 1e-30):.4f} "
              f"end share of the two={100 * share:.3f}%", flush=True)


if __name__ == "__main__":
    main()
