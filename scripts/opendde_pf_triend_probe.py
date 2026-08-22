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
     amplification arm, one level down: replay the reference sub-op in host fp32 on the
     DEVICE's own input to that sub-op. `impl` is the op's own error, `total` is what the
     naive tap reads, and `impl` is the number that belongs next to pass 1's 13%.

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


def dev_block_io(block, z, to_host):
    """The z half of `PairformerLayer.__call__`, tapping each sub-op's input and update.

    `ttnn.add_` writes into z, so the input has to come down before the add and the update
    before the next sub-op runs.
    """
    import ttnn
    ins, ups = {}, {}
    for name in SUB_OPS:
        ins[name] = to_host(z)
        update = dev_sub(block, name, z)
        ups[name] = to_host(update)
        z = ttnn.add_(z, update)
        ttnn.deallocate(update)
    return z, ins, ups


def main() -> None:
    import ttnn
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="_run/pf_trace")
    ap.add_argument("--blocks", default="0,8,24,36,47")
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
        shape = tuple(src.shape)
        n = shape[-2]
        rb, db = ref_blocks[i], dev_blocks[i]

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

        z_dev, ins_d, ups_d = dev_block_io(db, p.trunk._up(src.reshape(1, n, n, -1)),
                                           to_host)
        host_dev_out = p._to_host(z_dev, shape)
        ttnn.deallocate(z_dev)
        blk_e, blk_r = _stat(f"b{i} BLOCK z out", host_dev_out, z_ref)
        blk_pcc = pcc_ratio(host_dev_out.reshape(-1), z_ref.reshape(-1))[0]

        # reassembly: the device taps must rebuild the device block output, otherwise a tap
        # is not the sub-op's real result and nothing below it means anything.
        z_re = src.clone()
        for name in SUB_OPS:
            z_re = z_re + ups_d[name].reshape(shape).float()
        print(f"reassembled dev taps vs dev block out PCC="
              f"{pcc_ratio(z_re, host_dev_out)[0]:.8f}", flush=True)

        errs = {}
        for name in SUB_OPS:
            got = ups_d[name].reshape(shape).float()
            impl = ref_on(rb, name, ins_d[name].reshape(shape).float())
            e_impl, _ = _stat(f"b{i} {name} impl", got, impl)
            e_tot, _ = _stat(f"b{i} {name} total", got, ups_r[name])
            errs[name] = (e_impl, e_tot)
            if name == TRANSPOSED:
                # both frames, so the fix is named and not assumed
                _stat(f"b{i} {name} total [WRONG frame]", got, _T(ups_r[name]))

        quad_impl = sum(v[0] ** 2 for v in errs.values()) ** 0.5
        quad_tot = sum(v[1] ** 2 for v in errs.values()) ** 0.5
        print(f"b{i} SUMMARY block 1-PCC={1 - blk_pcc:.4e} block err rms={blk_e:.4f} | "
              f"quadrature impl={quad_impl:.4f} total={quad_tot:.4f} | "
              f"start impl={errs['tri_att_start'][0]:.4f} "
              f"end impl={errs[TRANSPOSED][0]:.4f} "
              f"end/start={errs[TRANSPOSED][0] / max(errs['tri_att_start'][0], 1e-30):.4f}",
              flush=True)


if __name__ == "__main__":
    main()
