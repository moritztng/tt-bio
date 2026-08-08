#!/usr/bin/env python3
"""One real PairformerLayer at a large-target shape: output dump + DRAM high-water.

The capacity instrument for the large-target OOM workstream. Builds ONE pairformer
block from the real checkpoint (opendde refiner: c_z=384, tri (32,12), att (48,8);
protenix-v2 trunk: c_z=256, tri (32,8), att (24,16)) and runs it on a random
(s, z) pair at a structural-scale token count N. Values must be compared across
code versions with --compare: every pair-track reordering behind
SEQ_LEN_MORE_CHUNKING is row-local, so before/after outputs are required to be
bit-identical, not approximately equal.

DRAM high-water goes to the TT_BIO_DRAM_PEAK file (tags "pairlayer <tag> ...").

    TT_VISIBLE_DEVICES=3 TT_BIO_DRAM_PEAK=/tmp/dram.log \
        python3 perf/large_target_oom/pairlayer_capacity.py --model opendde --n 1712 --out /tmp/leg_after.pt
    python3 perf/large_target_oom/pairlayer_capacity.py --compare /tmp/leg_before.pt /tmp/leg_after.pt
"""

import argparse

import torch

import ttnn
from tt_bio.tenstorrent import PairformerLayer, get_device, dram_peak

import os

OPENDDE_CKPT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--aurekaresearch--OpenDDE/"
    "snapshots/eddd563ce96571f784012edd8f045181c8f8627d/opendde_abag.pt")
PROTENIX_CKPT = os.path.expanduser("~/.boltz/protenix-v2.pt")


def build_layer(model, ckc):
    """Real block-0 weights; returns (layer, c_z). Mirrors the production configs."""
    if model == "opendde":
        from tt_bio.opendde import load_opendde_checkpoint, route_opendde_weights
        routed = route_opendde_weights(load_opendde_checkpoint(OPENDDE_CKPT, abag=True))
        blk = {k[len("layers.0."):]: v for k, v in routed["refiner"].items()
               if k.startswith("layers.0.")}
        # OPENDDE_CONFIG: c_z=384, refiner_tri_heads=12, refiner_att_heads=8
        return PairformerLayer(384 // 12, 12, 384 // 8, 8, True, blk, ckc), 384
    from tt_bio import protenix_weights as PW
    ck = torch.load(PROTENIX_CKPT, map_location="cpu", weights_only=True)
    ck = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
    blk = {k[len("pairformer_stack.blocks.0."):]: v for k, v in sd.items()
           if k.startswith("pairformer_stack.blocks.0.")}
    remapped = PW.remap_pairformer_block(blk)
    c_z = remapped["tri_mul_out.p_in.weight"].shape[1]
    return PairformerLayer(32, c_z // 32, 384 // 16, 16, True, remapped, ckc), c_z


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["opendde", "protenix"], default="opendde")
    ap.add_argument("--n", type=int, default=1712, help="token count (9ivj structural = 1712)")
    ap.add_argument("--tag", default="leg", help="dram_peak tag prefix")
    ap.add_argument("--out", default=None, help="write (s, z) outputs here for --compare")
    ap.add_argument("--compare", nargs=2, default=None, help="two --out files; bit-compare and exit")
    args = ap.parse_args()

    if args.compare:
        a_s, a_z = torch.load(args.compare[0], map_location="cpu")
        b_s, b_z = torch.load(args.compare[1], map_location="cpu")
        ok = True
        for name, a, b in (("s", a_s, b_s), ("z", a_z, b_z)):
            same = torch.equal(a, b)
            d = (a.float() - b.float()).abs().max().item()
            print(f"{name}: {'BIT-EXACT' if same else f'DIFFERS maxabs {d:.3e}'}")
            ok &= same
        print("COMPARE", "PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 1)

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    layer, c_z = build_layer(args.model, ckc)
    N = args.n
    torch.manual_seed(0)
    s = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    print(f"layer built: model={args.model} c_z={c_z} N={N}", flush=True)

    dram_peak(f"pairlayer {args.tag} enter")
    s, z = layer(s, z)   # warm (program cache, allocator pools)
    ttnn.synchronize_device(dev)
    dram_peak(f"pairlayer {args.tag} warm")
    s, z = layer(s, z)   # measured
    ttnn.synchronize_device(dev)
    dram_peak(f"pairlayer {args.tag} measured")
    # Fresh-input call: the comparison artifact. Running the layer twice on the same
    # tensors would feed a normed/updated (s, z) back in, which is a different input
    # distribution than a before/after code compare needs.
    s2 = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    z2 = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    s3, z3 = layer(s2, z2)
    ttnn.synchronize_device(dev)
    dram_peak(f"pairlayer {args.tag} fresh-input")

    mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    used = (mv.total_bytes_per_bank - mv.total_bytes_free_per_bank) * mv.num_banks / 2**30
    print(f"DRAM now: {used:.3f} GiB (see TT_BIO_DRAM_PEAK file for the high-water tags)")

    if args.out:
        torch.save((ttnn.to_torch(s3), ttnn.to_torch(z3)), args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
