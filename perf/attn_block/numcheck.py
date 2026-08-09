#!/usr/bin/env python3
"""One Pairformer block, fixed seed, dump s/z. Run once per code version and diff."""
import argparse, torch, ttnn
from tt_bio import protenix_weights as PW
from tt_bio.tenstorrent import PairformerLayer, get_device

CKPT = "/home/ttuser/.boltz/protenix-v2.pt"

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=320)
ap.add_argument("--out", required=True)
a = ap.parse_args()

dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
ck = torch.load(CKPT, map_location="cpu", weights_only=True)
ck = ck.get("model", ck)
sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
blk = {k[len("pairformer_stack.blocks.0."):]: v for k, v in sd.items()
       if k.startswith("pairformer_stack.blocks.0.")}
rm = PW.remap_pairformer_block(blk)
c_z = rm["tri_mul_out.p_in.weight"].shape[1]
layer = PairformerLayer(32, c_z // 32, 384 // 16, 16, True, rm, ckc)
N = a.n
torch.manual_seed(0)
s = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
z = ttnn.from_torch(torch.randn(1, N, N, c_z) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
s, z = layer(s, z)
ttnn.synchronize_device(dev)
torch.save({"s": ttnn.to_torch(s), "z": ttnn.to_torch(z)}, a.out)
print("wrote", a.out)
