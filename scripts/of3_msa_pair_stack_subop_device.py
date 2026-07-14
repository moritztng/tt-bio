"""Device bisect: localize which pair_stack sub-op the DEVICE loses precision in.

CPU leg (of3_msa_pair_stack_subop_golden.py) found the entire 18->270 z-magnitude jump
is in pair_transition (SwiGLU), and CPU-bf16 tracks every sub-op to >=0.99983. This
script runs the 5 device PairformerLayer sub-ops (tri_mul_out, tri_mul_in, tri_att_start,
tri_att_end, pair_transition) in order on the real block-0 pair_stack input, capturing z
after each, and compares to the CPU fp32 per-sub-op goldens -- isolating which sub-op's
DEVICE compute departs from fp32 (and how much worse than the CPU-bf16 0.99983 baseline,
which separates a device-only mechanism from generic bf16 rounding).

    TT_VISIBLE_DEVICES=2 python3 scripts/of3_msa_pair_stack_subop_device.py
"""
import os, pickle, torch, ttnn

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
GOLD = os.path.expanduser("~/of3_msa_pair_stack_subops.pkl")
_TRI_DIMS = (32, 4)  # c_hidden_pair_att=32, no_heads_pair=4 (OF3 msa pair_stack)


def pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def main():
    from tt_bio.tenstorrent import get_device, PairformerLayer
    from tt_bio.openfold3_weights import remap_msa_block, _sub

    g = pickle.load(open(GOLD, "rb"))
    z_in = g["z_in"]            # [N,N,c_z]
    fp32 = g["fp32"]            # {sub_op: z_after [N,N,c_z]}
    bf16_pcc = g["bf16_pcc"]    # {sub_op: cpu-bf16 pcc vs fp32}
    zsh = z_in.shape

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    block_remap = remap_msa_block(_sub(sd, "msa_module.blocks.0"))
    pl = PairformerLayer(*_TRI_DIMS, None, None, False, block_remap["pair_stack"], ckc,
                         fp32_softmax=True)

    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    z = ft(z_in.unsqueeze(0))

    sub_steps = [
        ("tri_mul_out", pl.triangle_multiplication_start, None),
        ("tri_mul_in", pl.triangle_multiplication_end, None),
        ("tri_att_start", pl.triangle_attention_start, None),
        ("tri_att_end", pl.triangle_attention_end, None),
        ("pair_transition", pl.transition_z, None),
    ]
    print(f"{'sub_op':<18}{'dev_z_std':>10}{'fp32_z_std':>12}{'dev_pcc':>10}{'cpu_bf16_pcc':>14}")
    results = {}
    for name, fn, mask in sub_steps:
        if name == "pair_transition":
            z_update = fn(z)
        elif name.startswith("tri_att"):
            z_update = fn(z, mask)
        else:
            z_update = fn(z, mask)
        z = ttnn.add_(z, z_update)
        zo = torch.Tensor(ttnn.to_torch(z)).float().reshape(zsh)
        ref = fp32[name].float()
        dev_pcc = pcc(zo, ref)
        results[name] = (float(zo.std()), dev_pcc)
        print(f"{name:<18}{float(zo.std()):>10.2f}{float(ref.std()):>12.2f}"
              f"{dev_pcc:>10.5f}{bf16_pcc[name]:>14.5f}")
    # Final pair_stack z_pcc (cumulative) vs the full fp32 pair_stack output
    final = torch.Tensor(ttnn.to_torch(z)).float().reshape(zsh)
    print(f"\nFinal pair_stack z_pcc (device vs fp32): {pcc(final, fp32['pair_transition'].float()):.5f}")
    print(f"(baseline test_of3_msa_block0 z_pcc ~0.708)")


if __name__ == "__main__":
    main()
