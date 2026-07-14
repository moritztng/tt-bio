"""Trace the pair_stack sub-op z-std trajectory with DEVICE OPM z_in vs CLEAN fp32 z_in.

The full-block bisect showed device-OPM z_in (pcc=1.0 vs fp32, std 18.179) feeds pair_stack
to output std 610, while clean fp32-cast-bf16 z_in (std 18.173) feeds it to std 295 -- a
2x over-amplification from a pcc-1.0 input perturbation. This script traces z std + PCC
after each pair_stack sub-op for BOTH z_in sources, localizing where the ill-conditioning
diverges (expected: pair_transition, the 13x amplifier).

    TT_VISIBLE_DEVICES=2 TT_BIO_LOGICAL_DEVICE_ID=0 TT_MESH_GRAPH_DESC_PATH=<p150.textproto> \
        PYTHONPATH=<worktree> <env>/python scripts/of3_msa_pair_stack_trajectory.py
"""
import os, pickle, torch, ttnn

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
BIS = os.path.expanduser("~/of3_msa_bisect.pkl")
_AVG_DIMS = (8, 8)
_TRI_DIMS = (32, 4)


def pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def main():
    from tt_bio.tenstorrent import get_device, OuterProductMean, PairWeightedAveraging, Transition, PairformerLayer
    from tt_bio.openfold3_weights import remap_msa_block, _sub

    b = pickle.load(open(BIS, "rb"))
    m_in = b["m"]; z_init = b["z_init"]
    gold_opm_z = b["subops_b0_fp32"]["opm"][1]
    fp32 = pickle.load(open(os.path.expanduser("~/of3_msa_pair_stack_subops.pkl"), "rb"))["fp32"]
    zsh = fp32["pair_transition"].shape

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    block_remap = remap_msa_block(_sub(sd, "msa_module.blocks.0"))
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    to = lambda t: torch.Tensor(ttnn.to_torch(t)).float().reshape(zsh)

    def trace(z_in_device, label):
        pl = PairformerLayer(*_TRI_DIMS, None, None, False, block_remap["pair_stack"], ckc)
        z = z_in_device
        steps = [("tri_mul_out", pl.triangle_multiplication_start),
                 ("tri_mul_in", pl.triangle_multiplication_end),
                 ("tri_att_start", pl.triangle_attention_start),
                 ("tri_att_end", pl.triangle_attention_end),
                 ("pair_transition", pl.transition_z)]
        print(f"\n== {label} ==")
        print(f"{'sub_op':<16}{'dev_std':>9}{'fp32_std':>10}{'pcc':>9}")
        for name, fn in steps:
            z = ttnn.add_(z, fn(z, None) if name != "pair_transition" else fn(z))
            zo = to(z)
            print(f"{name:<16}{float(zo.std()):>9.2f}{float(fp32[name].std()):>10.2f}"
                  f"{pcc(zo, fp32[name].float()):>9.4f}")
        return z

    # 1) device OPM z_in: run device OPM first
    m = ft(m_in.unsqueeze(0)); z = ft(z_init.unsqueeze(0))
    opm = OuterProductMean(block_remap["outer_product_mean"], ckc)
    z_dev_opm = ttnn.add(z, opm(m, None, None))
    # 2) clean fp32 OPM z_in cast bf16
    z_clean = ft(gold_opm_z.unsqueeze(0))

    trace(z_dev_opm, "device-OPM z_in")
    trace(z_clean, "clean fp32-cast-bf16 z_in")


if __name__ == "__main__":
    main()
