"""Full-block device bisect: replicate test_of3_msa_block0 and localize the dominant lever.

Runs the OF3 MSAModuleBlock on device (OPM + PWA + msa_transition + pair_stack, fused
SDPA -- the default path) on the real block-0 input, capturing z PCC after OPM and after
pair_stack vs the fp32 goldens (~/of3_msa_bisect.pkl subops_b0_fp32). This replicates the
xfailed test (z_pcc~0.708) and splits the loss into the device-OPM z error (amplified by
pair_stack) vs the device-pair_stack error with a clean fp32 z_in -- settling which is the
dominant lever.

    TT_VISIBLE_DEVICES=2 TT_BIO_LOGICAL_DEVICE_ID=0 TT_MESH_GRAPH_DESC_PATH=<p150.textproto> \
        PYTHONPATH=<worktree> <env>/python scripts/of3_msa_block_full_device.py
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
    m_in = b["m"]            # [1,76,64]
    z_init = b["z_init"]     # [76,76,128]
    gold_opm_z = b["subops_b0_fp32"]["opm"][1]          # [76,76,128]
    gold_pair_stack_z = b["subops_b0_fp32"]["pair_stack"][1]
    zsh = gold_pair_stack_z.shape

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    block_remap = remap_msa_block(_sub(sd, "msa_module.blocks.0"))

    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    to = lambda t: torch.Tensor(ttnn.to_torch(t)).float().reshape(zsh)

    m = ft(m_in.unsqueeze(0))
    z = ft(z_init.unsqueeze(0))

    opm = OuterProductMean(block_remap["outer_product_mean"], ckc)
    z = ttnn.add(z, opm(m, None, None))
    z_after_opm = to(z).clone()
    print(f"z after OPM:      pcc={pcc(z_after_opm, gold_opm_z.float()):.5f}  "
          f"std dev={float(z_after_opm.std()):.3f} fp32={float(gold_opm_z.std()):.3f}")

    pwa = PairWeightedAveraging(*_AVG_DIMS, block_remap["pair_weighted_averaging"], ckc)
    tm = Transition(block_remap["msa_transition"], ckc)
    m = ttnn.add(m, ttnn.reshape(pwa(m, ttnn.clone(z)), tuple(m.shape)))
    m = ttnn.add(m, ttnn.reshape(tm(m), tuple(m.shape)))

    pl = PairformerLayer(*_TRI_DIMS, None, None, False, block_remap["pair_stack"], ckc)
    z = pl(None, z)[1]
    z_final = to(z)
    print(f"z after pair_stack: pcc={pcc(z_final, gold_pair_stack_z.float()):.5f}  "
          f"std dev={float(z_final.std()):.3f} fp32={float(gold_pair_stack_z.std()):.3f}")
    print(f"(xfail test baseline z_pcc ~0.70847)")

    # Counterfactual: feed pair_stack the FP32 z_in (cast bf16) instead of device OPM z
    z_clean = ft(gold_opm_z.unsqueeze(0))
    z_clean = pl(None, z_clean)[1]
    z_clean_out = to(z_clean)
    print(f"pair_stack(clean fp32 z_in): pcc={pcc(z_clean_out, gold_pair_stack_z.float()):.5f}  "
          f"(isolates pair_stack error from device-OPM error)")


if __name__ == "__main__":
    main()
