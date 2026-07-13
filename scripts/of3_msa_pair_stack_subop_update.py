"""Measure each pair_stack sub-op's UPDATE quality directly (not the residual-added output).

The residual structure z = z + update(z) means pcc(z_dev, z_fp32) is dominated by the
common large-magnitude input z and INFLATED -- a wrong update can still score ~0.9994
because the input residual dominates the PCC. This script measures the UPDATE itself:
pcc(device_update, fp32_update), feeding the device the fp32 input at each point. This is
the clean per-primitive quality metric, and it settles whether the fused SDPA's bf16
softmax or the manual fp32-softmax is the better attention path here.

    TT_VISIBLE_DEVICES=2 TT_BIO_LOGICAL_DEVICE_ID=0 TT_MESH_GRAPH_DESC_PATH=<p150.textproto> \
        PYTHONPATH=<worktree> <env>/python scripts/of3_msa_pair_stack_subop_update.py
"""
import os, pickle, torch, ttnn

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
GOLD = os.path.expanduser("~/of3_msa_pair_stack_subops.pkl")
_TRI_DIMS = (32, 4)


def pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    if a.norm() == 0 or b.norm() == 0:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def main():
    from tt_bio.tenstorrent import get_device, PairformerLayer
    from tt_bio.openfold3_weights import remap_msa_block, _sub

    g = pickle.load(open(GOLD, "rb"))
    fp32 = g["fp32"]
    zsh = fp32["tri_mul_out"].shape

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    block_remap = remap_msa_block(_sub(sd, "msa_module.blocks.0"))
    ps = block_remap["pair_stack"]

    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    to = lambda t: torch.Tensor(ttnn.to_torch(t)).float().reshape(zsh)

    pl_fused = PairformerLayer(*_TRI_DIMS, None, None, False, ps, ckc, fp32_softmax=False)
    pl_man = PairformerLayer(*_TRI_DIMS, None, None, False, ps, ckc, fp32_softmax=True)

    # fp32 updates = difference of consecutive fp32 goldens
    fp32_upd = {
        "tri_mul_out": fp32["tri_mul_out"] - g["z_in"],
        "tri_mul_in": fp32["tri_mul_in"] - fp32["tri_mul_out"],
        "tri_att_start": fp32["tri_att_start"] - fp32["tri_mul_in"],
        "tri_att_end": fp32["tri_att_end"] - fp32["tri_att_start"],
        "pair_transition": fp32["pair_transition"] - fp32["tri_att_end"],
    }
    inputs = {
        "tri_mul_out": g["z_in"],
        "tri_mul_in": fp32["tri_mul_out"],
        "tri_att_start": fp32["tri_mul_in"],
        "tri_att_end": fp32["tri_att_start"],
        "pair_transition": fp32["tri_att_end"],
    }

    print(f"{'sub_op / mode':<34}{'upd_pcc':>9}{'upd_std_dev':>12}{'upd_std_fp32':>13}")
    def show(name, tag, dev_upd):
        print(f"{name + ' / ' + tag:<34}{pcc(dev_upd, fp32_upd[name].float()):>9.5f}"
              f"{float(dev_upd.std()):>12.3f}{float(fp32_upd[name].std()):>13.3f}")

    for name, fn in [("tri_mul_out", pl_fused.triangle_multiplication_start),
                     ("tri_mul_in", pl_fused.triangle_multiplication_end)]:
        z = ft(inputs[name].unsqueeze(0))
        upd = fn(z, None)
        show(name, "fused", to(upd))

    for ending, name, mod_f, mod_m in [
        (False, "tri_att_start", pl_fused.triangle_attention_start, pl_man.triangle_attention_start),
        (True, "tri_att_end", pl_fused.triangle_attention_end, pl_man.triangle_attention_end),
    ]:
        for tag, mod in [("fused", mod_f), ("fp32softmax", mod_m)]:
            z = ft(inputs[name].unsqueeze(0))
            upd = mod(z, None)
            show(name, tag, to(upd))

    z = ft(inputs["pair_transition"].unsqueeze(0))
    upd = pl_fused.transition_z(z)
    show("pair_transition", "fused", to(upd))


if __name__ == "__main__":
    main()
