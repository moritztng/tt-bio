"""P5 bisect (device side): localize the z_pcc loss across the 48-block stack and test
the physically-meaningful (LayerNorm-normalized) metric.

    TT_VISIBLE_DEVICES=1 python3 /tmp/of3_bisect_device.py
"""
import os, pickle, torch, ttnn

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
BIS = os.path.expanduser("~/of3_bisect.pkl")
_DIMS = (32, 4, 24, 16)


def pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a-a.mean())*(b-b.mean())).sum()/((a-a.mean()).norm()*(b-b.mean()).norm()))


def ln(x):  # per-position layernorm over channel dim, as z is consumed downstream
    return torch.nn.functional.layer_norm(x.float(), (x.shape[-1],))


def main():
    from tt_bio.tenstorrent import get_device, Pairformer
    from tt_bio.openfold3_weights import remap_pairformer_stack

    g = pickle.load(open(BIS, "rb"))
    s_init = g["s_init"]; z_init = g["z_init"]; traj_z = g["traj_z"]; traj_s = g["traj_s"]
    zsh = traj_z[0].shape; ssh = traj_s[0].shape
    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    pf = Pairformer(48, *_DIMS, True, remap_pairformer_stack(sd), cfg)

    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    s = ft(s_init.unsqueeze(0)); z = ft(z_init.unsqueeze(0))
    print("RESULT block | dev_z_std | cum_z_pcc(raw) | cum_z_pcc(LN) | cum_s_pcc")
    for i, blk in enumerate(pf.blocks):
        s, z = blk(s, z, None, None, None)
        zt = torch.Tensor(ttnn.to_torch(z)).float().reshape(zsh)
        st = torch.Tensor(ttnn.to_torch(s)).float().reshape(ssh)
        print(f"RESULT {i:5d} | {float(zt.std()):8.2f} | {pcc(zt, traj_z[i]):.5f} | "
              f"{pcc(ln(zt), ln(traj_z[i])):.5f} | {pcc(st, traj_s[i]):.5f}")

    # isolated last-block, perfect (reference fp32) input
    so, zo = pf.blocks[47](ft(traj_s[46].unsqueeze(0)), ft(traj_z[46].unsqueeze(0)), None, None, None)
    zo = torch.Tensor(ttnn.to_torch(zo)).float().reshape(zsh)
    print(f"RESULT ISO47 raw={pcc(zo, traj_z[47]):.5f} LN={pcc(ln(zo), ln(traj_z[47])):.5f}")


if __name__ == "__main__":
    main()
