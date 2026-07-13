"""Bisect the OF3 DiT 24-block stack PCC collapse: run the device stack block-by-block
against the reference per-block trajectory in ~/of3_ref_out.pkl, printing PCC + std at
each block to localize the divergence.

    TT_VISIBLE_DEVICES=1 TT_MESH_GRAPH_DESC_PATH=<...> PYTHONPATH=<worktree> \
      /home/ttuser/tt-bio/env/bin/python scripts/of3_diffusion_transformer_bisect.py
"""
import os, pickle, torch, ttnn

_CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
_GOLD = os.path.expanduser("~/of3_ref_out.pkl")


def pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    if a.norm() == 0 or b.norm() == 0:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def main():
    from tt_bio.tenstorrent import get_device, cleanup
    from tt_bio.openfold3_diffusion_transformer import OF3DiffusionTransformer
    from tt_bio.openfold3_weights import _sub

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    dt_sd = _sub(_sub(sd, "diffusion_module"), "diffusion_transformer")
    g = pickle.load(open(_GOLD, "rb"))["intermediates"]["diffusion_transformer_real"]
    a_in, s, z, tok = g["a_in"], g["s"], g["z"], g["token_mask"]
    traj = g["a_traj"]  # list of 24 reference per-block outputs

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    dit = OF3DiffusionTransformer(dt_sd, cfg, n_blocks=24)

    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    N = tok.shape[0]
    padded_N = ((N + 31) // 32) * 32
    tok_mask = tok.reshape(1, N)
    tok_mask_col = tok.reshape(1, N, 1)
    a_d = ft(a_in.unsqueeze(0)); s_d = ft(s.unsqueeze(0)); z_d = ft(z.unsqueeze(0))
    tm_d = ft(tok_mask); tmc_d = ft(tok_mask_col)
    # Pad to tile-aligned logical width (matches the stack __call__ padding).
    from tt_bio.openfold3_diffusion_transformer import _pad_single, _pad_pair
    a_d = _pad_single(a_d, padded_N); s_d = _pad_single(s_d, padded_N); z_d = _pad_pair(z_d, padded_N)
    tok_p = torch.zeros(padded_N); tok_p[:N] = tok
    tmc_p = tok_p.reshape(1, padded_N, 1); tmc_d = ft(tmc_p)
    mb = torch.full((1, 1, 1, padded_N), -1e9); mb[..., :N] = (tok - 1.0) * 1e9
    mb_d = ft(mb)

    print(f"ref a_in std={a_in.std():.3f}  padded_N={padded_N}")
    for b in range(24):
        a_d = dit.blocks[b](a_d, s_d, z_d, mb_d, tmc_d)
        ttnn.synchronize_device(dev)
        out = torch.Tensor(ttnn.to_torch(a_d)).float()[:, :N, :].reshape(a_in.shape)
        p = pcc(out, traj[b].float())
        print(f"block {b:2d}: pcc={p:.5f}  dev_std={out.std():.3f}  ref_std={traj[b].std():.3f}")
    cleanup()


if __name__ == "__main__":
    main()
