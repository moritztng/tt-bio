"""Decisive test: does full-fp32 attention stabilize tri_att's ill-conditioned softmax?

The trajectory bisect localized the lever: tri_att_start's update is 2.6x over-amplified
(std 23.22 vs fp32 18.91) when fed device-OPM z_in (pcc 1.0 to fp32, but bf16-rounded
differently). The attention scores have std ~23 (peaky softmax), so tiny bf16 score
perturbations tip the softmax peak. This script implements tri_att_start with FULL fp32
attention (fp32 qkv/g/o projections + fp32 scores + fp32 numerically-stable softmax +
fp32 attn@v) and measures the update PCC + output std with BOTH device-OPM z_in and
clean fp32 z_in. If fp32 attention lifts the device-OPM update to ~fp32 and collapses
the 23.22->18.91 std swing, score/softmax precision is the lever and fp32 attention is
the fix (gated, perf cost). If not, the ill-conditioning is beyond attention precision.

    TT_VISIBLE_DEVICES=2 TT_BIO_LOGICAL_DEVICE_ID=0 TT_MESH_GRAPH_DESC_PATH=<p150.textproto> \
        PYTHONPATH=<worktree> <env>/python scripts/of3_triatt_fp32_attention_test.py
"""
import os, pickle, torch, ttnn

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
BIS = os.path.expanduser("~/of3_msa_bisect.pkl")
SUBOPS = os.path.expanduser("~/of3_msa_pair_stack_subops.pkl")
_TRI_DIMS = (32, 4)


def pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def main():
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_weights import remap_msa_block, _sub
    import torch as T

    b = pickle.load(open(BIS, "rb"))
    sub = pickle.load(open(SUBOPS, "rb"))
    fp32 = sub["fp32"]
    zsh = fp32["tri_att_start"].shape
    gold_upd = fp32["tri_att_start"] - fp32["tri_mul_in"]   # fp32 tri_att_start update

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    ps = remap_msa_block(_sub(sd, "msa_module.blocks.0"))["pair_stack"]
    ta_w = _sub(ps, "tri_att_start")  # keys: layer_norm.*, linear.weight (bias=linear_z), mha.linear_q/k/v/g/o.weight

    head_dim, n_heads = 32, 4
    scale = head_dim ** -0.5

    # Load weights as FP32 device tensors (skip the bf16 cast).
    def fp32_dev(key, t):
        return ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
    ln_w = fp32_dev(None, ta_w["layer_norm.weight"])
    ln_b = fp32_dev(None, ta_w["layer_norm.bias"])
    qkv_w = T.cat([ta_w["mha.linear_q.weight"], ta_w["mha.linear_k.weight"], ta_w["mha.linear_v.weight"]], dim=0).t().contiguous()
    qkv_w = ttnn.from_torch(qkv_w.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
    g_w = ttnn.from_torch(ta_w["mha.linear_g.weight"].t().float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
    o_w = ttnn.from_torch(ta_w["mha.linear_o.weight"].t().float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
    bias_w = ttnn.from_torch(ta_w["linear.weight"].t().float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)

    def tri_att_fp32(z_in):
        # z_in: [1, N, N, c] device bf16. Cast to fp32, run tri_att_start in full fp32.
        z = ttnn.typecast(z_in, ttnn.float32)
        z = ttnn.reshape(z, tuple(z.shape)[1:])  # [N, N, c]
        # starting -> no transpose
        x = ttnn.layer_norm(z, weight=ln_w, bias=ln_b, epsilon=1e-5, compute_kernel_config=ckc)
        bias = ttnn.linear(x, bias_w, compute_kernel_config=ckc, dtype=ttnn.float32)  # [N,N,H]
        bias = ttnn.unsqueeze(bias, 0)
        bias = ttnn.permute(bias, (0, 3, 1, 2))  # [1,H,N,N]
        qkv = ttnn.linear(x, qkv_w, compute_kernel_config=ckc, dtype=ttnn.float32)  # [N,N,3*H*hd]
        qkv = ttnn.unsqueeze(qkv, 1)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=n_heads, num_kv_heads=n_heads, transpose_k_heads=False)
        ttnn.deallocate(qkv)
        sc = ttnn.matmul(q, ttnn.permute(k, (0, 1, 3, 2)), compute_kernel_config=ckc)
        sc = ttnn.multiply(sc, scale)
        sc = ttnn.add(sc, bias)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(bias)
        attn = ttnn.softmax(sc, dim=-1, numeric_stable=True)
        ttnn.deallocate(sc)
        o = ttnn.matmul(attn, v, compute_kernel_config=ckc)  # [1,H,N,hd]
        ttnn.deallocate(attn); ttnn.deallocate(v)
        o_heads = ttnn.experimental.nlp_concat_heads(o)
        ttnn.deallocate(o)
        o = ttnn.squeeze(o_heads, 1)  # [1,N,H*hd]
        g = ttnn.linear(x, g_w, compute_kernel_config=ckc, dtype=ttnn.float32)
        o = ttnn.multiply_(o, g, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        out = ttnn.linear(o, o_w, compute_kernel_config=ckc, dtype=ttnn.float32)  # [1,N,c] update
        return out

    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    to = lambda t: torch.Tensor(ttnn.to_torch(t)).float().reshape(zsh)

    # device-OPM z_in
    from tt_bio.tenstorrent import OuterProductMean
    opm = OuterProductMean(remap_msa_block(_sub(sd, "msa_module.blocks.0"))["outer_product_mean"], ckc)
    m = ft(b["m"].unsqueeze(0)); z0 = ft(b["z_init"].unsqueeze(0))
    z_dev_opm = ttnn.add(z0, opm(m, None, None))
    z_clean = ft(fp32["tri_mul_in"].unsqueeze(0))

    for label, zin in [("device-OPM z_in", z_dev_opm), ("clean fp32 z_in", z_clean)]:
        upd = tri_att_fp32(zin)
        upd_z = to(upd)
        # output = z_in + update; reconstruct output std
        zin_z = torch.Tensor(ttnn.to_torch(zin)).float().reshape(zsh)
        out_z = zin_z + upd_z
        print(f"{label}: update_pcc={pcc(upd_z, gold_upd.float()):.5f} "
              f"update_std={float(upd_z.std()):.3f} (fp32 {float(gold_upd.std()):.3f}) "
              f"output_std={float(out_z.std()):.3f} (fp32 {float(fp32['tri_att_start'].std()):.3f})")


if __name__ == "__main__":
    main()
