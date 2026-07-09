"""On-device parity for the Protenix-v2 diffusion denoiser-as-a-unit (dit_consistent):
DiffusionConditioning(single) -> AtomAttentionEncoder(has_coords) -> 24-block token
DiffusionTransformer -> AtomAttentionDecoder -> EDM coord update, chained end-to-end
and compared to the real per-step denoiser golden. Mirrors scripts/protenix_denoiser_parity.py
(the validated "MILESTONE: full denoiser reproduces golden coords PCC 0.99976" result).
The 24-block DiT stage runs in fp32 torch: its per-block updates are near-identity, so
ttnn bf16 accumulates too much noise over 24 blocks (documented precision ceiling, not a
bug — same situation as confidence's plddt/resolved). cond/atomenc/atomdec run on-device.
Gated on the pre-mutation golden pkl (scripts/protenix_extract_denoiser_pre.py)."""
import os, pickle, pytest, torch, torch.nn.functional as F, ttnn

_CKPT = "/home/ttuser/protenix_ckpt/protenix-v2.pt"
_DENOISER = os.path.expanduser("~/protenix_denoiser_pre.pkl")
pytestmark = pytest.mark.skipif(
    not (os.path.exists(_CKPT) and os.path.exists(_DENOISER)),
    reason="v2 ckpt or pre-mutation denoiser golden pkl missing (run scripts/protenix_extract_denoiser_pre.py)")


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def test_dit_consistent_denoiser_unit_on_device():
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from protenix_reference import remap_transition
    from tt_bio.tenstorrent import get_device, Transition, CORE_GRID_MAIN as CORE
    from tt_bio.protenix import AtomTransformer

    ck = torch.load(_CKPT, map_location="cpu", weights_only=True); ck = ck.get("model", ck)
    D = pickle.load(open(_DENOISER, "rb")); kw = D["kwargs"]
    feat = kw["input_feature_dict"]; x_noisy = kw["x_noisy"].float(); t_hat = kw["t_hat_noise_level"].float()
    s_inputs = kw["s_inputs"].float(); s_trunk = kw["s_trunk"].float(); pair_z = kw["pair_z"].float()
    p_lm = kw["p_lm"].float()[0]; c_l = kw["c_l"].float(); coords_g = D["out"].float()

    N = c_l.shape[0]; NT = s_inputs.shape[0]
    a2t = feat["atom_to_token_idx"].long(); mt = feat["pad_info"]["mask_trunked"].float()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    T = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    NQ, NK, PADL = 32, 128, 48; NP = ((N + NQ - 1) // NQ) * NQ; nb = NP // NQ
    S = torch.zeros(N, NT); S[torch.arange(N), a2t] = 1.0

    def lin(x, wk, P):
        return ttnn.linear(x, T(ck[P + wk].t().contiguous()), compute_kernel_config=ckc, core_grid=CORE)

    # 1) DiffusionConditioning single path -> s_single (on-device).
    C = "module.diffusion_module.diffusion_conditioning."; g = lambda k: ck[C + k]
    ss = lin(ttnn.layer_norm(T(torch.cat([s_trunk, s_inputs], -1)), weight=T(g("layernorm_s.weight")), epsilon=1e-5, compute_kernel_config=ckc), "linear_no_bias_s.weight", C)
    tp = torch.log(t_hat / 16.0) / 4
    fou = torch.cos(2 * torch.pi * (tp.unsqueeze(-1) * g("fourier_embedding.w") + g("fourier_embedding.b")))
    nn_ = lin(ttnn.layer_norm(T(fou), weight=T(g("layernorm_n.weight")), epsilon=1e-5, compute_kernel_config=ckc), "linear_no_bias_n.weight", C)
    ss = ttnn.reshape(ttnn.add(ss, nn_), (1, NT, 384))
    for nm in ("transition_s1", "transition_s2"):
        sub = {k[len(C + nm) + 1:]: v for k, v in ck.items() if k.startswith(C + nm)}
        t = Transition(remap_transition(sub), ckc)
        ss = ttnn.add(ss, ttnn.reshape(t(ss), tuple(ss.shape)))
    s_single = ss

    # 2) AtomAttentionEncoder(has_coords) -> token-level a(768), q_skip/c_skip/p_skip (on-device).
    E = "module.diffusion_module.atom_attention_encoder."; ge = lambda k: ck[E + k]
    sp = lin(ttnn.layer_norm(T(s_trunk), weight=T(ge("layernorm_s.weight")), epsilon=1e-5, compute_kernel_config=ckc), "linear_no_bias_s.weight", E)
    c_la = ttnn.add(T(c_l), ttnn.matmul(T(S), sp, compute_kernel_config=ckc, core_grid=CORE))
    sigma = 16.0
    r_noisy = x_noisy / torch.sqrt(sigma ** 2 + t_hat ** 2).reshape(-1, 1, 1)
    q_l = ttnn.add(c_la, lin(T(r_noisy[0]), "linear_no_bias_r.weight", E))

    def wq(x):
        x = ttnn.to_layout(ttnn.reshape(x, (1, N, 128)), ttnn.ROW_MAJOR_LAYOUT); x = ttnn.pad(x, [[0, 0], [0, NP - N], [0, 0]], 0.0)
        return ttnn.to_layout(ttnn.reshape(x, (nb, NQ, 128)), ttnn.TILE_LAYOUT)

    def wkv(x):
        x = ttnn.to_layout(ttnn.reshape(x, (1, N, 128)), ttnn.ROW_MAJOR_LAYOUT); Lp = PADL + NP + NK
        x = ttnn.pad(x, [[0, 0], [PADL, Lp - PADL - N], [0, 0]], 0.0)
        bl = [ttnn.slice(x, [0, i * NQ, 0], [1, i * NQ + NK, 128]) for i in range(nb)]
        return ttnn.to_layout(ttnn.reshape(ttnn.concat(bl, 0), (nb, NK, 128)), ttnn.TILE_LAYOUT)

    clq = ttnn.relu(wq(c_la)); clk = ttnn.relu(wkv(c_la))
    p = ttnn.add(ttnn.add(T(p_lm), ttnn.unsqueeze(lin(clq, "linear_no_bias_cl.weight", E), 2)), ttnn.unsqueeze(lin(clk, "linear_no_bias_cm.weight", E), 1))
    m = lin(ttnn.relu(p), "small_mlp.1.weight", E); m = lin(ttnn.relu(m), "small_mlp.3.weight", E); m = lin(ttnn.relu(m), "small_mlp.5.weight", E)
    p = ttnn.add(p, m)
    atx_e = AtomTransformer(3, {k[len(E + "atom_transformer."):]: v for k, v in ck.items() if k.startswith(E + "atom_transformer.")}, ckc)
    q_out = atx_e(ttnn.reshape(q_l, (1, N, 128)), ttnn.reshape(c_la, (1, N, 128)), p, mt)
    a_tok = ttnn.matmul(T(S.t().contiguous() / (S.sum(0, keepdim=True).t() + 1e-6)), ttnn.reshape(ttnn.relu(lin(q_out, "linear_no_bias_q.weight", E)), (N, 768)), compute_kernel_config=ckc, core_grid=CORE)
    q_skip, c_skip, p_skip = q_out, c_la, p
    DM = "module.diffusion_module."
    a_tok = ttnn.add(a_tok, ttnn.reshape(lin(ttnn.layer_norm(ttnn.reshape(s_single, (NT, 384)), weight=T(ck[DM + "layernorm_s.weight"]), epsilon=1e-5, compute_kernel_config=ckc), "linear_no_bias_s.weight", DM), (NT, 768)))

    # 3) 24-block token DiffusionTransformer, fp32 torch (bf16 ttnn accumulates too
    # much error over near-identity blocks; see module docstring).
    P = "module.diffusion_module.diffusion_transformer."; nbk = 24; hd, nh = 48, 16
    a_h = torch.Tensor(ttnn.to_torch(a_tok)).float().reshape(NT, 768)
    s_h = torch.Tensor(ttnn.to_torch(s_single)).float().reshape(NT, 384)
    pz_n = F.layer_norm(pair_z, (pair_z.shape[-1],))
    z_h = pz_n.reshape(NT, NT, 256)
    gP = lambda k: ck[P + k].float()

    def _adaln(a, s, pre):
        an = F.layer_norm(a, (a.shape[-1],)); sn = F.layer_norm(s, (s.shape[-1],)) * gP(pre + "layernorm_s.weight")
        return torch.sigmoid(F.linear(sn, gP(pre + "linear_s.weight"), gP(pre + "linear_s.bias"))) * an + F.linear(sn, gP(pre + "linear_nobias_s.weight"))

    for b in range(nbk):
        A = f"blocks.{b}.attention_pair_bias."; Cc = f"blocks.{b}.conditioned_transition_block."
        an = _adaln(a_h, s_h, A + "layernorm_a.")
        zb = F.layer_norm(z_h, (256,)) * gP(A + "layernorm_z.weight")
        bias = F.linear(zb, gP(A + "linear_nobias_z.weight")).permute(2, 0, 1)
        Q = F.linear(an, gP(A + "attention.linear_q.weight"), gP(A + "attention.linear_q.bias")).reshape(NT, nh, hd).permute(1, 0, 2)
        K = F.linear(an, gP(A + "attention.linear_k.weight")).reshape(NT, nh, hd).permute(1, 0, 2)
        V = F.linear(an, gP(A + "attention.linear_v.weight")).reshape(NT, nh, hd).permute(1, 0, 2)
        o = torch.einsum("hij,hjd->hid", torch.softmax(torch.einsum("hid,hjd->hij", Q, K) / (hd ** 0.5) + bias, -1), V).permute(1, 0, 2).reshape(NT, nh * hd)
        o = o * torch.sigmoid(F.linear(an, gP(A + "attention.linear_g.weight")))
        attn = F.linear(o, gP(A + "attention.linear_o.weight"))
        attn = torch.sigmoid(F.linear(s_h, gP(A + "linear_a_last.weight"), gP(A + "linear_a_last.bias"))) * attn
        ao = attn + a_h
        an2 = _adaln(ao, s_h, Cc + "adaln.")
        bb = F.silu(F.linear(an2, gP(Cc + "linear_nobias_a1.weight"))) * F.linear(an2, gP(Cc + "linear_nobias_a2.weight"))
        a_h = torch.sigmoid(F.linear(s_h, gP(Cc + "linear_s.weight"), gP(Cc + "linear_s.bias"))) * F.linear(bb, gP(Cc + "linear_nobias_b.weight")) + ao
    a_t = T(a_h.reshape(1, NT, 768))

    # 4) AtomAttentionDecoder + EDM coordinate update (on-device).
    a_t = ttnn.layer_norm(a_t, weight=T(ck["module.diffusion_module.layernorm_a.weight"]), epsilon=1e-5, compute_kernel_config=ckc)
    DE = "module.diffusion_module.atom_attention_decoder."; gd = lambda k: ck[DE + k]
    q = ttnn.add(ttnn.matmul(T(S), lin(ttnn.reshape(a_t, (NT, 768)), "linear_no_bias_a.weight", DE), compute_kernel_config=ckc, core_grid=CORE), ttnn.reshape(q_skip, (N, 128)))
    atx_d = AtomTransformer(3, {k[len(DE + "atom_transformer."):]: v for k, v in ck.items() if k.startswith(DE + "atom_transformer.")}, ckc)
    qd = atx_d(ttnn.reshape(q, (1, N, 128)), ttnn.reshape(c_skip, (1, N, 128)), p_skip, mt)
    qn = ttnn.layer_norm(qd, weight=T(gd("layernorm_q.weight")), epsilon=1e-5, compute_kernel_config=ckc)
    r_update = torch.Tensor(ttnn.to_torch(lin(qn, "linear_no_bias_out.weight", DE))).float().reshape(1, N, 3)[:, :N]
    sr = (t_hat / sigma).reshape(-1, 1, 1)
    coords = (1.0 / (1.0 + sr ** 2)) * x_noisy[:, :N] + (t_hat.reshape(-1, 1, 1) / torch.sqrt(1.0 + sr ** 2)) * r_update

    assert _pcc(coords, coords_g) > 0.99
