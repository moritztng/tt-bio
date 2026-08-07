"""On-device parity for the OpenFold3 MSAModuleEmbedder (s_input -> m).

Golden: ~/of3_ref_out.pkl["intermediates"]["msa_module_embedder_real"], captured by
scripts/of3_msa_embedder_golden.py. The golden carries the post-subsample msa_feat (the
stochastic subsample is captured on host via a linear_m input hook), so the device
embedder -- two bias-free linears + a broadcast add -- is gated against the exact
reference subsample, isolating the device linear precision from the subsample logic.

This extends the trunk validation past the InputEmbedder: s_input -> m here, complementing
the already-gated MSA stack (m, z -> z in tests/test_openfold3_msa.py).
"""
import os, pickle, pytest, torch, ttnn

_CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
_GOLD = os.path.expanduser("~/of3_ref_out.pkl")
pytestmark = pytest.mark.skipif(not (os.path.exists(_CKPT) and os.path.exists(_GOLD)),
                                reason="of3 ckpt or golden pkl missing")


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def _cfg(dev):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def test_of3_msa_module_embedder_on_device():
    """Device MSAModuleEmbedder (s_input -> m) vs the reference on real ubiquitin.
    m is the MSA single representation; gated tight (PCC > 0.98)."""
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_msa_embedder import MSAModuleEmbedder
    from tt_bio.openfold3_weights import _sub

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    me_sd = _sub(sd, "msa_module_embedder")
    g = pickle.load(open(_GOLD, "rb"))["intermediates"]["msa_module_embedder_real"]
    msa_feat, s_input, m_ref = g["msa_feat"], g["s_input"], g["m_ref"]

    dev = get_device()
    me = MSAModuleEmbedder(me_sd, _cfg(dev))

    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    # msa_feat [N_seq, N_token, 34] -> [1, N_seq, N_token, 34]; s_input [N_token, 449] -> [1, N_token, 449]
    m_d = me(ft(msa_feat.unsqueeze(0)), ft(s_input.unsqueeze(0)))
    m = torch.Tensor(ttnn.to_torch(m_d)).float().reshape(m_ref.shape)
    m_pcc = _pcc(m, m_ref.float())
    print(f"\nOF3 MSAModuleEmbedder: m_pcc={m_pcc:.5f}")
    assert m_pcc > 0.98, f"m_pcc={m_pcc:.5f} below 0.98"
