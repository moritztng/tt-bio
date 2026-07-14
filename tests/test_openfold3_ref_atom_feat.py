"""On-device parity for the OpenFold3 InputEmbedder atom-featurization leg.

Golden: ~/of3_ref_out.pkl["intermediates"]["input_embedder_ref_atom_feat_real"], captured
by scripts/of3_real_golden.py. The golden carries the per-atom reference features AND the
precomputed block inputs (dlm, vlm, inv_sq_dists) so the device pair linears are gated
against the exact reference block structure (the mask-derived gather in
convert_single_rep_to_blocks is captured, not re-derived).

This isolates RefAtomFeatureEmbedder (the 8 weight-only linears producing cl + plm) from
the AtomTransformer attention (-> ql) and the atom->token aggregation (-> ai), which are
gated in subsequent increments. cl is the per-atom conditioning [N_atom, c_atom]; plm is
the per-block pair conditioning [N_blk, N_q, N_k, c_atom_pair].
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


def test_of3_ref_atom_feature_embedder_on_device():
    """Device RefAtomFeatureEmbedder (single leg -> cl, pair leg -> plm) vs the reference
    on real ubiquitin. Both cl and plm are weighted outputs, gated tight (PCC > 0.98)."""
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3 import RefAtomFeatureEmbedder
    from tt_bio.openfold3_weights import _sub

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    rafe_sd = _sub(_sub(sd, "input_embedder.atom_attn_enc"), "ref_atom_feature_embedder")
    gold = pickle.load(open(_GOLD, "rb"))["intermediates"]["input_embedder_ref_atom_feat_real"]
    cl_ref, plm_ref = gold["out"]
    b = gold["in"]
    dlm = gold["dlm"]
    vlm = gold["vlm"]
    inv_sq_dists = gold["inv_sq_dists"]

    dev = get_device()
    rafe = RefAtomFeatureEmbedder(rafe_sd, _cfg(dev))

    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    ref_pos = ft(b["ref_pos"].unsqueeze(0))
    ref_charge = ft(torch.arcsinh(b["ref_charge"]).unsqueeze(0).unsqueeze(-1))
    ref_mask = ft(b["ref_mask"].unsqueeze(0).unsqueeze(-1))
    ref_element = ft(b["ref_element"].unsqueeze(0))
    ref_chars = ft(b["ref_atom_name_chars"].flatten(start_dim=-2).unsqueeze(0))
    cl_d, plm_d = rafe(ref_pos, ref_charge, ref_mask, ref_element, ref_chars,
                       ft(dlm.unsqueeze(0)), ft(vlm.unsqueeze(0)), ft(inv_sq_dists.unsqueeze(0)))

    cl = torch.Tensor(ttnn.to_torch(cl_d)).float().reshape(cl_ref.shape)
    plm = torch.Tensor(ttnn.to_torch(plm_d)).float().reshape(plm_ref.shape)
    cl_pcc = _pcc(cl, cl_ref.float())
    plm_pcc = _pcc(plm, plm_ref.float())
    print(f"\nOF3 RefAtomFeatureEmbedder: cl_pcc={cl_pcc:.5f} plm_pcc={plm_pcc:.5f}")
    assert cl_pcc > 0.98, f"cl_pcc={cl_pcc:.5f} below 0.98"
    assert plm_pcc > 0.98, f"plm_pcc={plm_pcc:.5f} below 0.98"
