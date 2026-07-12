"""On-device parity for the OpenFold3 InputEmbedder *glue* leg (s_input -> s, z).

Golden: ~/of3_ref_out.pkl["intermediates"]["input_embedder_real"], captured by
scripts/of3_real_golden.py (real of3-p2-155k.pt weights, real ubiquitin example via
tt_bio.openfold3_data.build_openfold3_features -> reference InputEmbedderAllAtom). The
golden now carries the reference ``relpos`` (OF3 ``relpos_complex``, 139-dim) so the
device glue is PCC-gated against the *exact* reference relative-position feature rather
than a re-computation.

This isolates the five weight-only glue linears (linear_s, linear_z_i, linear_z_j,
linear_relpos, linear_token_bonds) + the outer-sum z from the atom-encoder attention
leg (-> s_input), which is gated separately. The atom-encoder output ``ai`` is captured
as ``input_embedder_atom_enc_real`` for that subsequent gate.
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


def test_of3_input_embedder_glue_on_device():
    """Device glue linears + outer-sum z vs the reference InputEmbedder on real ubiquitin.
    s and z are both weighted outputs (linear_s / linear_z_{i,j} + linear_relpos +
    linear_token_bonds), so both are gated tight (PCC > 0.98)."""
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3 import InputEmbedderGlue
    from tt_bio.openfold3_weights import _sub

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    gold = pickle.load(open(_GOLD, "rb"))["intermediates"]["input_embedder_real"]
    s_input, s_ref, z_ref = gold["out"]
    relpos = gold["relpos"]
    token_bonds = gold["in"]["token_bonds"]

    dev = get_device()
    glue = InputEmbedderGlue(_sub(sd, "input_embedder"), _cfg(dev))

    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    s_in = ft(s_input.unsqueeze(0))
    rel = ft(relpos.unsqueeze(0))
    tb = ft(token_bonds.unsqueeze(0).unsqueeze(-1))
    s_d, z_d = glue(s_in, rel, tb)

    s = torch.Tensor(ttnn.to_torch(s_d)).float().reshape(s_ref.shape)
    z = torch.Tensor(ttnn.to_torch(z_d)).float().reshape(z_ref.shape)
    s_pcc = _pcc(s, s_ref.float())
    z_pcc = _pcc(z, z_ref.float())
    print(f"\nOF3 InputEmbedderGlue: s_pcc={s_pcc:.5f} z_pcc={z_pcc:.5f}")
    assert s_pcc > 0.98, f"s_pcc={s_pcc:.5f} below 0.98"
    assert z_pcc > 0.98, f"z_pcc={z_pcc:.5f} below 0.98"
