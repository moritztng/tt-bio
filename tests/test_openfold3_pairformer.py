"""On-device parity for OpenFold3 -> tt-bio Pairformer.

Golden: ~/of3_ref_out.pkl from scripts/of3_golden.py (real of3-p2-155k.pt weights fed
deterministic seeded trunk inputs). Remap: tt_bio.openfold3_weights (pure dict rename +
delegate to the proven protenix remap). Mirrors test_protenix_trunk_pairformer.py.
"""
import os, pickle, pytest, torch, ttnn

_CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
_GOLD = os.path.expanduser("~/of3_ref_out.pkl")
pytestmark = pytest.mark.skipif(not (os.path.exists(_CKPT) and os.path.exists(_GOLD)),
                                reason="of3 ckpt or golden pkl missing")


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def _run(sd, gold, dev, cfg):
    from tt_bio.tenstorrent import Pairformer
    from tt_bio.openfold3_weights import remap_pairformer_stack
    combined = remap_pairformer_stack(sd)
    nb = 1 + max(int(k.split(".")[1]) for k in combined)
    # OF3 pairformer dims: c_hidden_pair_att=32, no_heads_pair=4, c_hidden_pair_bias=24, no_heads_pair_bias=16
    pf = Pairformer(nb, 32, 4, 24, 16, True, combined, cfg)
    (s_in, z_in) = gold["in"]; (s_out, z_out) = gold["out"]
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    so, zo = pf(ft(s_in.unsqueeze(0)), ft(z_in.unsqueeze(0)))
    so = torch.Tensor(ttnn.to_torch(so)).float().reshape(s_out.shape)
    zo = torch.Tensor(ttnn.to_torch(zo)).float().reshape(z_out.shape)
    return _pcc(so, s_out.float()), _pcc(zo, z_out.float())


def test_of3_pairformer_stack_on_device():
    from tt_bio.tenstorrent import get_device
    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    gold = pickle.load(open(_GOLD, "rb"))["intermediates"]["pairformer_stack"]
    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    s_pcc, z_pcc = _run(sd, gold, dev, cfg)
    print(f"\nOF3 48-block pairformer_stack: s_pcc={s_pcc:.5f} z_pcc={z_pcc:.5f}")
    assert s_pcc > 0.98 and z_pcc > 0.97
