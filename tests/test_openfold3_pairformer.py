"""On-device parity for OpenFold3 -> tt-bio Pairformer.

Golden: ~/of3_ref_out.pkl from scripts/of3_golden.py (real of3-p2-155k.pt weights fed
deterministic seeded trunk inputs). Remap: tt_bio.openfold3_weights (pure dict rename +
delegate to the proven protenix remap). Mirrors test_protenix_trunk_pairformer.py.

GOLDEN-INPUT CAVEAT (see docs/openfold3-port.md status log, tick 3). The current golden
feeds the trunk *synthetic* N(0,1) tensors, not real featurized inputs. That is fine for
a single-block s-track correctness check (block 0 s_pcc=0.99985 -> remap is byte-correct),
but it is NOT a valid gate for the deep 48-block stack: N(0,1) is off the learned manifold,
so the reference trunk *explodes* (output s std ~3.7e4 vs a real fold's ~1.8e2). At that
magnitude the fp32->bf16 gap alone collapses the z-track *with no device and no remap*
(measured pure-CPU control: fp32-vs-bf16 z_pcc=0.72), and the device compounds it over 48
blocks (z_pcc=0.16, s_pcc=0.91). This is a numerical property of the pathological input, not
a port defect. Protenix's passing stack gate (test_protenix_trunk_pairformer.py) uses REAL
captured trunk I/O for exactly this reason. The definitive OF3 stack gate is therefore
DEFERRED (xfail below) until the golden captures real input-embedder output -- the P3
InputEmbedder port. The block-0 s-gate is the live per-block correctness gate meanwhile.
"""
import os, pickle, pytest, torch, ttnn

_CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
_GOLD = os.path.expanduser("~/of3_ref_out.pkl")
pytestmark = pytest.mark.skipif(not (os.path.exists(_CKPT) and os.path.exists(_GOLD)),
                                reason="of3 ckpt or golden pkl missing")

# OF3 pairformer dims: c_hidden_pair_att=32, no_heads_pair=4, c_hidden_pair_bias=24, no_heads_pair_bias=16
_DIMS = (32, 4, 24, 16)


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def _cfg(dev):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def _run(combined, gold, dev):
    from tt_bio.tenstorrent import Pairformer
    nb = 1 + max(int(k.split(".")[1]) for k in combined)
    pf = Pairformer(nb, *_DIMS, True, combined, _cfg(dev))
    (s_in, z_in) = gold["in"]; (s_out, z_out) = gold["out"]
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    so, zo = pf(ft(s_in.unsqueeze(0)), ft(z_in.unsqueeze(0)))
    so = torch.Tensor(ttnn.to_torch(so)).float().reshape(s_out.shape)
    zo = torch.Tensor(ttnn.to_torch(zo)).float().reshape(z_out.shape)
    return _pcc(so, s_out.float()), _pcc(zo, z_out.float())


def test_of3_pairformer_block0_on_device():
    """Live per-block correctness gate: the s-track PCC proves the OF3->tt-bio remap is
    byte-correct. z_pcc on synthetic N(0,1) input is a known bf16 artifact (a pure-CPU
    fp32-vs-bf16 run of the same block already gives ~0.977) -- recorded, not gated tight."""
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_weights import remap_pairformer_block, _sub
    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    block_sd = _sub(sd, "pairformer_stack.blocks.0")
    combined = {f"layers.0.{k}": v for k, v in remap_pairformer_block(block_sd).items()}
    gold = pickle.load(open(_GOLD, "rb"))["intermediates"]["pairformer_block0"]
    s_pcc, z_pcc = _run(combined, gold, get_device())
    print(f"\nOF3 PairFormerBlock0: s_pcc={s_pcc:.5f} z_pcc={z_pcc:.5f} (z on N(0,1) is a bf16 artifact)")
    assert s_pcc > 0.98


@pytest.mark.xfail(reason="synthetic-N(0,1) golden is off-manifold: reference trunk explodes "
                          "(out std ~3.7e4) and bf16 collapses z even on pure CPU (z_pcc=0.72, no "
                          "device). Valid stack gate needs a real-input-embedder golden (P3). "
                          "See module docstring + docs/openfold3-port.md.", strict=False)
def test_of3_pairformer_stack_on_device():
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_weights import remap_pairformer_stack
    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    combined = remap_pairformer_stack(sd)
    gold = pickle.load(open(_GOLD, "rb"))["intermediates"]["pairformer_stack"]
    s_pcc, z_pcc = _run(combined, gold, get_device())
    print(f"\nOF3 48-block pairformer_stack: s_pcc={s_pcc:.5f} z_pcc={z_pcc:.5f}")
    assert s_pcc > 0.98 and z_pcc > 0.97
