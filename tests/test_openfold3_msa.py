"""On-device parity for OpenFold3 -> tt-bio MSA module.

Golden: ~/of3_ref_out.pkl ("msa_block0_real"/"msa_stack_real" keys from
scripts/of3_real_golden.py) -- real of3-p2-155k.pt weights, real featurized ubiquitin
example (via P1's build_openfold3_features + the real InputEmbedderAllAtom/
MSAModuleEmbedder, see scripts/of3_real_golden.py). Remap: tt_bio.openfold3_weights
(pure dict rename + delegate to the proven protenix remaps for
OuterProductMean/PairWeightedAveraging/Transition/pair-stack).

ORDERING: OF3's msa_module.opm_first=True (OuterProductMean runs BEFORE the MSA
PairWeightedAveraging+Transition update), same convention as Protenix-v2 and the
opposite of tt_bio.tenstorrent.MSALayer's hardcoded opm-after-update order (which
matches Boltz-2). So this test composes the raw primitives directly in OF3's order
(mirrors tests/test_protenix_trunk_msa.py) instead of instantiating MSALayer -- using
MSALayer here would silently run PWA+transition before OPM, the wrong order for OF3.

RESOLVED (2026-08-07, production-merge pass): the z-track gap was two real bugs,
not intrinsic ill-conditioning. (1) The shared OuterProductMean scaled the raw outer
product by 1/S but added the proj_o bias FULL-STRENGTH, while the AF3/OF3 reference
divides the whole linear_out output (bias included) by the pair norm -- a structured
per-channel constant (~0.9 std) injected into every OPM update. Negligible on the
1024-row ubq golden (OPM update std ~18) but ~50% of the signal on 9BK6's 1108-row
MSA (update std ~1.8), where it compounded over 4 blocks x 4 cycles and the
pairformer amplified it ~7x in the low-magnitude inter-chain pair blocks -- the
complex-fold killer (9BK6 cross-RMSD 12.6 A, ipTM 0.15 vs reference 0.77). Fixed by
``OuterProductMean(scale_bias=True)``, wired on for OF3 only. (2) The tri_att pair
bias was sqrt(d)-scaled (Boltz convention) where OF3 pre-scales q and adds the bias
unscaled -- fixed by ``scale_pair_bias=False`` + the fp32-softmax path. With both,
block-0 z_out pcc=0.99999 and per-block stack z_out pcc>=0.99996 on the real 9BK6
input, and the trunk no longer compounds inter-chain error across cycles.
"""
import os, pickle, pytest, torch, ttnn

_CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
import of3_golden

_GOLD = os.path.expanduser("~/of3_ref_out.pkl")
pytestmark = pytest.mark.skipif(not (os.path.exists(_CKPT) and os.path.exists(_GOLD)),
                                reason="of3 ckpt or golden pkl missing")

# OF3 msa_module dims: c_hidden_msa_att=8, no_heads_msa=8; c_hidden_pair_att=32, no_heads_pair=4
_AVG_DIMS = (8, 8)
_TRI_DIMS = (32, 4)


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm()))


def _cfg(dev):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def _run_block(block_remap, m, z, ckc):
    """m, z are already device ttnn tensors; runs one OF3 MSAModuleBlock (opm_first=True)."""
    from tt_bio.tenstorrent import OuterProductMean, PairWeightedAveraging, Transition, PairformerLayer

    opm = OuterProductMean(block_remap["outer_product_mean"], ckc, scale_bias=True)
    z = ttnn.add(z, opm(m, None, None))

    if "pair_weighted_averaging" in block_remap:
        pwa = PairWeightedAveraging(*_AVG_DIMS, block_remap["pair_weighted_averaging"], ckc)
        tm = Transition(block_remap["msa_transition"], ckc)
        m = ttnn.add(m, ttnn.reshape(pwa(m, ttnn.clone(z)), tuple(m.shape)))
        m = ttnn.add(m, ttnn.reshape(tm(m), tuple(m.shape)))

    pl = PairformerLayer(*_TRI_DIMS, None, None, False, block_remap["pair_stack"], ckc,
                         scale_pair_bias=False, fp32_softmax=True)
    z = pl(None, z)[1]
    return m, z


def test_of3_msa_block0_on_device():
    """Single-block gate: exercises OPM + PWA + msa_transition + pair_stack (block 0
    is not the last block, so it has the full msa update)."""
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_weights import remap_msa_block, _sub
    dev = get_device()
    ckc = _cfg(dev)
    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    block_remap = remap_msa_block(_sub(sd, "msa_module.blocks.0"))
    gold = of3_golden.intermediates(_GOLD)["msa_block0_real"]
    (m_in, z_in) = gold["in"]; (m_gold, z_gold) = gold["out"]
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    m, z = _run_block(block_remap, ft(m_in.unsqueeze(0)), ft(z_in.unsqueeze(0)), ckc)
    mo = torch.Tensor(ttnn.to_torch(m)).float().reshape(m_gold.shape)
    zo = torch.Tensor(ttnn.to_torch(z)).float().reshape(z_gold.shape)
    m_pcc, z_pcc = _pcc(mo, m_gold.float()), _pcc(zo, z_gold.float())
    print(f"\nOF3 MSAModuleBlock0: m_pcc={m_pcc:.5f} z_pcc={z_pcc:.5f}")
    assert m_pcc > 0.98 and z_pcc > 0.98


def test_of3_msa_stack_on_device():
    """Full 4-block gate. Reference MSAModuleStack.forward returns z only (m is
    discarded after the last, skip_msa_update block) -- so only z is compared."""
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_weights import remap_msa_module
    dev = get_device()
    ckc = _cfg(dev)
    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    blocks = remap_msa_module(sd)
    gold = of3_golden.intermediates(_GOLD)["msa_stack_real"]
    (m_in, z_in) = gold["in"]; z_gold = gold["out"]
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    m, z = ft(m_in.unsqueeze(0)), ft(z_in.unsqueeze(0))
    for block_remap in blocks:
        m, z = _run_block(block_remap, m, z, ckc)
    zo = torch.Tensor(ttnn.to_torch(z)).float().reshape(z_gold.shape)
    z_pcc = _pcc(zo, z_gold.float())
    print(f"\nOF3 MSAModuleStack (4 blocks): z_pcc={z_pcc:.5f}")
    assert z_pcc > 0.97
