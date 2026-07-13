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

OPEN DEVICE-PRECISION GAP (both tests xfail, not remap bugs): the m-track (PWA +
msa_transition) is byte-correct (m_pcc=0.99999 on block 0 -- proves remap_msa_block's
key mapping and opm_first ordering are right). The z-track is not: block-0
z_pcc=0.708, 4-block-stack z_pcc=0.745. A pure-CPU fp32-vs-bf16 control on the exact
same real block-0 input (no device, no remap -- see scripts referenced in
docs/openfold3-port.md) gets z_pcc=0.9998, i.e. bf16 rounding alone tracks this block
almost perfectly. So the device is compounding real additional error beyond generic
bf16 rounding, on top of an already-large single-block magnitude jump specific to this
checkpoint (z std ~18 in, ~270 out -- a ~15x jump within one pair_stack call, versus
Protenix-v2's real MSA-stack gate at the same call pattern, which passes >0.99). This
is the same qualitative pattern as the pairformer-stack gate (see
test_openfold3_pairformer.py): OF3's real activations are simply much larger in
magnitude than Protenix's at equivalent trunk stages, and something about running that
regime through the device (not just bf16 itself) loses more precision than expected.

ROOT CAUSE (P8 tick 17 device bisect -- scripts/of3_msa_pair_stack_*.py,
of3_triatt_*_test.py; full writeup in docs/openfold3-port.md P8 tick 17): the loss is
NOT the DiffusionTransformer softmax-precision lever (P8 tick 14). A manual fp32-softmax
TriangleAttention path is strictly WORSE than the fused SDPA here (tri_att_start update
PCC 0.843 manual vs 0.991 fused) -- the fused SDPA's score computation is more precise
than a decomposed matmul+softmax, and the reference's own softmax is already bf16
(``softmax_no_cast``), so bf16 softmax is not the device-specific lever. Per-sub-op
update-PCC isolation (feeding each primitive the fp32 golden input) shows tri_mul
(0.99997) and pair_transition (0.99999) are essentially perfect; the seed error is in
tri_att (update PCC 0.991 vs CPU-bf16 ~0.9999). But the dominant effect is
ILL-CONDITIONING, not that 0.991: the OF3 pair_stack runs an extremely peaky softmax
(attention scores std ~23 at this checkpoint's magnitude) followed by the 13x
pair_transition magnitude amplifier, so the block's z output is hypersensitive to the
bf16 rounding PATTERN of the OPM z_in. A device-OPM z_in that is pcc=1.00000 to the fp32
OPM z (std 18.179 vs 18.173) feeds pair_stack to output std 611 / z_pcc 0.708, while the
fp32-OPM z cast to bf16 feeds the SAME device pair_stack to std 296 / z_pcc 0.921 -- a
2x output-magnitude swing and a 0.21 PCC swing from a pcc-1.0 input perturbation. Full
fp32 attention does NOT fix it: with device-OPM z_in it scores 0.72 (worse than bf16
fused 0.87, because the more-precise attention faithfully amplifies the tipped softmax
peak), and with clean z_in it scores 0.9999. So the only thing that stabilizes the
block is fp32 compute through the z-path (OPM + tri_att), a perf-regressing,
release-gated change -- the same intrinsic-bf16-ill-conditioning limit class as the
pairformer stack (P8 tick 11), not a fixable bug in the shared primitive. The fused-SDPA
TriangleAttention path is kept (it is the best available bf16 attention); the test stays
``xfail(strict=False)`` with this root cause on record.
"""
import os, pickle, pytest, torch, ttnn

_CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
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

    opm = OuterProductMean(block_remap["outer_product_mean"], ckc)
    z = ttnn.add(z, opm(m, None, None))

    if "pair_weighted_averaging" in block_remap:
        pwa = PairWeightedAveraging(*_AVG_DIMS, block_remap["pair_weighted_averaging"], ckc)
        tm = Transition(block_remap["msa_transition"], ckc)
        m = ttnn.add(m, ttnn.reshape(pwa(m, ttnn.clone(z)), tuple(m.shape)))
        m = ttnn.add(m, ttnn.reshape(tm(m), tuple(m.shape)))

    pl = PairformerLayer(*_TRI_DIMS, None, None, False, block_remap["pair_stack"], ckc)
    z = pl(None, z)[1]
    return m, z


@pytest.mark.xfail(reason="OPEN device-precision gap, not a remap bug: m_pcc=0.99999 "
                          "(remap + opm_first ordering are byte-correct) but z_pcc~0.71. "
                          "A pure-CPU fp32-vs-bf16 control on the same input gets "
                          "z_pcc=0.9998 -- see module docstring.", strict=False)
def test_of3_msa_block0_on_device():
    """Single-block gate: exercises OPM + PWA + msa_transition + pair_stack (block 0
    is not the last block, so it has the full msa update)."""
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_weights import remap_msa_block, _sub
    dev = get_device()
    ckc = _cfg(dev)
    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    block_remap = remap_msa_block(_sub(sd, "msa_module.blocks.0"))
    gold = pickle.load(open(_GOLD, "rb"))["intermediates"]["msa_block0_real"]
    (m_in, z_in) = gold["in"]; (m_gold, z_gold) = gold["out"]
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    m, z = _run_block(block_remap, ft(m_in.unsqueeze(0)), ft(z_in.unsqueeze(0)), ckc)
    mo = torch.Tensor(ttnn.to_torch(m)).float().reshape(m_gold.shape)
    zo = torch.Tensor(ttnn.to_torch(z)).float().reshape(z_gold.shape)
    m_pcc, z_pcc = _pcc(mo, m_gold.float()), _pcc(zo, z_gold.float())
    print(f"\nOF3 MSAModuleBlock0: m_pcc={m_pcc:.5f} z_pcc={z_pcc:.5f}")
    assert m_pcc > 0.98 and z_pcc > 0.98


@pytest.mark.xfail(reason="OPEN device-precision gap (see test_of3_msa_block0_on_device "
                          "and module docstring): z_pcc~0.75 over 4 blocks.", strict=False)
def test_of3_msa_stack_on_device():
    """Full 4-block gate. Reference MSAModuleStack.forward returns z only (m is
    discarded after the last, skip_msa_update block) -- so only z is compared."""
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_weights import remap_msa_module
    dev = get_device()
    ckc = _cfg(dev)
    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    blocks = remap_msa_module(sd)
    gold = pickle.load(open(_GOLD, "rb"))["intermediates"]["msa_stack_real"]
    (m_in, z_in) = gold["in"]; z_gold = gold["out"]
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    m, z = ft(m_in.unsqueeze(0)), ft(z_in.unsqueeze(0))
    for block_remap in blocks:
        m, z = _run_block(block_remap, m, z, ckc)
    zo = torch.Tensor(ttnn.to_torch(z)).float().reshape(z_gold.shape)
    z_pcc = _pcc(zo, z_gold.float())
    print(f"\nOF3 MSAModuleStack (4 blocks): z_pcc={z_pcc:.5f}")
    assert z_pcc > 0.97
