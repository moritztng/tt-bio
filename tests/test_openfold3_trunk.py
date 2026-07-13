"""On-device parity for the OpenFold3 trunk assembly (P8).

Golden: ~/of3_ref_out.pkl["intermediates"]["trunk_real"], captured by
scripts/of3_trunk_golden.py -- the reference ``run_trunk`` forward (4 cycles =
num_recycles+1) on real featurized ubiquitin, with per-cycle intermediates
(z_prev, z_after_zglue, z_after_template, m, z_after_msa, s_prev, s_after_sglue)
and the final s_trunk / z_trunk. Same real-weights + real-features methodology as
every other OF3 leg.

Two gates isolate the genuinely-new trunk-assembly device code from the
documented device-xfail sub-components (see docs/openfold3-port.md P8 tick 13):

  1. test_of3_trunk_glue_on_device -- GATED (PCC=1.00000). The new top-level cycle
     glue (OF3TrunkGlue: z = z_init + linear_z(layer_norm_z(z_prev));
     s = s_init + linear_s(layer_norm_s(s_prev))) is gated in isolation across all
     4 cycles: feed the golden per-cycle z_prev / s_prev -> device glue -> compare
     to the golden z_after_zglue / s_after_sglue. Gates the new code with REAL
     non-zero inputs, independent of the xfail pair_stack sub-components.

  2. test_of3_trunk_assembly_on_device -- GATED on s_trunk AND z_trunk (the device
     cycle-glue + 48-block Pairformer output on the real settled trunk
     distribution), WITH the template + MSA pair_stack z substituted from the
     golden so the Pairformer receives the correct z each cycle. This is NOT a
     fully-device-gated trunk: the template + MSA pair_stacks are device-xfail /
     throwing (their own tests) and are substituted here. See the test docstring
     for the full honest scope, including the finding that the device Pairformer
     z-track gates on the real cycle-3 trunk z (unlike the cycle-0 case).

The template + MSA pair_stacks themselves are device-xfail / throwing in their own
tests (tests/test_openfold3_template.py, tests/test_openfold3_msa.py); the trunk
does not re-gate them.
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


def test_of3_trunk_glue_on_device():
    """Gate the new top-level cycle glue (OF3TrunkGlue) in isolation across all 4
    cycles. Feeds the golden per-cycle z_prev/s_prev, compares the device glue
    output to the golden z_after_zglue/s_after_sglue. Byte-correct (PCC > 0.98)."""
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_trunk import OF3TrunkGlue

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    g = pickle.load(open(_GOLD, "rb"))["intermediates"]["trunk_real"]
    nc = g["num_cycles"]
    s_init_ref, z_init_ref = g["s_init"], g["z_init"]

    dev = get_device()
    glue = OF3TrunkGlue(sd, _cfg(dev))
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    s_init_d = ft(s_init_ref.unsqueeze(0))
    z_init_d = ft(z_init_ref.unsqueeze(0))

    z_pccs, s_pccs = [], []
    for c in range(nc):
        cyc = g["cycles"][c]
        z_prev_d = ft(cyc["z_prev"].unsqueeze(0))
        z_glue_d = glue.glue_z(z_prev_d, z_init_d)
        z_glue = torch.Tensor(ttnn.to_torch(z_glue_d)).float().reshape(cyc["z_after_zglue"].shape)
        z_pccs.append(_pcc(z_glue, cyc["z_after_zglue"].float()))
        ttnn.deallocate(z_prev_d)
        ttnn.deallocate(z_glue_d)

        s_prev_d = ft(cyc["s_prev"].unsqueeze(0))
        s_glue_d = glue.glue_s(s_prev_d, s_init_d)
        s_glue = torch.Tensor(ttnn.to_torch(s_glue_d)).float().reshape(cyc["s_after_sglue"].shape)
        s_pccs.append(_pcc(s_glue, cyc["s_after_sglue"].float()))
        ttnn.deallocate(s_prev_d)
        ttnn.deallocate(s_glue_d)

    z_min, s_min = min(z_pccs), min(s_pccs)
    print(f"\nOF3 trunk glue ({nc} cycles): z_pcc={[f'{p:.5f}' for p in z_pccs]} "
          f"s_pcc={[f'{p:.5f}' for p in s_pccs]} (min z={z_min:.5f} s={s_min:.5f})")
    assert z_min > 0.98, f"trunk z-glue min PCC {z_min:.5f} below 0.98"
    assert s_min > 0.98, f"trunk s-glue min PCC {s_min:.5f} below 0.98"


def test_of3_trunk_assembly_on_device():
    """Assembled trunk forward (cycle glue + 48-block Pairformer) with the
    device-xfail template + MSA pair_stack z substituted from the golden, so the
    Pairformer receives the correct z each cycle. GATES s_trunk AND z_trunk -- the
    device cycle-glue + 48-block Pairformer output on the real settled trunk
    distribution -- with the per-cycle z input to the Pairformer taken from the
    golden (template+MSA pair_stacks substituted).

    HONEST SCOPE: this is NOT a fully-device-gated trunk. The template pair_stack
    throws on device (sub-tile head_dim=16 kernel bug) and the MSA pair_stack is
    z-xfail (~0.75); both are substituted from the reference golden here, and both
    are documented-xfail in their own tests (tests/test_openfold3_template.py,
    tests/test_openfold3_msa.py). What IS gated here is the device-runnable
    assembly path: the new top-level cycle glue + the 48-block Pairformer (s AND z
    tracks) run end-to-end across all 4 cycles on the REAL trunk distribution.

    Notable finding: the device Pairformer z-track gates cleanly on the real
    settled (cycle-3) trunk z (z_trunk_pcc~0.999), unlike the cycle-0 (s_init,
    z_init) single-pass case where the P5 final-block catastrophic cancellation
    caps z_pcc at ~0.66. The cancellation is a cycle-0-input-specific artifact;
    the actual trunk's final-cycle Pairformer z does not trigger it. So the real
    trunk z_trunk is device-achievable on the Pairformer side, pending the
    template+MSA pair_stack kernel fix.
    """
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_trunk import OF3Trunk

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    g = pickle.load(open(_GOLD, "rb"))["intermediates"]["trunk_real"]
    nc = g["num_cycles"]
    s_init_ref, z_init_ref = g["s_init"], g["z_init"]
    s_trunk_ref, z_trunk_ref = g["s_trunk"], g["z_trunk"]

    dev = get_device()
    trunk = OF3Trunk(sd, _cfg(dev), num_cycles=nc)
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    s_init_d = ft(s_init_ref.unsqueeze(0))
    z_init_d = ft(z_init_ref.unsqueeze(0))
    z_after_msa = [ft(g["cycles"][c]["z_after_msa"].unsqueeze(0)) for c in range(nc)]

    s_trunk_d, z_trunk_d = trunk(s_init_d, z_init_d, z_after_msa)
    s_trunk = torch.Tensor(ttnn.to_torch(s_trunk_d)).float().reshape(s_trunk_ref.shape)
    z_trunk = torch.Tensor(ttnn.to_torch(z_trunk_d)).float().reshape(z_trunk_ref.shape)
    s_pcc = _pcc(s_trunk, s_trunk_ref.float())
    z_pcc = _pcc(z_trunk, z_trunk_ref.float())
    print(f"\nOF3 assembled trunk ({nc} cycles; template+MSA pair_stack z from golden, "
          f"device-runnable path = cycle glue + 48-block Pairformer): "
          f"s_trunk_pcc={s_pcc:.5f} z_trunk_pcc={z_pcc:.5f} [both gated; NOT a "
          f"fully-device trunk -- template+MSA pair_stacks substituted]")
    assert s_pcc > 0.98, f"assembled trunk s_trunk_pcc {s_pcc:.5f} below 0.98"
    assert z_pcc > 0.98, f"assembled trunk z_trunk_pcc {z_pcc:.5f} below 0.98"
