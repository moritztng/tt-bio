"""On-device parity for the OpenFold3 confidence heads (P10).

Golden: ~/of3_ref_out.pkl["intermediates"]["confidence_heads_real"], captured by
scripts/of3_confidence_golden.py -- the reference ``AuxiliaryHeadsAllAtom`` forward
(AF3 Algorithm 31) on real featurized ubiquitin with real of3-p2-155k.pt weights, fed
the real trunk (si_input/si_trunk/zij_trunk) and the real diffusion-sampler coords
(xl_final), use_zij_trunk_embedding=True (the reference eval-mode value).

The device ``OF3ConfidenceHead`` runs a hybrid confidence Pairformer: the z-path
(TriangleMultiplication / TriangleAttention / Transition on [N, N, 128], the heavy pair
compute) on device bf16 (HiFi4 + fp32 dest acc), and the s-path (LN + AttentionPairBias
+ Transition on [N, 384]) on host fp32 -- precision-motivated, because the confidence
Pairformer receives the trunk's raw ``si_trunk`` at ~196k magnitude (no glue/LayerNorm,
unlike the trunk Pairformer which starts each cycle at s~187 via the s-glue), where bf16
corrupts the small per-block s-updates (attention-amplified) and the plddt/resolved
LayerNorm exposes them. The z-embedding and the five output heads are host-fp32
(mirroring Protenix-v2's ConfidenceHead). All five heads PCC-gate vs the real golden.
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


def _run():
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_confidence import OF3ConfidenceHead

    sd = torch.load(_CKPT, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}
    g = pickle.load(open(_GOLD, "rb"))["intermediates"]["confidence_heads_real"]

    dev = get_device()
    head = OF3ConfidenceHead(aux, dev, _cfg(dev))
    out = head.forward(
        si_input=g["si_input"].float(),
        si_trunk=g["si_trunk"].float(),
        zij_trunk=g["zij_trunk"].float(),
        repr_x_pred=g["repr_x_pred"].float(),
        max_atom_per_token_mask=g["max_atom_per_token_mask"].float(),
        use_zij_trunk_embedding=g["use_zij_trunk_embedding"],
    )
    return g, out


_HEADS = ["plddt_logits", "experimentally_resolved_logits", "pae_logits",
          "pde_logits", "distogram_logits"]


def test_of3_confidence_heads_on_device():
    """Gate all five confidence heads + the confidence Pairformer (si_conf, zij_conf) vs
    the real OF3 reference golden. PCC > 0.98 per head."""
    g, out = _run()
    pccs = {h: _pcc(out[h].float(), g[h].float()) for h in _HEADS}
    si_pcc = _pcc(out["si_conf"].float(), g["si_conf"].float())
    zij_pcc = _pcc(out["zij_conf"].float(), g["zij_conf"].float())
    print("\nOF3 confidence heads (device z-path bf16 + host-fp32 s-path + host-fp32 heads):")
    for h in _HEADS:
        print(f"  {h:32s} PCC={pccs[h]:.5f}  shape={tuple(out[h].shape)}")
    print(f"  si_conf (conf Pairformer s)        PCC={si_pcc:.5f}")
    print(f"  zij_conf (conf Pairformer z, dev)  PCC={zij_pcc:.5f}")
    for h, p in pccs.items():
        assert p > 0.98, f"{h} PCC {p:.5f} below 0.98"
    assert si_pcc > 0.98, f"si_conf PCC {si_pcc:.5f} below 0.98"
    assert zij_pcc > 0.98, f"zij_conf PCC {zij_pcc:.5f} below 0.98"
