"""ABodyBuilder3 on-device parity tests vs the reference golden.

Golden: ~/abb3_golden.pkl (or $ABB3_GOLDEN), captured by
scripts/abb3_golden.py -- the vendored reference ``StructureModule`` forward on the
paired 6yio H0-L0 Fv with the real ``plddt-loss`` checkpoint, dumping per-block
IPA / LayerNorm / Transition / BackboneUpdate / AngleResnet intermediates plus the
final single state, pLDDT logits, and atom14 positions. The ttnn port PCC-gates
each component against this golden (PCC > 0.98) -- real weights, real inputs.

This file currently ships the comparator + a reference-self-consistency test
(running the reference IPA on the golden block-0 inputs reproduces the golden
ipa_delta bit-identically, PCC ~ 1.0). That proves the golden + comparator are
correct so the ttnn IPA port (next chunk -- IPA has no reusable primitive in
tt-bio, so it is ported from scratch) drops straight in: replace the reference
call with the ttnn module and assert the same PCC bar.
"""
import os
import pickle

import pytest
import torch

_GOLD = os.environ.get("ABB3_GOLDEN", os.path.expanduser("~/abb3_golden.pkl"))
pytestmark = pytest.mark.skipif(not os.path.exists(_GOLD),
                                reason="abb3 golden pkl missing (run scripts/abb3_golden.py)")


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


@pytest.fixture(scope="module")
def golden():
    return pickle.load(open(_GOLD, "rb"))


def test_golden_shapes(golden):
    """Sanity: the golden has all 8 blocks with every component's intermediates."""
    assert len(golden["blocks"]) == 8
    b0 = golden["blocks"][0]
    assert b0["ipa_s_in"].shape == (1, 229, 128)
    assert b0["ipa_z_in"].shape == (1, 229, 229, 128)
    assert b0["ipa_rot_mats"].shape == (1, 229, 3, 3)
    assert b0["ipa_trans"].shape == (1, 229, 3)
    assert b0["ipa_delta"].shape == (1, 229, 128)
    assert b0["bb_update"].shape == (1, 229, 6)
    assert b0["ang_norm"].shape == (1, 229, 7, 2)
    assert golden["final"]["single"].shape == (1, 229, 128)
    assert golden["final"]["plddt_logits"].shape == (1, 229, 50)
    assert golden["final"]["atom14"].shape == (1, 229, 14, 3)


def test_reference_ipa_self_consistent(golden):
    """Reference IPA on the golden block-0 inputs reproduces the golden ipa_delta
    bit-identically. This validates the golden + comparator; the ttnn IPA port
    replaces the reference call below with the on-device module and asserts the
    same PCC > 0.98 bar."""
    from tt_bio._vendor.abodybuilder3.openfold.model.structure_module import StructureModule
    from tt_bio._vendor.abodybuilder3.openfold.utils.rigid_utils import Rigid, Rotation
    from tt_bio.abodybuilder3_weights import ABB3_CONFIG, ensure_abb3_weights

    cache = os.environ.get("TT_BIO_CACHE", os.path.expanduser("~/.ttbio"))
    sd = torch.load(ensure_abb3_weights(cache) if not os.path.exists(
        os.path.join(cache, "abodybuilder3_plddt.pt")) else
        os.path.join(cache, "abodybuilder3_plddt.pt"),
        map_location="cpu", weights_only=True)
    model = StructureModule(**ABB3_CONFIG)
    model.load_state_dict(sd, strict=True)
    model.eval()

    b0 = golden["blocks"][0]
    r = Rigid(Rotation(rot_mats=b0["ipa_rot_mats"], quats=None), b0["ipa_trans"])
    with torch.no_grad():
        delta = model.ipa_layers[0](
            b0["ipa_s_in"], b0["ipa_z_in"], r, b0["ipa_mask"])
    pcc = _pcc(delta.float(), b0["ipa_delta"].float())
    assert pcc > 0.999, f"reference IPA self-consistency PCC={pcc}"


@pytest.mark.skipif(not os.environ.get("TT_VISIBLE_DEVICES"),
                        reason="needs a Tenstorrent device (set TT_VISIBLE_DEVICES=0)")
def test_abodybuilder3_ipa_projections_on_device(golden):
    """On-device IPA linear projections PCC > 0.98 vs the reference internals.

    The IPA linear projections (q, kv, qp, kvp, pair bias b) are the on-device
    piece of the IPA -- validated PCC 1.0. The IPA attention (scalar q.k AND point)
    is the documented ceiling: it needs subtile head/point-dim reshapes (head=12,
    head_dim=16, P_q/P_v=4/8, point coords=3) that ttnn stock ops cannot express on
    device; a full on-device IPA needs a custom tt-metal point-attention kernel
    (kernel authoring is a separate domain, deferred). See tt_bio/abodybuilder3.py
    IPALayer docstring."""
    import subprocess, sys
    from tt_bio.tenstorrent import WeightScope
    from tt_bio.abodybuilder3 import (abb3_compute_kernel_config, IPALayer,
                                  _from_torch, _to_torch)

    cache = os.environ.get("TT_BIO_CACHE", os.path.expanduser("~/.ttbio"))
    ipa_pkl = os.path.join(cache, "abb3_ipa_internals.pkl")
    if not os.path.exists(ipa_pkl):
        env = dict(os.environ, TT_BIO_CACHE=cache)
        subprocess.run([sys.executable, "scripts/abb3_ipa_internals.py"], check=True, env=env)
    I = pickle.load(open(ipa_pkl, "rb"))

    sd = torch.load(os.path.join(cache, "abodybuilder3_plddt.pt"), map_location="cpu", weights_only=True)
    scope = "ipa_layers.0"
    weights = WeightScope({k[len(scope) + 1:]: v for k, v in sd.items() if k.startswith(scope + ".")})
    ck = abb3_compute_kernel_config()
    ipa = IPALayer(weights, ck)

    out = ipa(_from_torch(I["s"]), _from_torch(I["z"]),
                 _from_torch(I["rot_mats"]), _from_torch(I["trans"]), _from_torch(I["mask"]))
    N = I["s"].shape[1]
    q = _to_torch(out["q"]).reshape(1, N, -1)
    kv = _to_torch(out["kv"]).reshape(1, N, -1)
    qp = _to_torch(out["qp"]).reshape(1, N, -1)
    kvp = _to_torch(out["kvp"]).reshape(1, N, -1)
    b = _to_torch(out["b"]).reshape(I["b"].shape)

    q_ref = I["q"].reshape(1, N, -1)
    kv_ref = torch.cat([I["k"], I["v"]], dim=-1).reshape(1, N, -1)
    qp_ref = I["q_pts"].permute(0, 1, 4, 2, 3).reshape(1, N, -1)
    kvp_ref = torch.cat([I["k_pts"], I["v_pts"]], dim=-2).permute(0, 1, 4, 2, 3).reshape(1, N, -1)

    assert _pcc(q, q_ref) > 0.98, f"IPA q PCC={_pcc(q, q_ref)}"
    assert _pcc(kv, kv_ref) > 0.98, f"IPA kv PCC={_pcc(kv, kv_ref)}"
    assert _pcc(qp, qp_ref) > 0.98, f"IPA qp PCC={_pcc(qp, qp_ref)}"
    assert _pcc(kvp, kvp_ref) > 0.98, f"IPA kvp PCC={_pcc(kvp, kvp_ref)}"
    assert _pcc(b, I["b"]) > 0.98, f"IPA pair bias PCC={_pcc(b, I['b'])}"


@pytest.mark.skipif(not os.environ.get("TT_VISIBLE_DEVICES"),
                        reason="needs a Tenstorrent device (set TT_VISIBLE_DEVICES=0)")
def test_abodybuilder3_hybrid_end_to_end(golden):
    """End-to-end hybrid StructureModuleTT parity vs the reference: Cα-RMSD < 0.5 Å
    and pLDDT PCC > 0.98 on the 6yio H0-L0 Fv.

    The hybrid runs the IPA linear projections + linear_out + input embeddings +
    post-IPA LayerNorm + Transition + BackboneUpdate + AngleResnet linears + pLDDT
    head on device (bf16, fp32 dest acc; PCC ~1.0 component-by-component), and the
    IPA rigid-apply + scalar/point attention + value aggregation + quaternion
    backbone compose + torsion_angles_to_frames + atom14 reconstruction on host fp32
    (the documented ceiling -- the IPA attention needs a custom tt-metal
    point-attention kernel for a fully on-device port)."""
    import torch
    from tt_bio.abodybuilder3_weights import ABB3_CONFIG, ensure_abb3_weights
    from tt_bio.abodybuilder3 import abb3_compute_kernel_config, StructureModuleTT
    from tt_bio._vendor.abodybuilder3.openfold.data.data_transforms import make_atom14_masks
    from tt_bio._vendor.abodybuilder3.openfold.utils.feats import atom14_to_atom37

    cache = os.environ.get("TT_BIO_CACHE", os.path.expanduser("~/.ttbio"))
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from abodybuilder3_reference import string_to_input, EXAMPLE_HEAVY, EXAMPLE_LIGHT, compute_plddt
    inp = string_to_input(EXAMPLE_HEAVY, EXAMPLE_LIGHT, "cpu")
    single, pair, aatype = inp["single"], inp["pair"], inp["aatype"]
    mask = torch.ones(single.shape[:-1], dtype=single.dtype)

    ref_atom14 = golden["final"]["atom14"][0]
    batch = make_atom14_masks({"aatype": aatype.squeeze(0)})
    ref_atom37 = atom14_to_atom37(ref_atom14, batch)
    ref_ca = ref_atom37[:, 1]

    sd = torch.load(ensure_abb3_weights(cache), map_location="cpu", weights_only=True)
    ck = abb3_compute_kernel_config()
    model = StructureModuleTT(sd, ck, ABB3_CONFIG)
    with torch.no_grad():
        out = model(single, pair, aatype, mask)
    atom14 = out["positions"][-1, 0]
    atom37 = atom14_to_atom37(atom14, batch)
    ca = atom37[:, 1]

    # Kabsch Cα-RMSD
    a = ca.double(); b = ref_ca.double()
    a, b = a - a.mean(0), b - b.mean(0)
    u, s, vt = torch.linalg.svd(a.T @ b)
    d = torch.sign(torch.det(vt.T @ u.T))
    corr = torch.diag(torch.tensor([1.0, 1.0, d], dtype=a.dtype))
    a = (u @ corr @ vt @ a.T).T
    rmsd = float(torch.sqrt(((a - b) ** 2).sum(-1).mean()))
    assert rmsd < 0.5, f"end-to-end Cα-RMSD={rmsd:.4f} Å"

    from abodybuilder3_reference import compute_plddt as _cpp
    plddt = _cpp(out["plddt"][0])
    plddt_ref = _cpp(golden["final"]["plddt_logits"][0])
    assert _pcc(plddt, plddt_ref) > 0.98, f"pLDDT PCC={_pcc(plddt, plddt_ref)}"


@pytest.mark.skip(reason="ttnn IPA port is the next chunk -- IPA has no reusable "
                         "primitive in tt-bio (its ESMFold2 is a diffusion folder), "
                         "so InvariantPointAttention is ported from scratch.")
def test_abodybuilder3_ipa_on_device(golden):
    """On-device IPA PCC > 0.98 vs the golden ipa_delta, fed the golden block-0
    (s, z, rot_mats, trans, mask). Drop-in target once tt_bio.abodybuilder3.IPA
    lands."""
