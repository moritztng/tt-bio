"""On-device (ttnn) parity for the ABodyBuilder3 standard structure-module
components vs the reference golden (scripts/abb3_golden.py).

Each component (post-IPA LayerNorm, Transition, BackboneUpdate, AngleResnet,
pLDDT head) is fed the exact golden input (real weights, real 6yio H0-L0 inputs)
and its ttnn output is PCC-compared against the golden reference output. The bar
is PCC > 0.98 per component (the tt-bio porting gate).

The novel InvariantPointAttention point-attention is NOT covered here — it is the
long pole of the port (no reusable primitive in tt-bio) and lands in a follow-on
chunk. See tests/test_abodybuilder3_parity.py for the IPA drop-in target.
"""
import os
import pickle

import pytest
import torch

from tt_bio.tenstorrent import get_device
from tt_bio.abodybuilder3_weights import ABB3_CONFIG, ensure_abb3_weights

_GOLD = os.environ.get("ABB3_GOLDEN", os.path.expanduser("~/abb3_golden.pkl"))
pytestmark = pytest.mark.skipif(
    not os.path.exists(_GOLD),
    reason="abb3 golden pkl missing (run scripts/abb3_golden.py)",
)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum()
                 / ((a - a.mean()).norm() * (b - b.mean()).norm()))


@pytest.fixture(scope="module")
def golden():
    return pickle.load(open(_GOLD, "rb"))


@pytest.fixture(scope="module")
def sd():
    cache = os.environ.get("TT_BIO_CACHE", os.path.expanduser("~/.ttbio"))
    path = os.path.join(cache, "abodybuilder3_plddt.pt")
    if not os.path.exists(path):
        path = ensure_abb3_weights(cache)
    return torch.load(path, map_location="cpu", weights_only=True)


@pytest.fixture(scope="module")
def ck():
    from tt_bio.abodybuilder3 import abb3_compute_kernel_config
    return abb3_compute_kernel_config()


def _slice(out, ref):
    """Match the ttnn (possibly tile-padded) output shape to the golden's."""
    return out.reshape(ref.shape)


def _run(mod_cls, sd, ck, key_prefix, in_keys, out_key, golden, block=None):
    from tt_bio.abodybuilder3 import _from_torch, _to_torch
    scope = key_prefix if block is None else f"{key_prefix}.{block}"
    from tt_bio.tenstorrent import WeightScope
    weights = WeightScope({k[len(scope) + 1:]: v for k, v in sd.items() if k.startswith(scope + ".")})
    mod = mod_cls(weights, ck)
    if isinstance(in_keys, tuple):
        args = tuple(_from_torch(golden["blocks"][block][k] if block is not None else golden["final"][k])
                     for k in in_keys)
    else:
        src = golden["blocks"][block] if block is not None else golden["final"]
        args = (_from_torch(src[in_keys]),)
    out = mod(*args)
    return _to_torch(out)


def test_ipa_layernorm(golden, sd, ck):
    from tt_bio.abodybuilder3 import IPALayerNorm
    out = _run(IPALayerNorm, sd, ck, "layer_norm_ipa_layers", "ln_s_in", "ln_s_out",
               golden, block=0)
    ref = golden["blocks"][0]["ln_s_out"]
    pcc = _pcc(_slice(out, ref), ref)
    assert pcc > 0.98, f"IPA LayerNorm PCC={pcc}"


def test_backbone_update(golden, sd, ck):
    from tt_bio.abodybuilder3 import BackboneUpdate
    out = _run(BackboneUpdate, sd, ck, "bb_update_layers", "bb_s_in", "bb_update",
               golden, block=0)
    ref = golden["blocks"][0]["bb_update"]
    pcc = _pcc(_slice(out, ref), ref)
    assert pcc > 0.98, f"BackboneUpdate PCC={pcc}"


def test_transition(golden, sd, ck):
    from tt_bio.abodybuilder3 import StructureModuleTransition
    out = _run(StructureModuleTransition, sd, ck, "transition_layers", "trans_s_in",
               "trans_s_out", golden, block=0)
    ref = golden["blocks"][0]["trans_s_out"]
    pcc = _pcc(_slice(out, ref), ref)
    assert pcc > 0.98, f"Transition PCC={pcc}"


def test_angle_resnet(golden, sd, ck):
    from tt_bio.abodybuilder3 import AngleResnet, normalize_angles
    from tt_bio.tenstorrent import WeightScope
    from tt_bio.abodybuilder3 import _from_torch, _to_torch
    scope = "angle_resnet_layers.0"
    weights = WeightScope({k[len(scope) + 1:]: v for k, v in sd.items() if k.startswith(scope + ".")})
    mod = AngleResnet(weights, ck)
    b0 = golden["blocks"][0]
    out = mod(_from_torch(b0["ang_s_in"]), _from_torch(b0["ang_s_initial"]))
    norm = normalize_angles(_to_torch(out).reshape(b0["ang_norm"].shape))
    pcc = _pcc(norm, b0["ang_norm"])
    assert pcc > 0.98, f"AngleResnet PCC={pcc}"


def test_plddt_head(golden, sd, ck):
    from tt_bio.abodybuilder3 import PLDDTHead
    out = _run(PLDDTHead, sd, ck, "plddt", "single", "plddt_logits", golden, block=None)
    ref = golden["final"]["plddt_logits"]
    pcc = _pcc(_slice(out, ref), ref)
    assert pcc > 0.98, f"pLDDT head PCC={pcc}"
