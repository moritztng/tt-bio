"""Every released RFD4 weight is read by exactly one of the host heads and the device trunk.

The heads load `strict=False`, because the trunk's weights share their `nn.` prefix. Without a
check in both directions a renamed or absent head weight (`sequence_readout`, `insertion_q_proj`)
stays at its init, and an unread tensor is silently ignored: the model designs plausible output
from the wrong weights. `engine.load_trunk` now refuses both, and this pins it card-free on both
ways the engine loads a model: the release export, and `base/` plus `adapter/`.

The check follows Magnus Bauer's weight-consumption gate on his `rfd4-experimental` branch
(`scripts/rfd4_release_weights.py`): weights read must equal weights given, and a test that
cannot fail proves nothing, so a dropped and an extra weight must each turn it red.
"""
from unittest import mock

import pytest

ttnn = pytest.importorskip("ttnn", reason="the trunk is built with ttnn's constructors")

from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import weights as W  # noqa: E402
from tt_bio.rfd4 import engine as E  # noqa: E402
from tt_bio.rfd4 import network as N  # noqa: E402
from tt_bio.rfd4 import trunk as TR  # noqa: E402

BUNDLE = W.resolve("rfd4")
pytestmark = pytest.mark.skipif(BUNDLE is None or not (BUNDLE / "base").is_dir(),
                                reason="the RFD4 bundle is not on this box")


def _host(t, *args, **kwargs):
    return t


def _load(state, config):
    """`engine.load_trunk` with the uploads kept on the host: the same reads, no card."""
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)
    with mock.patch.object(T, "get_device", lambda *a, **k: None), \
            mock.patch.object(TR, "compute_kernel_config", lambda: ckc), \
            mock.patch.object(ttnn, "from_torch", _host), mock.patch.object(ttnn, "to_device", _host):
        glue = N.FlowGlue(state, config)
        E.load_trunk(glue, state, config["nn"])
    return glue


@pytest.fixture(scope="module", params=["release", "base+adapter"])
def checkpoint(request):
    if request.param == "release":
        return N.load_checkpoint(BUNDLE)
    return N.load_checkpoint(BUNDLE / "base", adapter=BUNDLE / "adapter")


def test_the_heads_and_the_trunk_read_every_weight_once(checkpoint):
    state, config = checkpoint
    glue = _load(state, config)
    nn = set(N._sub(state, "nn."))
    assert glue.trunk_keys | set(glue.state_dict()) == nn
    assert not glue.trunk_keys & set(glue.state_dict())
    assert {"sequence_readout.1.weight", "insertion_q_proj.weight", "insertion_k_proj.weight"} <= set(glue.state_dict())


@pytest.mark.parametrize("key", ["nn.insertion_q_proj.weight", "nn.sequence_readout.1.weight"])
def test_a_missing_head_weight_fails_the_load(checkpoint, key):
    state, config = checkpoint
    with pytest.raises(KeyError, match=key.replace(".", r"\.")):
        _load({k: v for k, v in state.items() if k != key}, config)


def test_a_missing_trunk_weight_fails_the_load(checkpoint):
    state, config = checkpoint
    key = "nn.transformer_layers.7.mhba.mha.to_qkv.weight"
    assert key in state
    with pytest.raises(KeyError, match="to_qkv.weight"):
        _load({k: v for k, v in state.items() if k != key}, config)


@pytest.mark.parametrize("key", ["nn.transformer_layers.3.mhba.mha.extra.weight", "nn.insertion_head.weight"])
def test_a_weight_nothing_reads_fails_the_load(checkpoint, key):
    state, config = checkpoint
    with pytest.raises(ValueError, match="read by nothing: " + key.replace(".", r"\.")):
        _load({**state, key: state["nn.insertion_q_proj.weight"]}, config)


def test_a_renamed_head_weight_fails_the_load(checkpoint):
    """The rename that motivated this: the tensor is still there, under a name no module reads."""
    state, config = checkpoint
    renamed = {("nn.insertion_query_proj.weight" if k == "nn.insertion_q_proj.weight" else k): v
               for k, v in state.items()}
    with pytest.raises(KeyError, match=r"nn\.insertion_q_proj\.weight"):
        _load(renamed, config)


def test_a_recording_scope_names_root_keys_through_children():
    scope = T.WeightScope.recording({"a.b.w": 1, "a.c.w": 2})
    child = scope.child("a").child("b")
    assert child["w"] == 1 and child.get("missing") is None
    assert scope.reads == {"a.b.w"}
    assert T.WeightScope({"x": 1}).child("").reads == frozenset()
