"""The weights `tt-bio train` writes fold through `tt-bio predict`.

The optimizer's masters are in device coordinates, keyed by where each weight sits in the built
model; predict builds from an upstream-format state dict. `lineage.fold_back` turns one into the
other by differentiating the build, and a weights run writes the result as OUT/weights.pt. These
tests build a toy model with the transforms the real one uses (transpose, fused concatenation,
padding, a constant scale, one parameter stored under two keys) and check the round trip:
rebuild from the folded-back dict and every upload has moved by exactly what training moved it. Host-only: `ttnn.from_torch` is
replaced by a stand-in that keeps the tensor, because even a host upload brings the cluster up.
"""

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from tt_bio.train import lineage  # noqa: E402


class _Upload:
    """What the recording needs of a ttnn tensor: an identity, a shape, no buffer."""

    def __init__(self, t):
        self.t = t.detach().clone()
        self.shape = tuple(t.shape)

    def buffer_address(self):
        raise RuntimeError("host tensor")


@pytest.fixture(autouse=True)
def _host_uploads(monkeypatch):
    monkeypatch.setattr(ttnn, "from_torch", lambda t, *a, **k: _Upload(t))


def _canonical(k):
    return k[len("copy."):] if k.startswith("copy.") else k


def _build(sd):
    """Five uploads, one per transform family the OpenFold3 build uses."""
    up = lambda t: ttnn.from_torch(t)  # noqa: E731
    return {
        "lin": up(sd["lin.weight"].t().contiguous()),
        "qkv": up(torch.cat([sd["q.weight"], sd["k.weight"], sd["v.weight"]]).t()),
        "ln": up(torch.nn.functional.pad(sd["ln.weight"], (0, 3))),
        "scaled": up(sd["s.weight"] * 0.125),
        "aliased": up(sd["copy.a.weight"]),
    }


def _state_dict():
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    a = r(5, 4)
    return {"lin.weight": r(6, 4), "q.weight": r(4, 8), "k.weight": r(4, 8),
            "v.weight": r(4, 8), "ln.weight": r(5), "s.weight": r(3, 3),
            "a.weight": a, "copy.a.weight": a.clone(), "untouched.weight": r(2, 2),
            "step": torch.tensor(7)}


def _host(t):
    return t.t.double()


def _record(sd, **kw):
    with lineage.recording(sd, _canonical, **kw) as (traced, lin):
        model = _build(traced)
    walked = [(p, model, p, t) for p, t in model.items()]
    return model, lin, lin.by_path(walked, hosts=kw.get("differentiable", False))


def test_fold_back_reproduces_every_moved_upload():
    sd = _state_dict()
    before, lin, uploads = _record(sd, differentiable=True)
    g = torch.Generator().manual_seed(1)
    moved = {p: 1e-2 * torch.randn(tuple(t.shape), generator=g, dtype=torch.float64)
             for p, t in before.items() if p != "ln"}
    # A padded weight can only move where it has a source: the optimizer never writes the pad.
    ln = torch.zeros(tuple(before["ln"].shape), dtype=torch.float64)
    ln[:5] = 1e-2
    moved["ln"] = ln
    out = lineage.fold_back(sd, lin, uploads, moved, _canonical)

    after, _, _ = _record(out)
    for p in before:
        got = _host(after[p]) - _host(before[p])
        assert torch.allclose(got, moved[p], atol=1e-6), p
    # A parameter stored twice moves as one, and a key no upload reaches is the same object.
    assert torch.equal(out["a.weight"], out["copy.a.weight"])
    assert not torch.equal(out["a.weight"], sd["a.weight"])
    assert out["untouched.weight"] is sd["untouched.weight"]
    assert out["step"] is sd["step"]
    assert all(out[k].dtype == sd[k].dtype for k in sd)


def test_nothing_moved_is_the_same_state_dict():
    sd = _state_dict()
    _, lin, uploads = _record(sd, differentiable=True)
    out = lineage.fold_back(sd, lin, uploads, {}, _canonical)
    assert all(out[k] is sd[k] for k in sd)


def test_a_value_uploaded_twice_and_trained_apart_is_refused():
    sd = {"w": torch.randn(4, 4)}
    with lineage.recording(sd, differentiable=True) as (traced, lin):
        model = {"one": ttnn.from_torch(traced["w"]), "two": ttnn.from_torch(traced["w"].t())}
    uploads = lin.by_path([(p, model, p, t) for p, t in model.items()], hosts=True)
    moved = {"one": torch.full((4, 4), 1e-2, dtype=torch.float64)}
    with pytest.raises(ValueError, match="cannot be written back"):
        lineage.fold_back(sd, lin, uploads, moved)


def test_the_plain_recording_is_unchanged():
    """The training build's own recording keeps no host tensor and makes no leaf."""
    sd = _state_dict()
    _, lin, keys = _record(sd)
    assert lin.leaves == {}
    assert keys["qkv"] == frozenset({"q.weight", "k.weight", "v.weight"})
    assert keys["aliased"] == frozenset({"a.weight"})


def test_predict_loads_the_shipped_file_or_the_checkpoint_in_its_place(tmp_path):
    from tt_bio import worker

    pt = tmp_path / "shipped.pt"
    torch.save({"w": torch.arange(3.0)}, pt)
    assert torch.equal(worker._of3_state_dict({"of3_ckpt": str(pt)})["w"], torch.arange(3.0))
    trained = tmp_path / "weights.pt"
    torch.save({"w": torch.ones(3)}, trained)
    got = worker._of3_state_dict({"of3_ckpt": str(pt), "checkpoint": str(trained)})
    assert torch.equal(got["w"], torch.ones(3))


def test_predict_names_weights_pt_when_handed_a_resume_checkpoint(tmp_path):
    from tt_bio import worker

    with pytest.raises(RuntimeError, match="weights.pt"):
        worker._of3_state_dict({"of3_ckpt": "x", "checkpoint": str(
            tmp_path / "adapter-00000001.safetensors")})


def test_train_adapters_is_refused_for_openfold3_with_what_to_do():
    from click.testing import CliRunner
    from tt_bio.train.cli import train

    res = CliRunner().invoke(train, ["--model", "openfold3", "--train", "adapters",
                                     "--dry-run"])
    assert res.exit_code == 2
    assert "Drop the flag" in res.output and "--checkpoint" in res.output
    assert "--train" not in CliRunner().invoke(train, ["--help"]).output
