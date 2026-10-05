"""--head: a user's Python runs on the fold's trunk outputs and only reads them.

What can go wrong here is quiet: a head that sees padded tokens, one that mutates a tensor the
structure writer reads next, a spec that resolves to a different file inside the worker, and a
head silently ignored on a model that never calls it.
"""

import numpy as np
import pytest
import torch

from tt_bio import heads
from tt_bio.capabilities import FLAG_READERS

N, PAD, ATOMS, APAD, SAMPLES = 5, 3, 7, 2, 2


def _pred():
    g = torch.Generator().manual_seed(0)
    tok = torch.tensor([[1] * N + [0] * PAD], dtype=torch.float32)
    atom = torch.tensor([[1] * ATOMS + [0] * APAD], dtype=torch.float32)
    T = N + PAD
    return {
        "exception": False, "token_masks": tok, "masks": atom,
        "s": torch.randn(1, T, 384, generator=g), "z": torch.randn(1, T, T, 128, generator=g),
        "coords": torch.randn(SAMPLES, ATOMS + APAD, 3, generator=g),
        "plddt": torch.rand(SAMPLES, T, generator=g), "pae": torch.rand(SAMPLES, T, T, generator=g),
    }


def test_fold_view_strips_padding_on_every_axis():
    pred = _pred()
    f = heads.fold_view(pred, {})
    assert f.s.shape == (N, 384) and f.z.shape == (N, N, 128)
    assert f.coords.shape == (SAMPLES, ATOMS, 3)
    assert f.plddt.shape == (SAMPLES, N) and f.pae.shape == (SAMPLES, N, N) and f.pde is None
    assert torch.equal(f.z, pred["z"][0, :N, :N])


HEAD_SRC = '''
import torch
calls = []

def mean_z(fold):
    calls.append(1)
    return {"mean": fold.z.mean(-1), "n": len(fold.s)}

class Scaled(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(128))

    def forward(self, fold):
        fold.z.mul_(0)          # a careless head; must not reach the prediction
        return {"y": (fold.z * self.w).sum(-1)}
'''


@pytest.fixture
def head_file(tmp_path):
    p = tmp_path / "my_heads.py"
    p.write_text(HEAD_SRC)
    return p


def test_resolve_makes_a_file_spec_absolute(head_file, monkeypatch):
    monkeypatch.chdir(head_file.parent)
    assert heads.resolve("my_heads.py:mean_z") == f"{head_file}:mean_z"
    assert heads.resolve("pkg.mod:fn") == "pkg.mod:fn"
    with pytest.raises(ValueError):
        heads.resolve("my_heads.py")
    with pytest.raises(FileNotFoundError):
        heads.resolve("missing.py:fn")


def test_run_writes_one_npz_per_head(head_file, tmp_path):
    loaded = {heads.name(s): heads.load(heads.resolve(s))
              for s in (f"{head_file}:mean_z", f"{head_file}:Scaled")}
    pred = _pred()
    heads.run(loaded, pred, {}, tmp_path, "rec")
    a = np.load(tmp_path / "rec_mean_z.npz")
    assert a["mean"].shape == (N, N) and int(a["n"]) == N
    assert np.load(tmp_path / "rec_Scaled.npz")["y"].shape == (N, N)


def test_a_head_that_writes_in_place_cannot_change_the_prediction(head_file, tmp_path):
    pred = _pred()
    before = {k: v.clone() for k, v in pred.items() if torch.is_tensor(v)}
    heads.run({"Scaled": heads.load(f"{head_file}:Scaled")}, pred, {}, tmp_path, "rec")
    assert all(torch.equal(before[k], pred[k]) for k in before)


def test_a_head_must_return_arrays(tmp_path):
    with pytest.raises(TypeError, match="dict of arrays"):
        heads.run({"bad": lambda fold: fold.z}, _pred(), {}, tmp_path, "rec")


def test_no_head_writes_nothing(tmp_path):
    heads.run({}, _pred(), {}, tmp_path, "rec")
    assert list(tmp_path.iterdir()) == []


def test_only_boltz2_reads_it():
    assert FLAG_READERS["--head"] == ("boltz2",)


def test_predict_refuses_a_head_on_another_model(head_file):
    from click.testing import CliRunner

    from tt_bio.main import cli

    target = head_file.with_name("t.yaml")
    target.write_text("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MKV\n")
    r = CliRunner().invoke(cli, ["predict", str(target), "--model", "esmfold2",
                                 "--head", f"{head_file}:mean_z"])
    assert r.exit_code != 0 and "--head is not available for --model esmfold2" in r.output
