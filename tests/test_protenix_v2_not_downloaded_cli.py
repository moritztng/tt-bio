"""`tt-bio predict --model protenix-v2` on a host without the checkpoint is refused before
the MSA search, with the reason and the model to use instead. A host that has a copy is let
through. Nothing here downloads or opens a card."""

import zipfile

import pytest

click = pytest.importorskip("click")
from click.testing import CliRunner

from tt_bio import main, weights


class _PastTheCheck(Exception):
    pass


@pytest.fixture
def stop_at_msa(monkeypatch):
    def stop(*a, **k):
        raise _PastTheCheck
    monkeypatch.setattr(main, "_resolve_msa_default", stop)
    for var in weights.ARTIFACTS["protenix-v2"].env_vars:
        monkeypatch.delenv(var, raising=False)


def _predict(tmp_path):
    f = tmp_path / "t.yaml"
    f.write_text("sequences:\n  - protein:\n      id: A\n      sequence: MKTAYIAKQR\n")
    return CliRunner().invoke(main.cli, ["predict", str(f), "--model", "protenix-v2",
                                         "--cache", str(tmp_path), "--out_dir",
                                         str(tmp_path / "out")])


def test_refused_without_a_checkpoint(tmp_path, stop_at_msa):
    res = _predict(tmp_path)
    assert res.exit_code != 0 and not isinstance(res.exception, _PastTheCheck)
    assert "does not download it" in res.output and "protenix-v1" in res.output


def test_a_copy_on_disk_is_let_through(tmp_path, stop_at_msa):
    with zipfile.ZipFile(tmp_path / "protenix-v2.pt", "w") as z:
        z.writestr("data.pkl", b"x")
    assert isinstance(_predict(tmp_path).exception, _PastTheCheck)
