"""TrunkCache: one trunk for inputs that differ only in their constraint."""
import torch

from tt_bio.cache import TrunkCache


def _feats():
    g = torch.Generator().manual_seed(0)
    return {"restype": torch.randn(7, 32, generator=g), "token_bonds": torch.zeros(7, 7, dtype=torch.bool),
            "asym_id": torch.arange(7), "z": torch.randn(7, 7, 4, generator=g).bfloat16()}


def test_key_depends_on_every_feature_and_part(monkeypatch):
    monkeypatch.delenv("TT_BIO_LEVERS", raising=False)
    f = _feats()
    k = TrunkCache.key(f, "opendde", 10, "w", False)
    assert k == TrunkCache.key(_feats(), "opendde", 10, "w", False)
    for name in f:
        g = _feats()
        g[name] = g[name].clone()
        g[name].view(-1)[0] = 1 if g[name].dtype == torch.bool else g[name].view(-1)[0] + 1
        assert TrunkCache.key(g, "opendde", 10, "w", False) != k, name
    assert TrunkCache.key(f, "opendde", 9, "w", False) != k
    assert TrunkCache.key(f, "opendde", 10, "w", True) != k
    monkeypatch.setenv("TT_BIO_LEVERS", "x")
    assert TrunkCache.key(f, "opendde", 10, "w", False) != k


def test_round_trip_and_torn_entry(tmp_path):
    c = TrunkCache(tmp_path)
    v = {"s_inputs": torch.randn(3, 4), "z_trunk": torch.randn(3, 3, 2).bfloat16()}
    assert c.load("k") is None
    c.save("k", v)
    got = c.load("k")
    assert all(torch.equal(got[n], v[n]) and got[n].dtype == v[n].dtype for n in v)
    (tmp_path / "bad.pt").write_bytes(b"not a torch file")
    assert c.load("bad") is None
