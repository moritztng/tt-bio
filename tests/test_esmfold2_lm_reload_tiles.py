"""A released ESMC-6B comes back from host tiles kept in the process, not from the checkpoint.

On Wormhole fold_complex releases the 6B after its forward for an MSA fold above 768 aa. The next
fold's preload re-read and re-tiled the whole checkpoint: 96 s of a warm 1024 aa ESMFold2 fold's
201 s (wrl, .108, 1000 MHz). The first load now keeps the device-format host tiles and a reload only
uploads them.
"""
import pytest

pytest.importorskip("ttnn")


def test_dict_weight_cache_dumps_then_loads_without_weights(monkeypatch):
    from tt_bio import tenstorrent as T

    made, sent = [], []
    monkeypatch.setattr(T.ttnn, "from_torch", lambda x, **k: made.append(x) or ("host", x))
    monkeypatch.setattr(T.ttnn, "to_device", lambda h, d: sent.append(h) or ("dev", h))
    m = object.__new__(T.Module)
    m.device = "chip"
    m.weights = {"a": 1, "b": 2}
    tiles = {}
    with T.weight_cache(tiles, "dump"):
        first = [m.torch_to_tt(k, transform=lambda x: x * 10) for k in ("a", "b")]
    assert tiles == {0: ("host", 10), 1: ("host", 20)} and made == [10, 20]
    m.weights = {}  # a load reads no weight
    with T.weight_cache(tiles, "load"):
        again = [m.torch_to_tt(k) for k in ("a", "b")]
    assert again == first and made == [10, 20]
    assert T._weight_cache is None


@pytest.mark.parametrize("wormhole,tile_dir", [(True, None), (False, None), (True, "/c"), (False, "/c")])
def test_adapter_reloads_from_its_tiles_on_wormhole(monkeypatch, wormhole, tile_dir):
    """With a persistent tile cache every load reads it; without one Wormhole keeps the tiles in RAM."""
    from tt_bio import esmfold2_runtime as rt
    from tt_bio import tenstorrent as T

    calls = []

    class LM:
        @classmethod
        def from_pretrained(cls, repo, tile_dir=None):
            wc = T._weight_cache
            calls.append(("pretrained", wc and wc["mode"], tile_dir))
            if wc:
                wc["dir"][0] = "tile"
            return cls()

        def load_state_dict(self, sd, strict=True):
            wc = T._weight_cache
            calls.append(("load", wc["mode"], dict(wc["dir"]), sd))

    monkeypatch.setattr(rt, "ESMCLanguageModel", LM)
    monkeypatch.setattr(rt, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(rt, "dram_peak", lambda *a: None)
    monkeypatch.setattr(rt, "fetch", lambda k: "/snap")
    monkeypatch.setattr(rt, "esmc6b_tile_dir", lambda name, snap: tile_dir)
    a = rt._ESMCAdapter("biohub/ESMC-6B")
    a.preload()
    a.lm = None  # what the release after a fold leaves
    a.preload()
    if tile_dir:
        assert calls == [("pretrained", None, "/c")] * 2 and a._tiles is None
    elif wormhole:
        assert calls == [("pretrained", "dump", False), ("load", "load", {0: "tile"}, {})]
    else:
        assert calls == [("pretrained", None, False)] * 2 and a._tiles is None


def _fake_tiles(monkeypatch):
    """Host tiles as tuples and a tensorbin as a dict entry under the name torch_to_tt publishes."""
    import os
    from tt_bio import tenstorrent as T
    disk, tmp = {}, f".{os.getpid()}.tmp.tensorbin"

    def dump(p, h):
        disk[p[:-len(tmp)]] = h
        open(p, "w").close()

    monkeypatch.setattr(T.ttnn, "from_torch", lambda x, **k: ("host", x))
    monkeypatch.setattr(T.ttnn, "to_device", lambda h, d: ("dev", h))
    monkeypatch.setattr(T.ttnn, "dump_tensor", dump)
    monkeypatch.setattr(T.ttnn, "load_tensor", lambda p: disk[p])
    m = object.__new__(T.Module)
    m.device = "chip"
    return T, m


def test_tile_cache_builds_once_then_loads_without_weights(tmp_path, monkeypatch):
    T, m = _fake_tiles(monkeypatch)
    keys, runs = ("a", "b"), []

    def build():
        runs.append("build")
        m.weights = {"a": 1, "b": 2}
        return [m.torch_to_tt(k, transform=lambda x: x * 10) for k in keys]

    def load():
        runs.append("load")
        m.weights = {}
        return [m.torch_to_tt(k) for k in keys]

    first = T.tile_cached(str(tmp_path), build, load)
    again = T.tile_cached(str(tmp_path), build, load)
    assert runs == ["build", "load"] and again == first == [("dev", ("host", 10)), ("dev", ("host", 20))]
    assert T._weight_cache is None


def test_tile_cache_from_another_construction_is_rebuilt(tmp_path, monkeypatch):
    T, m = _fake_tiles(monkeypatch)
    runs = []

    def make(keys):
        def build():
            runs.append(("build", keys))
            m.weights = {k: 1 for k in keys}
            return [m.torch_to_tt(k, transform=lambda x: x) for k in keys]

        def load():
            runs.append(("load", keys))
            m.weights = {}
            return [m.torch_to_tt(k) for k in keys]
        return build, load

    T.tile_cached(str(tmp_path), *make(("a", "b")))
    T.tile_cached(str(tmp_path), *make(("a", "c")))  # same call count, a different weight second
    T.tile_cached(str(tmp_path), *make(("a", "c")))
    assert runs == [("build", ("a", "b")), ("load", ("a", "c")), ("build", ("a", "c")), ("load", ("a", "c"))]


def test_tile_cache_failed_build_leaves_no_done(tmp_path, monkeypatch):
    T, m = _fake_tiles(monkeypatch)

    def build():
        m.weights = {"a": 1}
        m.torch_to_tt("a", transform=lambda x: x)
        raise KeyError("b")

    with pytest.raises(KeyError):
        T.tile_cached(str(tmp_path), build, lambda: None)
    assert sorted(p.name for p in tmp_path.iterdir()) == [".lock"]


def test_wh_lm_release_rule_counts_drawn_rows():
    """Deep MSAs up to 768 aa keep the 6B resident: the encoder draws max_depth rows a loop (lmr)."""
    from types import SimpleNamespace as NS
    from tt_bio.esmfold2_runtime import wh_releases_lm

    def chains(tokens, depth):
        return [("A", "M" * tokens, NS(depth=depth) if depth else None)]

    # measured resident on a 12 GiB chip (lmr / lmv, 2026-10-20)
    for tokens, depth in [(512, 35), (512, 4096), (256, 16384), (512, 6144), (768, 4096)]:
        assert not wh_releases_lm(chains(tokens, depth), max_depth=1024), (tokens, depth)
    assert wh_releases_lm(chains(768, 8192), max_depth=1024)  # past the deepest point measured
    assert wh_releases_lm(chains(769, 64), max_depth=1024)  # pair track headroom
    assert wh_releases_lm(chains(512, 4096))  # no subsampling: every row is an encoder row
    assert not wh_releases_lm(chains(1088, 0)) and wh_releases_lm(chains(1089, 0))


def test_esmc6b_tile_dir_keys_on_the_real_ttnn(tmp_path, monkeypatch):
    """The key reads the installed ttnn, not a fake: a missing attribute crashed every load (wtc)."""
    from tt_bio import esmc

    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(b"0" * 64)
    monkeypatch.setenv("TT_BIO_CACHE", str(tmp_path / "cache"))
    # The key is under test, not this host's disk: below 10 GB free the cache is off (None).
    monkeypatch.setattr(esmc.shutil, "disk_usage", lambda _p: type("U", (), {"free": 1 << 50})())
    first = esmc.esmc6b_tile_dir("esmc-6b", snap)
    assert first and first == esmc.esmc6b_tile_dir("esmc-6b", snap)
    monkeypatch.setenv("TT_BIO_ESMC_PROBE", "1")
    assert esmc.esmc6b_tile_dir("esmc-6b", snap) != first
