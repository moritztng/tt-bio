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


@pytest.mark.parametrize("wormhole", [True, False])
def test_adapter_reloads_from_its_tiles_on_wormhole(monkeypatch, wormhole):
    from tt_bio import esmfold2_runtime as rt
    from tt_bio import tenstorrent as T

    calls = []

    class LM:
        @classmethod
        def from_pretrained(cls, repo):
            wc = T._weight_cache
            calls.append(("pretrained", wc and wc["mode"]))
            if wc:
                wc["dir"][0] = "tile"
            return cls()

        def load_state_dict(self, sd, strict=True):
            wc = T._weight_cache
            calls.append(("load", wc["mode"], dict(wc["dir"]), sd))

    monkeypatch.setattr(rt, "ESMCLanguageModel", LM)
    monkeypatch.setattr(rt, "is_wormhole", lambda: wormhole)
    monkeypatch.setattr(rt, "dram_peak", lambda *a: None)
    a = rt._ESMCAdapter("biohub/ESMC-6B")
    a.preload()
    a.lm = None  # what the release after a fold leaves
    a.preload()
    if wormhole:
        assert calls == [("pretrained", "dump"), ("load", "load", {0: "tile"}, {})]
    else:
        assert calls == [("pretrained", None), ("pretrained", None)] and a._tiles is None
