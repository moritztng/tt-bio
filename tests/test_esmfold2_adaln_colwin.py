"""ESMFold2's AdaLN reads its dit_shoist windows as the shared `_ColWin` (tensor, offset, width, row).

`_ColWin` grew its `row` field for Boltz-2's dit_cond_steps; ESMFold2 still unpacked three fields, so every
Wormhole --fast fold (dit_shoist is in that set) died on its first diffusion step.
"""
import types

import pytest

pytest.importorskip("ttnn")


def test_adaln_unpacks_colwin_and_passes_rows(monkeypatch):
    from tt_bio import esmfold2
    from tt_bio.tenstorrent import _ColWin

    seen = {}
    monkeypatch.setattr(esmfold2.ttnn, "layer_norm", lambda a, **k: ("ln", a))
    monkeypatch.setattr(esmfold2.ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(esmfold2, "addcmul_at", lambda *a, **k: seen.update(args=a, kw=k) or "out")
    me = types.SimpleNamespace(compute_kernel_config=None)
    terms = (_ColWin("sc", 768, 768, 2), _ColWin("sh", 1536, 768, 3))
    assert esmfold2.AdaLN.__call__(me, "a", "s", terms) == "out"
    assert seen["args"] == ("sh", ("ln", "a"), "sc", 768)
    assert seen["kw"] == {"x_off": 1536, "z_off": 768, "x_row": 3, "z_row": 2}
