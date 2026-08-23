"""The per-card baseline key must survive a host where tt-smi does not answer.

Both gates that carry per-card baselines (perf_regression, and release_gate's size-ladder
through it) look their numbers up under the board type detect_card_type() returns. tt-smi is
the canonical source, but it hangs on some hosts -- on qb1 2026-08-23 `tt-smi -s` did not
return inside 60 s against a 20 s timeout -- and then the sysfs fallback is the only thing
standing between the gate and a NO BASELINE failure that says nothing about the release.
"""
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None


def _load():
    path = REPO / "scripts" / "perf_regression.py"
    spec = importlib.util.spec_from_file_location("tt_bio_perf_regression_cardtest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _detect_with_sysfs(mod, sub, visible="0"):
    """detect_card_type() with tt-smi absent and sysfs reporting *sub*."""
    mod._resolve_tt_smi = lambda: None
    mod._sysfs_subsystem_device = lambda _dev: sub
    os.environ["TT_VISIBLE_DEVICES"] = visible
    return mod.detect_card_type()


def test_sysfs_fallback_names_p150():
    """0x0040 is p150a, the board most gate legs run on.

    Measured on two hosts: pc node 0 (the card every p150a baseline was recorded on) and
    all four qb1 nodes read subsystem 0x0040, and tt-smi calls that board p150a. Before
    this mapping existed the fallback returned 'unknown:0x0040' and the size-ladder failed
    'NO BASELINE for card type unknown:0x0040' in 23 s on a host whose tt-smi was slow.
    """
    assert _detect_with_sysfs(_load(), "0x0040") == "p150a"


def test_sysfs_fallback_still_names_p300():
    """The board the fallback already knew must keep resolving."""
    mod = _load()
    for sub in ("0x0044", "0x0045", "0x0046"):
        assert _detect_with_sysfs(mod, sub) == "p300c", sub


def test_unknown_subsystem_still_fails_loudly():
    """An unmapped board must NOT silently borrow another board's baseline.

    The whole point of the 'unknown:<sub>' key is that a missing baseline is louder than a
    wrong one, so widening the map must not turn into a default.
    """
    assert _detect_with_sysfs(_load(), "0xbeef") == "unknown:0xbeef"


def test_tt_smi_wins_over_sysfs():
    """sysfs is the fallback, not an override: a live tt-smi still decides."""
    mod = _load()
    mod._sysfs_subsystem_device = lambda _dev: "0x0040"
    os.environ["TT_VISIBLE_DEVICES"] = "0"

    class _Out:
        stdout = '{"device_info": [{"board_info": {"board_type": "p300c"}}]}'

    mod._resolve_tt_smi = lambda: "/nonexistent/tt-smi"
    mod.subprocess.run = lambda *a, **k: _Out()
    assert mod.detect_card_type() == "p300c"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name}  OK")
    print("card-type detection: OK")
