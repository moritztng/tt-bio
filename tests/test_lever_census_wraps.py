"""The seven wrap-counted levers must be counted whatever the host is doing.

Seven levers keep no `*_STATS` of their own and are counted by monkeypatching a helper. A
monkeypatch counts only the calls made after it lands, so when the install was driven by the
census's 3-second dump thread, a fold that got going before the first tick was simply not
counted, and whether that happened depended on how busy the machine was. Measured on the
boltz2-affinity fold at 256 aa: 11446 calls counted with the box idle, 7456 with three
concurrent folds, the 3990-call gap being exactly those seven levers at `0/0` while six of
the seven are served on the apo fold. Nothing in the artifact distinguishes that from a
lever genuinely going dark.

The install is now driven by an import hook, which introduces the failure this file mostly
exists to pin: a module is in `sys.modules` from the moment its execution STARTS, so the hook
can fire while `tt_bio.tenstorrent` is half-built. The old code set its did-this-already flag
BEFORE the first attribute access, so that would claim the flag, raise, get swallowed, and
leave all seven counters dead for the life of the process. Deterministically wrong instead of
load-dependently wrong.

Host-only: no device, no fold. The module under test is driven against a stand-in.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def lc():
    spec = importlib.util.spec_from_file_location(
        "lever_census_under_test", REPO_ROOT / "scripts" / "lever_census.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stub(ready: bool) -> types.ModuleType:
    """A stand-in for tt_bio.tenstorrent, either mid-import or fully executed."""
    m = types.ModuleType("tt_bio.tenstorrent")
    if not ready:
        return m

    class AdaLN:
        def s_terms(self, *a, **k):
            return ()

    class DiffusionModule:
        def _hoist_layer_bias(self, bias, transformer):
            return bias

    m.AdaLN = AdaLN
    m.DiffusionModule = DiffusionModule
    m._pair_transpose_impl = lambda t, mc: t
    m._transpose_memory_config = lambda *a, **k: None
    m._pair_proj_minimal_matmul = lambda *a, **k: None
    m._qkv_mm_config = lambda *a, **k: None
    m.ADALN_S_HOIST = True
    m._B2_ADALN_S_MEMO = False
    m._PT_ROW_MAJOR = True
    return m


@pytest.fixture
def install(monkeypatch):
    def _install(lc, mod):
        monkeypatch.setitem(sys.modules, "tt_bio.tenstorrent", mod)
        monkeypatch.setitem(sys.modules, "ttnn", types.ModuleType("ttnn"))
        lc._install_wraps()
    return _install


def test_a_half_imported_module_does_not_burn_the_only_install_attempt(lc, install):
    """The regression that makes the import hook safe. Claiming the flag against a module
    whose attributes do not exist yet would zero all seven counters for the process, and the
    census would report seven levers dark with the fold running perfectly."""
    partial = _stub(ready=False)
    install(lc, partial)
    assert not getattr(partial, "_census_wrapped", False)


def test_a_ready_module_gets_wrapped_and_counts(lc, install):
    mod = _stub(ready=True)
    install(lc, mod)
    assert getattr(mod, "_census_wrapped", False)
    before = list(lc.WRAP_COUNTS["ADALN_S_HOIST"])
    mod.AdaLN.s_terms(mod.AdaLN())
    assert lc.WRAP_COUNTS["ADALN_S_HOIST"] != before


def test_a_retry_after_the_module_finishes_importing_still_wraps(lc, install):
    """The point of not claiming the flag: the next call has to succeed. Otherwise declining
    to wrap a partial module would just be a quieter way of losing the counters."""
    mod = _stub(ready=False)
    install(lc, mod)
    assert not getattr(mod, "_census_wrapped", False)
    for k, v in vars(_stub(ready=True)).items():
        if not k.startswith("__"):
            setattr(mod, k, v)
    lc._install_wraps()
    assert getattr(mod, "_census_wrapped", False)


def test_wraps_are_installed_once(lc, install):
    """Half-installed wraps cannot be retried without double-counting, so the flag has to
    stop the second pass completely."""
    mod = _stub(ready=True)
    install(lc, mod)
    first = mod._pair_transpose_impl
    lc._install_wraps()
    assert mod._pair_transpose_impl is first


def test_every_wrap_counted_lever_has_a_counter_slot(lc):
    """WRAP_COUNTS and the `wrap` rows in LEVERS are two lists of the same seven levers, and
    a lever in one and not the other reads as permanently dark."""
    wrap = {f for f, _m, _a, _c, how in lc.LEVERS if how == "wrap"}
    assert wrap == set(lc.WRAP_COUNTS)
    assert len(wrap) == 7
