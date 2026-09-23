"""Regression: protenix-v2 must run its trunk at the spec 10 recycling cycles by default.

The shared ``--recycling_steps`` flag historically defaulted to 3 (the Boltz-2/AF3
convention), but ``protenix.Trunk.N_CYCLES = 10`` is Protenix-v2's spec. Running it at 3
under-recycled the trunk into a bimodal ensemble the confidence head then mis-ranked, so the
delivered structure on hard targets was much worse than a sample already in the ensemble
(7ROA delivered 3.47 A versus 2.35 A at 10).

ESMFold2/ESMFold2-Fast likewise default to 10, the ESMFold2 paper's benchmark protocol
(A.2.10: "For all benchmark results ... ESMFold2 uses 10 loops").

``predict`` now resolves the count per-model via ``_resolve_recycling_steps``; these tests pin
the contract so the fix can't silently revert. Host-only — no device, no network.
"""
from __future__ import annotations

import pytest

from tt_bio.main import _resolve_recycling_steps as _resolve
from tt_bio.protenix import Trunk


def test_protenix_default_is_the_spec():
    """Unset (None) -> protenix-v2 uses its spec, which is Trunk.N_CYCLES (10)."""
    assert _resolve(None, "protenix-v2") == 10
    assert _resolve(None, "protenix-v2") == Trunk.N_CYCLES  # stays in lockstep with the model


@pytest.mark.parametrize("model", ["esmfold2", "esmfold2-fast"])
def test_esmfold2_default_is_the_paper_protocol(model):
    """Unset -> esmfold2/esmfold2-fast use the ESMFold2 paper's 10 loops (A.2.10)."""
    assert _resolve(None, model) == 10


@pytest.mark.parametrize("model", ["boltz2"])
def test_boltz2_keeps_the_af3_default(model):
    """Unset -> boltz2 keeps the historical Boltz-2/AF3 count of 3 (unchanged behavior)."""
    assert _resolve(None, model) == 3


@pytest.mark.parametrize("model", ["protenix-v2", "boltz2", "esmfold2", "esmfold2-fast"])
@pytest.mark.parametrize("n", [1, 3, 5, 10, 20])
def test_explicit_value_overrides_for_every_model(model, n):
    """An explicit --recycling_steps is honored verbatim, regardless of model."""
    assert _resolve(n, model) == n


def test_explicit_overrides_even_when_equal_to_a_default():
    """An explicit 3 on protenix-v2 is honored (not silently bumped to 10), and an explicit
    10 on boltz2 is honored (not clamped to 3) — the override is unconditional."""
    assert _resolve(3, "protenix-v2") == 3
    assert _resolve(10, "boltz2") == 10


def test_worker_reads_an_explicit_zero_as_zero():
    """`cfg.get(...) or default` read 0 as unset: rf3 ran 10 cycles, openfold3 4."""
    from tt_bio.worker import _recycles
    assert _recycles({"recycling_steps": 0}, 10) == 0
    assert _recycles({"recycling_steps": None}, 10) == 10
    assert _recycles({}, 3) == 3


@pytest.mark.parametrize("model,ok", [("protenix-v2", False), ("protenix-v1", False),
                                      ("opendde", False), ("opendde-abag", False),
                                      ("rf3", False), ("boltz2", True), ("esmfold2", True),
                                      ("openfold3", True), ("openbind", True)])
def test_zero_recycles_refused_where_it_means_no_trunk(model, ok):
    """Models that count trunk cycles refuse 0; the N+1 models accept it. None always passes."""
    import click
    from tt_bio.main import _check_recycling_steps as check
    check(None, model)
    check(1, model)
    if ok:
        check(0, model)
    else:
        with pytest.raises(click.BadParameter, match="counts trunk cycles"):
            check(0, model)
    with pytest.raises(click.BadParameter):
        check(-1, model)
