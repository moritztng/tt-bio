"""A trace region is reserved only where a captured trace could be replayed.

The reservation is not free and the comment that said it was is the bug this guards. It comes off
EVERY DRAM bank, so on a 12-bank Wormhole Galaxy chip `_ESMC_TRACE_REGION_SIZE` costs 3 GiB of a
12.8 GiB part -- measured off the allocator's own refusal on j10glx02, "bank size is 805306336 B"
with it against "1073741792 B" without. On the sequence axis that was the difference between
esmc-300m refusing 65537 residues and saprot-35m, same code path and no reservation, embedding
73728 on the same part.

`_dispatch` captures on the SECOND sighting of a bucketed shape by design, so a single long
sequence pays the 3 GiB and never gets a trace. These tests pin the decision, and pin the wiring:
a predicate nothing calls decides nothing.

No device is opened here.
"""
import pytest

from tt_bio import esmc


def _width(n: int) -> int:
    """The bucketed token width esmc will pad an n-residue sequence to (<cls> and <eos> included)."""
    return ((n + 2 + esmc.BUCKET - 1) // esmc.BUCKET) * esmc.BUCKET


def test_one_sequence_never_pays_for_a_trace():
    """The capacity case. One forward, one shape, seen once -- a trace can never be replayed."""
    assert esmc.trace_pays({"only": "A" * 60000}) is False
    assert esmc.trace_pays("A" * 60000) is False
    assert esmc.trace_pays(["A" * 300]) is False


def test_a_repeated_bucket_does_pay():
    """The serving case, and the reason this is not simply `len(seqs) > 1`.

    Two sequences of DIFFERENT bucketed widths are two shapes, each seen once, so neither is ever
    captured -- exactly the one-sequence case twice over. Only a repeat can be replayed.
    """
    assert esmc.trace_pays(["A" * 100, "A" * 100]) is True
    assert esmc.trace_pays({"a": "A" * 100, "b": "C" * 110}) is True, (
        "100 and 110 residues land in the same 32-multiple bucket, so the shape does repeat")
    far = _width(100) + esmc.BUCKET * 4
    assert esmc.trace_pays(["A" * 100, "A" * far]) is False, (
        "two widths one bucket apart or more are two shapes, each seen once")


def test_the_bucket_is_the_models_own_and_is_a_multiple_of_32():
    """The predicate must group by the width the chip really allocates, not by raw length.

    If it compared raw lengths, 100 and 110 residues would read as two shapes where the device
    sees one, and the serving path would lose its trace.
    """
    assert esmc.BUCKET % 32 == 0, esmc.BUCKET
    assert _width(100) == _width(110)
    assert esmc.trace_pays(["A" * 100, "A" * 110]) is True


def test_the_cli_and_the_library_both_ask(monkeypatch):
    """The wiring assertion. A predicate nothing calls saves nothing.

    Both entry points that can see a user's sequences before the model loads must pass the answer
    to `load_esmc`; the serving worker deliberately does not, because it loads before any job
    arrives. Asserted by source inspection of the call rather than by running a device path.
    """
    import inspect
    from tt_bio import main
    for fn, who in ((main.embed_cmd.callback, "tt-bio embed"), (esmc.embed, "esmc.embed")):
        src = inspect.getsource(fn)
        assert "trace_pays" in src, (
            f"{who} loads the model without asking trace_pays, so a long single sequence still "
            f"reserves a trace region it can never replay")


def test_the_serving_path_still_traces_unconditionally():
    """The complement, and the reason no perf cell can move on any board.

    A lost trace would show up as a perf regression, and the only place that is measured is
    `scripts/perf_regression.py`'s `esmc-300m-single` leg. That leg does NOT go through the embed
    CLI: it opens the chip itself with `get_device(trace_region_size=...)` and then loads through
    `tt_bio.worker._WorkerState.load_model`, which calls `load_esmc` with `trace` at its default.
    `get_device` caches the first open, so the reservation is already made by the time any model
    code runs and `trace_pays` is never consulted on that path.

    That is deliberate: a serving worker loads before any job arrives, so it cannot see a
    sequence to decide on, and its shapes repeat by construction. Asserted here so the guarantee
    cannot rot into a comment -- if someone wires trace_pays into the worker, the perf gate's
    traced leg silently becomes an eager one and this fails first.
    """
    import inspect
    from tt_bio import worker
    src = inspect.getsource(worker._WorkerState.load_model)
    assert "load_esmc(" in src, "the worker no longer loads ESM-C here; re-point this test"
    assert "trace_pays" not in src, (
        "the serving worker now asks trace_pays, but it loads before any job arrives -- it would "
        "be deciding on an empty workload and the perf gate's traced leg would go eager")


def test_the_predicate_can_say_no_which_is_what_makes_it_a_check():
    """The negative control: a `trace_pays` stubbed to always return True fails these tests.

    Written out so the check cannot decay into one that only ever confirms the happy answer.
    """
    real = esmc.trace_pays
    try:
        esmc.trace_pays = lambda *a, **k: True
        with pytest.raises(AssertionError):
            test_one_sequence_never_pays_for_a_trace()
    finally:
        esmc.trace_pays = real
    assert esmc.trace_pays(["A" * 60000]) is False
