"""A recycling cycle DRAM refuses restarts the trunk from cycle 0 one mode lighter.

Why it exists. The guard used to keep each cycle's input pair allocated through the whole cycle
so a refused cycle could re-run from it. At 1024 Mol-1 tokens that pair is 0.94 GB, and it was
what a Wormhole chip ran out of in cycle 0 (64 MiB refused, 4.1 MiB largest block, 12.76 of
12.88 GB allocated). The device-rows site also forgot to keep it and re-ran on a freed buffer
("Buffer is not allocated", mol1-whl2). Restarting from cycle 0, whose inputs are rebuilt from
z_init and s_init, needs no kept pair at all. Card-free: no device is opened.
"""
import inspect

import pytest

from tt_bio.size_limits import restart_lighter

REFUSAL = RuntimeError(
    "TT_FATAL: Out of Memory: Not enough space to allocate 67108864 B DRAM buffer across 12 banks, "
    "where each bank needs to store 5593088 B, but bank size is 1073741792 B (allocated: 1063710080 B, "
    "free: 10031712 B, largest free block: 4319104 B)")
LADDER = {"table": "cut", "cut": "park", "chip": "host"}


def runner(refuse):
    calls = []

    def run(mode):
        calls.append(mode)
        if mode in refuse:
            raise RuntimeError(str(REFUSAL))
        return f"carry from {mode}"
    return run, calls


def test_a_fitting_run_runs_once():
    run, calls = runner(set())
    assert restart_lighter(run, "table", LADDER, lambda m: pytest.fail("stepped down")) == "carry from table"
    assert calls == ["table"]


def test_each_refusal_restarts_one_mode_lighter():
    run, calls = runner({"table", "cut"})
    stepped = []
    assert restart_lighter(run, "table", LADDER, stepped.append) == "carry from park"
    assert calls == ["table", "cut", "park"] and stepped == ["table", "cut"]


def test_the_step_down_runs_outside_the_handler():
    """The refused run's traceback pins every tensor it made; freeing them needs the handler gone."""
    import sys
    run, _ = runner({"chip"})
    seen = []
    restart_lighter(run, "chip", LADDER, lambda m: seen.append(sys.exc_info()[0]))
    assert seen == [None]


def test_a_refusal_in_the_lightest_mode_propagates():
    run, calls = runner({"park"})
    with pytest.raises(RuntimeError, match="Out of Memory"):
        restart_lighter(run, "park", LADDER, lambda m: pytest.fail("stepped down"))
    assert calls == ["park"]


def test_other_errors_are_never_rerun():
    def run(mode):
        raise RuntimeError("TT_THROW: Buffer is not allocated")
    with pytest.raises(RuntimeError, match="not allocated"):
        restart_lighter(run, "table", LADDER, lambda m: pytest.fail("stepped down"))


def test_no_cycle_keeps_its_input_pair():
    """Trunk.__call__ goes through restart_lighter and nothing keeps a cycle's inputs for a re-run."""
    import tt_bio.protenix as P
    src = inspect.getsource(P.Trunk.__call__)
    assert "restart_lighter(" in src
    assert "retry" not in src and "inputs = list(carry)" not in src


def test_a_refused_deep_draw_resumes_at_its_own_cycle():
    """The carry is kept before every cycle that draws deeper than all before it, so a refusal there
    resumes at that cycle. Seed 1101's draws at 1024 Mol-1 tokens (6835 rows), whose deepest at cycle 17
    redid 17 cycles from cycle 0 and made it the slowest fold of the WH gang."""
    from tt_bio.protenix import deeper_than_before
    assert deeper_than_before([3, 10, 27, 27, 25, 74, 58, 15, 52, 65, 31, 38, 44, 9, 58, 37, 49, 93, 86, 65]) == {1, 2, 5, 17}
    assert deeper_than_before([5, 9, 7], first=4) == {5}
    assert deeper_than_before([9]) == set() and deeper_than_before([]) == set()


def test_only_a_large_sampled_pair_keeps_marks():
    """The kept carry costs one DMA of the pair a mark: only where the pair is a 32nd of the chip."""
    import tt_bio.protenix as P
    src = inspect.getsource(P.Trunk.__call__)
    assert "32 * N * N * self.C_Z * 2 >= dram_bytes()" in src
    assert "cyc + 1 in marks" in src
