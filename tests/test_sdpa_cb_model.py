"""The guard for the SDPA circular-buffer L1 model in tt_bio/sdpa_generic.py.

`cb_bytes` prices the CB table `build` allocates, and `_tri_att_sdpa_at` / `triatt_sdpa.sdpa`
lean on it to skip a (q_chunk, k_chunk) pair that cannot fit rather than build the program and
catch the throw. A model that drifts from the factory is worse than no model: it retires a config
the device would have run, and the fold quietly takes a slower one.

So this asserts the model against every refusal actually measured, and it asserts them the only
way that means anything -- the model must reproduce the byte figure tt-metal printed, not merely
agree that the config was too big. `perf/bgsdpa/cb_model.MEASURED` carries the evidence and
`perf/bgsdpa/probe_cb.py` re-measures it on a device in seconds.

Opens no device. `plan` reads shapes and dtypes only.
"""

import pathlib
import sys

import pytest

ttnn = pytest.importorskip("ttnn")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "perf" / "bgsdpa"))
import cb_model as M                                                        # noqa: E402


@pytest.mark.parametrize("seq,heads,head_dim,q_chunk,k_chunk,reported,src", M.MEASURED)
def test_model_reproduces_every_measured_refusal(seq, heads, head_dim, q_chunk, k_chunk,
                                                 reported, src):
    p = M.plan_for(seq, heads, head_dim, q_chunk, k_chunk)
    assert M.reported_bytes(p) == reported, src
    assert not M.fits(p), f"{src}: the device refused this config, the model must too"


def test_the_measured_set_covers_both_probe_and_a_real_fold():
    """A model calibrated only on a 32-row probe misses `q_buffer_factor`, which doubles the q CB
    once a core owns more than one q chunk -- the difference between 1735168 B and 1782272 B at
    q_chunk 736. Both rows are in the set, so a regression on that term fails a test."""
    srcs = {row[-1] for row in M.MEASURED}
    assert {"probe_cb", "log_r2100"} <= srcs
    probe = M.plan_for(32, 4, 32, 736, 256)
    fold = M.plan_for(2208, 4, 32, 736, 256)
    assert probe["q_buffer_factor"] == 1 and fold["q_buffer_factor"] == 2
    assert M.reported_bytes(fold) - M.reported_bytes(probe) == 23 * 2048


def test_a_config_the_device_ran_is_not_refused_by_the_model():
    """The negative control. `probe_cb.py` ran these three on qb1 card 2 without a throw, so a
    model that calls them over L1 is wrong in the direction that costs performance."""
    for q_chunk, k_chunk in [(608, 256), (640, 256), (224, 736)]:
        assert M.fits(M.plan_for(32, 4, 32, q_chunk, k_chunk)), (q_chunk, k_chunk)


def test_the_fused_persistent_mask_is_priced_separately():
    """The fused K1/K2 kernel fronts `k_num_chunks * Sq_chunk_t * Sk_chunk_t` mask tiles where the
    stock op double-buffers one chunk, so it is a different budget and must not be read off the
    stock one. At 2208 padded tokens q96/k736 fits both and q736/k96 fits only the stock op."""
    for q_chunk, k_chunk, stock_ok, fused_ok in [(96, 736, True, True), (736, 96, True, False)]:
        p = M.plan_for(2208, 4, 32, q_chunk, k_chunk)
        pers = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
        assert M.fits(p) is stock_ok, (q_chunk, k_chunk)
        assert M.fits(p, mask_cb_tiles=pers) is fused_ok, (q_chunk, k_chunk)


def test_l1_per_core_is_taken_from_a_refusal_that_names_it():
    """The constant is a fallback, not a claim: a part whose per-core L1 is not 1.5 MiB corrects
    it from the device's own message the first time one is caught."""
    from tt_bio import sdpa_generic as SG
    before = SG.L1_PER_CORE
    try:
        SG.note_l1_refusal("grow to 9 B which is beyond max L1 size of 1048576 B")
        assert SG.L1_PER_CORE == 1048576
        SG.note_l1_refusal("some unrelated failure")
        assert SG.L1_PER_CORE == 1048576
    finally:
        SG.L1_PER_CORE = before
