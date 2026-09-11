"""The clash message says which allocation is resident, if you subtract.

Both messages here are verbatim from the OpenDDE 576 aa fold that set the published 544 ceiling
(state/ceiling-opendde.md, logs/576.log on the Galaxy, 2026-09-07). The numbers asserted below are
the ones that identify the buffer, so a regression in the parsing shows up as a wrong diagnosis
rather than a missing one.
"""
import re

import pytest

pytest.importorskip("ttnn", reason="tenstorrent.py imports ttnn at module scope")

from tt_bio.tenstorrent import (  # noqa: E402
    absorb_l1_refusal, describe_l1_clash, note_l1_clash)

CLASH = (
    "TT_THROW @ /project/tt_metal/impl/program/program.cpp:1052: tt::exception\n"
    "info:\nStatically allocated circular buffers in program 1060 clash with L1 buffers on core "
    "range [(x=0,y=0) - (x=7,y=8)]. L1 buffer allocated at 352256 and static circular buffer "
    "region ends at 382240"
)
OVERFLOW = (
    "Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=8)] grow to "
    "3822880 B which is beyond max L1 size of 1499136 B"
)

# moritztng/tt-bio#14, verbatim. The OVERFLOW form on a Blackhole 11x10 grid.
ISSUE14_THROW = (
    "TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)] "
    "grow to 1765888 B which is beyond max L1 size of 1572864 B (assert.hpp:104)"
)

# Measured on the Galaxy Wormhole chip this fold ran on, card 2, 2026-09-08.
WH_L1_TOP = 1499136
WH_L1_UNRESERVED = 1466080


def test_clash_names_the_resident_buffer():
    c = describe_l1_clash(CLASH, WH_L1_TOP, WH_L1_UNRESERVED)
    assert c["grid"] == (8, 9) and c["cores"] == 72
    # 1499136 - 352256: what the live tensors held on every core when the program was built.
    assert c["resident_per_core"] == 1146880
    # x 72 banks = the interleaved buffer's whole size, which is what identifies it. 82575360 B is
    # the ending triangle attention's transposed pair block, 480 x 672 x 128 bf16.
    assert c["resident_total"] == 82575360
    assert 480 * 672 * 128 * 2 == c["resident_total"]
    # The consumer's static CBs, and the room they had. The gap is the whole bug.
    assert c["cb_base"] == WH_L1_TOP - WH_L1_UNRESERVED == 33056
    assert c["cb_need"] == 349184
    assert c["cb_free"] == 319200
    assert c["shortfall"] == 29984 == c["cb_need"] - c["cb_free"]


def test_shortfall_needs_no_device_constants():
    c = describe_l1_clash(CLASH)
    assert c["shortfall"] == 29984
    assert c["cores"] == 72
    assert "cb_need" not in c or c.get("l1_top")  # omitted, not guessed


def test_overflow_form_implicates_no_tensor():
    c = describe_l1_clash(OVERFLOW)
    assert c["cores"] == 72
    assert c["resident_per_core"] == 0
    assert c["cb_need"] == 3822880 and c["l1_top"] == 1499136
    assert c["shortfall"] == 3822880 - 1499136


def test_note_learns_l1_top_from_the_overflow_form():
    import tt_bio.tenstorrent as T

    T._L1_TOP_SEEN.clear()
    T.L1_CLASH_CENSUS.clear()
    assert describe_l1_clash(CLASH).get("resident_per_core") is None
    note_l1_clash("probe", OVERFLOW)
    c = note_l1_clash("tri_att_end/layer_norm", CLASH)
    assert c["resident_per_core"] == 1146880
    assert [w for w, _ in T.L1_CLASH_CENSUS] == ["probe", "tri_att_end/layer_norm"]
    T._L1_TOP_SEEN.clear()
    T.L1_CLASH_CENSUS.clear()


def test_unrelated_message_is_not_a_clash():
    assert describe_l1_clash("Out of Memory: Not enough space to allocate 2015363072 B") is None
    assert note_l1_clash("x", "boom") is None


def test_captured_worker_throw_renders_its_census():
    """The launcher turns a worker's captured L1 throw into the census, not raw text.

    issue #14's throw is the OVERFLOW form, so `resident_per_core` must read 0: the
    program config alone does not fit an empty core and no tensor is implicated.
    """
    from tt_bio.main import _l1_census_line

    line = _l1_census_line(ISSUE14_THROW)
    assert "cb_need=1765888" in line
    assert "l1_top=1572864" in line
    assert "shortfall=193024" in line
    assert "resident_per_core=0" in line
    assert "cores=110" in line
    assert _l1_census_line("worker exited with code 14") == ""


def test_absorbed_refusal_labels_itself_on_the_stream_it_went_to(tmp_path, capfd):
    """A by-design refusal must not read as a crash.

    tt-metal writes its TT_THROW to fd 2 before the ladder ever sees the exception, and the
    ladder then retries a narrower plan and the fold completes. #14 is that throw, filed as a
    crash. The absorb has to mark the same fd, or a worker's captured stderr offers the throw
    to the launcher as a cause of death.
    """
    import tt_bio.tenstorrent as T

    T.L1_CLASH_CENSUS.clear()
    absorb_l1_refusal("tri_att_sdpa/last_q_chunk", RuntimeError(ISSUE14_THROW))
    err = capfd.readouterr().err
    assert "retrying a narrower one" in err and "not a crash" in err
    assert "cb_need=1765888" in err and "resident_per_core=0" in err
    # and it is attributable now, which the ladder's private sets never were
    assert [w for w, _ in T.L1_CLASH_CENSUS] == ["tri_att_sdpa/last_q_chunk"]
    T.L1_CLASH_CENSUS.clear()


def test_absorb_reraises_anything_that_is_not_an_l1_refusal():
    exc = RuntimeError("Out of Memory: Not enough space to allocate 2015363072 B")
    with pytest.raises(RuntimeError, match="Out of Memory"):
        absorb_l1_refusal("tri_att_sdpa/q_chunk", exc)


def test_launcher_does_not_render_a_census_twice():
    """A captured tail that already carries a census is printed as it stands."""
    from tt_bio.main import _l1_census_line

    assert _l1_census_line(ISSUE14_THROW) != ""
    assert _l1_census_line(ISSUE14_THROW + "\n[tt-bio] L1 census: grid=(11, 10)") == ""
