"""The one-sentence OOM summary has to reach the USER, not just exist in size_limits.

`describe_device_oom` landed at 2026-09-09T23:09:31Z. The esmfold2 1536 rung that motivated it
ran at 23:04:53Z, four and a half minutes earlier, so what the run actually printed was the raw
`TT_FATAL @ .../bank_manager.cpp:439: false` plus twenty C++ backtrace frames and the renderer
has never been exercised by a real refusal. Every test on it so far fed it a hand-written string.

This drives the real `_stream_run` failure path with the error string the worker ACTUALLY stored
for that fold (`perf/bh1536/runs/esmfold2_1536/.../results.json`, 2000 chars, elided in the
middle by `worker._err_text`) and asserts on what lands on stdout. The elision is the specific
risk: the renderer's regex needs the closing `(allocated: ... largest free block: N B)`
parenthetical, and `_err_text` cuts the message at a fixed budget and spends the rest of it on
backtrace frames. If a future budget change moves that cut one field earlier, the summary
silently disappears and the user is back to reading tt-metal source paths.

Host-only: no device, no controller, no worker. The client is a stub that reports one failed job.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tt_bio.main import _stream_run  # noqa: E402
from tt_bio.size_limits import describe_device_oom  # noqa: E402

#: The recorded fold whose error text this test replays. Kept as a file lookup rather than a
#: pasted literal so the test is scoring what the engine really wrote.
RECORDED = "perf/bh1536/runs/esmfold2_1536/**/results.json"


def _recorded_error() -> str:
    matches = glob.glob(str(REPO / RECORDED), recursive=True)
    if not matches:
        pytest.skip(f"no recorded esmfold2 1536 refusal at {RECORDED}")
    doc = json.loads(Path(matches[0]).read_text())
    rows = doc if isinstance(doc, list) else [doc]
    errors = [r["error"] for r in rows if isinstance(r, dict) and r.get("error")]
    if not errors:
        pytest.skip("the recorded results.json carries no error")
    return errors[0]


class _Client:
    """Reports one failed job carrying the recorded error, then a terminal run."""

    def __init__(self, error: str):
        self.error = error
        self.sent = False

    def events(self, run_id, after):
        if self.sent:
            return {"events": [], "status": "failed", "failed": 1}
        self.sent = True
        return {"events": [{"seq": 1, "event": "done", "name": "cdk2x2_1536",
                            "status": "failed", "error": self.error,
                            "row": {"id": "cdk2x2_1536", "status": "failed",
                                    "error": self.error}}],
                "status": "failed", "failed": 1}


def _run(capsys, error: str, debug: bool) -> str:
    failed = _stream_run(_Client(error), "r1", total=1, n_workers=1, debug=debug, log=False,
                         results_path=None, struct_dir=None, model="esmfold2",
                         local_procs=None)
    assert failed == 1, f"the failed job was not counted: {failed}"
    return capsys.readouterr().out


def test_the_stored_refusal_still_carries_the_field_the_summary_needs():
    """The elision guard. If `_err_text`'s budget ever cuts one field earlier, this fails here
    with a clear reason instead of silently degrading the message the user reads."""
    err = _recorded_error()
    assert len(err) == 2000, f"budget changed ({len(err)}); re-check the elision"
    assert "largest free block:" in err, "the elision removed the field the summary is read from"
    assert describe_device_oom(err) is not None, (
        "the stored error no longer renders; the user is back to the C++ assertion")


def test_a_real_refusal_prints_one_sentence_and_not_a_cpp_assertion(capsys):
    out = _run(capsys, _recorded_error(), debug=False)

    # What the user must now read: the class of wall, named.
    assert "fragmentation, not a full chip" in out, out[-2000:]
    assert "4.50 GiB was requested" in out
    assert "largest single free block is 480.7 MiB" in out
    assert "Re-run with --debug for the allocator trace." in out

    # And what must be gone. These four lines are verbatim what the 23:04Z run printed.
    for noise in ("bank_manager.cpp:439", "TT_FATAL", "backtrace:",
                  "tt::tt_metal::BankManager::allocate_buffer"):
        assert noise not in out, f"{noise!r} is still in the user-visible output:\n{out[-2000:]}"


def test_debug_still_gives_the_whole_trace(capsys):
    """The negative control on the summary: --debug must keep printing the raw text, or the
    person diagnosing the allocator has lost the only record of it."""
    out = _run(capsys, _recorded_error(), debug=True)
    assert "bank_manager.cpp:439" in out, "the trace is gone from --debug too"
    assert "tt::tt_metal::BankManager::allocate_buffer" in out
    assert "Re-run with --debug" not in out, "--debug should not tell you to re-run with --debug"


def test_a_failure_that_is_not_an_allocator_refusal_prints_unchanged(capsys):
    """The other negative control: the summary must not swallow every other error. A missing-MSA
    failure carries actionable guidance on its later lines and all of it has to survive."""
    plain = ("no MSA found for chain A\n"
             "Supply one with --msa_server or --msa_db_path, or fold single-sequence with "
             "--single_sequence.")
    out = _run(capsys, plain, debug=False)
    assert "no MSA found for chain A" in out
    assert "--single_sequence" in out, "the guidance on the second line was dropped"
    assert "fragmentation" not in out
