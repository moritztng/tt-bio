"""When an open tape keeps the channel move off the reblock kernel.

BindCraft 2 on Wormhole ran its trimul channel moves on the stock permute in the checkpoint's
untaped forward and in the backward, because the gate refused every caller while a tape was open.
With `TAPED_MOVE` on, only a recording forward that is not the taped shim is refused. Host only.
"""

import pytest

from tt_bio import ops
from tt_bio import reblock_permute as R


@pytest.mark.parametrize("move, recording, taped_ok, refuses", [
    (False, True, False, True),     # lever off: the old blanket refusal
    (False, False, True, True),
    (True, True, False, True),      # a direct caller on a recording tape still keeps the stock op
    (True, True, True, False),      # the taped shim records the node itself
    (True, False, False, False),    # checkpoint forward under NOGRAD_IS_INFERENCE: nothing to drop
])
def test_refusal(monkeypatch, move, recording, taped_ok, refuses):
    monkeypatch.setattr(ops, "declines_under_tape", lambda name: True)
    monkeypatch.setattr(ops, "recording", lambda: recording)
    monkeypatch.setattr(R, "TAPED_MOVE", move)
    for name in ("reblock_permute", "reblock_permute_back"):
        assert R._tape_refuses(name, taped_ok) is refuses


def test_no_tape_never_refuses(monkeypatch):
    monkeypatch.setattr(ops, "declines_under_tape", lambda name: False)
    monkeypatch.setattr(R, "TAPED_MOVE", False)
    assert not R._tape_refuses("reblock_permute", False)
