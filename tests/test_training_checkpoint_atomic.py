"""A save cut off mid-write must not become the checkpoint a resume picks up.

``Checkpointer.latest`` takes the newest file by name. If a host reset lands inside
``save_adapter`` and the file was written in place, the newest name is a truncated file and the
resume dies on it (or, worse, on a later read). The save goes to a side file and is renamed in.
"""
import os

import numpy as np
import pytest

from tt_bio.train import checkpoint as C


class _Opt:
    def __init__(self, v):
        self.master = {"w": np.full((4, 4), v, np.float32)}

    def state_families(self):
        return {"exp_avg": {"w": np.zeros((4, 4), np.float32)}}

    def state_dict(self):
        return {"steps": 1}


def test_a_save_cut_off_before_the_rename_leaves_the_previous_checkpoint_latest(tmp_path, monkeypatch):
    ck = C.Checkpointer(tmp_path, every=1, keep=3)
    first = ck.save(1, _Opt(1.0))
    assert ck.latest() == first

    def reset(*a, **k):
        raise KeyboardInterrupt("host reset")
    monkeypatch.setattr(os, "replace", reset)
    with pytest.raises(KeyboardInterrupt):
        ck.save(2, _Opt(2.0))
    assert ck.latest() == first
    from safetensors.numpy import load_file
    assert float(load_file(str(ck.latest()))["master|w"][0, 0]) == 1.0


def test_the_gate_fails_when_the_save_writes_in_place(tmp_path, monkeypatch):
    """Negative control: an in-place writer leaves the cut-off file as latest."""
    ck = C.Checkpointer(tmp_path, every=1, keep=3)
    ck.save(1, _Opt(1.0))
    (tmp_path / "adapter-00000002.safetensors").write_bytes(b"\x10\x00")  # truncated in-place write
    assert ck.latest().name == "adapter-00000002.safetensors"
