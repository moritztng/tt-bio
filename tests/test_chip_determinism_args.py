"""The determinism harness must accept a passthrough flag that starts with a dash.

Two cross-chip determinism legs (protenix-v2, openfold3) died before opening a device on
`argument --predict-args: expected one argument`, while the boltz2 leg with four passthrough
words ran fine. argparse only exempts a dash-leading value when it contains a space, so the
old `--predict-args "<string>"` flag worked or failed depending on how many words the caller
happened to pass. The fix splits argv on a bare `--` before argparse sees it.
"""
import argparse
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "chip_determinism", ROOT / "perf" / "wh-parity" / "chip_determinism.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    return ap


@pytest.mark.parametrize("passthrough", [
    ["--single_sequence"],                                     # the shape that broke
    ["--single_sequence", "--sampling_steps", "200"],          # the shape that worked
    ["--recycling_steps", "3", "--single_sequence"],
    [],
])
def test_passthrough_survives_regardless_of_word_count(passthrough):
    mod = _load()
    argv = ["--model", "protenix-v2"] + (["--"] + passthrough if passthrough else [])
    args, extra = mod.parse_with_passthrough(_parser(), argv)
    assert args.model == "protenix-v2"
    assert extra == passthrough


def test_options_after_the_separator_are_not_parsed_as_ours():
    """A passthrough --model must reach the fold, not rebind the harness's own --model."""
    mod = _load()
    args, extra = mod.parse_with_passthrough(
        _parser(), ["--model", "boltz2", "--", "--model", "somethingelse"])
    assert args.model == "boltz2"
    assert extra == ["--model", "somethingelse"]


def test_the_old_flag_shape_is_gone():
    """Negative control: the removed flag is what failed, and it is not still accepted."""
    mod = _load()
    with pytest.raises(SystemExit):
        mod.parse_with_passthrough(
            _parser(), ["--model", "protenix-v2", "--predict-args", "--single_sequence"])
