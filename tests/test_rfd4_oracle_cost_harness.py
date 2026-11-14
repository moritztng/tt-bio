"""The oracle-cost harness builds the commands it claims to (perf/rfd4/oracle_cost.py).

The protocol constants are the point of the harness: upstream's oracle runs 5 recycles over a
200-step rollout, the recorded size-ladder cells run 6 steps, and a number taken at one and
compared against the other is the error this row exists to avoid.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("oracle_cost", ROOT / "perf" / "rfd4" / "oracle_cost.py")
oc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oc)


def test_the_oracle_protocol_is_upstreams():
    assert oc.ORACLE == {"recycling_steps": 5, "sampling_steps": 200, "diffusion_samples": 1}
    assert oc.LADDER["sampling_steps"] == 6


@pytest.mark.parametrize("rung", [128, 256, 512])
def test_af2ig_rungs_exist_and_carry_their_token_count(rung):
    from tt_bio.af2ig import token_count
    from tt_bio.af2ig_input import read_af2ig_input

    path = oc.fixture("af2ig", rung)          # also repoints the YAML at this checkout
    assert path.exists()
    spec_ = read_af2ig_input(path)
    assert token_count(spec_) == rung
    assert len(spec_.binder_sequence) == 80


def test_cofold_command_carries_the_protocol():
    cmd = oc.fold_command("openfold3", 256, "oracle", Path("/tmp/out"))
    assert "--single_sequence" in cmd
    for flag, value in oc.ORACLE.items():
        assert cmd[cmd.index(f"--{flag}") + 1] == str(value)

    ladder = oc.fold_command("openfold3", 256, "ladder", Path("/tmp/out"))
    assert ladder[ladder.index("--sampling_steps") + 1] == "6"
    assert "--recycling_steps" not in ladder     # the ladder leaves the model's shipped default


def test_af2ig_command_sets_no_sampling_flags():
    """No MSA, no diffusion: the whole job is the trunk, so neither protocol applies to it."""
    cmd = oc.fold_command("af2ig", 256, "oracle", Path("/tmp/out"))
    assert "--sampling_steps" not in cmd and "--single_sequence" not in cmd
    assert cmd[cmd.index("--model") + 1] == "af2ig"
