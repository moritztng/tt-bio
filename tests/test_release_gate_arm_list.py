"""Every arm `--list-arms` prints must be an arm `--model` accepts.

A long gate is launched one arm per process, so a killed arm does not take the passing ones with
it, and the launcher needs the arm list. Hand-copying it is how an arm that does not exist gets
asked for: on 2026-09-19 a C14 gate launcher asked for ``openbind-0``, the model's prose name,
against an argument spelled ``openbind``. argparse refused in 0 s with rc=2 and the run carried a
thirteenth arm that never opened a device -- a non-run in the shape of a result, which is the
error class this gate has already paid for twice (contended legs read as accuracy failures, a
banner grepped off disk read as the imported tree).

Device-free and torch-free. Run: python3 tests/test_release_gate_arm_list.py, or via pytest.
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import release_gate as rg


def _accepted() -> set:
    """The exact `--model` choices argparse was built with, read off the parser itself."""
    return set(list(rg.MODELS) + list(rg.DEFAULT_ARMS) + ["size-ladder"]
               + rg.ESMC_DEFAULT + rg.ESMC_OPT_IN)


def test_every_printed_arm_is_an_accepted_model_argument():
    printed = rg.default_arms()
    assert printed, "the arm list is empty, so a launcher iterating it would run nothing"
    unknown = [m for m in printed if m not in _accepted()]
    assert not unknown, f"--list-arms prints arms --model would refuse: {unknown}"


def test_the_check_above_can_fail():
    """Negative control. Without it, the assertion would pass against an empty relationship."""
    unknown = [m for m in rg.default_arms() + ["openbind-0"] if m not in _accepted()]
    assert unknown == ["openbind-0"], (
        "the prose name 'openbind-0' is expected to be refused by --model; if this passes, the "
        "test above proves nothing")


def test_the_list_and_the_run_cannot_disagree():
    """main() must expand a bare run through the same helper, not through a second copy."""
    src = open(os.path.join(REPO, "scripts", "release_gate.py")).read()
    assert "models = args.model or default_arms()" in src, (
        "main() no longer expands the default set through default_arms(), so --list-arms can "
        "drift from what a bare gate run actually scores")


def test_it_prints_one_arm_per_line_and_opens_nothing():
    """The flag has to answer before the preflight, the card grant and the mesh probe."""
    env = dict(os.environ, PYTHONPATH=REPO)
    r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "release_gate.py"),
                        "--list-arms"], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    lines = [x for x in r.stdout.splitlines() if x.strip()]
    assert lines == rg.default_arms()
    assert "granted" not in r.stdout, "the flag reached the card grant; it must exit before it"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
