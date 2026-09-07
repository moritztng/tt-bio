"""The refusal, exercised through the real CLI commands rather than through the module.

The unit tests in tests/test_size_limits.py prove the table and the arithmetic. This file proves the
WIRING, which is the part that silently rots: a ceiling table nothing calls refuses nothing, and
that failure looks exactly like a passing test suite. Each command is invoked the way a user invokes
it, and the assertion is that the refusal lands BEFORE a device is opened.

No device is opened here, on any path -- which is also what these tests check.
"""

import pytest

click = pytest.importorskip("click")
from click.testing import CliRunner

from tt_bio import main, size_limits as sl


@pytest.fixture
def wormhole(monkeypatch):
    """Pretend this host is the Wormhole Galaxy the ceilings were measured on.

    Patched rather than skipped-unless-on-Wormhole so the wiring is covered on every box, including
    the Blackhole hosts most of this fleet runs on. `arch_name` is lru_cached and reads the real
    card, so the patch goes on the seam size_limits actually calls.
    """
    monkeypatch.setattr(sl, "current_arch", lambda: "wormhole_b0")


@pytest.fixture
def no_device(monkeypatch):
    """Make any device open an immediate, loud failure.

    This is the real assertion of this file. Checking only for a non-zero exit and a message would
    pass just as well if the guard ran AFTER the device was taken -- which is the outcome the guard
    exists to prevent, since an L1 throw can leave the chip wedged for the next job. So the test
    makes opening a device a distinguishable error and requires that it never happens.
    """
    opened = []

    def boom(*a, **k):
        opened.append(1)
        raise AssertionError("a device was opened before the size guard refused")

    for mod, name in (("tt_bio.tenstorrent", "get_device"), ("tt_bio.tenstorrent", "open_device")):
        try:
            import importlib
            m = importlib.import_module(mod)
            if hasattr(m, name):
                monkeypatch.setattr(m, name, boom)
        except Exception:
            pass
    return opened


class _PastTheGuard(Exception):
    """Raised immediately after the guard, so a test can prove execution got there and stop.

    Needed because the commands under test do real work: invoking `predict` with an input the guard
    ALLOWS runs the actual pipeline -- weights, MSA, a device. Letting that happen would make these
    tests a fold rather than a wiring check, so every path here terminates at the guard, whether it
    refused or let the input through.
    """


def _yaml_of(n: int) -> str:
    return f"sequences:\n  - protein:\n      id: A\n      sequence: {'A' * n}\n"


def test_predict_refuses_an_oversized_target(tmp_path, wormhole, no_device):
    f = tmp_path / "big.yaml"
    f.write_text(_yaml_of(1024))
    res = CliRunner().invoke(main.cli, ["predict", str(f), "--model", "opendde",
                                        "--out_dir", str(tmp_path / "out")])
    assert res.exit_code != 0
    msg = str(res.output) + str(res.exception)
    assert "1024" in msg and "544" in msg and "opendde" in msg
    assert not no_device


def test_predict_admits_a_target_at_the_cap(tmp_path, wormhole, monkeypatch):
    """A size MEASURED to work must not be stopped by the guard.

    The half that matters as much as the refusal: a ceiling set one rung too low is invisible unless
    something asserts the boundary from below.
    """
    def stop(*a, **k):
        raise _PastTheGuard
    monkeypatch.setattr(main, "_resolve_recycling_steps", stop)
    f = tmp_path / "ok.yaml"
    f.write_text(_yaml_of(544))
    res = CliRunner().invoke(main.cli, ["predict", str(f), "--model", "opendde",
                                        "--out_dir", str(tmp_path / "out")])
    assert isinstance(res.exception, _PastTheGuard), (
        f"expected to reach past the guard at the cap, got {res.exception!r}")


def test_predict_is_silent_on_blackhole(tmp_path, monkeypatch):
    """The same oversized input on the architecture where nobody measured a ceiling.

    OpenDDE caps at 544 on Wormhole and folded every rung to 1024 aa on a Blackhole p150a, so a
    refusal here would be this guard inventing a limit -- the exact failure the arch key prevents.
    This is also the Blackhole no-regression check that needs no Blackhole box: with no BH rows,
    check() returns before it reads a size, so the BH path cannot change behaviour.
    """
    def stop(*a, **k):
        raise _PastTheGuard
    monkeypatch.setattr(sl, "current_arch", lambda: "blackhole")
    monkeypatch.setattr(main, "_resolve_recycling_steps", stop)
    f = tmp_path / "big.yaml"
    f.write_text(_yaml_of(1024))
    res = CliRunner().invoke(main.cli, ["predict", str(f), "--model", "opendde",
                                        "--out_dir", str(tmp_path / "out")])
    assert isinstance(res.exception, _PastTheGuard), (
        f"a Blackhole run was stopped by a Wormhole ceiling: {res.exception!r}")


def test_design_refuses_an_oversized_rfd3_contig(tmp_path, wormhole, no_device):
    f = tmp_path / "spec.json"
    f.write_text('{"binder-1": {"input": "t.pdb", "contig": "A1-2,4000"}}')
    res = CliRunner().invoke(main.cli, ["design", str(f), "--model", "rfd3",
                                        "--out_dir", str(tmp_path / "out")])
    assert res.exit_code != 0
    msg = str(res.output) + str(res.exception)
    assert "4002" in msg and "490" in msg
    assert not no_device


def test_design_refuses_an_oversized_pxdesign_target(tmp_path, wormhole, no_device):
    f = tmp_path / "target.yaml"
    f.write_text("target:\n  file: t.cif\n  chains:\n    A:\n      crop: [\"1-900\"]\n"
                 "binder_length: 80\n")
    res = CliRunner().invoke(main.cli, ["design", str(f), "--model", "pxdesign",
                                        "--out_dir", str(tmp_path / "out")])
    assert res.exit_code != 0
    msg = str(res.output) + str(res.exception)
    assert "900" in msg and "768" in msg
    assert not no_device


def test_embed_refuses_an_oversized_bare_sequence(tmp_path, wormhole, no_device):
    res = CliRunner().invoke(main.cli, ["embed", "A" * 2000, "--model", "esmc-6b",
                                        "--out_dir", str(tmp_path / "out")])
    assert res.exit_code != 0
    msg = str(res.output) + str(res.exception)
    assert "2000" in msg and "1968" in msg
    assert not no_device


def test_every_wired_command_actually_calls_the_guard(monkeypatch, tmp_path):
    """The coverage assertion: every command that takes user input calls the guard.

    A ceiling table nothing calls refuses nothing, and that failure looks exactly like a green
    suite. The commands are named rather than discovered on purpose -- the point is to fail when a
    NEW input-taking command is added and nobody wires it, and a discovered list would just grow to
    match, asserting nothing.

    check_input is replaced with a raise, which does double duty: it records that the call happened
    AND stops each command right there, so no invocation runs off into weights, an MSA or a device.
    """
    seen = []

    def stop(data, model, **k):
        seen.append(model)
        raise _PastTheGuard

    monkeypatch.setattr(main.size_limits, "check_input", stop)
    f = tmp_path / "in.yaml"
    f.write_text(_yaml_of(10))
    for argv in (["predict", str(f), "--model", "opendde"],
                 ["design", str(f), "--model", "rfd3"],
                 ["embed", str(f), "--model", "esmc-6b"],
                 ["saprot", str(f), "--model", "saprot-35m"],
                 ["affinity", str(f), "--model", "nesso1"]):
        seen.clear()
        res = CliRunner().invoke(main.cli, argv + ["--out_dir", str(tmp_path / "out")])
        assert seen, (f"`tt-bio {argv[0]}` never called size_limits.check_input, so no ceiling "
                      f"can refuse anything on that command")
        assert isinstance(res.exception, _PastTheGuard)
