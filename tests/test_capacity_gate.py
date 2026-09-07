"""Guards for the capacity gate, and the trigger that makes a ceiling change re-run it.

THE HOLE THIS CLOSES. The platform's coverage matrix sizes each cell at the model's own advertised
ceiling. While the ceilings were 576/576/627 it folded 576 and correctly passed. Raising every
ceiling to 1024 on 2026-09-03 moved the gate's own target to 1024 and nothing re-ran it, so three
models broke in real traffic at 40-50%. The published ceilings live in the serving platform, which
tt-bio does not import, but the engine's own copy is tt_bio.size_limits.CEILINGS -- so that table
is fingerprinted here, and moving any row fails this test until the gate has been re-run at the
new size and docs/capacity_gate_baseline.json re-recorded.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import capacity_fixture                                                          # noqa: E402
import capacity_gate as cg                                                       # noqa: E402

from tt_bio import size_limits as sl                                             # noqa: E402

BASELINE = cg.BASELINE
#: One definition of the fingerprint, in the gate that records it. A second copy here would let
#: the recorder and the check drift apart, which is the one failure this pair cannot survive.
ceilings_fingerprint = cg.ceilings_fingerprint


def test_the_bar_is_bucket_aligned():
    """1500 pads to 1504 internally, so a gate that tests 1500 and reports 1500 hides a 4-token
    pad and an unmasked tail is a ~72x error. The bar has to be a size the hardware sees."""
    assert cg.TOKEN_BAR % cg.TOKEN_BUCKET == 0
    for rung in cg.BISECT_RUNGS:
        assert rung % cg.TOKEN_BUCKET == 0, f"bisect rung {rung} is not bucket-aligned"
        assert rung < cg.TOKEN_BAR


def test_every_shipped_model_is_covered_or_exempted_in_writing():
    """The roster is derived from main.py's own tuples, so a new port appears here by itself. If
    the gate cannot drive it, that must be a written reason and not a silent absence."""
    gaps = cg.coverage_gaps()
    assert not gaps, (
        f"shipped models the capacity gate neither runs nor exempts: {gaps}. Add a cell, or an "
        f"entry to capacity_gate.EXEMPT saying why the bar cannot be expressed for it.")


def test_no_exemption_for_a_model_that_is_not_shipped():
    """So a renamed or retired model does not leave a stale exemption standing in for coverage."""
    stale = sorted(set(cg.EXEMPT) - set(cg.roster()))
    assert not stale, f"exemptions for models no CLI --model choice reaches: {stale}"


def test_every_exemption_gives_a_real_reason():
    for model, reason in cg.EXEMPT.items():
        assert len(reason) > 60, f"{model}'s exemption is not a reason: {reason!r}"
        assert "TODO" not in reason


def test_a_screen_pass_is_never_reported_as_a_pass():
    """The whole safety argument for the layer-subset shortcut. Truncating block stacks can only
    REDUCE what runs, so a screen can miss a cumulative-residency failure but cannot invent one.
    That is fine only as long as a clean screen is INCONCLUSIVE and never PASS."""
    src = (ROOT / "scripts" / "capacity_gate.py").read_text()
    body = src[src.index("def _screen("):src.index("def _residency(")]
    assert '"PASS"' not in body, (
        "_screen must never emit PASS: a one-block run cannot see the cumulative residency and "
        "fragmentation class (RF3 dies at 630 tokens on a 103 MB request with DRAM 99% full).")
    assert '"INCONCLUSIVE"' in body


def test_the_residency_tier_watches_for_a_stall_not_only_an_exception():
    """OpenFold3's diffusion-side L1 refusal is retried rather than raised: it sat 2289 s at
    diffusion step 0. A gate that waits for an exception scores that green."""
    assert cg.STALL_S > 0
    body = (ROOT / "scripts" / "capacity_gate.py").read_text()
    assert '"STALL"' in body and "stall_s" in body


def test_host_oom_is_a_distinct_verdict_from_the_device_wall():
    """A 1504-token deep-MSA fold can OOM the HOST. That is a different failure from running out
    of device DRAM and must not be recorded as a capacity ceiling."""
    assert "HOST_OOM" in cg.VERDICTS
    assert "NO_WEIGHTS" in cg.VERDICTS, "an absent checkpoint is not a failed bar either"


def test_the_fixture_reaches_the_bar_at_real_depth(tmp_path):
    """Depth is a second axis, not a detail: for the OF3 family the failing tensor scales with
    tokens x rows. And the row count in the FILE is not what the model sees -- `_parse_a3m_to_msa`
    deduplicates by sequence, so a tiled alignment can arrive as a handful of rows."""
    f = capacity_fixture.build(cg.TOKEN_BAR, tmp_path)
    assert f["residues"] == cg.TOKEN_BAR
    assert len(f["yaml"].read_text().split("sequence: ")[1].strip()) == cg.TOKEN_BAR
    assert f["effective_depth"] > 1000, (
        f"only {f['effective_depth']} distinct rows reach the model; a shallow alignment cannot "
        f"exercise the tokens x rows tensor that walls the OF3 family")
    assert f["effective_depth"] >= 0.99 * f["file_depth"], "tandem repetition collapsed rows"


def test_the_fixture_reaches_the_bar_in_every_denomination(tmp_path):
    """size_limits sizes `predict` on summed residues and `embed`/`saprot` on the LONGEST single
    sequence, and it reads two different file formats to do it. Both must see 1504, or a cell
    passes having folded something smaller than the bar it reports.
    """
    f = capacity_fixture.build(cg.TOKEN_BAR, tmp_path)
    assert sl.scan_residues(f["yaml"].read_text()) == cg.TOKEN_BAR
    assert sl.scan_longest_sequence(f["fasta"].read_text()) == cg.TOKEN_BAR


def test_embed_and_saprot_are_handed_a_fasta_not_the_predict_yaml():
    """`embed` takes a FASTA, a bare sequence, or a YAML that is a flat {id: sequence} mapping.
    predict's `sequences:` document is none of those and parses to nothing, so the cell would
    complete having embedded zero residues and be scored PASS."""
    src = (ROOT / "scripts" / "capacity_gate.py").read_text()
    body = src[src.index("def build_argv("):src.index("def execute(")]
    head = body[:body.index('if cell.verb == "affinity"')]
    assert 'fixture["fasta"]' in head and 'fixture["yaml"]' not in head


def test_the_gate_says_out_loud_that_it_is_not_a_parity_gate():
    """pc card 0 miscomputes matmuls and runs custom 130-core firmware, so this gate can answer
    "allocates and completes" and can never answer "the output is right". A completed fold with a
    torn structure is a worse outcome than an OOM."""
    doc = cg.__doc__ or ""
    assert "NOT A PARITY GATE" in doc.upper()
    assert "full_parity_gate" in doc


@pytest.mark.skipif(not BASELINE.exists(), reason="no capacity baseline recorded yet")
def test_a_moved_ceiling_re_runs_the_capacity_gate():
    """THE TRIGGER. A ceiling change is exactly what let the production failures ship: the gate's
    target follows the ceiling, and raising the ceiling without re-running the gate leaves the new
    size untested. So the recorded run pins the ceiling table it was measured against."""
    recorded = json.loads(BASELINE.read_text()).get("ceilings_fingerprint")
    assert recorded, "the baseline records no ceiling fingerprint"
    assert recorded == ceilings_fingerprint(), (
        "tt_bio/size_limits.CEILINGS has changed since the capacity gate last ran, so the sizes "
        "tt-bio now advertises have not been capacity-tested. Re-run\n"
        "  TT_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python3 scripts/capacity_gate.py\n"
        "and re-record docs/capacity_gate_baseline.json. This is the check that was missing when "
        "the ceilings went to 1024 on 2026-09-03 and three models broke in traffic.")


@pytest.mark.skipif(not BASELINE.exists(), reason="no capacity baseline recorded yet")
def test_a_blackhole_ceiling_row_must_come_from_a_capacity_run():
    """Blackhole was never walked before this gate existed, and a fabricated row is the failure
    the arch key exists to prevent."""
    for model, per_arch in sl.CEILINGS.items():
        for arch, c in per_arch.items():
            if "blackhole" not in arch or not c.measured:
                continue
            assert "capacity_gate" in c.evidence, (
                f"{model}/{arch} publishes a measured Blackhole ceiling whose evidence does not "
                f"name the run that measured it")


# ---------------------------------------------------------------------------------------------
# The hook. These exist because the hook was silently inert for a whole campaign: `capacity_hook`
# was not on the spawned child's PYTHONPATH, the generated sitecustomize swallowed the
# ModuleNotFoundError, and every "screen" ran the full model while reporting itself a screen.
# ---------------------------------------------------------------------------------------------


def _fresh_hook(mode, tmp_path, monkeypatch):
    """A hook instance armed in-process, with its module-level state reset."""
    import importlib
    import capacity_hook
    h = importlib.reload(capacity_hook)
    monkeypatch.setenv("TT_BIO_CAPACITY_HOOK", mode)
    monkeypatch.setenv("TT_BIO_CAPACITY_HOOK_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("TT_BIO_CAPACITY_HOOK_BEAT", str(tmp_path / "beat"))
    h.install()
    return h


def _stack_module(h, name="tt_bio._captest"):
    """A module shaped like the real ports: a class holding `self.blocks = [...]` of a block class
    with its own `__call__`."""
    import types
    mod = types.ModuleType(name)

    class Block:
        def __call__(self, x):
            return x + 1

    class Stack:
        def __init__(self, n):
            self.blocks = [Block() for _ in range(n)]

        def __call__(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    Block.__module__ = Stack.__module__ = name
    mod.Block, mod.Stack = Block, Stack
    h._patch_module(mod)
    return mod


def test_the_screen_hook_actually_truncates_a_block_stack(tmp_path, monkeypatch):
    h = _fresh_hook("screen", tmp_path, monkeypatch)
    mod = _stack_module(h)
    s = mod.Stack(48)
    assert len(s.blocks) == 1, "the screen ran 48 blocks and would have reported itself a screen"
    assert s(0) == 1
    assert h._state["truncated"], "the hook truncated without recording it, so a no-op is invisible"
    assert h._state["truncated"][0][2] == 48, "the record must carry the ORIGINAL depth"


def test_the_hook_emits_a_heartbeat_per_block_call(tmp_path, monkeypatch):
    """The stall detector's sharp signal. Without it a legitimately slow block reads as a hang."""
    h = _fresh_hook("residency", tmp_path, monkeypatch)
    mod = _stack_module(h)
    mod.Stack(4)(0)
    beats = sum(f.stat().st_size for f in tmp_path.glob("beat.*"))
    assert beats >= 4, f"only {beats} block calls seen; the heartbeat is not wired"


def test_the_residency_hook_leaves_the_depth_alone(tmp_path, monkeypatch):
    """Tier 2's whole job is the cumulative residency, which a truncated stack cannot build up."""
    h = _fresh_hook("residency", tmp_path, monkeypatch)
    mod = _stack_module(h)
    assert len(mod.Stack(48).blocks) == 48
    assert not h._state["truncated"]


def test_the_hook_is_inert_without_its_env_var(tmp_path, monkeypatch):
    """The generated sitecustomize can outlive a run on a stale PYTHONPATH; it must do nothing."""
    import importlib
    import capacity_hook
    h = importlib.reload(capacity_hook)
    monkeypatch.delenv("TT_BIO_CAPACITY_HOOK", raising=False)
    h.install()
    assert h._state["mode"] is None
    mod = _stack_module(h)
    assert len(mod.Stack(48).blocks) == 48


def test_a_failed_hook_install_is_recorded_not_swallowed(tmp_path):
    """The bug itself: an ImportError in the generated sitecustomize left no trace, so a full-depth
    run was reported as a screen. The gate must be able to SEE that it happened."""
    src = (ROOT / "scripts" / "capacity_gate.py").read_text()
    body = src[src.index("def hook_dir("):src.index("def hook_findings(")]
    assert "install-failed" in body, "sitecustomize must record an install failure"
    assert "except Exception:\n" not in body.replace("except BaseException", "")
    assert "install_failed" in src[src.index("def hook_findings("):], \
        "the gate must read the install-failure marker"
    assert "Not a\n                     f\"screen." in src or "Not a" in src, \
        "a leg whose hook did not apply must not be called a screen"


def test_the_hook_reaches_a_spawned_child(tmp_path):
    """`predict` folds in a spawned worker. A hook that only installs in the launcher measures
    nothing, which is exactly what happened."""
    import subprocess
    d = tmp_path / "_hook"
    d.mkdir()
    (d / "capacity_hook.py").write_text((ROOT / "scripts" / "capacity_hook.py").read_text())
    (d / "sitecustomize.py").write_text(
        "import os\n"
        "if os.environ.get('TT_BIO_CAPACITY_HOOK'):\n"
        "    import capacity_hook; capacity_hook.install()\n")
    env = dict(os.environ, PYTHONPATH=str(d), TT_BIO_CAPACITY_HOOK="screen",
               TT_BIO_CAPACITY_HOOK_OUT=str(tmp_path / "o"),
               TT_BIO_CAPACITY_HOOK_BEAT=str(tmp_path / "b"))
    # `spawn` re-imports the child target's module, so it has to live in a real file.
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import multiprocessing as mp\n"
        "def child(q):\n"
        "    import capacity_hook as h\n"
        "    q.put(h._state['mode'])\n"
        "if __name__ == '__main__':\n"
        "    ctx = mp.get_context('spawn'); q = ctx.Queue()\n"
        "    p = ctx.Process(target=child, args=(q,)); p.start(); p.join(60)\n"
        "    print(q.get(timeout=10))\n")
    r = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True,
                       env=env, timeout=180)
    assert r.stdout.strip().endswith("screen"), (
        f"the hook did not arm in the spawned child: {r.stdout!r} {r.stderr[-400:]!r}")
