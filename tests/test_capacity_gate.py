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
import time
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
    """The bar has to be a size the hardware actually sees. 1536 is exactly 48x32, so it needs no
    padding at all; a "1500" bar would pad to 1504 and report a number 4 tokens smaller than the
    one that ran, and an unmasked tail is a ~72x error."""
    assert cg.TOKEN_BAR == 1536, "Moritz's number, and it needs no padding"
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
    """A 1536-token deep-MSA fold can OOM the HOST. That is a different failure from running out
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
    sequence, and it reads two different file formats to do it. Both must see 1536, or a cell
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
    size untested. So every recorded cell pins the ceiling table it was measured against.

    PER CELL, and it used to be per file. `record_baseline` stamps the current fingerprint on the
    whole file whether the run measured one model or twenty, so `--models boltz2 --record` would
    have turned this test green while twelve cells still held another engine's numbers -- and the
    2026-09-07 ceiling move arrived with 800 changed lines of tt_bio/tenstorrent.py, the RF3
    triangle-attention rewrite and OpenFold3's MSA embedder, every one of which moves DRAM at
    1536 tokens. The sweep is hours of card time and has to run in stages, so the check that
    matters is the one that can be satisfied a stage at a time and says what is left.
    """
    per_card = cg.read_baseline()
    assert per_card, "the baseline records no cells"
    unstamped = sorted(f"{card}/{m}" for card, blk in per_card.items()
                       for m, c in (blk.get("cells") or {}).items()
                       if not (c or {}).get("ceilings_fingerprint"))
    assert not unstamped, (
        f"these cells record no ceiling fingerprint at all, so nothing can tell whether they are "
        f"current: {unstamped}. Re-measure and --record them.")
    assert not cg.baseline_stale(), (
        f"tt_bio/size_limits.CEILINGS has changed since these cells were measured, so the sizes "
        f"tt-bio now advertises have not been capacity-tested: {cg.baseline_stale()}. Re-run\n"
        f"  TT_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python3 scripts/capacity_gate.py "
        f"--models <them> --record\n"
        f"This is the check that was missing when the ceilings went to 1024 on 2026-09-03 and "
        f"three models broke in traffic.")


def _record_into_tmp(report, *, prior_cells, partial):
    """`record_baseline` against a throwaway file, so the committed baseline is not touched."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "baseline.json"
        path.write_text(json.dumps({
            "bar_tokens": cg.TOKEN_BAR,
            "cards": {"p150a": {"geometry": {"board_type": "p150a"}, "cells": prior_cells}}}))
        real, cg.BASELINE = cg.BASELINE, path
        try:
            cg.record_baseline(report, partial=partial)
            return json.loads(path.read_text())
        finally:
            cg.BASELINE = real


def test_one_models_record_cannot_certify_another_models_cell():
    """The false green above, pinned. Recording a run that measured ONE model must leave every
    other cell reading exactly as stale as it was."""
    stale = {"verdict": "PASS", "tokens_requested": cg.TOKEN_BAR,
             "ceilings_fingerprint": "0000000000000000", "tree": "old"}
    report = {"bar_tokens": cg.TOKEN_BAR, "started": "now", "tree": "new", "dirty": False,
              "geometry": {"board_type": "p150a"}, "reductions": [],
              "results": [{"model": "esmc-300m", "verdict": "PASS",
                           "tokens_requested": cg.TOKEN_BAR}]}
    written = _record_into_tmp(report, prior_cells={"esmc-6b": stale}, partial=True)
    cells = written["cards"]["p150a"]["cells"]
    assert cells["esmc-6b"]["ceilings_fingerprint"] == "0000000000000000", \
        "a one-model record re-certified a cell it never measured"
    assert cells["esmc-300m"]["ceilings_fingerprint"] == ceilings_fingerprint()
    assert "ceilings_fingerprint" not in written, \
        "a file-level fingerprint is back; it is the thing that made the false green possible"
    assert "ceilings_fingerprint" not in written["cards"]["p150a"], \
        "a card-level fingerprint is the same false green one level down"


@pytest.mark.skipif(not BASELINE.exists(), reason="no capacity baseline recorded yet")
def test_a_baseline_from_a_different_bar_is_not_evidence():
    """The ceiling guard next door catches the ceilings moving under a fixed bar. Moving the BAR
    is the same hole from the other side: a baseline measured at 1504 says nothing about whether
    a model allocates at 1536, and pair tensors scale roughly quadratically, so the gap is not
    small. Caught for real when the bar was raised 1504 -> 1536 and every recorded cell stayed
    green."""
    b = json.loads(BASELINE.read_text())
    assert b.get("bar_tokens") == cg.TOKEN_BAR, (
        f"docs/capacity_gate_baseline.json was measured at {b.get('bar_tokens')} tokens but the "
        f"bar is now {cg.TOKEN_BAR}. Those cells are not evidence for this bar; re-run the gate "
        f"and re-record.")
    # The file-level stamp alone is not enough: a PARTIAL re-record (--models) merges into the
    # prior file and stamps it with the new bar, so a cell measured at the old bar can sit inside
    # a correctly-stamped file. That is exactly what happened at 1504 -> 1536.
    stale = {f"{card}/{m}": c.get("tokens_requested")
             for card, blk in cg.read_baseline().items()
             for m, c in (blk.get("cells") or {}).items()
             if c.get("tokens_requested") not in (None, cg.TOKEN_BAR)}
    assert not stale, (
        f"these baseline cells were measured at a different bar and are not evidence for "
        f"{cg.TOKEN_BAR}: {stale}")


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


@pytest.fixture(autouse=True)
def _leave_the_interpreter_as_we_found_it():
    """Disarm the hook after every test in this file, whatever the test did with it.

    `_fresh_hook` arms the real hook IN THIS PROCESS, and an armed hook is global state three ways
    over: a finder at `sys.meta_path[0]` that wraps the loader of every `tt_bio` module imported
    after it, a replaced `__init__` on every tt_bio class already imported, and an `atexit` flush.
    None of that used to come back, so this file left the interpreter different for whoever ran
    next: 7 tests in `test_sdpa_ragged_pad_defaults.py` and `test_triatt_sdpa_hifi_defaults.py`
    read `inspect.signature(...).parameters[param].default` off a tt_bio class, saw the wrapper's
    parameters, and failed only in full-suite order -- reported against those two files, which had
    nothing wrong with them. Worse than the false failures: a leaked `screen` hook truncates the
    block stack of anything constructed afterwards, so a later in-process fold would run 1 block
    deep and say nothing about it.

    Autouse rather than a teardown inside `_fresh_hook`, because a new test that arms the hook any
    other way is covered too.
    """
    yield
    import capacity_hook
    capacity_hook.uninstall()


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


def test_the_hook_never_opens_the_device_it_samples(tmp_path, monkeypatch):
    """An instrument must not acquire the thing it measures.

    The DRAM sampler read the allocator through `get_device()`, which OPENS a card and takes
    an exclusive host-wide lease on it first. In a run with no card of its own that blocked
    TT_BIO_LEASE_TIMEOUT (120 s) behind whoever legitimately held the card, once per sampled
    block -- which is how `TT_VISIBLE_DEVICES= pytest tests/` reached its 1800 s wall on this
    very test with nothing on stdout, and got misread as a collection hang. The handle it
    wants is the one already open; in a real capacity run the device is open before the first
    block executes, which is the only time this samples at all.
    """
    import tt_bio.tenstorrent as tt
    opened = []
    monkeypatch.setattr(tt, "get_device", lambda *a, **kw: opened.append(1))
    monkeypatch.setattr(tt, "_device", None)
    h = _fresh_hook("residency", tmp_path, monkeypatch)
    mod = _stack_module(h)
    mod.Stack(4)(0)
    assert not opened, "the DRAM sampler opened (and leased) the device"
    assert not h._state["errors"], h._state["errors"]
    assert h._state["samples"] == 4, "the sampler stopped counting blocks"


def test_the_residency_hook_leaves_the_depth_alone(tmp_path, monkeypatch):
    """Tier 2's whole job is the cumulative residency, which a truncated stack cannot build up."""
    h = _fresh_hook("residency", tmp_path, monkeypatch)
    mod = _stack_module(h)
    assert len(mod.Stack(48).blocks) == 48
    assert not h._state["truncated"]


def _filtering_loader_module(h, name="tt_bio._captest_sig"):
    """A module shaped like the real loaders: a class whose classmethod constructor reads its own
    `__init__` signature to decide which checkpoint hyperparameters to pass on. This is boltz2's
    and boltzgen's actual contract, reproduced in miniature."""
    import types
    mod = types.ModuleType(name)

    class Model:
        def __init__(self, atom_s, atom_z, token_s, token_z, num_bins, extra=1):
            self.hp = (atom_s, atom_z, token_s, token_z, num_bins, extra)
            self.blocks = [object() for _ in range(48)]

        @classmethod
        def from_pretrained(cls, hparams):
            import inspect
            valid = set(inspect.signature(cls.__init__).parameters) - {"self"}
            return cls(**{k: v for k, v in hparams.items() if k in valid})

    Model.__module__ = name
    mod.Model = Model
    h._patch_module(mod)
    return mod


CKPT_HPARAMS = {"atom_s": 1, "atom_z": 2, "token_s": 3, "token_z": 4, "num_bins": 5,
                "not_a_parameter": 9}


def test_the_hook_does_not_change_what_a_class_looks_like(tmp_path, monkeypatch):
    """Defect 12: the instrument was not signature-transparent, and it broke the model it measured.

    A bare `(self, *a, **kw)` wrapper over `__init__` makes `inspect.signature(cls.__init__)`
    report the WRAPPER's parameters. A loader that filters checkpoint hyperparameters against that
    set therefore drops all of them and constructs the class with nothing. That is what killed
    boltz2 8 s into Tier 1 -- `Boltz2.__init__() missing 5 required positional arguments` --
    against 172 s of healthy unhooked construction, and the gate recorded the invented failure as
    a capacity FAIL at 1536.

    Negative control: drop the `functools.wraps` in `_wrap_init` and this fails with that exact
    TypeError, which is the whole point of the test existing.
    """
    h = _fresh_hook("screen", tmp_path, monkeypatch)
    mod = _filtering_loader_module(h)
    import inspect
    params = list(inspect.signature(mod.Model.__init__).parameters)
    assert params == ["self", "atom_s", "atom_z", "token_s", "token_z", "num_bins", "extra"], (
        f"the hook changed the class's signature to {params}; a loader that filters on it will "
        f"pass nothing")
    m = mod.Model.from_pretrained(CKPT_HPARAMS)
    assert m.hp == (1, 2, 3, 4, 5, 1), "the hyperparameters did not survive the wrapper"
    assert len(m.blocks) == 1, "signature transparency must not cost the truncation"
    assert h._state["truncated"], "the screen silently stopped applying"


def test_the_hook_reaches_the_same_construction_hooked_and_unhooked(tmp_path, monkeypatch):
    """The shape of the check the brief asks for: construct through the loader with the hook armed
    and with it disarmed, and require the same model out. A verdict is only about the model if the
    instrument is a no-op on everything except depth."""
    off = _fresh_hook("", tmp_path, monkeypatch)
    plain = _filtering_loader_module(off, "tt_bio._captest_sig_off").Model.from_pretrained(
        CKPT_HPARAMS)
    on = _fresh_hook("screen", tmp_path, monkeypatch)
    hooked = _filtering_loader_module(on, "tt_bio._captest_sig_on").Model.from_pretrained(
        CKPT_HPARAMS)
    assert hooked.hp == plain.hp, (
        f"hooked construction produced {hooked.hp}, unhooked {plain.hp}; the instrument is "
        f"altering construction semantics, not observing them")
    assert len(plain.blocks) == 48 and len(hooked.blocks) == 1, "depth is the only allowed change"


def test_a_strict_state_dict_load_survives_the_screens_depth_cut(tmp_path, monkeypatch):
    """Defect 13. Cutting `layers` to one element leaves the checkpoint's `layers.1.*` with
    nowhere to go, and `load_state_dict(strict=True)` raises "Unexpected key(s) in state_dict".
    Measured on boltz2, whose atom-encoder DiffusionTransformer holds 3 layers.

    The block that still runs must keep its REAL weights: a screen against re-initialised weights
    would allocate the right shapes for the wrong reasons and could not be trusted about anything
    else either.
    """
    torch = pytest.importorskip("torch")
    import types
    nn = torch.nn
    h = _fresh_hook("screen", tmp_path, monkeypatch)

    class Inner(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(n)])

    class Top(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.enc = Inner(n)

    unhooked = Top(3)                     # built before the hook sees the module: the checkpoint
    ckpt = unhooked.state_dict()
    Inner.__module__ = Top.__module__ = "tt_bio._captest_sd"
    mod = types.ModuleType("tt_bio._captest_sd")
    mod.Inner, mod.Top = Inner, Top
    h._patch_module(mod)

    hooked = mod.Top(3)
    assert len(hooked.enc.layers) == 1, "the cut did not apply, so this proves nothing"
    hooked.load_state_dict(ckpt, strict=True)          # the whole test
    assert torch.equal(hooked.enc.layers[0].weight, unhooked.enc.layers[0].weight), (
        "the block that runs did not get the checkpoint's weights")


def test_the_gate_does_not_read_its_own_depth_cut_as_a_bad_checkpoint():
    """The misattribution defect 13 caused: the no-weights classifier matches "Unexpected key(s)
    in state_dict", so a cut the GATE made came back as a fact about the ARTIFACT -- boltz2 scored
    NO_WEIGHTS, which reads as "not our problem". A genuinely wrong checkpoint must still read
    NO_WEIGHTS, so the discriminator is the recorded cut, not the message."""
    cut = [["tt_bio.boltz2.DiffusionTransformer", "layers", 3]]
    ours = ('Unexpected key(s) in state_dict: '
            '"input_embedder.atom_attention_encoder.atom_encoder.diffusion_transformer.layers.1'
            '.adaln.s_norm.weight"')
    assert cg._hook_cut_these_weights(ours, cut), "the gate still blames the checkpoint"
    assert cg._no_weights(ours), "and the classifier it has to win against still matches"

    theirs = 'Unexpected key(s) in state_dict: "atom_encoder.extra_head.weight"'
    assert not cg._hook_cut_these_weights(theirs, cut), (
        "a checkpoint that is genuinely wrong for the module must still report NO_WEIGHTS")
    assert not cg._hook_cut_these_weights(ours, []), "with no cut recorded there is nothing to own"


def test_a_mechanismless_screen_fail_under_truncation_is_not_scored():
    """Defect 14, and the one that breaks Tier 1's founding claim. `Module.__call__` in
    tt_bio/tenstorrent.py slices a per-layer bias tensor as `z.shape[1] // len(self.layers)`, so
    the cut from 3 layers to 1 turned a 4-head slice into a 12-head one. A block stack's LENGTH is
    load-bearing for tensors outside the stack, so truncation CAN invent a failure.

    A real capacity failure names a mechanism -- rf3's is an allocator refusal for a single
    18530435072 B DRAM buffer. A shape mismatch names none, so it is not scored."""
    invented = {"verdict": "FAIL", "mechanism": None, "stacks_truncated": 8,
                "tail": "The size of tensor a (4) must match the size of tensor b (12)"}
    assert cg.screen_reduction_is_unsafe(invented), (
        "a shape mismatch under the gate's own depth cut would be published as a 1536 ceiling")

    real = {"verdict": "FAIL", "mechanism": "dram", "stacks_truncated": 8,
            "tail": "Out of Memory: Not enough space to allocate 18530435072 B DRAM buffer"}
    assert not cg.screen_reduction_is_unsafe(real), (
        "rf3's measured FAIL must stay definitive; re-running it full-depth costs 20 min and "
        "cannot change a shape verdict")

    untruncated = dict(invented, stacks_truncated=0)
    assert not cg.screen_reduction_is_unsafe(untruncated), (
        "with no cut applied the screen cannot be the cause, so the leg is the model's own")
    assert not cg.screen_reduction_is_unsafe(dict(invented, verdict="INCONCLUSIVE"))


def test_no_loader_introspects_in_a_way_the_hook_cannot_survive():
    """Keeps the test above honest against the real tree. `functools.wraps` sets `__wrapped__`,
    which `inspect.signature` follows and `inspect.getfullargspec` does NOT. So the transparency
    holds for every loader that uses `signature`, and would silently break for one that reached
    for `getfullargspec` instead. That is a claim about tt_bio's source, so it is checked there
    rather than asserted in a docstring."""
    import subprocess
    hits = subprocess.run(
        ["grep", "-rn", "-e", "getfullargspec", "-e", "getargspec", "--include=*.py", "tt_bio"],
        cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    live = [h for h in hits if "/_vendor/" not in h and "/reference" not in h]
    assert not live, (
        f"these read a signature in a form functools.wraps does not make transparent, so the "
        f"capacity hook can break them the way it broke boltz2: {live}")
    sigs = subprocess.run(
        ["grep", "-rn", "inspect.signature", "--include=*.py", "tt_bio"],
        cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    assert [s for s in sigs if "__init__" in s], (
        "no loader filters on an __init__ signature any more; if that is real, the guards above "
        "are modelling a contract the tree no longer has and should be re-grounded")


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


# ---------------------------------------------------------------------------------------------
# THIS FILE MUST NOT LEAVE THE INTERPRETER DIFFERENT FOR WHOEVER RUNS NEXT.
#
# It arms the real hook in the pytest process, and until `uninstall()` existed nothing took it
# back down. The bill: 7 tests in the two files below failed in full-suite order and passed alone,
# and were reported against those files rather than against this one. They read
# `inspect.signature(cls.__init__).parameters[param].default`, which is exactly what a replaced
# `__init__` destroys.
#
# `functools.wraps` on `_wrap_init` (the boltz2 fix) makes that particular read transparent again,
# so the symptom is currently masked on main. It is not the fix: a leaked `screen` hook still cuts
# the block stack of every object built afterwards, and a model that quietly runs 1 block deep is
# not a test-order nuisance. Both halves are pinned here -- the in-process one because it names
# the mechanism, the session one because it is what a reader actually sees.
# ---------------------------------------------------------------------------------------------

#: The two files the leak actually broke. Named, not discovered: the guard has to fail if the
#: mechanism comes back, and a search for "files that read a signature" would quietly find none.
_COLLATERAL = ("tests/test_sdpa_ragged_pad_defaults.py",
               "tests/test_triatt_sdpa_hifi_defaults.py")


def _wrapped_inits(mod) -> list[str]:
    """Classes in `mod` whose own `__init__` is currently the hook's wrapper."""
    out = []
    for name in dir(mod):
        cls = getattr(mod, name, None)
        if not isinstance(cls, type):
            continue
        init = cls.__dict__.get("__init__")
        if getattr(init, "_capacity_wrapped", False):
            out.append(name)
    return out


def test_uninstall_takes_back_every_change_install_made(tmp_path, monkeypatch):
    """Each assertion has its own arm-side check, so a hook that failed to arm cannot pass this."""
    h = _fresh_hook("screen", tmp_path, monkeypatch)
    armed = _stack_module(h)
    assert len(armed.Stack(48).blocks) == 1, "the hook did not arm; nothing below proves anything"
    assert [f for f in sys.meta_path if type(f).__name__ == "_Finder"], \
        "install() put no finder on sys.meta_path"
    assert _wrapped_inits(sl), \
        "install() patched no class in the already-imported tt_bio.size_limits, so the restore " \
        "check below would pass on an empty set"

    h.uninstall()

    assert not [f for f in sys.meta_path if type(f).__name__ == "_Finder"], \
        "a finder is still on sys.meta_path, so every later `import tt_bio.*` is still wrapped"
    assert not _wrapped_inits(sl), "tt_bio classes still carry the hook's __init__"
    assert h._state["mode"] is None
    after = _stack_module(h, "tt_bio._captest_after")
    assert len(after.Stack(48).blocks) == 48, \
        "a stack built after uninstall is still being truncated"


def test_uninstall_takes_the_atexit_flush_back_down(tmp_path):
    """Checked by what the flush DOES, not by `atexit._ncallbacks()`, which on 3.10 keeps counting
    a callback `unregister` has already removed and would pass either way."""
    import subprocess
    prog = (
        "import os, sys, capacity_hook as h\n"
        "h.install()\n"
        "if 'uninstall' in sys.argv: h.uninstall()\n"
    )
    for argv, expect in ((["armed"], True), (["uninstall"], False)):
        out = tmp_path / argv[0]
        r = subprocess.run([sys.executable, "-c", prog, *argv], cwd=ROOT,
                           env=dict(os.environ, PYTHONPATH=str(ROOT / "scripts"),
                                    TT_BIO_CAPACITY_HOOK="screen",
                                    TT_BIO_CAPACITY_HOOK_OUT=str(out)),
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        wrote = bool(list(tmp_path.glob(f"{argv[0]}.*.json")))
        assert wrote is expect, (
            f"armed hook wrote its findings at exit: {expect}, observed {wrote}. The 'armed' leg "
            f"is the control: if it does not write, this test proves nothing about uninstall.")


@pytest.mark.skipif(bool(os.environ.get("TT_BIO_CAPACITY_GATE_NO_RECURSE")),
                    reason="the inner session this guard spawned; it must not spawn its own")
def test_this_file_does_not_break_the_files_that_run_after_it():
    """One pytest session: this file, then the two it used to break. Both halves have to be green.

    A subprocess because the property is about a fresh interpreter's import state, which cannot be
    observed from inside the session that already polluted it.
    """
    import subprocess
    for rel in _COLLATERAL:
        assert (ROOT / rel).exists(), (
            f"{rel} is gone, so this guard is now vacuous. Re-ground it on a file that reads a "
            f"tt_bio signature -- grep for `inspect.signature` under tests/.")
    env = dict(os.environ, TT_BIO_CAPACITY_GATE_NO_RECURSE="1", TT_VISIBLE_DEVICES="",
               PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:randomly", "--tb=no",
                        "-rf", "tests/test_capacity_gate.py", *_COLLATERAL],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    out = r.stdout + r.stderr
    assert " passed" in out, f"the inner session did not run at all:\n{out[-4000:]}"
    broke = [ln for ln in out.splitlines()
             if ln.startswith("FAILED") and any(c in ln for c in _COLLATERAL)]
    assert not broke, (
        "test_capacity_gate.py ran first and took these down with it, which is the import-state "
        f"leak coming back: {broke}\nThey pass on their own. See capacity_hook.uninstall().")
    assert r.returncode == 0, (
        "the inner session is red, but not from the pollution this guard is about -- every "
        f"failure is in test_capacity_gate.py itself:\n{out[-4000:]}")


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


# ---------------------------------------------------------------------------------------------
# The --workers fan-out. p1 wired it and never validated it, because qb1 and qb2 were unpowered.
# Validating it is what showed it did not fan out at all: the loop round-robined the ASSIGNMENT
# and then ran the cell inline, so four cards cost exactly what one card cost. These run on one
# host with no device, which is the only way the claim gets checked while the other boxes are down.
# ---------------------------------------------------------------------------------------------


def _sweep_probe(n_workers, n_cells, hold=0.15, retire=None):
    """Run cg.sweep with fake workers and a cell that just sleeps, and record who ran what when."""
    import threading
    import time as _t
    seen, lock = [], threading.Lock()
    workers = [f"w{k}" for k in range(n_workers)]

    def run_one(w, cell):
        t0 = _t.monotonic()
        _t.sleep(hold)
        return {"model": cell, "verdict": "PASS", "worker": w, "t0": t0,
                "t1": _t.monotonic()}

    published = {}

    def publish(i, rec):
        with lock:
            published[i] = rec
            seen.append((rec["worker"], rec["model"]))

    t0 = _t.monotonic()
    unrun = cg.sweep(list(range(n_cells)), workers, run_one, publish, retire)
    return published, unrun, _t.monotonic() - t0, seen


def test_the_worker_fanout_actually_runs_cells_concurrently():
    """The defect: `workers[i % len(workers)]` chose a different card per iteration and the loop
    still waited for it, so --workers changed WHICH card ran a cell and never how many ran at
    once. Four cards would have left the 19.9 min sweep at 19.9 min.

    Wall-clock is the only honest check here, so it is the one used: 8 cells holding 0.15 s each
    is 1.2 s serial and ~0.3 s on four cards. The bound is deliberately loose (0.75 s) so a busy
    host does not fail it, and it is still far below serial.
    """
    published, unrun, wall, _seen = _sweep_probe(4, 8)
    assert not unrun and len(published) == 8, "cells were lost"
    assert wall < 0.75, (
        f"8 cells x 0.15 s took {wall:.2f}s on 4 workers; serial is 1.2s, so the fan-out is "
        f"still running one cell at a time")
    overlap = max(sum(1 for o in published.values() if o["t0"] <= r["t0"] < o["t1"])
                  for r in published.values())
    assert overlap > 1, "no two cells were ever in flight together"
    assert len({r["worker"] for r in published.values()}) == 4, "not every card was used"


def test_the_fanout_balances_by_pulling_rather_than_by_index():
    """Cell cost spans 10 s to 175 s here, so a static round-robin hands one card openfold3 and
    protenix-v2 while another finishes saprot-35m and idles. A shared queue lets the card that is
    free take the next cell, which is why more cells than cards is not a problem."""
    published, unrun, _wall, _seen = _sweep_probe(2, 7, hold=0.05)
    assert not unrun and len(published) == 7
    per = {}
    for r in published.values():
        per[r["worker"]] = per.get(r["worker"], 0) + 1
    assert sum(per.values()) == 7 and len(per) == 2
    assert all(v >= 1 for v in per.values()), f"a card sat idle through the whole sweep: {per}"


def test_a_retired_card_hands_its_remaining_cells_to_a_live_one():
    """A wedge a reset cannot clear must cost that card, not the run. p1's recovery already does
    this serially; under the fan-out the retiring thread has to leave the queue for the others."""
    def retire(w, rec):
        return w == "w0"                       # w0 dies on its very first cell
    published, unrun, _wall, _seen = _sweep_probe(2, 6, hold=0.02, retire=retire)
    assert not unrun, f"cells were dropped when a card retired: {unrun}"
    assert len(published) == 6
    by = {}
    for r in published.values():
        by[r["worker"]] = by.get(r["worker"], 0) + 1
    assert by.get("w0", 0) == 1, f"w0 kept taking work after retiring: {by}"
    assert by.get("w1", 0) == 5, f"w1 did not pick up the rest: {by}"


def test_every_cell_is_reported_when_all_cards_retire():
    """With no card left, the cells still queued were never measured. They come back as unrun so
    main() can record CARD_DIRTY, rather than vanishing from the report."""
    published, unrun, _wall, _seen = _sweep_probe(2, 6, hold=0.02, retire=lambda w, r: True)
    assert len(published) == 2, "each card should have managed exactly one cell"
    assert len(unrun) == 4, f"4 cells were owed and {len(unrun)} came back"
    assert sorted([i for i, _c in unrun] + list(published)) == list(range(6)), (
        "the reported and unrun cells do not add up to the sweep")


def test_the_fanout_vets_every_card_not_just_the_first():
    """A gate that probes workers[0] and fans out over four records every cell that landed on an
    unhealthy card 3 as a capacity failure. That is the exact lie the card-0 probe exists to stop,
    so the check has to iterate."""
    import inspect
    # Code only: the comment above the check names the workers[0] bug it replaced. And scoped to
    # the health check -- `geometry(workers[0])` is a different, correct use of the first card.
    src = "\n".join(l for l in inspect.getsource(cg.main).splitlines()
                    if not l.strip().startswith("#"))
    probes = [l for l in src.splitlines() if "probe_card" in l or "card_healthy" in l]
    assert probes, "main no longer vets the cards at all"
    for line in probes:
        assert "workers[0]" not in line, (
            f"only the first card is vetted ({line.strip()}); cells landing on an unhealthy "
            f"sibling would be scored as failed bars")
    sick = next(l for l in src.splitlines() if l.strip().startswith("sick ="))
    assert "for w in workers" in sick and "probe_card(w)" in sick, (
        f"the pre-flight card check does not probe every worker: {sick.strip()}")


def test_concurrent_cells_do_not_race_on_the_shared_fixture():
    """Models sharing a residue count share the fixture path, and the MSA target is keyed by
    sequence hash, so two threads write one file while a third reads it. Serialised in
    fixture_for, which is seconds against runs of minutes."""
    import inspect
    src = inspect.getsource(cg.fixture_for)
    assert "_FIXTURE_LOCK" in src, "fixture construction is not serialised under the fan-out"
    assert "capacity_fixture.build" in src.split("_FIXTURE_LOCK")[1]


def test_a_failing_model_does_not_wedge_the_rest_of_the_run():
    """Measured: rf3's 1504-token OOM (a TT_FATAL from the allocator) left card 0 accepting an
    open and then never dispatching, and the next cell sat 10 minutes at 100% CPU inside
    tt_bio's own `_assert_local_dispatch` with no log line and no progress event.

    Provoking that refusal is THE JOB of this gate, so recovery is part of the gate. Without it
    the first model that legitimately fails the bar turns every model after it into an invented
    failure -- a real one-line result wrapped in a cascade of noise."""
    import inspect
    # The recovery moved into main()'s `retire` collaborator when the fan-out was made real; the
    # policy it encodes is unchanged, so this reads where it lives now rather than being deleted.
    loop = inspect.getsource(cg.main)
    assert "recover_card(w, workers)" in loop, (
        "the loop must re-check the card after a fail-like verdict; polling a wedged chip cannot "
        "fix it, only a reset can")
    for v in ('"FAIL"', '"STALL"'):
        assert v in loop, f"recovery must trigger on {v}"
    assert 'dead[repr(w)]' in loop and '"CARD_DIRTY"' in loop, (
        "a card the gate gave up on must report its owed cells CARD_DIRTY -- nothing was "
        "measured on them, so recording FAIL would publish a ceiling nobody walked")
    assert "retire" in inspect.signature(cg.sweep).parameters, (
        "the scheduler must be able to retire a card, or one wedge poisons every cell after it")


def test_every_bisect_rung_keeps_its_own_evidence():
    """A bisect screens up to seven rungs and the ceiling it reports rests on which rung refused.
    All of them wrote `screen_<model>.log`, so each rung overwrote the last and the only surviving
    proof was the final one. A ceiling nobody can re-read is a ceiling nobody can check."""
    import inspect
    src = inspect.getsource(cg._screen)
    assert 'f"screen_{cell.model}.log"' not in src, "every rung still writes one shared log"
    assert 'tokens_requested' in src and '{tok}.log' in src, (
        "the screen log must be keyed by size, the way the residency log already is")


def test_the_bisect_re_probes_the_card_between_rungs():
    """Defect 15. p1's recovery runs after each CELL, but a bisect provokes rf3's TT_FATAL
    allocator refusal once per RUNG inside one cell -- the densest run of refusals this gate ever
    produces, and it had no recovery in it. Every rung below the first failing one was running on
    a card the rung above may have left accepting an open and never dispatching, so the ceiling
    those rungs report is exactly the kind of number that gets published without being walked."""
    import inspect
    src = inspect.getsource(cg._bisect)
    assert "recover" in inspect.signature(cg._bisect).parameters
    assert src.count("settle(") >= 3, (
        "the coarse walk, its residency leg and the refinement must each re-probe the card")
    for v in ('"FAIL"', '"HOST_OOM"', '"STALL"'):
        assert v in src, f"recovery must trigger on {v} between rungs"
    # and main() must actually supply it, or the parameter is decoration
    m = inspect.getsource(cg.main)
    assert "recover_card(ww, workers)" in m and "no_card_reset" in m, (
        "run_cell is called without a recovery, so the bisect still runs on a card nobody checked")


def test_a_reset_is_refused_when_the_run_does_not_own_the_host():
    """`tt-smi -r` resets the BOARD PAIR, not the chip, so a reset issued for card 0 of a p300c
    also takes down card 1. Under --workers fan-out that is somebody else's in-flight leg."""
    a = cg.Worker("pc", 0, True)
    b = cg.Worker("pc", 1, True)
    assert cg._may_reset(a, [a]) is True
    assert cg._may_reset(a, [a, b]) is False, "two cards on one host: a reset hits the sibling"
    assert cg._may_reset(cg.Worker("qb1", 0, False), [cg.Worker("qb1", 0, False)]) is False, (
        "the reset path is local-only; it does not ssh a reset at a remote host")


def test_a_sigkilled_worker_is_the_host_not_the_card():
    """Measured: esmfold2 at 1504 residues was killed by the kernel OOM killer at 22.8 GiB
    anon-rss on this 30 GB host. The launcher only ever saw `SpawnProcess-1 exit -9` and no
    device error, and the RSS-floor sampler missed the spike because the kill lands between two
    samples. Scored as FAIL that publishes a Blackhole ceiling that nothing on the card set."""
    assert cg._host_killed(-9, "") is True
    assert cg._host_killed(137, "") is True
    assert cg._host_killed(1, "every local worker exited before the run finished "
                              "(SpawnProcess-1 exit -9); no job can be served") is True
    assert cg._host_killed(0, "all good") is False
    assert cg._host_killed(1, "Out of Memory: Not enough space to allocate 18530435072 B "
                              "DRAM buffer across 8 banks") is False, (
        "a device allocator refusal is the bar failing, not the host")


def test_an_unusable_checkpoint_is_not_a_failed_bar():
    """nesso1's artifact on this host carries atom-encoder layers the module does not declare, so
    from_pretrained dies in load_state_dict(strict=True) before one tensor reaches the card. An
    absent checkpoint was already excluded; an unusable one is the same non-result."""
    assert cg._no_weights("RuntimeError: Error(s) in loading state_dict for Nesso1:\n\t"
                          "Unexpected key(s) in state_dict: \"input_embedder.atom_attention\"")
    assert cg._no_weights("size mismatch for trunk.weight: copying a param with shape ...")
    assert not cg._no_weights("Out of Memory: Not enough space to allocate 18530435072 B DRAM")


def test_the_screen_only_path_reclassifies_a_failing_leg_too():
    """The guard here read `!= "FAIL"`, which skipped every case the reclassification exists for:
    a missing or broken checkpoint is precisely what makes the leg exit nonzero."""
    src = (ROOT / "scripts" / "capacity_gate.py").read_text()
    body = src[src.index("def _screen_only("):]
    assert '!= "FAIL" and _no_weights' not in body
    assert '("FAIL", "HOST_OOM") and _no_weights' in body


def test_the_gates_own_instrument_breaking_a_model_is_not_a_failed_bar():
    """Measured: the Tier 1 hook wraps __init__ on every class in every non-vendored tt_bio
    module, and that broke boltz2's construction outright -- "Boltz2.__init__() missing 5 required
    positional arguments", 8 s in, scored FAIL. Without the hook the same fixture gets 172 s into
    the model, so the FAIL was a capacity verdict invented by the gate's own instrument.

    A model the gate cannot even build has not failed the bar, and this must be loud rather than
    quietly excluded: GATE_BUG makes the run exit nonzero without counting as a capacity result."""
    assert "GATE_BUG" in cg.VERDICTS
    assert cg._HOOK_BROKE.search("TypeError: Boltz2.__init__() missing 5 required positional "
                                 "arguments: 'atom_s', 'atom_z', 'token_s', 'token_z'")
    assert not cg._HOOK_BROKE.search("Out of Memory: Not enough space to allocate 18530435072 B")
    src = (ROOT / "scripts" / "capacity_gate.py").read_text()
    assert 'not report["counts"]["GATE_BUG"]' in src, (
        "a gate that cannot instrument a model must not exit 0")


def test_the_host_oom_corroboration_can_actually_read_the_kernel_log():
    """kernel.dmesg_restrict=1 on this host, so the plain `dmesg` call failed with "Operation not
    permitted" and the corroborating evidence came back None on every HOST_OOM the gate recorded,
    silently. The SIGKILL stays the primary signal; this is the second opinion."""
    src = (ROOT / "scripts" / "capacity_gate.py").read_text()
    body = src[src.index("def _oom_killer_fired("):src.index("def _bisect(")]
    assert '["sudo", "-n", "dmesg"]' in body
    assert "returncode == 0" in body, "a failed dmesg must not read as 'no kill found'"


# ---------------------------------------------------------------------------------------------
# Recording. A sweep at this bar is hours and a bisect alone can be hours, so it runs in stages,
# and the two defects below both lose a measurement that cost card time to get.
# ---------------------------------------------------------------------------------------------


def _cells(path, card="p150a"):
    """The recorded cells for one board. The file is keyed by board type: p150a and p300c are
    both "blackhole" to ttnn, so a cell filed under no card is a cell about no card. Not on a
    geometry difference: stock boards of both types read 110 L1 banks on an (x=11,y=10) grid.
    The 130-bank figure this used to cite came from pc and its custom 130-core firmware."""
    return json.loads(Path(path).read_text())["cards"][card]["cells"]


def _bisect_report(bar=None, model="rf3"):
    """A report shaped like the one a bisect writes: no completing ceiling at any rung, so the
    alloc ceiling is the only number it produced."""
    return {
        "bar_tokens": cg.TOKEN_BAR if bar is None else bar,
        "started": "2026-09-07T00:00:00Z", "tree": "deadbeef", "dirty": False,
        "geometry": {"dram_banks": 8, "board_type": "p150a"}, "reductions": {},
        "results": [{"model": model, "verdict": "FAIL", "tokens_requested": cg.TOKEN_BAR,
                     "ceiling_tokens": None, "alloc_ceiling_tokens": 1344,
                     "alloc_ceiling_note": "1344 allocates, 1376 does not",
                     "mechanism": "TT_FATAL bank_manager.cpp:439", "wall_s": 64.7}],
    }


def test_the_baseline_keeps_the_only_ceiling_number_a_bisect_produced(tmp_path, monkeypatch):
    """`ceiling_tokens` is None for a model that never completes a residency run at any rung, and
    rf3 is exactly that model. Recording only the completing ceiling threw away the entire result
    of a multi-hour bisect and left the cell reading as if nothing had been measured."""
    monkeypatch.setattr(cg, "BASELINE", tmp_path / "baseline.json")
    cg.record_baseline(_bisect_report(), partial=True)
    cell = _cells(tmp_path / "baseline.json")["rf3"]
    assert cell["alloc_ceiling_tokens"] == 1344, (
        "the baseline dropped alloc_ceiling_tokens, which for a model with no completing rung is "
        "the only ceiling the bisect measured")
    assert cell["alloc_ceiling_note"], "the number is recorded without what it means"


def test_a_finished_run_can_be_recorded_without_rerunning_it(tmp_path, monkeypatch):
    """The stages of one campaign are separate processes, so a stage that finished before anyone
    thought about `--record` could otherwise only be recorded by spending its card time twice."""
    monkeypatch.setattr(cg, "BASELINE", tmp_path / "baseline.json")
    p = tmp_path / "report.json"
    p.write_text(json.dumps(_bisect_report()))
    assert cg._record_from(p) == 0
    assert _cells(tmp_path / "baseline.json")["rf3"]["verdict"] == "FAIL"


def test_recording_another_bars_report_is_refused(tmp_path, monkeypatch):
    """p1's defect 9 through a new door, and the easiest mistake to make here: a bisect's rungs
    are all BELOW the bar, so its per-rung reports are other-bar reports. Folding one in keeps the
    1536 cells (they match TOKEN_BAR) and restamps the file with the rung's number, which is a
    baseline that reads as current evidence for a bar nothing in it was measured at."""
    monkeypatch.setattr(cg, "BASELINE", tmp_path / "baseline.json")
    p = tmp_path / "rung.json"
    p.write_text(json.dumps(_bisect_report(bar=1408)))
    assert cg._record_from(p) == 2, "a report from another bar was folded into this bar's baseline"
    assert not (tmp_path / "baseline.json").exists(), "it wrote a baseline anyway"


def test_recording_an_unreadable_or_empty_report_is_refused(tmp_path, monkeypatch):
    """Truncated report = the run was killed. Recording it would publish a partial sweep as one."""
    monkeypatch.setattr(cg, "BASELINE", tmp_path / "baseline.json")
    bad = tmp_path / "half.json"
    bad.write_text('{"bar_tokens": 1536, "results": [{"model": "rf3"')      # killed mid-write
    assert cg._record_from(bad) == 2
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps(dict(_bisect_report(), results=[])))
    assert cg._record_from(empty) == 2
    assert not (tmp_path / "baseline.json").exists()


@pytest.mark.skipif(not BASELINE.exists(), reason="no capacity baseline recorded yet")
def test_every_runnable_model_has_a_recorded_cell():
    """Coverage that leaves no record behind is not coverage.

    The roster guard asks whether a shipped model is covered by the gate. Nothing asked whether
    the covered model actually has a measured cell, so the baseline sat at 8 of 16 through two
    passes with every test green: boltz2's PASS at 1536 was measured, written up in prose and
    never recorded, and its report lived in gitignored scratch inside a worktree that was later
    torn down. The claim outlived the evidence, which is the one failure a capacity baseline
    exists to prevent.
    """
    missing = cg.baseline_gaps()
    assert not missing, (
        f"these models are runnable by the gate but have no cell in "
        f"docs/capacity_gate_baseline.json, so their verdict is prose and not evidence: "
        f"{missing}. Run the gate for them and --record (or --record-from a finished report). "
        f"A model that genuinely cannot be measured anywhere needs a written EXEMPT reason "
        f"instead, the way saprot-1.3b has one.")


def test_a_legs_wall_is_not_rounded_up_to_the_poll_interval(tmp_path):
    """The gate's ~60 s/model budget verdict is decided on `wall_s`.

    The watch loop slept 5 s and then asked whether the process had exited, while `wall` was taken
    after the loop, so a leg was over-reported by 0-5 s -- which is why every recorded Tier 1 wall
    was a multiple of 5. A leg that takes a fifth of a second must not read as five seconds.
    """
    w = cg.Worker("local", 0, True)
    r = cg.execute(w, [sys.executable, "-c", "import time; time.sleep(0.2)"],
                   tmp_path / "leg.log", mode="", hook_out=tmp_path / "o",
                   hookdir=tmp_path / "hd", out_dir=tmp_path / "out", timeout=60, stall_s=60)
    assert r["rc"] == 0, r
    assert r["wall_s"] < 3.0, (
        f"a 0.2 s leg reported {r['wall_s']} s: the wall is being rounded up to the poll "
        f"interval, which inflates every per-model Tier 1 number the budget is judged on")


def test_a_leg_cannot_inherit_the_previous_runs_output_or_dram_peak(tmp_path):
    """A warm work dir turned a re-run into a PASS that folded nothing.

    `tt_bio.main predict` skips a target whose results already exist, so a second leg at the same
    (model, tokens) exits 0 in seconds and prints "All predictions complete"; `hook_findings` then
    globs `<hook_out>.*.json` and returns the EARLIER run's numbers. Measured 2026-09-08 re-running
    the boltz2 cell against this repo's own perf/capacity/work: PASS in 2.6 s at 1536 tokens,
    carrying the previous run's 5.78 GiB peak. With --record that banks a cell nobody measured, and
    the wall is the only tell -- the verdict and the peak both look right.
    """
    out = tmp_path / "out_resid_boltz2_1536"
    (out / "boltz2_results_cap_1536").mkdir(parents=True)
    (out / "boltz2_results_cap_1536" / "results.json").write_text('{"ok": true}')
    stale = tmp_path / "o.4242.json"
    stale.write_text(json.dumps({"dram_peak_bytes": 6203490304, "truncated": [],
                                 "instrumented": ["stale"]}))

    r = cg.execute(cg.Worker("local", 0, True), [sys.executable, "-c", "pass"],
                   tmp_path / "leg.log", mode="", hook_out=tmp_path / "o",
                   hookdir=tmp_path / "hd", out_dir=out, timeout=60, stall_s=60)

    assert not out.exists(), (
        "the leg ran against the previous run's output directory, so tt_bio skips every target "
        "that already has results and the leg passes without folding anything")
    assert not stale.exists(), (
        "the previous run's hook json survived, and hook_findings globs the prefix -- so this "
        "leg's DRAM peak can be the last leg's")
    assert not (r["hook"] or {}).get("instrumented"), (
        f"this leg folded nothing yet reports hook findings, which can only have come from an "
        f"earlier run: {r['hook']}")


def _bisect_probe(screen_fail_above=None, screen_verdicts=None, residency_verdicts=None,
                  screen_mechanism="dram", screen_sequence=None):
    """Drive _bisect with canned leg outcomes, so the rung bookkeeping is testable without
    spending a rung of real card time on it (a real rung is minutes to tens of minutes).

    `screen_sequence` maps a size to the verdicts its successive screens return, so a rung that
    loses the card and is re-walked can be canned. The last entry repeats.
    """
    screen_verdicts = screen_verdicts or {}
    residency_verdicts = residency_verdicts or {}
    seq = {k: list(v) for k, v in (screen_sequence or {}).items()}
    rec = {"legs": []}
    calls = []

    def screen(worker, cell, f, work, hookdir):
        calls.append(("screen", f))
        if seq.get(f):
            v = seq[f].pop(0) if len(seq[f]) > 1 else seq[f][0]
            return {"verdict": v, "mechanism": {"CARD_DIRTY": "card"}.get(v, screen_mechanism)
                    if v not in ("PASS", "INCONCLUSIVE") else None, "wall_s": 1.0}
        v = screen_verdicts.get(f)
        if v is None:
            v = "FAIL" if (screen_fail_above is not None and f > screen_fail_above) else "PASS"
        return {"verdict": v, "mechanism": screen_mechanism if v == "FAIL" else None,
                "wall_s": 1.0}

    def residency(worker, cell, f, work, hookdir, tokens):
        calls.append(("residency", tokens))
        return {"verdict": residency_verdicts.get(tokens, "FAIL"), "wall_s": 1.0,
                "dram_peak_bytes": 1}

    import unittest.mock as m
    with m.patch.object(cg, "_screen", screen), m.patch.object(cg, "_residency", residency), \
         m.patch.object(cg, "fixture_for", lambda cell, t, work, depth: t):
        ceiling = cg._bisect(None, None, None, None, None, rec)
    return ceiling, rec, calls


def test_a_bisect_that_completes_no_rung_still_reports_what_it_walked():
    """rf3 is this case. Every rung's shapes were refused, `lo` stayed None, and the function
    returned before recording anything -- so seven rungs of card time came back as an empty cell
    that reads as if the bisect had never run. The walk found something and has to say so."""
    ceiling, rec, _ = _bisect_probe(screen_verdicts={t: "FAIL" for t in cg.BISECT_RUNGS})
    assert ceiling is None, "nothing completed, so there is no completing ceiling"
    assert "alloc_ceiling_note" in rec, "the walk recorded nothing at all"
    assert str(min(cg.BISECT_RUNGS)) in rec["alloc_ceiling_note"], (
        f"the note does not say how low the walk actually went: {rec['alloc_ceiling_note']}")


def test_a_walk_where_the_allocator_never_refused_is_not_a_ceiling():
    """A size wall is size-dependent. A walk that dies the same way at every rung has not found
    one, and must not publish a bound below the lowest rung it burned.

    Measured on esmfold2 on qb2's p300c, 2026-09-10: the Tier 1 truncation hook leaves a
    downstream reshape inconsistent, so 1408, 1280, 1024, 896, 768, 640 and 512 all died in 4-6 s
    with the identical `shape '[1, 1, 3, 1]' is invalid for input of size 81` and no allocator
    message at all. The cell published "The ceiling is below 512 tokens if there is one at all"
    for a model whose size ladder folds 768 on that same card in 96.6 s.

    Third instrument breakage after _HOOK_BROKE and _UNEXPECTED_KEY, both of which are matched by
    message. A fourth message would not match a fourth regex either, so this guard reads the
    shape of the evidence: an error that does not change with size is not evidence about size.
    """
    _, rec, _ = _bisect_probe(screen_verdicts={t: "FAIL" for t in cg.BISECT_RUNGS},
                              screen_mechanism=None)
    note = rec["alloc_ceiling_note"]
    assert "decided nothing" in note, (
        f"a walk in which the allocator never refused published a ceiling anyway: {note}")
    assert rec["alloc_ceiling_tokens"] is None
    assert f"below {min(cg.BISECT_RUNGS)}" not in note, \
        f"it still offers the lowest rung as a bound: {note}"


def test_a_residency_failure_does_not_lower_the_allocation_ceiling():
    """The two ceilings are different questions and were sharing one bound. If the screen at a
    size is clean, the shapes allocate at that size -- whatever the residency run then does. Using
    the residency failure to lower the allocation bound throws away a measured screen result."""
    # Shapes allocate at 1408 and below, nothing completes. 1408 is the allocation ceiling and
    # the completing ceiling does not exist -- two different answers from one walk.
    top = max(cg.BISECT_RUNGS)
    ceiling, rec, _ = _bisect_probe(screen_fail_above=top, residency_verdicts={})
    assert ceiling is None, "no residency passed, so there is no completing ceiling"
    assert rec["alloc_ceiling_tokens"] == top, (
        f"the allocation ceiling came back {rec.get('alloc_ceiling_tokens')} even though the "
        f"screen at {top} allocated cleanly and only the residency failed")


#: Quoted verbatim from a real nesso1 residency leg on pc's p150a, 2026-09-07.
_NESSO1_CB_OVERFLOW = (
    "TT_THROW: Statically allocated circular buffers on core range "
    "[(x=0,y=0) - (x=12,y=9)] grow to 3424768 B which is beyond max L1 size of 1572864 B")
_NESSO1_INPUT_REFUSED = "Error: No protein or ligand tokens found in the batch"


def test_a_circular_buffer_overflow_is_recognised_and_named_l1():
    """The pattern for this was written as ".*exceed" and labelled "dram". tt-metal says "grow to
    N B which is BEYOND max L1 size", so it could never fire for the message it existed for, and
    a statically allocated circular buffer lives in L1 rather than DRAM anyway -- so it would have
    named the wrong memory if it had."""
    assert cg.classify(_NESSO1_CB_OVERFLOW) == "l1", (
        f"the L1 circular-buffer wall classified as {cg.classify(_NESSO1_CB_OVERFLOW)!r}")


def test_a_model_rejecting_its_input_is_not_a_failed_bar():
    """The MODEL refusing the input is not the card refusing the size, and scoring the first as
    the second publishes a ceiling nobody walked -- p1's defects 7 and 8 in a third guise. nesso1
    is an affinity model and this gate's fixture is polymer-only, so it never got a valid input at
    any size."""
    assert cg._input_rejected(_NESSO1_INPUT_REFUSED)
    # And it must not swallow a real capacity failure: rf3's refusal stays a refusal.
    assert not cg._input_rejected(
        "Out of Memory: Not enough space to allocate 19327352832 B DRAM buffer across 8 banks")


def test_a_bad_fixture_is_neither_a_pass_nor_silently_ignored():
    """It has to fail the run the way GATE_BUG does. A non-result that exits zero is a non-result
    nobody looks at, and this one means the gate needs a new fixture."""
    report = {"results": [{"verdict": "BAD_FIXTURE"}], "coverage_gaps": []}
    cg._finish(report)
    n = report["counts"]
    assert n["PASS"] == 0, "a rejected input counted as a pass"
    assert n["fail_like"] == 0, "a rejected input counted as a failed bar"
    assert n["BAD_FIXTURE"] == 1
    ok = (n["fail_like"] == 0 and not n["GATE_BUG"] and not n["BAD_FIXTURE"]
          and not report["coverage_gaps"])
    assert not ok, "the run would have exited 0 with a model whose input was never valid"


def test_nesso1_is_exempt_for_a_reason_that_names_the_ligand():
    """It is the roster's only affinity model and the fixture is polymer-only, so it is a
    structural gap like the design models and not a capacity result."""
    assert "nesso1" in cg.EXEMPT
    assert "ligand" in cg.EXEMPT["nesso1"].lower()
    assert "nesso1" not in cg.runnable()
    assert "nesso1" not in cg.coverage_gaps(), "exempt in writing, so not a coverage gap"


# --- the card probe's two phases ---------------------------------------------------------------
#
# Detecting a wedged chip used to cost up to 600 s, because one timeout covered python starting,
# torch and ttnn importing, the device opening AND the dispatch. A bisect provokes an allocator
# refusal on every rung and the gate re-probes after each one, so that timeout was paid over and
# over. These pin the split: the child says CARD_OPEN when the host-side half is done, and only the
# dispatch is held to the short budget.
#
# Every one of these runs a real child process through the real Worker.popen path -- no device, and
# no mock of the thing under test.

def _probe_with(body, **kw):
    """Run probe_card against a local Worker whose probe script is `body`."""
    import capacity_gate
    real = capacity_gate._CARD_PROBE
    capacity_gate._CARD_PROBE = body
    try:
        return capacity_gate.probe_card(cg.Worker(cg.local_host(), 0, True), **kw)
    finally:
        capacity_gate._CARD_PROBE = real


def test_both_device_probes_set_the_p300_mesh_descriptor_before_they_open():
    """A lone p300 chip is a CUSTOM topology to tt-metal, and `ttnn.open_device` refuses it
    outright without a 1x1 mesh-graph descriptor. Both probes here open ttnn directly instead of
    through the CLI, so neither inherits the descriptor tt_bio sets per worker.

    Measured on qb2 card 3, 2026-09-10: the health probe died after 1.0 s with "Custom fabric mesh
    graph descriptor path must be specified for CUSTOM cluster type", the gate scored that as a
    wedge and printed "Reset (tt-smi -r) and re-run". On a p300c that reset takes the BOARD PAIR
    down and can kill a sibling worker's card, so a bare-open failure turned a question about one
    healthy chip into a destructive instruction aimed at two. Every p300c cell in the baseline was
    unreachable for the same reason, which is why the file still holds only p150a numbers.
    """
    for name, src in (("_CARD_PROBE", cg._CARD_PROBE), ("_GEOM_PROBE", cg._GEOM_PROBE)):
        assert "ensure_p300_mesh_descriptor()" in src, (
            f"{name} opens a device without asking tt_bio for the p300 descriptor, so it cannot "
            f"run on a p300c at all and its failure there says nothing about the card")
        assert src.index("ensure_p300_mesh_descriptor()") < src.index("open_device"), (
            f"{name} sets the descriptor after the open, which is too late: the env is read when "
            f"the device opens")


def test_a_healthy_probe_reports_done_and_its_dispatch_cost():
    p = _probe_with("print('CARD_OPEN', flush=True)\nprint('CARD_HEALTHY', flush=True)\n")
    assert p and p.phase == "done", p
    assert p.seconds < 30, "a probe that dispatched immediately should not report a long wall"
    assert "dispatches" in p.why()


def test_a_chip_that_opens_and_never_dispatches_is_caught_by_the_short_budget():
    """THE case this split exists for: a chip left dirty by a TT_FATAL accepts the open and then
    hangs, so the old single timeout charged the full 300-420 s to notice."""
    t0 = time.monotonic()
    p = _probe_with("import time\nprint('CARD_OPEN', flush=True)\ntime.sleep(600)\n",
                    timeout=300, dispatch_budget=1)
    took = time.monotonic() - t0
    assert not p and p.phase == "dispatch", p
    assert took < 60, (
        f"the dispatch budget did not apply: took {took:.0f}s under a 300 s open timeout, which "
        f"is the 600 s-to-notice behaviour this replaced")
    assert "dirty-chip" in p.why()


def test_a_probe_that_never_opens_is_a_different_finding_from_one_that_never_dispatches():
    """'Never opened' is a busy card, a held lease or a missing driver; 'opened and would not
    dispatch' is the wedge. Reporting them as one verdict sent every one of them to tt-smi -r."""
    p = _probe_with("import time\ntime.sleep(600)\n", timeout=1, dispatch_budget=1)
    assert not p and p.phase == "open", p
    assert "never opened" in p.why()


def test_a_probe_that_exits_without_dispatching_is_not_read_as_a_timeout():
    p = _probe_with("print('CARD_OPEN', flush=True)\nraise SystemExit(1)\n")
    assert not p and p.phase == "exit", p


def test_a_probe_that_dispatches_and_exits_in_the_same_breath_is_not_a_wedge():
    """The race the phase loop has to survive: the child prints CARD_HEALTHY and exits before the
    reader thread has drained the pipe. Reading process-exit first scores a healthy card wedged,
    and a wedged card gets tt-smi -r, which on a p300c takes the board pair down."""
    for _ in range(12):
        p = _probe_with("print('CARD_OPEN')\nprint('CARD_HEALTHY')\n")
        assert p and p.phase == "done", f"a healthy probe read as {p.phase}"


def test_a_chatty_probe_does_not_deadlock_on_its_own_stderr():
    """A ttnn process writes ~50 lines of driver log per device open, on stderr. A pipe nobody
    drains fills and blocks the child, so the probe would hang inside the very phase it times."""
    body = ("import sys\n"
            "for i in range(20000): print('driver log line %d' % i, file=sys.stderr)\n"
            "print('CARD_OPEN', flush=True)\n"
            "print('CARD_HEALTHY', flush=True)\n")
    t0 = time.monotonic()
    p = _probe_with(body, timeout=90, dispatch_budget=60)
    assert p and p.phase == "done", f"{p.phase}: a chatty child blocked on its own output"
    assert time.monotonic() - t0 < 60


def test_the_dispatch_budget_is_justified_by_a_measured_healthy_cost():
    """It cannot just be tightened to taste: a false 'cannot dispatch' triggers tt-smi -r, which
    on a p300c resets the board pair and can kill a sibling leg's card. The number is safe because
    the phase split puts every host-side cost before it, and the comment has to say so."""
    import inspect
    src = inspect.getsource(cg)
    doc = src.split("_DISPATCH_BUDGET_S")[0].rsplit("#:", 1)[-1] + src.split(
        "_DISPATCH_BUDGET_S = ")[0].split("#: Seconds allowed")[-1]
    assert cg._DISPATCH_BUDGET_S >= 30, "too tight to absorb any variance at all"
    assert "0.726" in doc and "p150a" in doc, (
        "the dispatch budget must cite the measured healthy probe cost it is derived from")
    assert "tt-smi -r" in doc, "the cost of a false negative must be written next to the number"


def test_the_probe_prints_its_open_marker_before_it_dispatches():
    """The marker has to sit between open_device and the first dispatch, or the split measures
    nothing: after the dispatch it never prints on a wedged chip, before the open it charges the
    import to the dispatch budget."""
    lines = [l for l in cg._CARD_PROBE.splitlines() if l.strip()]
    opened = next(i for i, l in enumerate(lines) if "open_device" in l)
    marker = next(i for i, l in enumerate(lines) if "CARD_OPEN" in l)
    dispatch = next(i for i, l in enumerate(lines) if "ttnn.add" in l)
    assert opened < marker < dispatch, cg._CARD_PROBE
    assert "flush=True" in lines[marker], "an unflushed marker arrives after the hang it precedes"


# --- never reset a card somebody else is computing on -------------------------------------------
#
# The fleet dispatcher granted card 0 on pc to this gate AND to worker:ceiling-rfd3 at the same
# time, mid-campaign. tt-bio's own lease handled the collision correctly: the residency leg was
# refused at the device open and the cell scored CONTENDED, which is a non-result and not a failed
# bar. What was not handled is the step after it. A contended card fails the health probe, a failed
# probe reads as a wedge, and a wedge gets `tt-smi -r` -- which on a p300c takes the board pair
# down, and on any card destroys whatever the other worker was measuring.

def _lease(tmp_path, monkeypatch, **fields):
    d = tmp_path / "leases"
    d.mkdir(exist_ok=True)
    meta = {"host": "pc", "card": "0", "holder": "worker:someone-else",
            "pid": os.getpid(), "acquired": 0.0, "released": None}
    meta.update(fields)
    (d / "pc-card0.json").write_text(json.dumps(meta))
    monkeypatch.setattr(cg.device_lease, "lease_dir", lambda: str(d))
    return cg.Worker("pc", 0, True)


def test_a_card_leased_by_another_worker_is_never_reset(tmp_path, monkeypatch):
    w = _lease(tmp_path, monkeypatch)
    monkeypatch.setenv("TT_BIO_LEASE_HOLDER", "worker:this-gate")
    assert cg.co_tenant(w) == f"worker:someone-else (pid {os.getpid()})"

    reset = []
    monkeypatch.setattr(cg.subprocess, "run", lambda *a, **k: reset.append(a) or None)
    monkeypatch.setattr(cg, "probe_card", lambda *a, **k: pytest.fail(
        "the lease already answered this; probing costs 300 s to learn nothing"))
    ok, how = cg.recover_card(w, [w])
    assert not ok and not reset, "the gate reset a card another worker was computing on"
    assert "worker:someone-else" in how and "must not reset" in how


def test_our_own_lease_is_not_a_co_tenant(tmp_path, monkeypatch):
    """A straggler of ours holding the lease on a wedged chip is exactly what a reset is for.
    Reading every lease as a co-tenant would disable recovery entirely, and post-OOM recovery is
    what keeps one real FAIL from wrapping itself in a cascade of invented ones."""
    w = _lease(tmp_path, monkeypatch, holder="worker:this-gate")
    monkeypatch.setenv("TT_BIO_LEASE_HOLDER", "worker:this-gate")
    assert cg.co_tenant(w) is None


def test_a_stale_lease_is_not_a_co_tenant(tmp_path, monkeypatch):
    """A released lease, or one whose holder is gone, must not stand in the way of a reset: a
    process killed on a wedged card leaves exactly that behind, and it is the case recovery
    exists for."""
    monkeypatch.setenv("TT_BIO_LEASE_HOLDER", "worker:this-gate")
    assert cg.co_tenant(_lease(tmp_path, monkeypatch, released=1.0)) is None
    dead = 2 ** 22 - 1                                     # above the default pid_max
    assert cg.co_tenant(_lease(tmp_path, monkeypatch, pid=dead)) is None
    assert cg.co_tenant(cg.Worker("no-such-host", 9, True)) is None, "absent lease file"


def test_the_preflight_tells_the_operator_to_wait_not_to_reset(tmp_path, monkeypatch):
    """The old message on an unusable card was "cannot dispatch a trivial program. Reset
    (tt-smi -r) and re-run", which against a co-tenant is a direct instruction to break the other
    job."""
    import inspect
    src = inspect.getsource(cg.main)
    busy = next(l for l in src.splitlines() if l.strip().startswith("busy ="))
    assert "co_tenant(w)" in busy and "for w in workers" in busy
    after = src.split("busy =")[1]
    assert "do NOT reset it" in after.split("sick =")[0]
    assert after.index("if busy:") < after.index("sick ="), (
        "the co-tenant check must come BEFORE the probe, or the gate pays 300 s per card to learn "
        "what the lease file already said")


# ---------------------------------------------------------------------------------------------
# Defect 22: a screen sweep could overwrite a Tier 2 PASS with INCONCLUSIVE and print success.
# The campaign shell avoided it by never passing --record to the warm sweep, which is a rule
# living in a comment. The screen is the cheap leg, so it is the one that gets re-run.
# ---------------------------------------------------------------------------------------------


def _cell_report(model, verdict, decided_by, bar=None, **kw):
    return {
        "bar_tokens": cg.TOKEN_BAR if bar is None else bar,
        "started": "2026-09-07T00:00:00Z", "tree": "deadbeef", "dirty": False,
        "geometry": {"dram_banks": 8, "board_type": "p150a"}, "reductions": {},
        "results": [dict({"model": model, "verdict": verdict, "decided_by": decided_by,
                          "tokens_requested": cg.TOKEN_BAR}, **kw)],
    }


def test_a_screen_sweep_does_not_overwrite_a_residency_pass(tmp_path, monkeypatch):
    """The exact run that would do it: `--tier screen` over the roster, then `--record`. A clean
    screen is INCONCLUSIVE by design, so every PASS in the file becomes a non-verdict and the
    most expensive results in the baseline are gone with a success message printed."""
    monkeypatch.setattr(cg, "BASELINE", tmp_path / "baseline.json")
    cg.record_baseline(_cell_report("openbind", "PASS", "residency",
                                    dram_peak_bytes=6203490304, wall_s=280.3), partial=True)
    msg = cg.record_baseline(_cell_report("openbind", "INCONCLUSIVE", "screen", wall_s=90.5),
                             partial=True)
    cell = _cells(tmp_path / "baseline.json")["openbind"]
    assert cell["verdict"] == "PASS", (
        "a screen cell overwrote a residency PASS; the run that decided nothing replaced the one "
        "that decided")
    assert cell["dram_peak_bytes"] == 6203490304, "the PASS survived but its measurement did not"
    assert "openbind" in msg and "kept" in msg, (
        f"the cell was kept silently, which reads exactly like it was updated: {msg!r}")


def test_a_full_roster_screen_record_does_not_wipe_every_pass(tmp_path, monkeypatch):
    """partial=False empties `cells` before the merge, so the prior cell has to be read off the
    file rather than off the working dict. A whole-roster screen sweep is the worst case."""
    monkeypatch.setattr(cg, "BASELINE", tmp_path / "baseline.json")
    cg.record_baseline(_cell_report("esmc-6b", "PASS", "residency", dram_peak_bytes=12789007360),
                       partial=True)
    cg.record_baseline(_cell_report("esmc-6b", "INCONCLUSIVE", "screen"), partial=False)
    assert _cells(tmp_path / "baseline.json")["esmc-6b"]["verdict"] == "PASS", \
        "a full-roster screen record wiped a residency PASS"


def test_a_card_that_was_never_available_does_not_erase_a_verdict(tmp_path, monkeypatch):
    """CONTENDED and CARD_DIRTY are cells where no model code ran at all. openbind's Tier 2 stage
    returned CONTENDED for real this campaign, with another worker holding card 0."""
    monkeypatch.setattr(cg, "BASELINE", tmp_path / "baseline.json")
    cg.record_baseline(_cell_report("rf3", "FAIL", "screen", mechanism="dram",
                                    alloc_ceiling_tokens=1088), partial=True)
    for dud in ("CONTENDED", "CARD_DIRTY"):
        cg.record_baseline(_cell_report("rf3", dud, "screen"), partial=True)
        cell = _cells(tmp_path / "baseline.json")["rf3"]
        assert cell["verdict"] == "FAIL" and cell["alloc_ceiling_tokens"] == 1088, (
            f"{dud} erased a measured FAIL and the ceiling the bisect spent hours on")


def test_a_leg_that_never_opened_the_device_writes_no_cell(tmp_path, monkeypatch):
    """The other half of the arm above, and the one that actually bit.

    That arm covers a non-measurement arriving over a measurement. This is a non-measurement
    arriving over NOTHING, which is what a re-measure of a dropped cell always is: chain7 re-ran
    opendde on 2026-09-10, the leg died at firmware init in 27 min without running a line of
    model code, and `--record` filed CARD_DIRTY as the model's cell. `_would_lose_evidence`
    returns False when there is no prior cell, so the guard that exists for exactly this verdict
    waved it through.
    """
    monkeypatch.setattr(cg, "BASELINE", tmp_path / "baseline.json")
    for dud in ("CONTENDED", "CARD_DIRTY"):
        msg = cg.record_baseline(_cell_report("opendde", dud, "residency", wall_s=16.6),
                                 partial=True)
        assert "opendde" not in _cells(tmp_path / "baseline.json"), (
            f"{dud} was written as opendde's cell; the card never opened, so this publishes the "
            f"absence of a measurement where the coverage check reads a measurement")
        assert "opendde" in msg and "no cell was written" in msg, (
            f"the cell was dropped silently, which reads like it was recorded: {msg!r}")


def test_a_cell_the_card_never_produced_is_not_coverage(tmp_path, monkeypatch):
    """`baseline_gaps` asked whether a cell was PRESENT, so a CARD_DIRTY cell answered it.

    Both directions, because a gap check that reports everything is as useless as one that
    reports nothing: the model with a real PASS must stay out of the list.
    """
    monkeypatch.setattr(cg, "BASELINE", tmp_path / "baseline.json")
    monkeypatch.setattr(cg, "runnable", lambda: ["boltz2", "opendde"])
    cg.record_baseline(_cell_report("boltz2", "PASS", "residency", wall_s=444.8), partial=True)
    assert cg.baseline_gaps() == ["p150a/opendde"], (
        f"a model with no cell at all is the case this check was built for: "
        f"{cg.baseline_gaps()}")
    # Now give opendde a cell the card never produced. Presence must not close the gap.
    raw = json.loads((tmp_path / "baseline.json").read_text())
    raw["cards"]["p150a"]["cells"]["opendde"] = {"verdict": "CARD_DIRTY",
                                                 "tokens_requested": cg.TOKEN_BAR}
    (tmp_path / "baseline.json").write_text(json.dumps(raw))
    assert cg.baseline_gaps() == ["p150a/opendde"], (
        "a CARD_DIRTY cell closed the coverage gap, so a model whose leg never opened the "
        "device reads as measured")
    # The control in the other direction: a real verdict there does close it.
    raw["cards"]["p150a"]["cells"]["opendde"]["verdict"] = "FAIL"
    (tmp_path / "baseline.json").write_text(json.dumps(raw))
    assert cg.baseline_gaps() == [], \
        "a measured FAIL is coverage and must not be reported as a gap"


def test_a_real_verdict_still_replaces_whatever_stands(tmp_path, monkeypatch):
    """The control. Keeping the stronger cell must not turn the baseline read-only: a re-measured
    verdict, including one that goes PASS -> FAIL, has to land. Otherwise a model that regresses
    keeps publishing a ceiling it no longer walks."""
    monkeypatch.setattr(cg, "BASELINE", tmp_path / "baseline.json")
    cg.record_baseline(_cell_report("protenix-v2", "PASS", "residency"), partial=True)
    cg.record_baseline(_cell_report("protenix-v2", "FAIL", "residency", mechanism="dram"),
                       partial=True)
    assert _cells(tmp_path / "baseline.json")["protenix-v2"]["verdict"] == "FAIL", \
        "the guard froze the cell instead of protecting it"
    # And an INCONCLUSIVE over an INCONCLUSIVE is a legitimate refresh, not a downgrade.
    cg.record_baseline(_cell_report("openfold3", "INCONCLUSIVE", "screen", wall_s=87.3),
                       partial=True)
    cg.record_baseline(_cell_report("openfold3", "INCONCLUSIVE", "screen", wall_s=12.0),
                       partial=True)
    assert _cells(tmp_path / "baseline.json")["openfold3"]["wall_s"] == 12.0, \
        "a same-strength re-measurement was refused"


#: Quoted verbatim from work/resid_opendde_1536.log, qb2 card 3, 2026-09-10T02:31:24Z. The whole
#: log is 73 lines: the banner and this throw at the top, then the outer CLI traceback.
_P300C_FW_INIT_OPEN_FAIL = """tt-bio worker tt-quietbox2:tt3: device open failed
Traceback (most recent call last):
  File "tt_bio/worker.py", line 1793, in run_worker_loop
    _get_device()
  File "tt_bio/tenstorrent.py", line 4005, in _open_device_locked
    dev = ttnn.open_device(device_id=device_id, **kwargs)
RuntimeError: TT_THROW @ /project/tt_metal/impl/device/firmware/risc_firmware_initializer.cpp:1115: tt::exception
info:
Device 0 init: failed to initialize FW! Try resetting the board.
backtrace:
 --- tt::tt_metal::RiscFirmwareInitializer::initialize_and_launch_firmware(int)
RuntimeError: every local worker exited before the run finished (SpawnProcess-1 exit 1); no job can be served. The worker's own traceback above says why."""

#: Same cause, different message: work/resid_protenix-v1_1536.log, 2026-09-10T04:35:08Z.
_P300C_SYSMEM_PIN_OPEN_FAIL = """tt-bio worker tt-quietbox2:tt3: device open failed
Traceback (most recent call last):
  File "tt_bio/tenstorrent.py", line 4005, in _open_device_locked
    dev = ttnn.open_device(device_id=device_id, **kwargs)
RuntimeError: TT_THROW @ /project/tt_metal/third_party/umd/device/chip_helpers/silicon_sysmem_manager.cpp:326: tt::exception
info:
Proceeding could lead to undefined behavior
backtrace:
 --- tt::umd::SiliconSysmemManager::pin_or_map_iommu()
 --- tt::umd::LocalChip::start_device()"""


def test_a_leg_whose_worker_never_opened_the_device_decided_nothing():
    """Three of the four p300c FAIL cells recorded on qb2 on 2026-09-10 were this: opendde,
    opendde-abag and protenix-v1 all came back FAIL at the 1536 bar in 9.9-16.6 s with
    mechanism null, 0 progress events and 0 blocks instrumented, and their logs say `device open
    failed` on line 1. No model code ran at any of them.

    The gate already had the right idea for this -- `contention` exists because "a killed leg
    whose spawned fold worker outlived the kill keeps the lease" is not a capacity result -- but
    it only knew the one message. A p300c whose previous leg was killed on the stall timeout does
    not report a busy device; it throws out of ttnn.open_device at firmware init or at the sysmem
    pin. So the arm matches the PHASE, which every future open-time error also has.
    """
    for name, log in (("firmware init", _P300C_FW_INIT_OPEN_FAIL),
                      ("sysmem pin", _P300C_SYSMEM_PIN_OPEN_FAIL)):
        assert cg.classify(log) == "card", (
            f"a {name} failure at device open classified as {cg.classify(log)!r}, so the leg "
            f"scores as a capacity FAIL and the cell publishes a wall nobody walked")
        assert cg.classify(log) not in cg.ALLOC_MECHANISMS, \
            f"a {name} failure at device open counts as an allocator refusal"
    assert cg.NOTHING_RAN["card"] == "CARD_DIRTY"
    assert cg.NOTHING_RAN_VERDICTS <= cg.UNDECIDED, (
        "a verdict that means no model code ran is not in UNDECIDED, so it can still decide a "
        "bar")


def test_a_real_allocator_refusal_still_outranks_the_card_arm():
    """The negative control the `card` arm needs. It must not swallow a leg that opened the card,
    ran, and then hit the allocator -- which is the only thing a ceiling is allowed to be made
    of. The arm sits after the allocator rows for exactly this reason."""
    real = ("Out of Memory: Not enough space to allocate 4831838208 B DRAM buffer across 8 "
            "banks, where each bank needs to store 603979776 B, but bank size is 4278190016 B")
    assert cg.classify(real) == "dram"
    # And a log holding both -- a leg that opened, ran, and was killed after its refusal -- is
    # still the allocator's statement, because that is the one that reached the model.
    assert cg.classify(real + "\n" + _P300C_FW_INIT_OPEN_FAIL) == "dram", \
        "a measured DRAM refusal was demoted to a card-dirty leg"


def test_a_rung_that_lost_the_card_moves_neither_bisect_bound():
    """opendde-abag's cell reported "1344 is the largest bucket-aligned size whose shapes ALLOCATE
    (screen); 1376 is the smallest that does not". The 1376 screen ran 15.6 s and never opened
    the device. The ceiling was read off a size the allocator was never asked about.

    So: recover and re-walk the rung once, and if the card is still gone, walk past it. It must
    not lower alloc_hi (a wall that was never measured) and must not raise alloc_lo (shapes that
    were never allocated)."""
    top = cg.BISECT_RUNGS[0]
    below = cg.BISECT_RUNGS[1]
    _, rec, calls = _bisect_probe(screen_sequence={top: ["CARD_DIRTY"]},
                                  residency_verdicts={below: "PASS"})
    assert [c for c in calls if c == ("screen", top)][1:], \
        f"the rung that lost the card was never re-walked: {calls}"
    note = rec["alloc_ceiling_note"]
    assert f"{top} is the smallest that does not" not in note, (
        f"a rung whose worker never opened the device was published as the wall: {note}")
    assert rec["alloc_ceiling_tokens"] != top, (
        f"a rung whose worker never opened the device was published as allocating: {note}")


def test_a_rung_that_really_failed_to_allocate_is_still_the_wall():
    """The other direction of the same guard: an identical walk where the top rung genuinely got
    an allocator refusal must still name it. Otherwise the fix above just blinds the bisect."""
    top = cg.BISECT_RUNGS[0]
    below = cg.BISECT_RUNGS[1]
    _, rec, _ = _bisect_probe(screen_sequence={top: ["FAIL"]},
                              residency_verdicts={below: "PASS"}, screen_mechanism="dram")
    assert f"{top} is the smallest that does not" in rec["alloc_ceiling_note"], (
        f"a real allocator refusal at {top} stopped being the wall: "
        f"{rec['alloc_ceiling_note']}")


def test_a_failed_leg_records_the_first_error_it_threw():
    """A leg that dies in the spawned fold worker records `mechanism: null` and a 25-line tail
    holding only the outer click traceback, whose last line is literally "The worker's own
    traceback above says why". The cause is on line 1 of the log and the report threw it away, so
    all three of Finding 9's cells said FAIL without saying why.
    """
    fw = cg.first_error(_P300C_FW_INIT_OPEN_FAIL)
    assert fw and "risc_firmware_initializer.cpp:1115" in fw, (
        f"the firmware-init throw did not survive into the record: {fw!r}")
    assert "failed to initialize FW" in fw, (
        f"the sentence under `info:` is the human-readable half and was dropped: {fw!r}")
    pin = cg.first_error(_P300C_SYSMEM_PIN_OPEN_FAIL)
    assert pin and "silicon_sysmem_manager.cpp:326" in pin, (
        f"the sysmem pin throw did not survive into the record: {pin!r}")
    # It is the FIRST error, not the loudest or the last: the outer wrapper says nothing useful.
    for got in (fw, pin):
        assert "every local worker exited" not in got

    # A capacity refusal is the case this must not garble, because that one becomes a ceiling.
    oom = cg.first_error(
        "RuntimeError: Out of Memory: Not enough space to allocate 4831838208 B DRAM buffer "
        "across 8 banks")
    assert oom and "4831838208 B DRAM" in oom, oom
    assert cg.first_error("nothing was thrown here at all") is None
