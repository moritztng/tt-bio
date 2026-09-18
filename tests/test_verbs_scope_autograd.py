"""A tt-bio verb must not turn off its caller's autograd.

``torch.set_grad_enabled(False)`` is process-wide and nothing restores it. In a
standalone ``tt-bio predict`` process that is harmless. Reached in-process -- from a
script, a notebook or a test -- it is permanent: the caller loses autograd for the rest
of its life. The shipped verbs did exactly that, and the casualty was
tests/test_train_interface.py's two gradcheck invariants, which failed in every
full-suite run and passed in isolation because tests/test_predict_exit_code.py leaks the
flag merely by invoking ``predict`` through CliRunner. It matters more now that the same
package is becoming a training framework: fold a structure, then train, and get no
gradients from a library call you had no reason to suspect.

Each case drives one entry point far enough to reach a collaborator it calls and stops
there. The collaborator records ``torch.is_grad_enabled()``, which gives both halves of
the contract from a single call:

* recorded False -- the body still runs with grad off, so no inference behaviour moves;
* True afterwards -- the setting stopped at the verb's edge.

Recording is also what keeps this honest. A verb that died before reaching its probe
leaves the recorder empty and fails here, rather than passing because nothing ran.
"""
from __future__ import annotations

import importlib

import pytest
import torch
from click.testing import CliRunner

from tt_bio.main import cli

_FASTA = ">a\nMKVL\n"
_YAML = "version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MKVL\n"


class _Stop(SystemExit):
    """Raised by a probe. The verb is answered; nothing past this point needs to run."""


def _probe(recorder):
    def probe(*_args, **_kwargs):
        recorder.append(torch.is_grad_enabled())
        raise _Stop("probe")
    return probe


# (id, argv, "module:attribute" the body calls, files the run needs).
# Each probe sits inside the verb: check_input and set_float32_matmul_precision at or
# before the line that used to set the flag, the rest immediately after it.
CLI_VERBS = [
    ("predict", ["predict", "in.yaml", "--accelerator", "cpu", "--out_dir", "o"],
     "torch:set_float32_matmul_precision", {"in.yaml": _YAML}),
    ("warmup", ["warmup", "--max_seq", "64", "--max_msa", "64"],
     "tt_bio.token_axis:msa_ladder", {}),
    ("embed", ["embed", "s.fasta", "--out_dir", "o"],
     "tt_bio.esmc:load_sequences", {"s.fasta": _FASTA}),
    ("affinity", ["affinity", "in.yaml", "--accelerator", "cpu", "--out_dir", "o"],
     "tt_bio.nesso1:screen", {"in.yaml": _YAML}),
    ("saprot", ["saprot", "s.fasta", "--out_dir", "o"],
     "tt_bio.saprot:load_sequences_with_structure", {"s.fasta": _FASTA}),
    ("design", ["design", "in.yaml", "--model", "rfd3", "--out_dir", "o"],
     "tt_bio.size_limits:check_input", {"in.yaml": "s1: {}\n"}),
]


@pytest.mark.parametrize("verb,argv,target,files", CLI_VERBS, ids=[c[0] for c in CLI_VERBS])
def test_cli_verb_scopes_autograd_to_its_own_body(verb, argv, target, files, monkeypatch):
    modname, attr = target.split(":")
    recorded: list[bool] = []
    monkeypatch.setattr(importlib.import_module(modname), attr, _probe(recorded))

    torch.set_grad_enabled(True)
    runner = CliRunner()
    with runner.isolated_filesystem():
        for name, text in files.items():
            with open(name, "w") as fh:
                fh.write(text)
        runner.invoke(cli, argv, catch_exceptions=True)

    assert recorded, (
        f"`tt-bio {verb}` never reached {target}, so this test proved nothing. Repoint the "
        f"probe at something the body still calls.")
    assert recorded[0] is False, (
        f"`tt-bio {verb}` ran its body with autograd ON. Inference wants it off; scoping the "
        f"flag must not have turned it off altogether.")
    assert torch.is_grad_enabled() is True, (
        f"`tt-bio {verb}` left torch's global autograd switch off. It is process-wide and "
        f"nothing restores it, so every caller downstream silently loses gradients.")


def test_pxdesign_run_design_scopes_autograd(monkeypatch):
    """tt_bio/pxdesign/design.py -- reached in-process by `tt-bio design --model pxdesign`."""
    from tt_bio.pxdesign import design as px
    import tt_bio.pxdesign.inputs as px_inputs

    recorded: list[bool] = []
    monkeypatch.setattr(px_inputs, "design_inputs_from_yaml", _probe(recorded))
    torch.set_grad_enabled(True)
    try:
        with pytest.raises(SystemExit):
            px.run_design("target.yaml", "o", None, 1, 1, 0)
    finally:
        after = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    assert recorded == [False], f"run_design body saw grad {recorded}, expected [False]"
    assert after is True, "run_design left the caller's autograd switch off"


def test_boltzgen_predict_run_scopes_autograd():
    """tt_bio/boltzgen/task/predict/predict.py -- the BoltzGen fold and inverse-fold steps."""
    from tt_bio.boltzgen.task.predict.predict import Predict

    recorded: list[bool] = []

    class _EmptySet:
        def __len__(self):
            recorded.append(torch.is_grad_enabled())
            return 0            # nothing to predict, so run() returns without a model

    class _Data:
        predict_set = _EmptySet()

    task = object.__new__(Predict)
    task.data = _Data()

    torch.set_grad_enabled(True)
    try:
        task.run()
    finally:
        after = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    assert recorded == [False], f"Predict.run body saw grad {recorded}, expected [False]"
    assert after is True, "Predict.run left the caller's autograd switch off"
