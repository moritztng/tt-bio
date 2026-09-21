#!/usr/bin/env python3
"""`site_softmax` must reach the host float64 softmax through the tape hook, in the COMPOSED tree.

A default is only the shipped answer, and `host_f64_softmax_site` reads an environment variable a
person can set. `assert_new_levers_default_off.py` checks the default and the call sites; this
checks the necessary condition beside it, which is the one D137 added: the path opens on
`ops.host_softmax_hook()`, a slot only `autograd.install` fills, so an inference fold has no route
to it whatever `TT_BIO_HOST_F64_SOFTMAX_AB` says.

This lives in the row's own directory rather than in the orchestrator's gate, per STANDING pass
272: a row that can edit the gate checking its own lever does not have a gate. The state doc hands
the orchestrator the same check in the terms its file should read, and the orchestrator decides.

  usage: assert_gate_is_on_the_tape.py [tree]        default: the tree this file is in
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                    pathlib.Path(__file__).resolve().parents[2])


def site_softmax_gated_on_the_tape(text):
    """The gate must be read from the hook, and must not be read by importing the tape."""
    m = re.search(r"\ndef site_softmax\(.*?\n(?=\n\ndef )", text, re.S)
    if not m:
        return "site_softmax is not in tenstorrent.py"
    body = m.group(0)
    if "host_softmax_hook()" not in body:
        return ("site_softmax does not consult ops.host_softmax_hook(), so the site selector "
                "alone decides the path and an inference fold can reach it")
    if re.search(r"from \.(autograd|taped_ttnn)|from \. import (autograd|taped_ttnn)", body):
        return ("site_softmax imports the tape directly, which puts the training stack back on "
                "every model's inference path")
    return None


def hook_needs_a_live_tape(text):
    """The slot alone must not answer: `tape()` leaves it filled after the block closes."""
    m = re.search(r"\ndef host_softmax_hook\(.*?\n(?=\n\n)", text, re.S)
    if not m:
        return "ops.py has no host_softmax_hook"
    if "grad_hook()" not in m.group(0):
        return ("host_softmax_hook returns the slot without checking grad_hook(), so the path "
                "stays reachable for the rest of the process after a tape block closes")
    return None


# Break control: each check must FAIL on a tree that has the defect, or its silence on the real
# tree says nothing. A17 -- a negative control breaks exactly what the check reads.
_probe = [
    site_softmax_gated_on_the_tape(
        "\ndef site_softmax(x, dim=-1, *, host_f64=False, **kw):\n"
        "    from .autograd import host_f64_softmax\n"
        "    return host_f64_softmax(x, dim) if host_f64 else ttnn.softmax(x, dim=dim)\n"
        "\n\ndef next_one():\n    pass\n"),
    hook_needs_a_live_tape(
        "\ndef host_softmax_hook():\n    return _HOST_SOFTMAX\n\n\n"),
]
if not all(_probe):
    print("REFUSING: the checks do not fire on a known-bad tree, so their silence on the real "
          "one is uninformative")
    raise SystemExit(2)

tens = ROOT / "tt_bio/tenstorrent.py"
ops = ROOT / "tt_bio/ops.py"
if not tens.is_file() or not ops.is_file():
    print("cannot read the engine files under %s" % ROOT)
    raise SystemExit(2)

fail = [e for e in (site_softmax_gated_on_the_tape(tens.read_text()),
                    hook_needs_a_live_tape(ops.read_text())) if e]
if fail:
    print("THE HOST FLOAT64 SOFTMAX IS REACHABLE FROM INFERENCE:")
    for f in fail:
        print("  " + f)
    raise SystemExit(1)
print("the host float64 softmax opens only on an installed tape (both probes fired)")
