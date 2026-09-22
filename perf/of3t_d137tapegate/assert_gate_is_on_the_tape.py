#!/usr/bin/env python3
"""Every host float64 softmax entry point must open on the tape hook, in the COMPOSED tree.

A default is only the shipped answer, and `host_f64_softmax_site` reads an environment variable a
person can set. `assert_new_levers_default_off.py` checks the default and the call sites; this
checks the necessary condition beside it, which is the one D137 added: the path opens on
`ops.host_softmax_hook()`, a slot only `autograd.install` fills, so an inference fold has no route
to it whatever `TT_BIO_HOST_F64_SOFTMAX_AB` says.

WIDENED by `of3t-f64route`: the gate itself now lives in `host_softmax_or_none`, because the
fp32-softmax attention tail runs `softmax_in_place` and cannot call `site_softmax`, and D225 is
what a second entry point that skips the gate costs. So this checks the gate function, checks
that `site_softmax` and `_fp32_softmax_tail` both go through it, and checks that no caller of
the hook reads it anywhere else -- one gate, and every route arriving at it.

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


GATE = "host_softmax_or_none"

#: Every function that may route a construction site to the host float64 softmax. Each one must
#: reach it through `GATE` and never through the hook or the tape directly. `_fp32_softmax_tail`
#: is on the list because it is the route D225 was missing: eight sites take it and it consulted
#: nothing.
ROUTES = ("site_softmax", "_fp32_softmax_tail")


def _body(text, name):
    """A function's CODE: its docstring and its comments removed.

    Both of these functions describe the hook in prose, and a check that greps the raw text
    reads that prose as a call -- `field-edit-anchored-on-index-matches-the-name-in-prose`, the
    same failure one level down. What the gate is about is what the code does.
    """
    m = re.search(r"\ndef %s\(.*?\n(?=\n\ndef )" % re.escape(name), text, re.S)
    if not m:
        return None
    body = re.sub(r'"""(?:.|\n)*?"""', "", m.group(0))
    return "\n".join(re.sub(r"\s#.*$", "", ln) for ln in body.split("\n")
                     if not ln.lstrip().startswith("#"))


def gate_reads_the_hook(text):
    """The one gate must be read from the hook, and must not read the tape by importing it."""
    body = _body(text, GATE)
    if body is None:
        return "%s is not in tenstorrent.py" % GATE
    if "host_softmax_hook()" not in body:
        return ("%s does not consult ops.host_softmax_hook(), so the site selector alone "
                "decides the path and an inference fold can reach it" % GATE)
    if re.search(r"from \.(autograd|taped_ttnn)|from \. import (autograd|taped_ttnn)", body):
        return ("%s imports the tape directly, which puts the training stack back on every "
                "model's inference path" % GATE)
    return None


def every_route_goes_through_the_gate(text):
    """A route that reads the hook itself is a second gate, and a second gate goes stale."""
    for name in ROUTES:
        body = _body(text, name)
        if body is None:
            return "%s is not in tenstorrent.py" % name
        if "host_softmax_hook()" in body:
            return ("%s reads ops.host_softmax_hook() itself instead of going through %s, so "
                    "the census and the gate now live in two places" % (name, GATE))
        if GATE + "(" not in body:
            return ("%s never consults %s, so a site that selected the host float64 softmax "
                    "cannot reach it from there -- this is D225 exactly" % (name, GATE))
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
_GOOD_GATE = ("\ndef host_softmax_or_none(host_f64):\n"
              "    from . import ops\n"
              "    return ops.host_softmax_hook() if host_f64 else None\n"
              "\n\ndef next_one():\n    pass\n")
_GOOD_ROUTES = ("\ndef site_softmax(x, dim=-1, *, host_f64=False, **kw):\n"
                "    h = host_softmax_or_none(host_f64)\n"
                "    return h(x, dim) if h else ttnn.softmax(x, dim=dim)\n"
                "\n\ndef _fp32_softmax_tail(sc0, host_f64=False):\n"
                "    h = host_softmax_or_none(host_f64)\n"
                "    return h(sc0, -1) if h else ttnn.softmax_in_place(sc0)\n"
                "\n\ndef next_one():\n    pass\n")
_probe = [
    # the gate importing the tape instead of reading the slot
    gate_reads_the_hook(
        "\ndef host_softmax_or_none(host_f64):\n"
        "    from .autograd import host_f64_softmax\n"
        "    return host_f64_softmax if host_f64 else None\n"
        "\n\ndef next_one():\n    pass\n"),
    # D225's own shape: the tail takes the route and never consults the gate
    every_route_goes_through_the_gate(
        _GOOD_GATE + _GOOD_ROUTES.replace(
            "    h = host_softmax_or_none(host_f64)\n"
            "    return h(sc0, -1) if h else ttnn.softmax_in_place(sc0)\n",
            "    return ttnn.softmax_in_place(sc0)\n")),
    # a second gate: the route reads the hook itself
    every_route_goes_through_the_gate(
        _GOOD_GATE + _GOOD_ROUTES.replace(
            "    h = host_softmax_or_none(host_f64)\n"
            "    return h(sc0, -1) if h else ttnn.softmax_in_place(sc0)\n",
            "    h = ops.host_softmax_hook()\n"
            "    return h(sc0, -1) if h else ttnn.softmax_in_place(sc0)\n")),
    hook_needs_a_live_tape(
        "\ndef host_softmax_hook():\n    return _HOST_SOFTMAX\n\n\n"),
]
# ... and the positive control, so a check that fires on everything is caught too.
_clean = _GOOD_GATE + _GOOD_ROUTES
if gate_reads_the_hook(_clean) or every_route_goes_through_the_gate(_clean):
    print("REFUSING: the checks fire on a tree that has no defect, so a failure on the real "
          "one would say nothing")
    raise SystemExit(2)
if not all(_probe):
    print("REFUSING: the checks do not fire on a known-bad tree, so their silence on the real "
          "one is uninformative")
    raise SystemExit(2)

tens = ROOT / "tt_bio/tenstorrent.py"
ops = ROOT / "tt_bio/ops.py"
if not tens.is_file() or not ops.is_file():
    print("cannot read the engine files under %s" % ROOT)
    raise SystemExit(2)

fail = [e for e in (gate_reads_the_hook(tens.read_text()),
                    every_route_goes_through_the_gate(tens.read_text()),
                    hook_needs_a_live_tape(ops.read_text())) if e]
if fail:
    print("THE HOST FLOAT64 SOFTMAX IS REACHABLE FROM INFERENCE:")
    for f in fail:
        print("  " + f)
    raise SystemExit(1)
print("the host float64 softmax opens only on an installed tape, through one gate "
      "(%d negative probes fired, positive control clean)" % len(_probe))
