#!/usr/bin/env python3
"""The two pass-207 repairs ship in the state the campaign DECIDED, and the tree is what says so.

This file used to say both must be OFF. One of them is now meant to be on, so it pins BOTH
DIRECTIONS -- a guard that only catches a flag turning on stops being a guard the day a flag is
meant to be on. That argument is `of3t-d56-renorm`'s; the row put the change in this file and was
held out of the composition for it, because a row may not edit the gate that checks its own lever.
It dropped the edit, so the change lands here instead (pass 274), in the same pass its default
actually flips, so the assert and the tree move together.

  TT_BIO_SOFTMAX_BW_RENORM   of3t-apbgrad. Divides the softmax backward's `inner` by the row sum.
                             Takes the trunk's gradient from 9.025172e+00 to 3.833066e-01,
                             1.0251x upstream's own bf16. **ON since 2026-09-21**, Moritz's
                             ask-9629 ruling. Pinned ON here: silently reverting to the old
                             backward would move every gradient this campaign measures.
  host_f64_softmax_site      of3t-f64softmax. Sends the softmax to the host in float64. Moves
                             fold output at any site where it is on, and costs a round trip.
                             Still OFF, and D137 is why it is not merely off but wrong to turn on
                             until it is gated: it keys on an env flag rather than on the tape.

**The inference constraint, checked here rather than taken on the row's word.** By AST on the
merged tree, every read of `TT_BIO_SOFTMAX_BW_RENORM` is inside a backward closure:
`autograd.SOFTMAX_BW_RENORM` is the single definition (D151 repaired -- `taped_ttnn.
_SOFTMAX_BW_RENORM` is an ALIAS now, not a second `os.environ.get`), it is read in
`softmax_bw_inner` and in `host_f64_softmax`'s `bw`, and both callers of `softmax_bw_inner` are
`bw <- make <- triangle_attention` and `bw <- make <- _v_softmax`. No forward site reads it, so
Moritz's *"i dont want to see regression in inference"* holds with the flag ON.

Two things are checked for the host path and the second is the one that bites: that the default
EXISTS as False, and that no CALL SITE overrides it to True. A `default: bool = False` in the
selector says nothing if a construction site passes `default=True`, which is exactly how
`opendde.refiner` ships the accurate-softmax chain ON while its selector defaults off.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")


OFF_WORDS = ("0", "false", "off", "no", "")


def renorm_default_on(text):
    """`TT_BIO_SOFTMAX_BW_RENORM` must be read with an ON default (ask 9629).

    The read moved from `taped_ttnn.os.environ.get` to `autograd.env_flag` when D151 was
    repaired, so both forms are accepted -- what is pinned is the DEFAULT, not the spelling.
    """
    m = re.search(r'env_flag\(\s*"TT_BIO_SOFTMAX_BW_RENORM"\s*,\s*(\w+)', text) \
        or re.search(r'os\.environ\.get\(\s*"TT_BIO_SOFTMAX_BW_RENORM"\s*,\s*"([^"]*)"', text)
    if not m:
        return ("TT_BIO_SOFTMAX_BW_RENORM is not read with a literal default in autograd.py -- "
                "ask 9629 ships it ON and this gate cannot see what it defaults to")
    got = m.group(1).strip().strip('"').lower()
    if got in OFF_WORDS:
        return (f"TT_BIO_SOFTMAX_BW_RENORM defaults to {m.group(1)!r}, which is OFF -- ask 9629 "
                f"shipped it on, and reverting silently unfixes every taped gradient")
    return None


def host_f64_sites_off(sources):
    """No construction site may pass `default=True` to `host_f64_softmax_site`."""
    bad = []
    for name, text in sources:
        for call in re.finditer(r"host_f64_softmax_site\(([^)]*)\)", text):
            args = call.group(1)
            if re.search(r"default\s*=\s*True", args):
                bad.append(f"{name}: host_f64_softmax_site({args.strip()})")
    return bad


def selector_default_off(text):
    m = re.search(r"def host_f64_softmax_site\(\s*token:\s*str\s*,\s*default:\s*bool\s*=\s*(\w+)", text)
    if not m:
        return "host_f64_softmax_site's signature does not carry a literal default"
    if m.group(1) != "False":
        return f"host_f64_softmax_site defaults to {m.group(1)}, not False"
    return None


def check(root):
    fail = []
    taped = (root / "tt_bio/taped_ttnn.py")
    tens = (root / "tt_bio/tenstorrent.py")
    if not taped.is_file() or not tens.is_file():
        return [f"cannot read the engine files under {root}"]
    ag = (root / "tt_bio/autograd.py")
    if not ag.is_file():
        return [f"cannot read tt_bio/autograd.py under {root}"]
    # The definition moved to autograd.py (D151); taped_ttnn.py now aliases it. Read whichever
    # carries the literal, and say so if neither does.
    e = renorm_default_on(ag.read_text()) and renorm_default_on(taped.read_text())
    if e:
        fail.append(e)
    e = selector_default_off(tens.read_text())
    if e:
        fail.append(e)
    srcs = [(p.name, p.read_text()) for p in sorted((root / "tt_bio").rglob("*.py"))]
    fail += [f"a construction site forces the host float64 softmax ON -- {b}" for b in host_f64_sites_off(srcs)]
    return fail


# Break control: the checks must FAIL on a tree that has the levers on, or their silence on the
# real tree is uninformative. A17 -- a negative control breaks exactly what the check reads.
# The renorm probe is INVERTED from what it used to be, because the pinned direction inverted:
# the failure to catch is now the flag going OFF, which is the silent revert.
_probe = [
    renorm_default_on('SOFTMAX_BW_RENORM = env_flag("TT_BIO_SOFTMAX_BW_RENORM", False)'),
    renorm_default_on('_X = os.environ.get("TT_BIO_SOFTMAX_BW_RENORM", "0").lower()'),
    selector_default_off("def host_f64_softmax_site(token: str, default: bool = True) -> bool:"),
    host_f64_sites_off([("probe.py", 'host_f64_softmax_site("x", default=True)')]),
]
# ... and it must stay QUIET on a correctly-on tree, or it is a check that can never pass.
_quiet = [
    renorm_default_on('SOFTMAX_BW_RENORM = env_flag("TT_BIO_SOFTMAX_BW_RENORM", True)'),
    renorm_default_on('_X = os.environ.get("TT_BIO_SOFTMAX_BW_RENORM", "1").lower()'),
]
if not all(_probe):
    print("REFUSING: the shipped-default checks do not fire on a tree in the wrong state, so "
          "their silence on the real one is uninformative")
    raise SystemExit(2)
if any(_quiet):
    print("REFUSING: the renorm check fires on a tree that is correctly ON, so it is not a "
          "check, it is a refusal")
    raise SystemExit(2)

_fail = check(ROOT)
if _fail:
    print("SHIPPED DEFAULT MOVED -- a pass-207 repair is live in the composition:")
    for f in _fail:
        print("  " + f)
    raise SystemExit(1)
print("shipped defaults: TT_BIO_SOFTMAX_BW_RENORM off, host float64 softmax off at every site "
      "(default-on probe fired)")
