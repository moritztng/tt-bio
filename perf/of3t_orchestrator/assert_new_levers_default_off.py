#!/usr/bin/env python3
"""The two pass-207 repairs must be OFF by default in the composed tree.

Both are release-gated, both are real fixes, and both would change numbers a user gets if they
were live. `off by default` is a property of the COMPOSED BRANCH, not of a row's write-up, so it
is read from the tree the way `assert_confidence_forward_signature.py` reads its own.

  TT_BIO_SOFTMAX_BW_RENORM   of3t-apbgrad. Divides the softmax backward's `inner` by the row sum.
                             Takes the trunk's gradient from 9.025172e+00 to 3.833066e-01,
                             1.0251x upstream's own bf16. Backward only, so it cannot move a
                             shipped inference result -- but it DOES move every taped gradient,
                             which is most of what this campaign measures.
  host_f64_softmax_site      of3t-f64softmax. Sends the softmax to the host in float64. Moves
                             fold output at any site where it is on, and costs a round trip.

Two things are checked per lever and the second is the one that bites: that the default EXISTS as
False, and that no CALL SITE overrides it to True. A `default: bool = False` in the selector says
nothing if a construction site passes `default=True`, which is exactly how `opendde.refiner` ships
the accurate-softmax chain ON while its selector defaults off.

And one more for the float64 softmax, because a default is only the shipped answer and the
selector reads an environment variable a person can set. `site_softmax` must reach the path
through `ops.host_softmax_hook()`, which only `autograd.install` fills, so an inference fold has
no route to it whatever `TT_BIO_HOST_F64_SOFTMAX_AB` says (D137).
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")


def renorm_default_off(text):
    """`TT_BIO_SOFTMAX_BW_RENORM` must be read with an off default."""
    m = re.search(r'os\.environ\.get\(\s*"TT_BIO_SOFTMAX_BW_RENORM"\s*,\s*"([^"]*)"', text)
    if not m:
        return "TT_BIO_SOFTMAX_BW_RENORM is not read with a literal default in taped_ttnn.py"
    if m.group(1).strip().lower() not in ("0", "false", "off", ""):
        return f"TT_BIO_SOFTMAX_BW_RENORM defaults to {m.group(1)!r}, not off"
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


def site_softmax_gated_on_the_tape(text):
    """`site_softmax` must ask the hook, and must not import the tape to do it."""
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


def check(root):
    fail = []
    taped = (root / "tt_bio/taped_ttnn.py")
    tens = (root / "tt_bio/tenstorrent.py")
    if not taped.is_file() or not tens.is_file():
        return [f"cannot read the engine files under {root}"]
    e = renorm_default_off(taped.read_text())
    if e:
        fail.append(e)
    e = selector_default_off(tens.read_text())
    if e:
        fail.append(e)
    e = site_softmax_gated_on_the_tape(tens.read_text())
    if e:
        fail.append(e)
    srcs = [(p.name, p.read_text()) for p in sorted((root / "tt_bio").rglob("*.py"))]
    fail += [f"a construction site forces the host float64 softmax ON -- {b}" for b in host_f64_sites_off(srcs)]
    return fail


# Break control: the checks must FAIL on a tree that has the levers on, or their silence on the
# real tree is uninformative. A17 -- a negative control breaks exactly what the check reads.
_probe = [
    renorm_default_off('_X = os.environ.get("TT_BIO_SOFTMAX_BW_RENORM", "1").lower()'),
    selector_default_off("def host_f64_softmax_site(token: str, default: bool = True) -> bool:"),
    host_f64_sites_off([("probe.py", 'host_f64_softmax_site("x", default=True)')]),
    site_softmax_gated_on_the_tape(
        "\ndef site_softmax(x, dim=-1, *, host_f64=False, **kw):\n"
        "    from .autograd import host_f64_softmax\n"
        "    return host_f64_softmax(x, dim) if host_f64 else ttnn.softmax(x, dim=dim)\n"
        "\n\ndef next_one():\n    pass\n"),
]
if not all(_probe):
    print("REFUSING: the default-off checks do not fire on a known-on tree, so their silence on "
          "the real one is uninformative")
    raise SystemExit(2)

_fail = check(ROOT)
if _fail:
    print("SHIPPED DEFAULT MOVED -- a pass-207 repair is live in the composition:")
    for f in _fail:
        print("  " + f)
    raise SystemExit(1)
print("shipped defaults: TT_BIO_SOFTMAX_BW_RENORM off, host float64 softmax off at every site "
      "and gated on an installed tape (both probes fired)")
