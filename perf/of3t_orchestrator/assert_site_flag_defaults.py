#!/usr/bin/env python3
"""Every construction site that ships a `_site_flag` lever ON by CALL-SITE default is pinned here.

WHY THIS EXISTS
---------------
`tt_bio.tenstorrent` has five per-site levers built on `_site_flag`:

    accurate_softmax_site   TT_BIO_ACCURATE_SOFTMAX_AB
    triatt_sdpa_hifi_site   TT_BIO_TRIATT_SDPA_HIFI_AB
    softmax_precise_site    TT_BIO_SOFTMAX_PRECISE_AB
    host_f64_softmax_site   TT_BIO_HOST_F64_SOFTMAX_AB
    sdpa_ragged_pad_site    TT_BIO_SDPA_RAGGED_PAD_AB

Each resolves PER CONSTRUCTION SITE and its default is an argument at the call, so **there is no
module-level constant holding the shipped value**. A lever census that walks known flags to their
resolved value therefore has no row for the family at all — it does not report "off", it reports
nothing. `openfold3.trunk` flipped to the fused HiFi SDPA path by default in `3a31dcdd1` (+11.564 s
at 512 aa, 1.5123x) and the census was silent, which is the fleet note this check comes from.

`assert_new_levers_default_off.py` covers ONE member of the family, `host_f64_softmax_site`, and
covers it well. This covers the other four the same way: by reading the CALL SITES.

WHAT IT CHECKS
--------------
A shrink-only ratchet over every call passing `default=True`. Three exist and each is pinned with
its model; a NEW one fails the compose, and a pinned one that goes away must be removed from the
list. It says nothing about whether ON is right — `opendde.refiner` shipping the accurate-softmax
chain ON is a deliberate, recorded choice. What it refuses is a default-ON site arriving unseen.

CPU only. Reads the composed tree it is pointed at.
"""
from __future__ import annotations

import ast
import pathlib
import sys

FLAGS = ("accurate_softmax_site", "triatt_sdpa_hifi_site", "softmax_precise_site",
         "host_f64_softmax_site", "sdpa_ragged_pad_site")

#: "file:callee" -> why it is on. Shrink-only. Verified at pass 301 against the composition.
PINNED = {
    "tt_bio/opendde.py:accurate_softmax_site":
        "opendde.refiner ships the accurate-softmax chain ON; the campaign's own "
        "assert_new_levers_default_off.py cites this site as the worked example of a selector "
        "defaulting off while a construction site passes default=True",
    "tt_bio/protenix.py:accurate_softmax_site":
        "two protenix sites (1467, 2511) ship it ON by call-site default. Not an OF3 model, so "
        "outside this campaign's scope to judge -- pinned so it cannot change unnoticed while "
        "protenix-v2 carries digest claims from D137/D155",
}


def default_true_sites(root: pathlib.Path):
    found = {}
    for f in sorted((root / "tt_bio").rglob("*.py")):
        try:
            tree = ast.parse(f.read_text(errors="replace"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            fn = n.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else None)
            if name not in FLAGS:
                continue
            for k in n.keywords:
                if k.arg == "default" and isinstance(k.value, ast.Constant) and k.value.value is True:
                    key = "%s:%s" % (f.relative_to(root), name)
                    found.setdefault(key, []).append(n.lineno)
    return found


def main(argv):
    root = pathlib.Path(argv[1] if len(argv) > 1 else ".").resolve()
    found = default_true_sites(root)

    # Probe: a synthetic default=True must be seen, and default=False must not.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        t = pathlib.Path(td); (t / "tt_bio").mkdir()
        (t / "tt_bio" / "p.py").write_text(
            "a = accurate_softmax_site('x', default=True)\n"
            "b = accurate_softmax_site('y', default=False)\n"
            "c = accurate_softmax_site('z')\n")
        got = default_true_sites(t)
        if "tt_bio/p.py:accurate_softmax_site" not in got:
            print("BROKEN a default=True call site is not seen", file=sys.stderr)
            return 2
        if len(got["tt_bio/p.py:accurate_softmax_site"]) != 1:
            print("BROKEN default=False or a bare call counted as ON", file=sys.stderr)
            return 2

    new = sorted(k for k in found if k not in PINNED)
    gone = sorted(k for k in PINNED if k not in found)
    if new:
        for k in new:
            print("  DRIFT %s ships a site lever ON by call-site default at line(s) %s, unpinned"
                  % (k, ", ".join(str(x) for x in found[k])))
        print("FAIL %d construction site(s) newly default-ON. A per-site default has no module "
              "constant, so nothing else sees it." % len(new))
        return 1
    if gone:
        for k in gone:
            print("  DRIFT %s is pinned as default-ON and no longer is -- remove it" % k)
        print("FAIL the ratchet has %d stale entr(y/ies); it may only shrink" % len(gone))
        return 1
    n_lines = sum(len(v) for v in found.values())
    print("ok    %d construction site(s) across %d file/lever pair(s) ship a `_site_flag` lever ON "
          "by call-site default, all pinned (probe fires on a synthetic one; default=False and a "
          "bare call do not)" % (n_lines, len(found)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
