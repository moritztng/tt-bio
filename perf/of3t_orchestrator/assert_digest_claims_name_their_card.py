#!/usr/bin/env python3
"""An artifact claiming digest equality must identify the HOST it ran on, and must not be pc card 0.

WHY THIS EXISTS (D155)
----------------------
`of3t-d137ab` reported `protenix-v2` moving between folds at a fixed seed and I filed it as a
USER-FACING defect against the model. It is **pc card 0**, a faulty card root-caused on 2026-08-17
by a row literally called `protenix-v2-nondeterminism-rootcause`: the fault is matmul-only,
location-keyed, probabilistic, at every size, with a matched qb1 control 15/15 clean. The standing
instruction is that **pc card 0 must not host hash-equality or bit-exact gating at any size, for
any model**, and that a clean run on it proves nothing about the next one.

Nothing in the artifact would have stopped me. It records `"card": 0` and no host -- and card 0 is
a different piece of hardware on pc, qb1 and qb2. **A card number without a host is not an
identification**, which is the smaller lesson inside the bigger one.

WHAT IT CHECKS
--------------
For every of3t JSON carrying a digest-equality field:

  1. the HOST is identifiable -- a `host`/`hostname`/`machine` field, or a card string that names
     one (`of3t_auxfind` writes "tt-quietbox2 card 0, Blackhole p300c", which identifies fine);
  2. it is not pc card 0.

Measured before shipping: 6 of3t artifacts carry such a field, 3 identify their host and 3 do not
-- and those 3 are exactly the ones this defect is about. They belong to a CONCLUDED row, so they
are FROZEN rather than rewritten: an orchestrator that edits another row's evidence to make its own
check pass has broken something worse than the check. A NEW anonymous digest claim fails, and the
list may only shrink -- a frozen entry that gets fixed and stays listed is also a failure.

What it does NOT do is judge whether the digests are right. It refuses to let a digest claim be
anonymous about the hardware that produced it, which is the one thing that would have caught D155.

CPU only. Reads the composed tree it is pointed at.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

DIGEST_FIELDS = ("base_equals_off", "base_equals_on", "digest_stable_within_arm", "digests_equal",
                 "byte_identical", "bit_identical", "digest_identical")
HOST_FIELDS = ("host", "hostname", "machine")
KNOWN_HOSTS = ("pc", "qb1", "qb2", "tt-quietbox", "tt-quietbox2", "whglx", "galaxy")
#: The exclusion. Root-caused 2026-08-17; matmul-only, location-keyed, probabilistic, every size.
BANNED = re.compile(r"\bpc\b[^\n]{0,24}\bcard\s*0\b|\bpc-card0\b", re.I)

#: path -> why frozen. Shrink-only. These three are `of3t-d137ab`'s, which CONCLUDED; its own
#: state doc names pc card 0 in prose (line 3), so the hardware is recorded where a reader looks
#: and missing only from the machine-readable half. D155's correction carries the consequence:
#: the timing half stands, the digest half wants one re-run on a clean card.
FROZEN = {
    "perf/of3t_d137ab/INFERENCE_AB_openfold3.json": "of3t-d137ab, concluded; pc card 0",
    "perf/of3t_d137ab/INFERENCE_AB_protenix-v2.json": "of3t-d137ab, concluded; pc card 0",
    "perf/of3t_d137ab/INFERENCE_AB_opendde.json": "of3t-d137ab, concluded; pc card 0",
}


def host_of(d):
    for k in HOST_FIELDS:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    card = d.get("card", d.get("cards"))
    if isinstance(card, str):
        for h in KNOWN_HOSTS:
            if re.search(r"\b%s\b" % re.escape(h), card, re.I):
                return card
    return None


def is_pc_card0(host, card):
    """pc card 0, however the artifact spells it: one string, or a host field plus a card field."""
    blob = "%s %s" % (host, card)
    if BANNED.search(blob):
        return True
    # host and card in separate fields is the common shape and the one that caught nobody.
    host_is_pc = bool(re.search(r"\bpc\b", str(host), re.I))
    card_is_0 = str(card).strip() in ("0", "[0]", "card 0")
    return host_is_pc and card_is_0


def offenders(root: pathlib.Path):
    out = []
    for f in sorted(root.glob("perf/**/*.json")):
        if "of3t" not in str(f):
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        fields = [k for k in DIGEST_FIELDS if k in d]
        if not fields:
            continue
        rel = str(f.relative_to(root))
        h = host_of(d)
        if h is None:
            out.append((rel, fields[0], "claims %s and does not say which HOST it ran on; card %r "
                                        "is a different card on pc, qb1 and qb2"
                        % (fields[0], d.get("card", d.get("cards")))))
        elif is_pc_card0(h, d.get("card", d.get("cards"))):
            out.append((rel, fields[0], "was taken on pc card 0, which must not host "
                                        "hash-equality gating at any size for any model"))
    return out


def main(argv):
    root = pathlib.Path(argv[1] if len(argv) > 1 else ".").resolve()
    bad = offenders(root)

    # Probes: anonymous fires, pc card 0 fires, an identified clean host does not.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        t = pathlib.Path(td)
        (t / "perf" / "of3t_probe").mkdir(parents=True)
        def w(name, obj):
            (t / "perf" / "of3t_probe" / name).write_text(json.dumps(obj))
        w("anon.json", {"base_equals_off": True, "card": 0})
        w("pc0.json", {"base_equals_off": True, "host": "pc", "card": 0})
        w("clean.json", {"base_equals_off": True, "host": "qb1", "card": 0})
        got = {f.split("/")[-1] for f, _k, _w in offenders(t)}
        if "anon.json" not in got:
            print("BROKEN an anonymous digest claim does not fire", file=sys.stderr)
            return 2
        if "pc0.json" not in got:
            print("BROKEN a digest claim from pc card 0 does not fire", file=sys.stderr)
            return 2
        if "clean.json" in got:
            print("BROKEN an identified digest claim on a good card fires", file=sys.stderr)
            return 2

    new = [(r, f, w) for r, f, w in bad if r not in FROZEN]
    healed = [r for r in FROZEN if r not in {b[0] for b in bad}]
    if new:
        for rel, _f, why in new:
            print("  DRIFT %s %s (D155)" % (rel, why))
        print("FAIL %d NEW digest claim(s) that cannot be attributed to healthy hardware"
              % len(new))
        return 1
    if healed:
        for r in healed:
            print("  DRIFT %s now names its hardware but is still frozen -- remove it (D155)" % r)
        print("FAIL the ratchet has %d stale entr(y/ies); it may only shrink" % len(healed))
        return 1
    n = sum(1 for _f in root.glob("perf/**/*.json") if "of3t" in str(_f))
    print("ok    %d of3t digest claim(s) cannot be attributed to healthy hardware, all %d frozen "
          "(probes: anonymous fires, pc card 0 fires, an identified good card does not); a new "
          "one fails" % (len(bad), len(FROZEN)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
