#!/usr/bin/env python3
"""Score the confirmatory region-T session against the prediction registered before it ran.

The pre-registration is `regiont_quiet_finding.md`, committed at `9c19bc77d` before the session
started at 23:16:40Z. It is transcribed here as constants so the scoring cannot drift into
whatever the data happens to say, and the transcription is asserted against the file's own text --
if the prose and the constants disagree, this fails rather than scoring the wrong claim.

    python3 score_prereg.py [regiont_quiet_ab2.json]

Exit 0 = every registered prediction holds, 1 = at least one does not. Opens no device.
"""
import itertools
import json
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRE = HERE / "regiont_quiet_finding.md"

DROP_BLOCK = 0                      # discarded unconditionally, as a session warm-up
BASE_BAND = (14.60, 14.70)
ON_BAND = (14.40, 14.50)
RATIO_BAND = (1.010, 1.018)
FLOOR_CEILING = 1.006
DIGESTS = {"base": "45781db716ebf020", "on": "791696c3e443676d"}
PLDDT = {"base": 0.845919, "on": 0.845902}

# The transcription control. A pre-registration that the scorer can silently diverge from is not
# a pre-registration, so each constant must still be findable in the prose that registered it.
_txt = PRE.read_text()
for probe in ("block 0 is discarded unconditionally", "14.60-14.70", "14.40-14.50",
              "1.010-1.018", "under 1.006", DIGESTS["base"], DIGESTS["on"]):
    assert probe in _txt, f"pre-registration does not contain {probe!r} -- constants have drifted"


def main(path):
    d = json.loads(Path(path).read_text())
    arms, digs, plddts = {}, {}, {}
    for b in d.get("blocks", []):
        r = b.get("result")
        if not r:
            continue
        digs.setdefault(b["arm"], set()).update(f["cif_sha256"][:16] for f in r["folds"])
        plddts.setdefault(b["arm"], set()).update(round(f["plddt"], 6) for f in r["folds"])
        if b["block"] == DROP_BLOCK:
            continue
        arms.setdefault(b["arm"], []).append([f["fold_s"] for f in r["folds"]])

    flat = lambda bs: [x for b in bs for x in b]
    base, on = arms.get("base", []), arms.get("on", [])
    if len(base) < 2 or len(on) < 2:
        print(f"INCOMPLETE: {len(base)} base and {len(on)} on blocks after dropping block "
              f"{DROP_BLOCK}")
        return 1
    bm, om = [st.median(x) for x in base], [st.median(x) for x in on]
    mb, mo = st.median(flat(base)), st.median(flat(on))
    ratio = mb / mo
    floor = max(max(x, y) / min(x, y) for i, x in enumerate(bm) for y in bm[i + 1:])
    gap = min(flat(base)) - max(flat(on))

    allm, k = bm + om, len(bm)
    obs = st.median(bm) - st.median(om)
    hits = tot = 0
    for c in itertools.combinations(range(len(allm)), k):
        a = [allm[i] for i in c]
        z = [allm[i] for i in range(len(allm)) if i not in c]
        tot += 1
        hits += st.median(a) - st.median(z) >= obs

    # how many admissible blocks had the on arm ahead -- the registered negative case is "on median
    # above base median in two or more admissible blocks"
    against = sum(1 for x, y in zip(bm, om) if y > x)
    holders = [(b["arm"], b["block"], sum(len(f.get("foreign_device_holders") or [])
                                          for f in b["result"]["folds"]))
               for b in d.get("blocks", []) if b.get("result")]
    loud = [(a, n, h) for a, n, h in holders if h and n != DROP_BLOCK]

    checks = [
        ("base median in 14.60-14.70", BASE_BAND[0] <= mb <= BASE_BAND[1], f"{mb:.3f} s"),
        ("on median in 14.40-14.50", ON_BAND[0] <= mo <= ON_BAND[1], f"{mo:.3f} s"),
        ("ratio in 1.010-1.018 (confirms ~+0.2 s, refutes +0.6640 s)",
         RATIO_BAND[0] <= ratio <= RATIO_BAND[1], f"{ratio:.5f}"),
        ("A/A floor under 1.006", floor < FLOOR_CEILING, f"{floor:.5f}"),
        ("effect clears its own floor", ratio > floor, f"{(ratio-1)/(floor-1):.2f}x"),
        ("registered negative case absent (<2 blocks against)", against < 2, f"{against} against"),
        ("base digest unchanged", digs.get("base") == {DIGESTS["base"]}, str(digs.get("base"))),
        ("on digest unchanged", digs.get("on") == {DIGESTS["on"]}, str(digs.get("on"))),
        ("base plDDT unchanged", plddts.get("base") == {PLDDT["base"]}, str(plddts.get("base"))),
        ("on plDDT unchanged", plddts.get("on") == {PLDDT["on"]}, str(plddts.get("on"))),
        ("no foreign device holder in any scored block", not loud, str(loud)),
    ]
    print(f"session {path}")
    print(f"  admissible blocks: base {[round(x,3) for x in bm]}  on {[round(x,3) for x in om]}")
    print(f"  base {mb:.3f} s  on {mo:.3f} s  delta {mb-mo:+.4f} s  ratio {ratio:.5f}  "
          f"floor {floor:.5f}")
    print(f"  separation gap {gap:+.4f} s (positive = every on fold beat every base fold)  "
          f"perm p = {hits}/{tot}")
    cl = [f["clock"] for b in d["blocks"] if b.get("result") for f in b["result"]["folds"]
          if f.get("clock", {}).get("aiclk_n")]
    print(f"  clock {min(c['aiclk_min'] for c in cl)}-{max(c['aiclk_max'] for c in cl)} MHz over "
          f"{sum(c['aiclk_n'] for c in cl)} during-fold samples")
    bad = 0
    for name, ok, got in checks:
        bad += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {got}")
    print("PRE-REGISTRATION HELD" if not bad else f"PRE-REGISTRATION BROKEN on {bad} check(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else HERE / "regiont_quiet_ab2.json"))
