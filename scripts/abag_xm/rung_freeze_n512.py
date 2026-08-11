"""Direct check of the PHASE 0b invariant on the rungs that carry published numbers.
The freeze gate flags rung 16 and cost annotations; this asks the narrower, load-bearing
question: did any rung <= 256 that a published table reads move at all?"""
import json

SNAP = json.load(open("analysis_curves.pre-n512.json"))
PUBLISHED_RUNGS = ["50", "64", "128", "256"]
LADDER = ["50->64", "64->128", "128->256"]
SCIENTIFIC = ("gain_ci", "common_targets", "degenerate", "doublings")

bad, checked = [], 0
for m in ["boltz2", "esmfold2", "opendde-abag", "protenix-v2"]:
    new = json.load(open("final_%s.json" % m))
    # per-rung summary rows
    for r in PUBLISHED_RUNGS:
        so, no = SNAP.get(m, {}).get(r), new.get(m, {}).get(r)
        if so is None or no is None:
            print("  %-14s N=%-4s absent in %s" % (m, r, "snapshot" if so is None else "final"))
            continue
        for k in sorted(set(so) | set(no)):
            checked += 1
            a, b = so.get(k), no.get(k)
            if k == "card_h":          # repriced on purpose by pass 36
                continue
            if a != b:
                bad.append("%s N=%s %s: %r -> %r" % (m, r, k, a, b))
    # ladder pairs: scientific payload only
    sp, np_ = SNAP.get(m + "__pairwise_gain_ci", {}), new.get(m + "__pairwise_gain_ci", {})
    for pair in LADDER:
        s, n = sp.get(pair), np_.get(pair)
        if s is None or n is None:
            print("  %-14s pair %-10s absent in %s" % (m, pair, "snapshot" if s is None else "final"))
            continue
        for k in SCIENTIFIC:
            if k not in s and k not in n:
                continue
            checked += 1
            a, b = s.get(k), n.get(k)
            if k == "gain_ci":
                # the snapshot predates the 'gap' metric; compare only shared metrics
                shared = set(a or {}) & set(b or {})
                for mk in sorted(shared):
                    checked += 1
                    if a[mk] != b[mk]:
                        bad.append("%s %s gain_ci[%s]: %r -> %r" % (m, pair, mk, a[mk], b[mk]))
                continue
            if a != b:
                bad.append("%s %s %s: %r -> %r" % (m, pair, k, a, b))

print("\n%d frozen scalars compared on rungs %s and pairs %s"
      % (checked, "/".join(PUBLISHED_RUNGS), ", ".join(LADDER)))
if bad:
    print("MOVED: %d" % len(bad))
    for x in bad:
        print("   ", x)
else:
    print("NO PUBLISHED-RUNG VALUE MOVED vs the pre-512 snapshot (card_h excluded: repriced by pass 36)")
