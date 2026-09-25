"""Composition of a designed binder, against the reference BindCraft 2 accepted a binder from.

The acceptance filters never see a sequence whose trajectory died before scoring, so a trajectory
terminated at a design stage leaves only its metrics behind. Composition is the cheapest
independent read on whether those metrics were describing a plausible protein.

Usage: seq_character.py NAME=SEQ [NAME=SEQ ...]
"""
import sys

HYDROPHOBIC = set("AVILMFWY")
AROMATIC = set("FWY")
CHARGED = set("DEKR")


def profile(seq):
    n = len(seq)
    counts = {a: seq.count(a) / n for a in set(seq)}
    return {
        "len": n,
        "distinct_aa": len(set(seq)),
        "hydrophobic": sum(counts.get(a, 0) for a in HYDROPHOBIC),
        "aromatic": sum(counts.get(a, 0) for a in AROMATIC),
        "charged": sum(counts.get(a, 0) for a in CHARGED),
        "M": counts.get("M", 0.0),
        "W": counts.get("W", 0.0),
        "V": counts.get("V", 0.0),
        "top3": ",".join("%s%.0f%%" % (a, p * 100) for a, p in
                         sorted(counts.items(), key=lambda kv: -kv[1])[:3]),
    }


COLS = ["len", "distinct_aa", "hydrophobic", "aromatic", "charged", "M", "W", "V"]


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    rows = []
    for item in argv:
        name, _, seq = item.partition("=")
        seq = seq.strip().upper()
        if not seq:
            sys.stderr.write("%s: no sequence\n" % name)
            return 1
        rows.append((name, profile(seq)))

    header = "arm".ljust(26) + "".join(c.rjust(13) for c in COLS)
    print(header + "   top3")
    for name, p in rows:
        line = name.ljust(26)
        for c in COLS:
            v = p[c]
            line += str(v).rjust(13) if isinstance(v, int) else ("%11.1f%% " % (v * 100))
        print(line + "   " + p["top3"])

    print()
    print("natural globular protein, for orientation: hydrophobic ~30-35%, aromatic ~8%,")
    print("charged ~25%, M ~2.4%, W ~1.3%")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
