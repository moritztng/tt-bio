"""Sanity-check the DockQ label distribution across all produced labels.

Ab-Ag co-folding is genuinely hard, so low DockQ is expected -- but an all-zero distribution is
also exactly what a chain-mapping bug looks like (see dockq-multicopy-chain-mapper-false-zero).
The two are told apart by the SHAPE: a real distribution has a tail with some targets reaching
acceptable/medium quality, whereas a mapping bug pins essentially everything at ~0 regardless of
target.

CAPRI bands: incorrect <0.23, acceptable >=0.23, medium >=0.49, high >=0.80.
"""
import collections
import json
import pathlib
import statistics
import sys

BANDS = (("high", 0.80), ("medium", 0.49), ("acceptable", 0.23))


def band(v):
    for name, lo in BANDS:
        if v >= lo:
            return name
    return "incorrect"


def main():
    per_model = collections.defaultdict(list)
    best_per_fold = []
    capri_seen = collections.Counter()
    for d in sys.argv[1:]:
        for f in sorted(pathlib.Path(d).glob("*.json")):
            doc = json.loads(f.read_text())
            model = f.name.rsplit("_", 1)[0]
            vals = []
            for rec in doc.get("samples", []):
                sub = rec.get("dockq")
                if isinstance(sub, dict) and isinstance(sub.get("dockq"), (int, float)):
                    vals.append(sub["dockq"])
                    if sub.get("capri"):
                        capri_seen[str(sub["capri"])] += 1
            if vals:
                per_model[model].extend(vals)
                best_per_fold.append((max(vals), doc.get("target"), model))

    print("per-sample DockQ by generator")
    for m, v in sorted(per_model.items()):
        v = sorted(v)
        print("  %-14s n=%4d  min=%.3f  p50=%.3f  p90=%.3f  max=%.3f  >=0.23: %d (%.1f%%)"
              % (m, len(v), v[0], statistics.median(v), v[int(0.9 * (len(v) - 1))], v[-1],
                 sum(1 for x in v if x >= 0.23), 100.0 * sum(1 for x in v if x >= 0.23) / len(v)))

    print("\nBEST-of-50 per fold (the oracle number this dataset is about)")
    best_per_fold.sort(reverse=True)
    bands = collections.Counter(band(b) for b, _, _ in best_per_fold)
    print("  folds=%d  " % len(best_per_fold) + "  ".join(
        "%s=%d" % (k, bands.get(k, 0)) for k in ("high", "medium", "acceptable", "incorrect")))
    print("  top 8:", ", ".join("%s/%s %.3f" % (t, m.replace("_", "-"), b)
                                for b, t, m in best_per_fold[:8]))
    if capri_seen:
        print("\n  capri labels seen:", dict(capri_seen.most_common(6)))


if __name__ == "__main__":
    main()
