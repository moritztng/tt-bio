"""Make a deeper alignment from a shallow one, so the ladder's win can be measured AGAINST depth.

The ladder saves `slope * (bucket_1024(d) - ladder(d))` per MSALayer, which is a function of the
TRUE alignment depth d and goes to exactly zero when the two buckets coincide. A win measured only
at the perf fixture's 35 rows is a fixture-shaped number
(`metric-spec-from-one-example-encodes-its-accidents`), so the curve needs real folds at real
depths.

Plain repetition does NOT work and the first attempt at this was wrong: the Boltz-2 featurizer
DEDUPLICATES the alignment, so a 256-row file built from 35 distinct rows arrives at the model as
35 (measured -- `prepare_features` returned `feats["msa"]` of shape (35, 512) for a 256-row a3m,
and the fold digest came back bit-identical to the shallow one, which is what gave it away). Each
copy therefore gets one substituted column so the rows are distinct.

That changes the model's answer -- more, and partly synthetic, evidence -- and that is fine and
deliberate: these fixtures exist to put a given number of rows through the MSA track, not to be
biologically better alignments. The query row is never touched, so `seed_msa_cache`'s "a3m query
row matches the target sequence" assertion still holds.
"""
import argparse
from pathlib import Path


def deepen(src: Path, dst: Path, depth: int) -> int:
    rows = src.read_text().rstrip("\n").split("\n")
    assert len(rows) % 2 == 0, f"{src} is not strict header/row pairs"
    pairs = [(rows[i], rows[i + 1]) for i in range(0, len(rows), 2)]
    query, rest = pairs[0], pairs[1:]
    out = [query]
    i = 0
    while len(out) < depth:
        h, s = rest[i % len(rest)]
        copy = i // len(rest)
        if copy:
            # one substituted column per copy, so the featurizer's dedup keeps the row
            pos = (i * 7 + copy * 13) % len(s)
            sub = "G" if s[pos] == "A" else "A"
            s = s[:pos] + sub + s[pos + 1:]
            h = f"{h}_r{copy}"
        out.append((h, s))
        i += 1
    dst.write_text("\n".join(f"{h}\n{s}" for h, s in out) + "\n")
    return len(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, required=True)
    a = ap.parse_args()
    n = deepen(Path(a.src), Path(a.out), a.depth)
    print(f"{a.out}: {n} sequences")
