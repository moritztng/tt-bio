"""Balance the 164 Tier-A targets across cards by COST, not by count.

Fold cost is close to linear in token count (the opendde slab measured 1.72-2.11 s/token), and
the campaign spans 230-1095 tokens -- a 4.8x spread. Slicing by index therefore leaves one card
running long after the others idle. Assignment is longest-processing-time-first: sort targets
descending by tokens and give each to the currently least-loaded card, which is the standard
makespan heuristic and gets within 4/3 of optimal.

Emits slices for 4 cards (qb1 alone) and 8 cards (qb1+qb2), as comma-separated --targets values.
"""
import json
import pathlib
import sys

import yaml

WT = pathlib.Path(__file__).resolve().parent.parent  # this checkout, not a torn-down slug worktree
YAMLS = WT / "examples/abag_xm"

tokens = {}
for f in sorted(YAMLS.glob("*.yaml")):
    doc = yaml.safe_load(f.read_text())
    n = 0
    for s in (doc.get("sequences") or []):
        p = s.get("protein") if isinstance(s, dict) else None
        if p and p.get("sequence"):
            n += len(p["sequence"])
    if n:
        tokens[f.stem] = n

print("targets: %d   tokens min %d median %d max %d   total %d"
      % (len(tokens), min(tokens.values()),
         sorted(tokens.values())[len(tokens) // 2], max(tokens.values()), sum(tokens.values())))

out = {"tokens": tokens}
for ncards in (4, 8):
    load = [0] * ncards
    slices = [[] for _ in range(ncards)]
    for tid, n in sorted(tokens.items(), key=lambda kv: -kv[1]):
        i = load.index(min(load))
        slices[i].append(tid)
        load[i] += n
    spread = (max(load) - min(load)) / (sum(load) / ncards) * 100
    print("\n%d cards: token load per card %s" % (ncards, load))
    print("   imbalance (max-min)/mean = %.1f%%  |  naive index-slicing imbalance for comparison:"
          % spread)
    ids = sorted(tokens)
    naive = [sum(tokens[t] for t in ids[k::ncards]) for k in range(ncards)]
    print("   %s -> %.1f%%" % (naive, (max(naive) - min(naive)) / (sum(naive) / ncards) * 100))
    for i, s in enumerate(slices):
        print("   card %d (%2d targets, %5d tok): %s" % (i, len(s), load[i], ",".join(sorted(s))))
    out["slices_%d" % ncards] = {str(i): sorted(s) for i, s in enumerate(slices)}
    out["load_%d" % ncards] = load

json.dump(out, open(sys.argv[1], "w"), indent=1)
print("\nwrote", sys.argv[1])
