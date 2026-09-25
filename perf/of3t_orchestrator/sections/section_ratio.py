#!/usr/bin/env python3
"""Per-section ratio of an arm's rel to upstream bf16's rel, from a score.py artifact.

The GRADIENTS charter clause read only global figures, so it passed GO384 while
aux_heads.pairformer_embedding sat at 36.7x its bf16 (D266). This writes the per-section
test as data the charter's "all" op can read: every section's `within_3x` must be true.

    section_ratio.py SCORE.json ARM OUT.json
"""
import json
import sys

LIMIT = 3.0

score, arm, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(score))
ours, bf16 = d["arms"][arm]["by_section"], d["arms"]["BF16"]["by_section"]
sections = {}
for name in sorted(bf16):
    ratio = ours[name]["rel"] / bf16[name]["rel"]
    sections[name] = {"rel": ours[name]["rel"], "bf16_rel": bf16[name]["rel"],
                      "ratio": ratio, "within_3x": ratio <= LIMIT}
assert set(ours) == set(bf16), sorted(set(ours) ^ set(bf16))
json.dump({"source": score, "arm": arm, "limit": LIMIT, "sections": sections},
          open(out, "w"), indent=2)
for name, s in sections.items():
    if not s["within_3x"]:
        print(f"{name}: {s['ratio']:.2f}x its bf16")
