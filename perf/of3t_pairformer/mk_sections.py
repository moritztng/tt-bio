#!/usr/bin/env python3
"""A per-block sections file for agreement.py.

`agreement.py` splits a scope by the LONGEST section key a tensor name starts with, so handing it
the 48 block prefixes instead of the one `pairformer_stack` prefix turns its per-section table
into a per-block table, each set carrying its own mass share. A23 wants the per-block split beside
the stack headline: block 47 alone is 18.3 % of the stack, and a stack figure without the split
cannot say whether one block carries it.
"""
import json
import sys

out = {"what": __doc__.strip().splitlines()[0],
       "why": "A23: every set statistic carries its set's mass share, and the stack headline "
              "hides a single dominant block unless the per-block split sits beside it.",
       "sections_pct_of_model": {f"pairformer_stack.blocks.{i}": None for i in range(48)}}
mass = json.load(open(sys.argv[1]))["per_block"]
for i in range(48):
    out["sections_pct_of_model"][f"pairformer_stack.blocks.{i}"] = {
        "n": mass[str(i)]["n"], "pct": mass[str(i)]["pct_of_model"],
        "pct_of_the_stack": mass[str(i)]["pct_of_stack"]}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("wrote", sys.argv[2], len(out["sections_pct_of_model"]), "blocks")
