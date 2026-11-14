#!/usr/bin/env python3
"""Triangle attention seconds per Mol-1 prediction, from the op bench cells.

Upstream defaults (state/mol1/recon/arch_census.md): 20 recycles, each with 4 MSA-module pair blocks at c_z 448 / 14
heads and 40 Pairformer blocks at 320 / 10 heads (the 8 PairMixer blocks have no triangle attention), plus the 4-block
confidence head at 448 / 14 once per sample, 5 samples. Each block runs a start and an ending attention.
Usage: project.py cells.jsonl [more.jsonl ...]  (later files override earlier cells for the same arm)
"""
import json
import sys

RECYCLES, SAMPLES = 20, 5
CALLS = {(320, 10): RECYCLES * 40, (448, 14): RECYCLES * 4 + SAMPLES * 4}


def main(paths):
    ms = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if "ms_min" in r:
                ms[(r["arm"], r["S"], r["c_z"], r["heads"], r["ending"])] = r["ms_min"]
    out = {}
    for arm in sorted({k[0] for k in ms}):
        for S in sorted({k[1] for k in ms if k[0] == arm}):
            try:
                s = sum(n * (ms[(arm, S, c, h, False)] + ms[(arm, S, c, h, True)]) for (c, h), n in CALLS.items()) / 1e3
            except KeyError:
                continue
            out[f"{arm}@{S}"] = round(s, 2)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
