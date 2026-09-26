#!/usr/bin/env python3
"""What the fused backward actually removes, by ttnn verb, from the tri-att backward family.

The round A/B says -1.117 s and the byte census says -484.5 GB. Neither says whether the kernel
replaced the backward's WHOLE op sequence or only its score traffic, and the campaign's
arithmetic turns on that: `state/perf10/bcx-CALLS.md` has triangle attention as the largest
caller of `multiply`, `add`, `slice`, `sum` and `concat`, so a kernel that takes out the op
sequence takes out more than the byte term it was projected on.

Reads the two `perf/bcx_p10_devmap/devmap.py block` blobs. Verb counts are deterministic.

    python3 perf/bcx_p10_tabwire/verbdelta.py perf/bcx_p10_tabwire/out/bytes
"""
import collections
import json
import pathlib
import sys

#: One block's two triangle attentions, over the blob's reps. Divides the totals back to a call.
CALLS_PER_BLOB = 2 * 2
#: Fused-backward calls a BindCraft 2 round makes, counted at the round boundary, not derived
#: from the block count. perf/bcx_p10_stack/out/bwab.
ROUND_CALLS = 108


def verbs(path, stack="evo"):
    d = json.load(open(path))
    agg = collections.Counter()
    for k, n in d["verb"]["calls"].items():
        tag, _, verb = k.partition("||")
        st, _blk, dr, fam = tag.split("|")
        if st == stack and dr == "bwd" and fam.startswith("tri_att"):
            agg[verb] += n
    return agg


def main(root):
    root = pathlib.Path(root)
    a = verbs(root / "blk288_tabwire_hifi.json")
    b = verbs(root / "blk288_tabwire_hifibw.json")
    print("ttnn verbs in ONE triangle-attention backward, n=288\n")
    print(f"  {'verb':<32}{'chunked':>9}{'fused':>7}{'delta':>7}{'a round':>10}")
    for v in sorted(set(a) | set(b), key=lambda v: -(a[v] - b[v])):
        x, y = a[v] / CALLS_PER_BLOB, b[v] / CALLS_PER_BLOB
        print(f"  {v:<32}{x:>9.1f}{y:>7.1f}{y - x:>7.1f}"
              f"{(y - x) * ROUND_CALLS:>10.0f}")
    ta, tb = sum(a.values()) / CALLS_PER_BLOB, sum(b.values()) / CALLS_PER_BLOB
    print(f"  {'TOTAL':<32}{ta:>9.1f}{tb:>7.1f}{tb - ta:>7.1f}"
          f"{(tb - ta) * ROUND_CALLS:>10.0f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "perf/bcx_p10_tabwire/out/bytes")
