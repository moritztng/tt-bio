#!/usr/bin/env python3
"""maskmap's 2x2 -> the per-round device cost of the pair masks and the second MSA row.

Same subtraction as `bcx-p10-devmap`: `device = synced - enqueue - lambda * calls`, per tag,
where a tag is `stack|block|dir|family`. The step walls in the artifact are NOT used: the
harness's own host bookkeeping (a `gc.collect()` over five loaded AF2 models, the tape, the
leaf uploads) is seconds per step and dwarfs the card, which is exactly why devmap timed the
verbs rather than the step.

The round runs 2 taped forwards and 1 backward over 48 Evoformer and 4 extra-MSA blocks; the
harness runs 1 forward and 1 backward over K. So

    round = blocks/K * (2 * fwd + 1 * bwd)
"""
import collections
import json
import statistics as st
import sys

BLOCKS = {"evo": 48, "extra": 4}


def per_arm(recs, floor):
    """(arm, stack) -> {dir: {family: device seconds for the K blocks this step ran}}."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        key = (r["arm"], r["stack"], r["K"])
        acc[key][r["mode"]].append((r["wall"], r["calls"]))
    out = {}
    for key, modes in acc.items():
        if "sync" not in modes or "free" not in modes:
            continue
        tags = set()
        for m in modes.values():
            for wall, _ in m:
                tags |= set(wall)
        dirs = collections.defaultdict(lambda: collections.defaultdict(float))
        calls = collections.defaultdict(float)
        for tag in tags:
            s = st.median([w.get(tag, 0.0) for w, _ in modes["sync"]])
            f = st.median([w.get(tag, 0.0) for w, _ in modes["free"]])
            n = st.median([float(c.get(tag, 0)) for _, c in modes["sync"]])
            _, _, d, fam = tag.split("|")
            dirs[d][fam] += s - f - floor * n
            calls[d] += n
        out[key] = {"dev": {d: dict(v) for d, v in dirs.items()},
                    "calls": dict(calls)}
    return out


def main(path):
    d = json.load(open(path))
    floor = d["sync_floor_s"]["median"]
    arms = per_arm(d["records"], floor)
    rounds = {}
    for (arm, stack, K), got in sorted(arms.items()):
        scale = BLOCKS[stack] / K
        fwd = sum(got["dev"].get("fwd", {}).values())
        bwd = sum(got["dev"].get("bwd", {}).values())
        rnd = scale * (2 * fwd + bwd)
        rounds[(arm, stack)] = rnd
        fams = collections.Counter()
        for dd in got["dev"].values():
            for k, v in dd.items():
                fams[k] += v
        print(f"{arm:10s} {stack:5s} K={K}  fwd {fwd:7.4f}  bwd {bwd:7.4f}  "
              f"calls {int(sum(got['calls'].values())):5d}  -> round {rnd:7.3f} s")
        print("             " + "  ".join(f"{k}={v:.4f}" for k, v in fams.most_common(6)))

    print("\n-- per-round device seconds, and what each variable costs --")
    for stack in ("evo", "extra"):
        have = {a: v for (a, s), v in rounds.items() if s == stack}
        if not have:
            continue
        print(f"[{stack}] " + "  ".join(f"{a}={v:.3f}" for a, v in sorted(have.items())))
        if "d1_nomask" in have and "d1_mask" in have:
            print(f"      masks at depth 1 : {have['d1_mask'] - have['d1_nomask']:+.3f} s")
        if "d2_nomask" in have and "d2_mask" in have:
            print(f"      masks at depth 2 : {have['d2_mask'] - have['d2_nomask']:+.3f} s")
        if "d1_nomask" in have and "d2_nomask" in have:
            print(f"      2nd MSA row      : {have['d2_nomask'] - have['d1_nomask']:+.3f} s")
        if "d1_nomask" in have and "d2_mask" in have:
            print(f"      devmap -> shipped: {have['d2_mask'] - have['d1_nomask']:+.3f} s")


if __name__ == "__main__":
    main(sys.argv[1])
