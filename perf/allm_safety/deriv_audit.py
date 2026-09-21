#!/usr/bin/env python3
"""Replay every key the pass-1 census counted through the candidate's derivation rule.

The candidate (ce7467175) deletes six written-down fused keys from _MM_BLOCK and derives them
instead. A derivation serves keys the dict refused, so the ONLY behaviour that can change is on a
key a model presented and was DECLINED for. This replays the census's counted keys against both
the old dict and the new dict+rule and prints, per model, every key whose answer moves.
"""
import json, sys
from pathlib import Path

OUT = Path(sys.argv[1])

# _MM_BLOCK as it stands on origin/main (before the candidate)
OLD = {
    (8, 24): (4, 8, 1, 4, 1), (8, 8): (4, 8, 1, 4, 1),
    (4, 12): (4, 4, 1, 4, 1), (4, 4): (4, 4, 1, 4, 1),
    (4, 16): (4, 4, 1, 4, 1), (4, 17): (4, 4, 1, 4, 1),
    (8, 32): (4, 8, 1, 4, 1), (8, 33): (4, 8, 1, 4, 1),
    (2, 12): (4, 2, 1, 4, 1), (2, 2): (4, 2, 1, 4, 1),
    (2, 6): (4, 2, 1, 4, 1), (2, 8): (4, 2, 1, 4, 1), (2, 9): (4, 2, 1, 4, 1),
    (12, 36): (4, 12, 1, 2, 1), (12, 12): (8, 12, 1, 2, 1),
}
# _MM_BLOCK after the candidate: BASE widths only
NEW = {
    (8, 24): (4, 8, 1, 4, 1), (8, 8): (4, 8, 1, 4, 1),
    (4, 12): (4, 4, 1, 4, 1), (4, 4): (4, 4, 1, 4, 1),
    (2, 12): (4, 2, 1, 4, 1), (2, 2): (4, 2, 1, 4, 1), (2, 6): (4, 2, 1, 4, 1),
    (12, 36): (4, 12, 1, 2, 1), (12, 12): (8, 12, 1, 2, 1),
}
# the two entries the table's own comment records as NOT bit-exact
NOT_BITEXACT = {(12, 36), (12, 12)}


def fused(kt, nt, tbl):
    """Verbatim reimplementation of _mm_fused_block from ce7467175."""
    widths = sorted({n for (k, n) in tbl if k == kt})
    best = None
    pair = None
    for i, a in enumerate(widths):
        for b in widths[i:]:
            if nt in (a + b, a + b + 1) and (best is None or max(a, b) > best):
                best, pair = max(a, b), (a, b)
    if best is None:
        return None, None
    return tbl[(kt, best)], pair


def resolve_new(kt, nt):
    blk = NEW.get((kt, nt))
    if blk is not None:
        return blk, None, "registered"
    blk, pair = fused(kt, nt, NEW)
    return blk, pair, ("derived" if blk is not None else "none")


print("=" * 78)
print("PART 1 -- do the six deleted literals reproduce?")
print("=" * 78)
ok = True
for k in [(4, 16), (4, 17), (8, 32), (8, 33), (2, 8), (2, 9)]:
    blk, pair, how = resolve_new(*k)
    same = blk == OLD[k]
    ok &= same and how == "derived"
    print(f"  {str(k):10s} old={OLD[k]}  new={blk} via {how} {pair}  {'MATCH' if same else '*** DIFFERS ***'}")
print(f"  => all six reproduce byte-for-byte: {ok}")

print()
print("=" * 78)
print("PART 2 -- every key the census counted, per model, where the answer MOVES")
print("=" * 78)
totals = {"newly_served": 0, "lost": 0}
rows = []
for f in sorted(OUT.glob("mmkeys_*.json")):
    d = json.loads(f.read_text())
    model = f.stem.replace("mmkeys_", "")
    moved = []
    for ck, n in sorted(d["keys"].items()):
        reader, key, verdict = ck.split("|")
        kt, nt = (int(x) for x in key.split(","))
        if reader != "tenstorrent":
            continue  # allow-list readers gate on their own closed set, handled in PART 3
        old = OLD.get((kt, nt))
        new, pair, how = resolve_new(kt, nt)
        if (old is None) != (new is None) or old != new:
            moved.append((kt, nt, n, verdict, old, new, pair, how))
    rows.append((model, d["keys"], moved))
    if moved:
        print(f"\n  {model}:")
        for kt, nt, n, verdict, old, new, pair, how in moved:
            flag = ""
            src = (kt, max(pair)) if pair else None
            if src in NOT_BITEXACT:
                flag = "   <== inherits a NOT-BIT-EXACT block"
            selfpair = pair and pair[0] == pair[1]
            if selfpair:
                flag += "   <== SELF-PAIR derivation (operand fused with itself)"
            print(f"    ({kt},{nt}) x{n} was {verdict}({old}) -> {how} {new} from {pair}{flag}")
            totals["newly_served"] += n if old is None and new is not None else 0
            totals["lost"] += n if old is not None and new is None else 0
    else:
        print(f"\n  {model}: no key moves -- behaviour unchanged by construction")
print()
print(f"  TOTAL newly-served calls a fold across all counted models: {totals['newly_served']}")
print(f"  TOTAL calls that LOSE a config: {totals['lost']}")

print()
print("=" * 78)
print("PART 3 -- the two allow-list readers (swiglu_fused, trimul_tail)")
print("=" * 78)
SWIGLU = {(4, 16)}
TRIMUL = {(8, 8), (4, 4)}
for name, allow in (("swiglu_fused", SWIGLU), ("trimul_tail", TRIMUL)):
    for k in sorted(allow):
        old = OLD.get(k)
        new, pair, how = resolve_new(*k)
        note = "KeyError on main+candidate-without-70b2c4796" if old is not None and NEW.get(k) is None else ""
        print(f"  {name} {str(k):8s} old={old} new={new} via {how} {pair}  {note}")

print()
print("=" * 78)
print("PART 4 -- keys the rule newly makes derivable that NO counted model presents")
print("=" * 78)
for kt in sorted({k for k, _ in NEW}):
    widths = sorted({n for (k, n) in NEW if k == kt})
    derivable = set()
    for i, a in enumerate(widths):
        for b in widths[i:]:
            derivable.add((a + b, (a, b)))
            derivable.add((a + b + 1, (a, b)))
    presented = set()
    for _, keys, _ in rows:
        for ck in keys:
            reader, key, _v = ck.split("|")
            k2, n2 = (int(x) for x in key.split(","))
            if reader == "tenstorrent" and k2 == kt:
                presented.add(n2)
    unseen = sorted({(nt, p) for nt, p in derivable if nt not in presented and (kt, nt) not in NEW})
    if unseen:
        print(f"  kt={kt} (registered widths {widths}):")
        for nt, p in unseen:
            src = (kt, max(p))
            tag = "  NOT-BIT-EXACT source" if src in NOT_BITEXACT else ""
            sp = "  SELF-PAIR" if p[0] == p[1] else ""
            print(f"    ({kt},{nt}) <- {p} => {NEW[src]}{tag}{sp}")

print()
print("=" * 78)
print("PART 5 -- is the A/B's `off` arm actually main? (perf/allm_gates/fused_key_ab.py)")
print("=" * 78)
print("  The harness sets `_mm_fused_block = lambda: None` for `off` and calls that arm main.")
print("  But the candidate DELETED six literals from _MM_BLOCK, so under `off` those six resolve")
print("  to None too -- configs main serves today. Per model, the calls `off` wrongly declines:")
DELETED = {(4, 16), (4, 17), (8, 32), (8, 33), (2, 8), (2, 9)}
dirty = {}
for f in sorted(OUT.glob("mmkeys_*.json")):
    d = json.loads(f.read_text())
    model = f.stem.replace("mmkeys_", "")
    bad = []
    for ck, n in sorted(d["keys"].items()):
        reader, key, verdict = ck.split("|")
        kt, nt = (int(x) for x in key.split(","))
        if reader == "tenstorrent" and verdict == "served" and (kt, nt) in DELETED:
            bad.append((kt, nt, n))
    dirty[model] = bad
    if bad:
        tot = sum(n for _, _, n in bad)
        detail = ", ".join(f"({kt},{nt}) x{n}" for kt, nt, n in bad)
        print(f"    {model:20s} DIRTY  {tot:5d} calls/fold wrongly declined: {detail}")
    else:
        print(f"    {model:20s} clean  -- presents none of the six deleted literals")
nd = sum(1 for v in dirty.values() if v)
print(f"\n  => `off` is NOT main on {nd} of {len(dirty)} counted models.")
print("  => A correct control restores the six deleted literals in the `off` arm and disables")
print("     derivation only for keys main never had.")
