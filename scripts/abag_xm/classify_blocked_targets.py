"""Separate the AbAg-XM targets that hit a real DRAM ceiling from the ones that merely failed.

Five of the 164 targets produced no structures on the Galaxy, and the campaign carried four of
them as "capacity-blocked". Two independent checks disagree with that label on two of the four,
so the exclusion list a released dataset states should be re-derived here rather than asserted.

**Check 1 -- does the refused byte count correspond to a tensor?**  A ttnn OOM reports the exact
size it could not allocate.  If that number decomposes into this model's own dimensions, the
failure names a tensor and is a size problem; if no decomposition exists, it is not a single
logical tensor and the size is not the story.  The search is exhaustive over the real dimension
catalogue (exact and tile-padded token counts, exact atom counts, MSA depth and its padded and
chunked forms, the channel widths in the stack, head / n_queries / n_keys), in bf16 and fp32.
Note TILE_LAYOUT pads the *last two* axes, so a logical (N, N, c) pair tensor occupies
``N * pad32(N) * c`` -- missing that convention is why an earlier pass found no match for 9j4c.

**Check 2 -- is the target bigger than everything that folds?**  Rank each target's resident
footprint (``m_feat`` plus the pair representation) against the 159 that produced structures.  A
target above every folding target has a ceiling argument; one sitting below dozens of them does
not, whatever its refused count says.  This is a ranking, not a threshold: the footprint does
*not* separate success from failure in general, and this script prints how many folding targets
sit above each failure precisely so that is visible.

The two checks are unrelated -- one is byte arithmetic, one is an empirical ranking over measured
outcomes -- so where they agree, the classification is worth acting on.

Usage:
    python3 scripts/abag_xm/classify_blocked_targets.py [--panel scripts/abag_xm/mfeat_panel_v2.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

C_Z = 384          # opendde pair channel
CHANNELS = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768)
ATTN = (4, 8, 16, 32, 128)          # heads, n_queries, n_keys

# Refused allocation sizes, copied from each target's own OOM line. Blank stays blank: a target
# that failed without an OOM (9mns produces no allocation failure at all) has no entry here.
REFUSED = {
    "9j4c": 941875200,
    "9i3p": 520093696,
    "9q7y": 654311424,
    "9ivj": 2272002048,
}
MISSING = ("9j4c", "9i3p", "9q7y", "9ivj", "9mns")


def atom_count(target: str, examples: Path) -> int | None:
    """Exact heavy-atom count, from the model's own per-residue atom sets.

    Estimating this is not good enough -- the whole point of check 1 is that a match is exact or
    it is not a match, and an approximate atom count can neither confirm nor rule out a family.
    """
    import re
    import sys

    # Run from anywhere: sys.path[0] is this script's directory, not the repo root.
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from tt_bio.data import const
    except ImportError:
        return None
    path = examples / f"{target}.yaml"
    if not path.exists():
        return None
    per_letter = {
        letter: len(const.ref_atoms[res])
        for res, letter in const.prot_token_to_letter.items()
        if res in const.ref_atoms
    }
    seqs = re.findall(r"sequence:\s*([A-Z]+)", path.read_text())
    if not seqs or any(c not in per_letter for s in seqs for c in s):
        return None
    return sum(per_letter[c] for s in seqs for c in s)


def decompositions(cells: int, dims: dict[str, int]) -> list[str]:
    """Every exact way to write `cells` as dim x dim x channel (optionally x an attention dim)."""
    # Keyed on the *unordered* dim pair: `tok x pad x c` and `pad x tok x c` are one tensor read
    # two ways, and counting both would double every hit.
    found = {}
    for na, va in dims.items():
        if not va or cells % va:
            continue
        rest = cells // va
        for nb, vb in dims.items():
            if not vb or rest % vb:
                continue
            tail = rest // vb
            pair = tuple(sorted(((na, va), (nb, vb))))
            if tail in CHANNELS:
                found[(pair, tail, 1)] = f"{na}({va}) x {nb}({vb}) x c={tail}"
            else:
                for c in CHANNELS:
                    if tail % c == 0 and tail // c in ATTN:
                        found[(pair, c, tail // c)] = (
                            f"{na}({va}) x {nb}({vb}) x c={c} x att={tail // c}")
        if rest in CHANNELS:
            found[(((na, va),), rest, 1)] = f"{na}({va}) x c={rest}"
    return sorted(found.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", type=Path,
                    default=Path("scripts/abag_xm/mfeat_panel_v2.json"))
    ap.add_argument("--examples", type=Path, default=Path("examples/abag_xm"))
    args = ap.parse_args()

    panel = {r["target"]: r for r in json.loads(args.panel.read_text())}
    footprint = {
        t: r["mfeat_gib"] + r["tokens"] * r["pad"] * C_Z * 2 / 2 ** 30
        for t, r in panel.items()
    }
    folded = [t for t in panel if t not in MISSING]
    ceiling_t = max(folded, key=footprint.__getitem__)

    print(f"panel {len(panel)} targets, {len(folded)} folded, {len(MISSING)} missing")
    print(f"highest footprint that folds: {ceiling_t} at {footprint[ceiling_t]:.3f} GiB\n")
    print(f"{'target':<7}{'tok':>6}{'depth':>7}{'mfeat':>8}{'pair':>7}{'sum':>8}"
          f"{'above it':>10}  refused decomposition")

    verdicts = {}
    for t in MISSING:
        r = panel[t]
        pair = r["tokens"] * r["pad"] * C_Z * 2 / 2 ** 30
        total = footprint[t]
        above = sum(1 for o in folded if footprint[o] > total)

        dims = {
            "tok": r["tokens"], "pad": r["pad"], "depth": r["depth"],
            "depth32": -(-r["depth"] // 32) * 32, "chunk512": 512,
        }
        atoms = atom_count(t, args.examples)
        if atoms:
            dims["atom"], dims["apad"] = atoms, -(-atoms // 32) * 32

        refused = REFUSED.get(t)
        if refused is None:
            note = "no OOM (this target does not fail on an allocation)"
        else:
            hits = []
            for size in (2, 4):
                if refused % size == 0:
                    hits += [f"{h} [{'bf16' if size == 2 else 'fp32'}]"
                             for h in decompositions(refused // size, dims)]
            note = hits[0] if len(hits) == 1 else (
                "; ".join(hits) if hits else "NONE -- not a single tensor")
            if not dims.get("atom"):
                note += "  (atom axis NOT checked: tt_bio import failed)"

        # Capacity-bound needs both: nothing larger folds, and the refusal names a tensor.
        verdicts[t] = "capacity" if above == 0 and refused and "NONE" not in note else "other"
        print(f"{t:<7}{r['tokens']:>6}{r['depth']:>7}{r['mfeat_gib']:>8.3f}{pair:>7.3f}"
              f"{total:>8.3f}{above:>10}  {note}")

    print("\nA decomposition is one reading of the cell count, not automatically the tensor: 9i3p's"
          "\n512 x 992 x 512 is the same number of cells as OuterProductMean's per-I-block matmul"
          "\nresult (rows*c_a, c_b*tokens) at 256 rows, which is the form with a mechanism behind it."
          "\nWhat matters here is only whether ANY reading exists -- for 9q7y and 9ivj none does.")

    cap = [t for t, v in verdicts.items() if v == "capacity"]
    other = [t for t, v in verdicts.items() if v != "capacity"]
    print(f"\ncapacity-bound  ({len(cap)}): {', '.join(cap)}")
    print(f"  above every folding target AND the refused size names a real tensor -- these need an"
          f" engineering fix, and each needs its own (9i3p's is OuterProductMean, 9j4c's is the"
          f" pair representation).")
    print(f"not capacity-bound ({len(other)}): {', '.join(other)}")
    print(f"  each fails while strictly larger targets fold, and none names a tensor -- a transient"
          f" or allocator effect, so a retry is the first thing to try, not a fix.")


if __name__ == "__main__":
    main()
