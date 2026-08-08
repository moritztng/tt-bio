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

**Both checks were run, both were wrong about two targets, and a device run settled it.** The
arithmetic found no decomposition for 9q7y and 9ivj, and the footprint ranking put them below
dozens of targets that fold; together that read as "transient, retry it". Retried, both reproduced
their refused byte counts exactly. The missing dimension was the structural-token axis: the
expander produces ~1.9x as many tokens as there are residues, and the pair tensor at that scale is
~3.7x the trunk's. With Ns in the catalogue both refusals factor immediately.

So the verdicts printed at the end are the MEASURED ones, and the two checks are kept for what they
legitimately show -- which allocation fails, and at which scale -- not as a way of guessing a
verdict. The footprint columns rank on the residue-scale pair tensor only, so a target can look
comfortable there and still fail in the structural-token stage.

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

# What a device run actually showed, rather than what this script's arithmetic would guess. An
# earlier version classified 9q7y and 9ivj as transients on the strength of their residue-scale
# footprint and offered "retry first"; retried, both reproduced their refused byte counts exactly.
#
# Post-fix truth (2026-08-08, branch wk/tt-bio-large-target-oom-rootcause): the four OOMs were
# one tensor family (the bf16 pair representation) at three sites -- pair-op multiplicity in the
# trunk/refiner, deep-MSA co-residency, and per-bank contiguity after trunk churn. The fix
# row-blocks the pair ops, host-offloads the pristine MSA features, byte-caps the tri_att row
# chunk, assembles oversized concats on the host, and frees the structural pair tensor plus the
# sampler's pair bias before the residue-axis confidence stage. All four targets now fold on
# all four models (WH Galaxy, 12 GiB/chip, campaign config, seed 42, fold time / DRAM peak):
MEASURED = {
    "9j4c": ("FIXED, folds everywhere -- boltz2 472 s/5.50 GiB, esmfold2 782 s/9.82, "
             "protenix-v2 1521 s/8.77, opendde 2902 s/8.84 (was: capacity, residue-scale "
             "pair tensor, trunk at ~43 s)"),
    "9i3p": ("FIXED, folds everywhere -- boltz2 353 s/4.91 GiB, esmfold2 660 s/9.28, "
             "protenix-v2 1145 s/7.18, opendde 2199 s/9.10 (was: capacity, residue-scale pair "
             "tensor, trunk at ~44 s)"),
    "9q7y": ("FIXED, folds everywhere -- boltz2 289 s/4.35 GiB, esmfold2 535 s/8.80, "
             "protenix-v2 949 s/5.70, opendde 1716 s/8.98 (was: capacity, STRUCTURAL-scale "
             "pair tensor, refiner block 0 at ~1320 s)"),
    "9ivj": ("FIXED, folds everywhere -- boltz2 291 s/4.35 GiB, esmfold2 588 s/9.01, "
             "protenix-v2 918 s/6.09, opendde 1744 s/9.62 (was: capacity, STRUCTURAL-scale "
             "pair tensor, refiner block 0 at ~1308 s)"),
    "9mns": "no OOM at all -- does not finish inside 3000 s; unexplained",
}


def structural_token_count(target: str, examples: Path) -> int | None:
    """Ns, the structural-token axis the expander produces -- ~1.9x the residue tokens.

    One backbone token per residue plus one sidechain token per residue that has a sidechain, so
    ``Ns = 2 * tokens - glycines``. Checked against the two values a device run actually printed:
    9q7y 853 tokens / 62 Gly -> 1644, and 9ivj 891 / 70 -> 1712, both exact.

    This axis is why an earlier version of this script reported "no decomposition" for those two.
    Every dimension it searched was residue-scale, and the tensor they die on is not.
    """
    import re

    path = examples / f"{target}.yaml"
    if not path.exists():
        return None
    seqs = re.findall(r"sequence:\s*([A-Z]+)", path.read_text())
    if not seqs:
        return None
    return sum(2 * len(s) - s.count("G") for s in seqs)


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
        ns = structural_token_count(t, args.examples)
        if ns:
            dims["ns"], dims["nspad"] = ns, -(-ns // 32) * 32

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

        verdicts[t] = MEASURED.get(t, "unmeasured")
        print(f"{t:<7}{r['tokens']:>6}{r['depth']:>7}{r['mfeat_gib']:>8.3f}{pair:>7.3f}"
              f"{total:>8.3f}{above:>10}  {note}")

    print("\nA decomposition is one reading of the cell count, not automatically the tensor: 9i3p's"
          "\n512 x 992 x 512 is the same number of cells as OuterProductMean's per-I-block matmul"
          "\nresult at 256 rows, which is the form with a mechanism behind it. What the arithmetic"
          "\nestablishes is that a reading EXISTS and at which scale.")

    print("\nverdicts (measured on a Wormhole Galaxy, not inferred from the table above):")
    for t in MISSING:
        print(f"  {t:<6} {MEASURED.get(t, 'unmeasured')}")
    print("\nAll four that OOM'd die on the same tensor family, the pair representation "
          "(N, pad32(N), 384) in bf16.\nTwo did so at residue scale in the trunk, two at "
          "structural-token scale where Ns is ~1.9x larger.\nThe footprint columns rank on the "
          "RESIDUE-scale pair tensor only, which is why 9q7y and 9ivj\nlook comfortable there and "
          "still failed: at structural scale their pair tensor is ~3.7x bigger.\nPost-fix all "
          "four fold on all four models; the campaign exclusions for them are unnecessary.")


if __name__ == "__main__":
    main()
