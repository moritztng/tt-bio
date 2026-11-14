"""Pin an annotated CIF's expandable segment to one length.

`RealizeExpandableSegments` samples each expandable segment's length from its [min, max], so
`dna_binder.cif` (chain C, 120-150) designs a different-sized binder at every seed and ours
never matches upstream's at the same seed. Nothing at or after `preprocess` can repair that:
the token count is already decided. Pinning min=max=<upstream's realized length> makes both
sides design the same problem, which is what a paired grade requires. The model still does the
full design on a binder of that length, so this is a pairing control and not a shortened run.

    python perf/rfd4/cond_chem/pin_segment.py --cif <in> --length 126 --out <out>
"""
import argparse
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif", required=True)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # The two integers before the <M> marker on the HETATM row are the segment's min and max.
    pat = re.compile(r"(\s)(\d+)(\s+)(\d+)(\s+<M>)")
    text = open(a.cif).read()
    out, n = pat.subn(lambda m: f"{m.group(1)}{a.length}{m.group(3)}{a.length}{m.group(5)}", text)
    if n != 1:
        raise SystemExit(f"expected exactly one expandable segment, matched {n}")
    open(a.out, "w").write(out)
    print(f"pinned 1 segment to {a.length}: {a.out}")


if __name__ == "__main__":
    main()
