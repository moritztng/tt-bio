"""Generate the RF3 size-ladder inputs, committed so the GPU and TT arms fold the same thing.

The sequence is the fleet's own size fixture: CDK2 (PDB 1HCL) apo, 298 aa, tiled and truncated
to N. Same construction as `perf/wh-correctness/matrix.py`, so an RF3 rung at N aa is directly
comparable with the esmfold2 / boltz2 / protenix / opendde rungs already on the perf page.

Single protein chain, no MSA, no templates. That is deliberate: MSA depth is a second axis and a
paired-MSA path is a host-side cost the TT port does not own yet, so folding both arms
single-sequence removes the confound. `perf-page-matched-batch-protocol-recurrence` has shipped
three times in this org; the inputs and the batch have to be identical on both sides or the
comparison is void.

    python perf/rf3/make_inputs.py          # writes perf/rf3/inputs/rf3_<N>.json
"""

import json
import pathlib

# CDK2 (PDB 1HCL) apo, 298 aa. Verbatim from perf/wh-correctness/matrix.py:CDK2_298.
CDK2_298 = ("MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEF"
            "LHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVV"
            "TLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQ"
            "DFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL")

# 298 is the fleet fixture at its own natural length -- no tiling, no truncation -- and it is the
# only rung where RF3 serves triangle attention at a k_chunk narrower than the padded key
# (shipped k = 64 at padded 320), so it is the only rung a k_chunk probe has anything to vary.
SIZES = (128, 256, 298, 512, 768, 1024)


def cdk2(n: int) -> str:
    """The fleet fixture's sequence at length n: tile the 298 aa domain, truncate."""
    return (CDK2_298 * (n // len(CDK2_298) + 1))[:n]


def spec(n: int) -> dict:
    return {"name": "cdk2_%d" % n,
            "components": [{"seq": cdk2(n), "chain_id": "A"}]}


def main() -> None:
    out = pathlib.Path(__file__).parent / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    for n in SIZES:
        p = out / ("rf3_%d.json" % n)
        p.write_text(json.dumps(spec(n), indent=2) + "\n")
        print("%s  %d aa" % (p, n))


if __name__ == "__main__":
    main()
