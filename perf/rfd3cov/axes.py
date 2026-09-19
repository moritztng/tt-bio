"""Both axes of an RFD3 rung, read off the model input rather than off the fixture's name.

The bar is in TOKENS and the second axis is alignment DEPTH. For this model one of those does
not exist, and that is worth proving rather than repeating from a blurb: a single-sequence pass
proves nothing for a model that HAS an MSA track, so the claim "there is no track" has to be a
measurement too.

    python3 perf/rfd3cov/axes.py > perf/rfd3cov/axes.json

No device. `featurize` is host-side, so this is the same tensor dict the sampler is handed.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tt_bio.rfd3.featurize import featurize            # noqa: E402
from tt_bio.rfd3.input import (InputSpecification, contig_residue_count,  # noqa: E402
                               parse_contig)

RUNGS = [
    ("control", "perf/ceilrfd3/targets/laczc_1008.cif", "A1-924,100"),
    ("bar route A", "perf/ceilrfd3/targets/laczc_1008.cif", "A1-1008,528"),
    ("bar route B", "perf/bhdesign/targets/big_1831.cif", "A1-1008,B1-448,80"),
]

#: Every name an alignment could arrive under. Checked against the feature dict rather than
#: against the source, because a track can exist in code and never be built for this path.
MSA_NAMES = ("msa", "alignment", "a3m", "profile", "deletion")


def rung(name: str, target: str, contig: str) -> dict:
    comps = parse_contig(contig)
    spec = InputSpecification.from_dict({"input": str(ROOT / target), "contig": contig})
    spec.validate()
    feats = featurize(ROOT / target, spec)
    shapes = {k: list(getattr(v, "shape", ())) for k, v in sorted(feats.items())}
    # The token axis, three ways: the engine's pre-device count, the featurizer's own token
    # plan, and the designed part of it. They have to agree.
    tokens = int(feats["restype"].shape[0]) if "restype" in feats else None
    return {
        "rung": name, "target": target, "contig": contig,
        "contig_residue_count": contig_residue_count(comps),
        "designed_residues": contig_residue_count(
            [c for c in comps if getattr(c, "chain", None) is None]),
        "feature_tokens": tokens,
        "feature_keys": sorted(feats),
        "feature_shapes": shapes,
        "msa_like_keys": sorted(k for k in feats
                                if any(n in k.lower() for n in MSA_NAMES)),
    }


def main() -> int:
    src = sorted((ROOT / "tt_bio" / "rfd3").glob("*.py"))
    hits = {p.name: len(re.findall(r"\bmsa\b", p.read_text(), re.I)) for p in src}
    out = {
        "rungs": [rung(*r) for r in RUNGS],
        # The source-side half of the depth claim. The feature dict above is the half that
        # matters (a track can exist and go unused); this says there is nothing to use.
        "msa_mentions_in_tt_bio_rfd3": {k: v for k, v in hits.items() if v},
        "msa_mentions_total": sum(hits.values()),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
