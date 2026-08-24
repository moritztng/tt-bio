"""p120 -- Anchor 1's committed bars, applied on R4, where the block-sparse arm is actually live.

Why this exists. `p118` ran the release gate's own RFD3 leg on both arms and both passed, but the
on-arm run reported `0 blocked, 1791 dense-fallback`: the arm was DARK. The gate's fixture is
`examples/rfd3_binder.json`, 1350 atoms -> 1376 padded -> 43 tiles, and `block_sparse.plan()`
needs the padded query axis to be divisible by Q=1216. It is not, so every step took the dense
fallback and `gate=True` on the "on" arm scored the shipped chain twice. That is
`pcc-gate-can-pass-without-the-op-it-names`.

So the arm has to be scored where it is live. R4 (6051 atoms -> 6080 padded -> 190 tiles = 5x1216)
is the fixture every measurement in this lineage used. This script applies the gate's OWN scoring
and the gate's OWN committed thresholds -- `release_gate._rfd3_score_cif`, `RFD3_MIN_INBAND`,
`RFD3_MAX_BREAKS`, `RFD3_MAX_CLASHES`, `RFD3_MIN_DISTINCT_AA`, `RFD3_MAX_UNK` -- to R4 CIFs from
either arm. No new bar, no new threshold, no re-derived metric; only the fixture moves, because on
the gate's fixture there is nothing to score.

Usage: p120_anchor1_r4.py <spec.json> <out.json> <label>=<cif_dir> [<label>=<cif_dir> ...]
"""
import importlib.util
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
SPEC = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
ARMS = [a.split("=", 1) for a in sys.argv[3:]]

spec_rg = importlib.util.spec_from_file_location("rg", REPO / "scripts" / "release_gate.py")
rg = importlib.util.module_from_spec(spec_rg)
sys.modules["rg"] = rg
spec_rg.loader.exec_module(rg)

from tt_bio.rfd3.design import _chain_label            # noqa: E402
from tt_bio.rfd3.featurize import DESIGNED_RESTYPE_IDX, featurize  # noqa: E402
from tt_bio.rfd3.input import InputSpecification       # noqa: E402

raw = json.loads(SPEC.read_text())
spec_id = next(iter(raw))
ispec = InputSpecification.from_dict(raw[spec_id])
f = featurize(str(REPO / ispec.input), ispec)
rt = f["restype"]
rt = rt.argmax(-1) if rt.ndim == 2 else rt
mask = (rt == DESIGNED_RESTYPE_IDX) & f["is_protein"].bool()
asym, resid = f["asym_id"].tolist(), f["residue_index"].tolist()
designed = {(_chain_label(int(asym[t])), int(resid[t]))
            for t in range(len(rt)) if bool(mask[t])}
print("[p120] %s:%s designed residues from the host featurizer: %d"
      % (SPEC.name, spec_id, len(designed)), flush=True)

geom = rg._load_geometry_harness()
res = {"spec": str(SPEC), "spec_id": spec_id, "n_designed": len(designed),
       "thresholds": {"RFD3_MIN_INBAND": rg.RFD3_MIN_INBAND,
                      "RFD3_MAX_BREAKS": rg.RFD3_MAX_BREAKS,
                      "RFD3_MAX_CLASHES": rg.RFD3_MAX_CLASHES,
                      "RFD3_MIN_DISTINCT_AA": rg.RFD3_MIN_DISTINCT_AA,
                      "RFD3_MAX_UNK": rg.RFD3_MAX_UNK,
                      "RFD3_MIN_CLEAN_RATE": rg.RFD3_MIN_CLEAN_RATE},
       "arms": {}}

for label, d in ARMS:
    cifs = sorted(pathlib.Path(d).glob("*.cif"))
    rg._parse_gate(cifs, name="rfd3-%s" % label)          # same parse gate the leg runs
    per = [rg._rfd3_score_cif(c, designed, geom) for c in cifs]
    for m in per:
        m["clean"] = (m["breaks"] <= rg.RFD3_MAX_BREAKS
                      and m["in_band"] >= rg.RFD3_MIN_INBAND)
    row = {
        "dir": d, "n_designs": len(cifs), "per_design": per,
        "clean_rate": sum(m["clean"] for m in per) / len(per),
        "in_band": min(m["in_band"] for m in per),
        "breaks": max(m["breaks"] for m in per),
        "clashes": max(m["clashes"] for m in per),
        "clash_frac": max(m["clash_frac"] for m in per),
        "distinct_aa": min(m["distinct_aa"] for m in per),
        "unk": max(m["unk"] for m in per),
    }
    # Same aggregation and same conjunction the leg uses, minus `determinism`, which is a
    # two-fresh-process check and needs the card. p107 already establishes it for this arm.
    row["gate_ex_determinism"] = (row["clean_rate"] >= rg.RFD3_MIN_CLEAN_RATE
                                  and row["clashes"] <= rg.RFD3_MAX_CLASHES
                                  and row["distinct_aa"] >= rg.RFD3_MIN_DISTINCT_AA
                                  and row["unk"] <= rg.RFD3_MAX_UNK)
    res["arms"][label] = row
    print("[p120] %-4s n=%d in_band %.4f breaks %d clashes %d distinct_aa %d unk %d "
          "clean_rate %.2f -> %s"
          % (label, row["n_designs"], row["in_band"], row["breaks"], row["clashes"],
             row["distinct_aa"], row["unk"], row["clean_rate"],
             "PASS" if row["gate_ex_determinism"] else "FAIL"), flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=2, default=str) + "\n")
print("wrote", OUT, flush=True)
