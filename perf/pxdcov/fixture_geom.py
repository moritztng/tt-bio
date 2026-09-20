"""Why the two-chain 1536-token route has a 93 A target fit, and the one-chain route 0.15 A.

`fit_rmsd` is the residual of one RIGID fit of the model's reconstruction of the conditioned
tokens onto the coordinates it was conditioned on. The conditioning is a 64-bin distogram over
2-22 A (`featurize.TEMPL_BIN_MAX`), so every pair further apart than 22 A lands in the top bin
and carries NO geometry. Two chains the fixture placed side by side are therefore unplaced
relative to each other as far as the model can see: it packs them somewhere, and one rigid fit
over the union is large no matter how well each chain is reproduced.

This measures that, rather than asserting it: per rung, the conditioned coordinates straight out
of `design_inputs_from_yaml`, the chain separation, and the fraction of conditioned pairs the
distogram cannot represent -- in total and across the chain boundary.

    TT_VISIBLE_DEVICES= python3 perf/pxdcov/fixture_geom.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "bhdesign"))

from ladder import pxdesign_fixture  # noqa: E402
from tt_bio.pxdesign.featurize import TEMPL_BIN_MAX  # noqa: E402
from tt_bio.pxdesign.inputs import design_inputs_from_yaml, read_design_yaml  # noqa: E402

PX = ROOT / "perf" / "pxdesign" / "targets" / "laczc_1008.cif"
BIG = ROOT / "perf" / "bhdesign" / "targets" / "big_1831.cif"
RUNGS = [(960, 64, PX), (1008, 528, PX), (1008, 80, BIG), (1456, 80, BIG)]


def main() -> int:
    work = pathlib.Path(__file__).parent / "work_tokens"
    work.mkdir(parents=True, exist_ok=True)
    out = []
    for tres, binder, target in RUNGS:
        y = pxdesign_fixture(work, tres, target, binder)
        spec = read_design_yaml(y)
        feats = design_inputs_from_yaml(y)
        cond = feats["condition"]
        keep = (torch.tensor([n != "xpb" for n in cond["res_name"]])
                & cond["is_resolved"].bool())
        xyz = cond["coord"][keep].double()
        # The chain label lives in the model input, not in `condition`: `cond` carries only
        # coordinates and residue identity. Both are per TOKEN, so one mask indexes both.
        asym = feats["asym_id"][keep]
        d = torch.cdist(xyz, xyz)
        n = d.shape[0]
        offdiag = ~torch.eye(n, dtype=torch.bool)
        far = (d > TEMPL_BIN_MAX) & offdiag
        row = {"target_residues": tres, "binder": binder, "target_cif": target.name,
               "chains_conditioned": sorted(spec["chains"]),
               "conditioned_tokens": n,
               "pairs_beyond_bin_max_frac": round(float(far.sum() / offdiag.sum()), 4)}
        if len(set(asym.tolist())) > 1:
            same = asym[:, None] == asym[None, :]
            cross = ~same
            row.update({
                "cross_chain_min_dist_a": round(float(d[cross].min()), 1),
                "cross_chain_pairs_beyond_bin_max_frac": round(
                    float((far & cross).sum() / cross.sum()), 4),
                "within_chain_pairs_beyond_bin_max_frac": round(
                    float((far & same & offdiag).sum() / (same & offdiag).sum()), 4)})
        out.append(row)
        print(json.dumps(row))
    (pathlib.Path(__file__).parent / "fixture_geom.json").write_text(
        json.dumps({"templ_bin_max_a": TEMPL_BIN_MAX, "rungs": out}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
