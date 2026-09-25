#!/usr/bin/env python3
"""of3t-confpfe: which frame is right at auxgrad's boundary.

    boundary_check.py --boundary B.pt --probe conf_bf16_tape.pt --checkpoint CK --out OUT.json

Three resolved / plddt / pae readings at upstream's captured boundary, all on the logits
upstream's own head gathered (existing atoms, real pairs): upstream's captured outputs (the
function auxgrad scored against), conf_split's float64 formula at the same inputs, and the device
head (head_probe.py on this boundary).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conf_split as cs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from openfold3.core.utils.atomize_utils import (broadcast_token_feat_to_atoms,
                                                    get_token_representative_atoms)
    dt = torch.float64
    B = torch.load(a.boundary, map_location="cpu", weights_only=False)
    kw, pos = B["inputs"]["kwargs"], B["inputs"]["args"]
    batch = kw["batch"] if "batch" in kw else pos[0]
    si_input = kw["si_input"] if "si_input" in kw else pos[1]
    outd = kw["output"] if "output" in kw else pos[2]
    s, z = outd["si_trunk"], outd["zij_trunk"]
    xpred = outd["atom_positions_predicted"].to(dtype=s.dtype)
    repr_x, repr_mask = get_token_representative_atoms(batch=batch, x=xpred,
                                                       atom_mask=batch["atom_mask"])
    tok = batch["token_mask"]
    n_tok = int(s.shape[-2])
    mapm = broadcast_token_feat_to_atoms(token_mask=tok, num_atoms_per_token=batch[
        "num_atoms_per_token"], token_feat=tok, max_num_atoms_per_token=23).reshape(-1).bool()
    real = tok.reshape(-1).bool()
    cfg, model, _d, _c = cs.ref_step.load(dt, a.checkpoint, 20260919, None)
    sq = lambda t: t.reshape(1, n_tok, *t.shape[-1:]).to(dt)  # noqa: E731
    with torch.no_grad():
        mine = cs.heads(model, sq(si_input), sq(s), z.reshape(1, n_tok, n_tok, -1).to(dt),
                        repr_x.reshape(n_tok, 3).to(dt), repr_mask.reshape(1, n_tok).to(dt), None)
    ref = B["outputs"]
    dev = torch.load(a.probe, weights_only=False)["outputs"]
    gather = lambda t, c: t.reshape(n_tok * 23, c)[mapm].to(dt)  # noqa: E731
    pair = lambda t: t.reshape(n_tok, n_tok, -1)[real][:, real].to(dt)  # noqa: E731
    rec = {"boundary": str(a.boundary), "probe": str(a.probe)}
    for name, key, c in (("resolved", "experimentally_resolved_logits", 2), ("plddt", "plddt_logits", 50)):
        up = ref[key].reshape(-1, c).to(dt)
        rec[name] = {"mine_vs_upstream": cs.rel(gather(mine[f"{name if name == 'plddt' else 'resolved'}_logits"], c), up),
                     "device_vs_upstream": cs.rel(gather(dev[key], c), up),
                     "device_vs_mine": cs.rel(gather(dev[key], c),
                                              gather(mine[f"{name if name == 'plddt' else 'resolved'}_logits"], c))}
    up = ref["pae_logits"]
    rec["pae"] = {"mine_vs_upstream": cs.rel(pair(mine["pae_logits"]), pair(up)),
                  "device_vs_upstream": cs.rel(pair(dev["pae_logits"]), pair(up))}
    rec["upstream_output_shapes"] = {k: list(v.shape) for k, v in ref.items() if torch.is_tensor(v)}
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
