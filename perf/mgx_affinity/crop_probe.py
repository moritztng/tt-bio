#!/usr/bin/env python3
"""The pocket crop Nesso-1 picks after its first trunk pass, device or CPU, same featurization.

    crop_probe.py <input.yaml> <out.json> [--cpu] [--seed N] [--force <crop.json>]

Runs `nesso1.screen` as the CLI does and stops at the crop: it records the kept token indices
and each protein token's expected distance to the ligand, then aborts the record. On a tiled
target the kept residues say which copy of the repeat the pocket landed on.

--force takes another run's crop json, uses its kept tokens instead of this run's own choice and
scores the record to the end, so the affinity heads see exactly the other run's pocket.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402

from tt_bio import nesso1  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("out")
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--seed", type=int, default=nesso1.DEFAULT_SEED)
ap.add_argument("--force")
a = ap.parse_args()
forced = json.loads(Path(a.force).read_text())["keep"] if a.force else None

got = {}
orig = nesso1.Nesso1._select_pocket_indices


class Stop(Exception):
    pass


def record(self, pdistogram, feats, *, cutoff, max_tokens):
    keep = orig(self, pdistogram, feats, cutoff=cutoff, max_tokens=max_tokens)
    d_exp = nesso1.compute_expected_distance(pdistogram, max_dist=self.max_dist)
    mt = feats["mol_type"][0]
    pad = feats["token_pad_mask"][0].bool()
    prot = torch.where((mt == nesso1.const.chain_type_ids["PROTEIN"]) & pad)[0]
    lig = torch.where((mt == nesso1.const.chain_type_ids["NONPOLYMER"]) & pad)[0]
    got.update(keep=keep.cpu().tolist(), n_protein=int(prot.numel()),
               min_d=d_exp[0][prot[:, None], lig[None, :]].min(dim=1).values.float().cpu().tolist())
    if forced is None:
        raise Stop
    got.update(own_keep=got["keep"], keep=forced, forced_from=a.force)
    return torch.tensor(forced, device=pdistogram.device, dtype=torch.long)


nesso1.Nesso1._select_pocket_indices = record
if not a.cpu:
    from tt_bio.main import _require_ttnn, ensure_p300_mesh_descriptor
    _require_ttnn()
    ensure_p300_mesh_descriptor()
src = Path(tempfile.mkdtemp())
(src / Path(a.input).name).write_text(Path(a.input).read_text())
rows = nesso1.screen(src, src / "out", use_tenstorrent=not a.cpu, seed=a.seed)
got.update(input=a.input, device=not a.cpu, seed=a.seed)
if forced is not None:
    got.update(row=rows[0])
Path(a.out).write_text(json.dumps(got) + "\n")
print("kept", len(got.get("keep", [])), "tokens")
