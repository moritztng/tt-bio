"""Why four PoseBusters complexes die in the Kabsch SVD while upstream docks them.

Docks one complex with `tt_bio.diffdock.sampling._repair_nans` wrapped, printing per step how
many entries of each score head are NaN and how many are infinite, and whether the all-NaN
fallback fired. Upstream's guard and ours both key on NaN only, so an overflow to +/-inf passes
straight through to `modify_conformer`, where the rigid-transform SVD raises; this says which
of the two it is, on the real embedding, without changing the run.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from pose_eval import PB, smiles_of  # noqa: E402
from tt_bio.diffdock import run, sampling  # noqa: E402
from tt_bio.diffdock.job import DockJob, DockOptions  # noqa: E402


def census(name, t):
    if t is None:
        return f"{name}=none"
    nan = int(torch.isnan(t).sum())
    inf = int(torch.isinf(t).sum())
    finite = t[torch.isfinite(t)]
    mx = float(finite.abs().max()) if finite.numel() else float("nan")
    return f"{name} nan={nan} inf={inf} n={t.numel()} absmax_finite={mx:.4g}"


STEP = [0]
_orig = sampling._repair_nans


def probe(tr, rot, tor):
    i = STEP[0]
    STEP[0] += 1
    bad = any(x is not None and not bool(torch.isfinite(x).all()) for x in (tr, rot, tor))
    if bad:
        print(f"step {i}: " + " | ".join(census(n, t) for n, t in
                                         (("tr", tr), ("rot", rot), ("tor", tor))), flush=True)
    out = _orig(tr, rot, tor)
    if bad:
        print(f"step {i}: after repair " + " | ".join(census(n, t) for n, t in
                                                      zip(("tr", "rot", "tor"), out)), flush=True)
    return out


sampling._repair_nans = probe

cid = sys.argv[1]
out = Path(sys.argv[2])
job = DockJob(name=cid, receptor_path=Path(f"{PB}/{cid}/{cid}_protein.pdb"), ligand=smiles_of(cid))
summary = run.dock([job], DockOptions(seed=42), out, isolate=True)
print("SUMMARY", summary["failed"] or "docked", flush=True)
