#!/usr/bin/env python3
"""r2's tripwire: score a checkpoint on the validation split so the run is falsifiable in days.

    TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=0,2 TT_BIO_LEASE_HOLDER=worker:train-b3-train \\
    PYTHONPATH=$PWD python3 scripts/abb3_port/tripwire.py --checkpoint runs/base/checkpoints/step-000005805.safetensors

r2's rule is that a folding model reaches ~90 % of its final accuracy inside the first few
percent of its budget. At 193,512 steps that puts the read at **step 5,805 (3 %)** with a
confirmation at **19,351 (10 %)**, and at the measured 19.5 s/step on one board pair the first
one is 31 hours in -- so a run that is going nowhere is knowable in a day instead of a month.

**"90 % of final accuracy" is stated as a fraction of the distance travelled, because the metric
is an error.** Taking 90 % of an RMSD would be a different and much weaker claim: an untrained
model already scores a finite number, so "within 10 % of 2.7 A" can be satisfied by a model that
has learned almost nothing. What has to be 90 % complete is the journey:

    progress = (untrained - now) / (untrained - target)

Both ends are measured in this same run rather than assumed. ``untrained`` is the seeded
initialisation scored on the same structures, and ``target`` is **their own released
predictions** scored by the same instrument on the same structures.

**The target is their UNREFINED prediction, and that is the apples-to-apples one.** Their
published 2.714 A is after OpenMM refinement; ours is unrefined, and refining 150 structures at
every checkpoint would cost more than the training between them. `train-b2-abb3-port`'s fold
gate already made this comparison the same way. The tripwire is about the trajectory, so the
refinement step -- which is the same for both sides at the end -- is out of it. The final bar
stays 2.714 A on the refined 250 and is not this script's job.

Everything structural goes through `train-b1-instrument`'s merged instrument via
``abodybuilder3_output.score_fv``, which is one line over ``antibody_rmsd.score_pdb_pair``. The
published per-region backbone is N/CA/C/**CB**, not N/CA/C/O, and the default here is the
instrument's own constant rather than a literal, so this file cannot drift from it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tt_bio import autograd as ag
from tt_bio.abodybuilder3 import to_device_fp32
from tt_bio.abodybuilder3_output import (atom14_from_frames, device_outputs_to_host, score_fv,
                                         write_fv_pdb)
from tt_bio.abodybuilder3_reference import ABB3Config, single_and_pair_features
from tt_bio.af2_data import RESTYPE_ATOM14_MASK
from tt_bio.tenstorrent import get_device
from tt_bio.train.abb3_checkpoint import load_run_state
from tt_bio.train.abb3_dataset import resolve_split
from tt_bio.train.hostreduce import master_hash
from tt_bio.train.abodybuilder3_step import TrainStep

#: r2's rule, as fractions of the schedule. Kept here and in ``abb3_run.TRIPWIRE_FRACTIONS``
#: for the same reason the checkpoints at those steps are protected from pruning.
FRACTIONS = (0.03, 0.10)
#: The fraction of the untrained-to-target distance that must be covered at 3 %.
BAR = 0.90


def fold_one(model, rec: dict, cfg: ABB3Config, tmp: Path, name: str) -> Path:
    """One Fv through the device model, written as a PDB their instrument can read.

    Under ``no_grad`` because this is evaluation: the tape would grow for a backward that never
    comes. It is also the SAME forward training uses -- grad-off is bit-identical to grad-on on
    all 27 gradchecked ops -- so the structure scored here is the structure being trained.
    """
    aatype = rec["aatype"].long()
    is_heavy = rec["is_heavy"].long()
    single, pair = single_and_pair_features(aatype.unsqueeze(0), is_heavy.unsqueeze(0),
                                            rec["residue_index"].long().unsqueeze(0))
    mask = torch.ones(1, len(aatype))
    square = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).unsqueeze(1)
    with ag.no_grad():
        out = model(to_device_fp32(single), to_device_fp32(pair), to_device_fp32(square),
                    to_device_fp32(cfg.inf * (square - 1.0)))
        host = device_outputs_to_host(out, cfg.no_angles, taped=True)
    atom14 = atom14_from_frames(host["frames"], host["angles"], aatype.unsqueeze(0))[0]
    atom_mask = torch.as_tensor(RESTYPE_ATOM14_MASK)[aatype]
    return write_fv_pdb(tmp / f"{name}.pdb", aatype, atom14, atom_mask, int(is_heavy.sum()))


def score_split(model, ids, structures: Path, released: Path, cfg, tmp: Path,
                label: str) -> dict:
    """Fold and score every id, beside their own released unrefined prediction."""
    ours, theirs, skipped = [], [], []
    t0 = time.perf_counter()
    for i, name in enumerate(ids):
        rec = torch.load(structures / f"{name}.pt", weights_only=False)
        true_pdb = released / "true" / f"{name}.pdb"
        if not true_pdb.is_file():
            skipped.append(name)
            continue
        pred = fold_one(model, rec, cfg, tmp, name)
        ours.append(score_fv(true_pdb, pred, rec["region"])["rmsd_cdrh3"])
        ref = released / "pred_unfixed" / f"{name}.pdb"
        if ref.is_file():
            theirs.append(score_fv(true_pdb, ref, rec["region"])["rmsd_cdrh3"])
        if (i + 1) % 25 == 0:
            print(f"  [{label}] {i + 1}/{len(ids)}  mean so far "
                  f"{statistics.mean(ours):.3f} A", flush=True)
    return {"label": label, "n": len(ours), "cdrh3": statistics.mean(ours) if ours else None,
            "cdrh3_sd": statistics.stdev(ours) if len(ours) > 1 else None,
            "theirs_unrefined": statistics.mean(theirs) if theirs else None,
            "skipped": skipped, "seconds": time.perf_counter() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="the checkpoint to score; omit to score the untrained init only")
    ap.add_argument("--split", default="valid", choices=("valid", "test", "train"))
    ap.add_argument("--n", type=int, default=0, help="limit the structures, 0 for all")
    ap.add_argument("--structures",
                    default="/home/ttuser/abb3_data/data/structures/structures")
    ap.add_argument("--released", default="/home/ttuser/abb3/base-loss")
    ap.add_argument("--split-csv",
                    default="/home/ttuser/abb3_src/ABodyBuilder3/data/split.csv")
    ap.add_argument("--steps", type=int, default=193_512, help="the full schedule, for the %")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    released = Path(args.released)
    ids = resolve_split(args.split_csv,
                        released / "true" if (released / "true").is_dir() else None)[args.split]
    if args.n:
        ids = ids[:args.n]
    structures = Path(args.structures)
    tmp = Path("/tmp/abb3_tripwire")
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = ABB3Config(use_plddt=False, no_blocks=args.blocks)

    dev = get_device()
    try:
        # Both arms share one process and one device context, so the untrained baseline and the
        # checkpoint are scored by the same instrument on the same structures on the same card.
        # Two processes would put a device-open and a fresh compile cache between them.
        torch.manual_seed(args.seed)
        from tt_bio.abodybuilder3_reference import ABB3StructureModule
        step = TrainStep(ABB3StructureModule(cfg).state_dict(), cfg,
                         accumulate=16, seed=args.seed)
        print(f"scoring {len(ids)} {args.split} structures\n")
        base = score_split(step.model, ids, structures, released, cfg, tmp, "untrained")
        print(f"\nuntrained : mean CDR-H3 {base['cdrh3']:.3f} A over {base['n']}")
        print(f"target    : their unrefined {base['theirs_unrefined']:.3f} A over the same "
              f"structures\n")

        now = None
        digests = None
        if args.checkpoint:
            # A checkpoint early in the run scores the SAME as the untrained model, correctly
            # and uninformatively -- which is exactly what a checkpoint that silently failed to
            # load also looks like. So the load is proved rather than inferred from the score:
            # the masters are hashed either side of it and an unchanged hash is a hard failure.
            before = master_hash([m.detach().cpu().numpy() for m in step.mirror])
            state = load_run_state(Path(args.checkpoint), step)
            after = master_hash([m.detach().cpu().numpy() for m in step.mirror])
            if before == after:
                raise SystemExit(
                    f"{args.checkpoint} left every master weight untouched ({after.hex()[:12]}), "
                    f"so this would have scored the untrained model twice and reported it as a "
                    f"checkpoint. Either the file holds the initialisation or the load is broken")
            digests = {"untrained": before.hex(), "loaded": after.hex()}
            print(f"checkpoint loaded: masters {before.hex()[:12]} -> {after.hex()[:12]}\n")
            gs = state.step
            now = score_split(step.model, ids, structures, released, cfg, tmp,
                              f"step {gs}")
            print(f"\nstep {gs:,}: mean CDR-H3 {now['cdrh3']:.3f} A over {now['n']}")
        else:
            gs = 0

        report = {"split": args.split, "n": base["n"], "untrained": base, "checkpoint": now,
                  "global_step": gs, "schedule": args.steps, "bar": BAR,
                  "fractions": list(FRACTIONS), "master_digests": digests}
        if now is not None:
            span = base["cdrh3"] - base["theirs_unrefined"]
            progress = (base["cdrh3"] - now["cdrh3"]) / span if span else float("nan")
            pct = 100.0 * gs / args.steps
            report.update({"progress": progress, "percent_of_schedule": pct, "span": span})
            print(f"\nTRIPWIRE at {pct:.2f} % of the schedule (step {gs:,} of {args.steps:,})")
            print(f"  untrained {base['cdrh3']:.3f} A -> target {base['theirs_unrefined']:.3f} A "
                  f"is a span of {span:.3f} A")
            print(f"  this checkpoint is at {now['cdrh3']:.3f} A, i.e. "
                  f"{100 * progress:.1f} % of the way")
            if pct >= 100 * FRACTIONS[0]:
                ok = progress >= BAR
                print(f"  {'PASS' if ok else 'FAIL'}: the bar at "
                      f"{100 * FRACTIONS[0]:.0f} % of budget is {100 * BAR:.0f} % of the span. "
                      + ("" if ok else "A run this far behind at this point does not catch up "
                                       "-- stop and report rather than spending the month."))
            else:
                print(f"  below the {100 * FRACTIONS[0]:.0f} % read point, so this is a "
                      f"trajectory sample and not yet a verdict")
        out = Path(args.report) if args.report else tmp / "tripwire.json"
        out.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"\nreport: {out}")
        return 0
    finally:
        import ttnn
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
