#!/usr/bin/env python3
"""Determinism hammer for the OpenFold3 reference-conformer pool. No device, no fold.

`ref_pos` is a model input, and it comes from `_pooled_reference_molecules` running one
ETKDG embedding per residue on 24 threads. A residue that needs a SECOND embedding -- its
first strategy timed out or failed to embed -- draws a second seed, and where that seed
comes from decides whether the blast radius is one residue or the whole complex
(wk/wh-correctness-parity-p3, tt-bio c8fdc1b8).

Hammering the end-to-end fold cannot measure that. A clean fold of examples/fkg_ligand.yaml
draws exactly 108 seeds, one per job, so no residue retries and the fixed and unfixed trees
are bit-identical -- 200 canonical folds would say nothing about the retry path. This runs
the pool itself, thousands of times, and reports the number of DISTINCT conformer hashes.
1 means the pool is a pure function of its seeds; more than 1 is the bug.

Two knobs, and the difference between them matters:

  --budget N   Scales the wall-clock budget the `default`/`random_init` ETKDG strategies
               get (production: 120 s, query.py). A small budget makes a real embedding
               exceed a real budget, so `func_timeout` raises the same FunctionTimedOut a
               starved host raises, at the same point -- after the pooled seed has been
               consumed. The timeout is natural; only the budget is scaled. This is how a
               ~1e-4 event becomes measurable.
  --ligand CCD Hammers one ligand's conformer instead of the protein. A hard ligand
               (macrocycle, fused rings) fails the `default` strategy on its own merits
               and advances the chain with no budget change at all.
  --force-n N  Times out the first N DISTINCT pooled seeds, once each. This is the arm
               that isolates the bug, and --budget cannot replace it: a small budget
               makes the SET of residues that time out a wall-clock race, so the output
               varies run to run on any tree, fixed or not, and the measurement cannot
               tell an RNG-stream defect from a different set of residues having stalled.
               Forcing by seed holds that set fixed, so the only thing left that can move
               the output is where the retry's seed came from.

Run it against two checkouts to get a negative control: the unfixed tree must report more
than one hash under the same conditions, or the arm proves nothing.

    scripts/conformer_pool_hammer.py --iters 200 --budget 0.002 --label fix
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# examples/fkg_ligand.yaml -- the target both production anomalies were folds of.
FKG_SEQ = ("GVQVETISPGDGRTFPKRGQTCVVHYTGMLEDGKKFDSSRDRNKPFKFMLGKQEVIRGWEEGVAQMSVGQRAK"
           "LTISPDYAYGATGHPGIIPPHATLVFDVELLKLE")


def _per_mol_hashes(mols) -> list:
    """One hash per reference molecule, so a divergence can be sized as well as detected.

    The distinction the whole pass turns on: the unfixed tree threw the pooled result away
    and recomputed every residue, so ONE stalled residue moved all of them. The fixed tree
    can only move the residues that actually stalled. Same verdict on "did it diverge",
    opposite answers on "how much".
    """
    out = []
    for m in mols:
        conf = m.mol.GetConformer(0)
        h = hashlib.sha256()
        for i in range(m.mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            h.update(f"{p.x:.6f},{p.y:.6f},{p.z:.6f};".encode())
        out.append(h.hexdigest()[:12])
    return out


def _hash_ref_mols(mols) -> str:
    """sha256 over every reference molecule's conformer coordinates, in job order.

    These coordinates ARE ref_pos. Rounding to 1e-6 A keeps the hash from reporting
    float noise that no downstream feature can see; the bug moves atoms by 0.1-0.3 A.
    """
    h = hashlib.sha256()
    for m in mols:
        conf = m.mol.GetConformer(0)
        for i in range(m.mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            h.update(f"{p.x:.6f},{p.y:.6f},{p.z:.6f};".encode())
        h.update(b"|")
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0,
                    help="global RNG seed, re-applied before every iteration: the pool's "
                         "seed list is then identical every time, so any difference in "
                         "the output is the bug and not a different draw")
    ap.add_argument("--budget", type=float, default=None,
                    help="seconds for the default/random_init ETKDG strategies "
                         "(production 120)")
    ap.add_argument("--ligand", default=None, help="hammer this CCD code instead")
    ap.add_argument("--force-n", type=int, default=0,
                    help="time out the first N distinct pooled seeds, once each, so the "
                         "retry set is identical every iteration")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None, help="write the per-iteration record here")
    args = ap.parse_args()

    from tt_bio._vendor.openfold3.core.data.primitives.structure import conformer as C
    from tt_bio._vendor.openfold3.core.data.primitives.structure import query as Q

    if args.budget is not None:
        # Reach the hard-coded production budget through the one function that reads it,
        # rather than editing the vendored file: the chain, the strategies and the
        # exception all stay exactly what a fold runs.
        orig = Q.multistrategy_compute_conformer
        b = args.budget

        def scaled(mol, **kw):
            if kw.get("timeouts"):
                kw["timeouts"] = {k: b for k in kw["timeouts"]}
            return orig(mol, **kw)

        Q.multistrategy_compute_conformer = scaled

    if args.force_n:
        # Verbatim in mechanism from wh-correctness-parity-p3's probe: raise the same
        # FunctionTimedOut func_timeout raises, from the same call, AFTER the pooled seed
        # has been consumed onto the strategy. Confined to the pool phase -- _POOL.retries
        # is thread-local and only a worker thread that took a pooled seed has it set, so
        # the unfixed tree's sequential whole-set recompute is never forced.
        orig_ft = C.func_timeout
        forced: dict = {}
        import threading as _th
        flock = _th.Lock()

        def ft(timeout, func, args=(), kwargs=None):
            strategy = args[1] if len(args) > 1 else None
            seed = getattr(strategy, "randomSeed", None)
            pooled = (getattr(C._POOL, "retries", None) is not None
                      or getattr(C._POOL, "active", False))
            if seed is not None and pooled:
                with flock:
                    if seed not in forced and len(forced) < args_force_n[0]:
                        forced[seed] = 1
                        raise C.FunctionTimedOut(
                            f"hammer: forced ETKDG timeout on pooled seed {seed}")
            return orig_ft(timeout=timeout, func=func, args=args, kwargs=kwargs or {})

        args_force_n = [args.force_n]
        C.func_timeout = ft

    draws = {"n": 0}
    orig_seed_fn = C._etkdg_seed

    def counted():
        draws["n"] += 1
        return orig_seed_fn()

    C._etkdg_seed = counted

    hashes: Counter = Counter()
    records = []
    t0 = time.time()
    # Iteration -1 is the clean draw: the same seed with nothing forced. It is the baseline
    # every later iteration is sized against, so "moved" means "moved away from the fold a
    # host that never stalled would have produced" -- the quantity that matters -- and not
    # "differs from whatever the first forced iteration happened to give".
    baseline = None
    for it in range(-1, args.iters):
        force_this = args.force_n if it >= 0 else 0
        if args.force_n:
            args_force_n[0] = force_this
        random.seed(args.seed)
        draws["n"] = 0
        if args.force_n:
            forced.clear()
        t1 = time.time()
        if args.ligand:
            swm = Q.structure_with_ref_mol_from_ccd_code(args.ligand, chain_id="B")
            mols = [swm.processed_reference_mols] if not isinstance(
                swm.processed_reference_mols, list) else swm.processed_reference_mols
        else:
            from tt_bio._vendor.openfold3.core.data.resources.residues import MoleculeType
            swm = Q.structure_with_ref_mols_from_sequence(
                FKG_SEQ, MoleculeType.PROTEIN, "A")
            mols = swm.processed_reference_mols
        h = _hash_ref_mols(mols)
        per = _per_mol_hashes(mols)
        if baseline is None:
            baseline = per
            clean_hash = h
            print(f"{args.label} clean-baseline hash={h} draws={draws['n']} "
                  f"jobs={len(mols)}", flush=True)
            continue
        hashes[h] += 1
        moved = sum(1 for a, b in zip(baseline, per) if a != b)
        records.append({"iter": it, "hash": h, "n_draws": draws["n"],
                        "n_jobs": len(mols), "mols_moved_vs_clean": moved,
                        "secs": round(time.time() - t1, 3)})
        print(f"{args.label} it={it} hash={h} draws={draws['n']} jobs={len(mols)} "
              f"moved={moved}/{len(mols)} {records[-1]['secs']}s", flush=True)

    retried = sum(1 for r in records if r["n_draws"] > r["n_jobs"])
    moved = [r["mols_moved_vs_clean"] for r in records]
    summary = {
        "label": args.label, "iters": args.iters, "budget": args.budget,
        "ligand": args.ligand, "seed": args.seed,
        "distinct_hashes": len(hashes), "hashes": hashes.most_common(),
        "iters_with_a_retry": retried,
        "force_n": args.force_n,
        "clean_hash": clean_hash,
        "matches_clean": hashes.get(clean_hash, 0),
        "mols_moved_vs_clean": {"max": max(moved) if moved else 0,
                              "mean": round(sum(moved) / len(moved), 2) if moved else 0,
                              "n_jobs": records[0]["n_jobs"]},
        "draws_seen": sorted({r["n_draws"] for r in records}),
        "total_secs": round(time.time() - t0, 1),
        "commit": os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip(),
    }
    print("\nSUMMARY " + json.dumps(summary))
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"summary": summary, "records": records}, indent=2))
    # Exit code is the verdict: one hash is determinism, more than one is the bug.
    return 0 if len(hashes) == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
