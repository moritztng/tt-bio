#!/usr/bin/env python3
"""Boltz-2 binding-affinity implementation parity: tt-bio device vs the official Boltz-2 reference.

The structure legs of docs/implementation-parity.md compare predicted *coordinates*
(Kabsch CA-RMSD) across seeds. Affinity prediction instead emits a scalar
(``affinity_pred_value`` = log10(IC50) in uM, MW-corrected ensemble mean over the
``--diffusion_samples_affinity`` samples and the two affinity heads; plus
``affinity_probability_binary``). A scalar has no alignment step, so the
distance is the absolute delta |device - reference|, and the same R/D/X
noise-floor framework the rest of the benchmark uses applies directly:

  R = |ref(seed i) - ref(seed j)|   across reference-seed pairs   (ref self-floor)
  D = |dev(seed i) - dev(seed j)|   across device-seed pairs      (dev self-floor)
  X = |dev(seed i) - ref(seed j)|   across all dev x ref pairs    (the parity question)

Parity holds when X sits within max(R, D): the device-vs-reference delta is
indistinguishable from the run-to-run diffusion sampling spread each
implementation already exhibits with itself. Reported as a distribution
(mean/std/min/max/n), never one number, via the shared statistical core
(`pharma_parity.noise_floor_verdict`).

Both sides run Boltz-2 affinity mode on the SAME input (a real protein-ligand
complex, msa: empty so single-sequence, no network). The reference is the
official `boltz` package (torch + pytorch-lightning, CPU); the device is the
ttnn port via `tt-bio predict --model boltz2 --affinity_mw_correction`. Both
hardcode affinity recycling_steps=5 and use --recycling_steps 3 for the
upstream structure, so the inputs and model settings are identical.

Reference output layout (official boltz):  <out>/boltz_results_<id>/predictions/<id>/affinity_<id>.json
Device output layout (tt-bio):              <out>/boltz2_results_<id>/results.json  (list, one entry per target)

Usage:
  python3 scripts/boltz2_affinity_parity.py \
      --ref-dirs /path/to/ref_seed0 /path/to/ref_seed1 /path/to/ref_seed2 \
      --dev-dirs /path/to/dev_seed0 /path/to/dev_seed1 /path/to/dev_seed2 \
      --target-id affinity_fkg [--out /tmp/affinity_parity.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

import gemmi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pharma_parity import noise_floor_verdict, summarize  # noqa: E402


# standard amino-acid three-letter codes (for ligand / pocket separation)
_AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())


def _load_atoms(path):
    st = gemmi.read_structure(str(path))
    lig = {}      # (chain, resname, atom, altloc) -> xyz  for non-AA residues
    ca = {}       # (chain, seqid) -> xyz  for CA atoms of AA residues
    for ch in st[0]:
        for res in ch:
            is_aa = res.name in _AA
            for atom in res:
                key = (ch.name, res.name, res.seqid.num, atom.name, atom.altloc)
                xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                if is_aa and atom.name == "CA":
                    ca[(ch.name, res.seqid.num)] = xyz
                elif not is_aa:
                    lig[key] = xyz
    return lig, ca


def _kabsch_rmsd(A, B):
    ca, cb = A.mean(0), B.mean(0)
    A0, B0 = A - ca, B - cb
    U, _, Vt = np.linalg.svd(A0.T @ B0)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    A_al = (R @ A0.T).T
    return float(np.sqrt(((A_al - B0) ** 2).sum() / len(A)))


def _lddt(pred, ref, cutoff=15.0):
    """lDDT over a set of atoms (pred vs ref), Mariani 4-threshold (0.5/1/2/4 A)."""
    L = len(ref)
    if L < 2:
        return 0.0
    dref = np.sqrt(((ref[:, None, :] - ref[None, :, :]) ** 2).sum(-1))
    dpred = np.sqrt(((pred[:, None, :] - pred[None, :, :]) ** 2).sum(-1))
    thr = (0.5, 1.0, 2.0, 4.0)
    per_res = np.zeros(L)
    for i in range(L):
        nb = np.where((dref[i] < cutoff) & (np.arange(L) != i))[0]
        if len(nb) == 0:
            per_res[i] = 1.0
            continue
        dd = np.abs(dref[i, nb] - dpred[i, nb])
        per_res[i] = float(np.mean([np.mean(dd < t) for t in thr]))
    return float(per_res.mean())


def _pose_metrics(dA, dB, tid):
    """Ligand-pose RMSD + pocket-lDDT for a (device=A, reference=B) run pair.

    Ligand-pose RMSD: Kabsch RMSD over the matched ligand (chain B, SB3) heavy
    atoms — how well the device places the ligand relative to the reference,
    after optimal rigid-body superposition of the ligand alone (the metric a
    pharma customer evaluating a binding pose actually feels).

    Pocket-lDDT: CA-lDDT over the pocket = ligand heavy atoms + every protein
    CA within 10 A of any ligand heavy atom in the REFERENCE structure, with
    the protein-ligand and ligand-ligand pairwise distances as the preserved
    contacts. Alignment-invariant, so no superposition is applied. Reports
    the local protein-ligand interface geometry the scalar affinity cannot.
    """
    # device layout: <dir>/structures/<tid>.cif ; reference (committed fixture): same
    a_cif = Path(dA) / "structures" / f"{tid}.cif"
    b_cif = Path(dB) / "structures" / f"{tid}.cif"
    if not a_cif.exists() or not b_cif.exists():
        return None
    a_lig, a_ca = _load_atoms(a_cif)
    b_lig, b_ca = _load_atoms(b_cif)
    lig_keys = [k for k in a_lig if k in b_lig]
    if len(lig_keys) < 3:
        return None
    A_lig = np.array([a_lig[k] for k in lig_keys])
    B_lig = np.array([b_lig[k] for k in lig_keys])
    lig_rmsd = _kabsch_rmsd(A_lig, B_lig)
    # pocket: ligand atoms + protein CA within 10 A of any ligand atom in the reference
    b_lig_pts = np.array(list(b_lig.values()))
    pocket_ca = [k for k in b_ca if np.min(np.linalg.norm(b_ca[k] - b_lig_pts, axis=1)) <= 10.0]
    # matched pocket atoms: ligand (by lig_keys) + pocket CA (by (chain, seqid))
    pk_a, pk_b = [], []
    for k in lig_keys:
        pk_a.append(a_lig[k]); pk_b.append(b_lig[k])
    for k in pocket_ca:
        if k in a_ca:
            pk_a.append(a_ca[k]); pk_b.append(b_ca[k])
    pk_a, pk_b = np.array(pk_a), np.array(pk_b)
    pocket_lddt = _lddt(pk_a, pk_b) if len(pk_a) >= 2 else 0.0
    return {"ligand_rmsd": lig_rmsd, "1-pocket_lddt": 1.0 - pocket_lddt}


AFFINITY_KEYS = ["affinity_pred_value", "affinity_probability_binary"]


def _find_ref_affinity(ref_dir: Path, target_id: str) -> Path:
    """Official boltz writes <out>/boltz_results_<id>/predictions/<id>/affinity_<id>.json."""
    cand = list(ref_dir.rglob(f"affinity_{target_id}.json"))
    if cand:
        return cand[0]
    cand = list(ref_dir.rglob("affinity_*.json"))
    if cand:
        return cand[0]
    raise FileNotFoundError(f"no affinity_*.json under reference dir {ref_dir}")


def _load_device_results(dev_dir: Path):
    """tt-bio writes results.json (a list with one entry per target)."""
    cand = list(dev_dir.rglob("results.json"))
    if not cand:
        raise FileNotFoundError(f"no results.json under device dir {dev_dir}")
    return json.loads(cand[0].read_text())


def _extract_ref(ref_dir: Path, target_id: str) -> dict:
    d = json.loads(_find_ref_affinity(ref_dir, target_id).read_text())
    return {k: float(d[k]) for k in AFFINITY_KEYS if k in d}


def _extract_dev(dev_dir: Path, target_id: str) -> dict:
    rows = _load_device_results(dev_dir)
    if isinstance(rows, dict):
        rows = [rows]
    row = None
    for r in rows:
        if r.get("id") == target_id or str(r.get("id", "")).endswith(target_id):
            row = r
            break
    if row is None and rows:
        row = rows[0]
    if row is None:
        raise FileNotFoundError(f"no results entry in {dev_dir}")
    return {k: float(row[k]) for k in AFFINITY_KEYS if k in row}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dirs", nargs="+", required=True,
                    help="official-boltz reference output dirs, one per seed")
    ap.add_argument("--dev-dirs", nargs="+", required=True,
                    help="tt-bio device output dirs, one per seed")
    ap.add_argument("--target-id", default="affinity_fkg",
                    help="target id (the yaml stem / record id)")
    ap.add_argument("--out", default="")
    ap.add_argument("--paired", action="store_true",
                    help="Also report the same-seed (diagonal dev_i vs ref_i) "
                         "distances for every metric, alongside the all-pairs cross "
                         "mean. The pass-4 RNG fix made the device and reference "
                         "share the same torch.randn stream per seed, so the "
                         "diagonal IS the shared-RNG-draw distance: if it is no "
                         "smaller than the cross mean the residual is seed-"
                         "independent (systematic bf16 arithmetic divergence), "
                         "not RNG stochasticity — the rigorous distinction between "
                         "a port defect and a bf16-precision-floor artifact.")
    args = ap.parse_args()

    ref_vals = [_extract_ref(Path(d), args.target_id) for d in args.ref_dirs]
    dev_vals = [_extract_dev(Path(d), args.target_id) for d in args.dev_dirs]

    print(f"### Boltz-2 binding-affinity parity: {args.target_id}\n")
    print(f"reference seeds: {len(ref_vals)}   device seeds: {len(dev_vals)}\n")
    print("Per-seed affinity_pred_value (log10 IC50 uM, MW-corrected) / affinity_probability_binary:")
    print("| side | seed | affinity_pred_value | affinity_probability_binary |")
    print("|---|---|---|---|")
    for i, v in enumerate(ref_vals):
        print(f"| ref | {i} | {v.get('affinity_pred_value', float('nan')):.4f} "
              f"| {v.get('affinity_probability_binary', float('nan')):.4f} |")
    for i, v in enumerate(dev_vals):
        print(f"| dev | {i} | {v.get('affinity_pred_value', float('nan')):.4f} "
              f"| {v.get('affinity_probability_binary', float('nan')):.4f} |")
    print()

    report = {"mode": "affinity", "target": args.target_id, "metrics": {}}
    print("| metric | dev-vs-ref (X) | ref-floor (R) | dev-floor (D) | X/floor | within floor |")
    print("|---|---|---|---|---|---|")
    for key in AFFINITY_KEYS:
        r = [v[key] for v in ref_vals if key in v]
        d = [v[key] for v in dev_vals if key in v]
        if not r or not d:
            print(f"| {key} | - | - | - | - | - |")
            continue
        cross = [abs(di - ri) for di, ri in itertools.product(d, r)]
        ref_floor = [abs(a - b) for a, b in itertools.combinations(r, 2)]
        dev_floor = [abs(a - b) for a, b in itertools.combinations(d, 2)]
        v = noise_floor_verdict(cross, ref_floor, dev_floor, key)
        report["metrics"][key] = v
        print(f"| {key} | {v['cross']['mean']:.4f}+/-{v['cross']['std']:.4f} "
              f"| {v['ref_floor']['mean']:.4f} "
              f"| {v['dev_floor']['mean']:.4f} "
              f"| {v['cross_over_floor']:.2f} "
              f"| {'yes' if v['within_noise_floor'] else 'NO'} |")
        if args.paired and len(d) == len(r) and d:
            diag = [abs(di - ri) for di, ri in zip(d, r)]
            pm = sum(diag) / len(diag)
            cm = sum(cross) / len(cross)
            seed_indep = pm >= 0.9 * cm
            print(f"  same-seed diagonal: X_diag {pm:.4f} (n={len(diag)}) vs "
                  f"all-pairs X {cm:.4f} -> "
                  f"{'seed-independent (systematic bf16)' if seed_indep else 'RNG-stochastic'}")
            report["metrics"][key]["same_seed_diagonal"] = {
                "n": len(diag), "mean": pm, "all_pairs_mean": cm,
                "seed_independent": seed_indep,
            }

    # ---- ligand-pose accuracy (P3): ligand-pose RMSD + pocket-lDDT from the
    # best-sample structure CIFs (device vs reference), scored through the same
    # R/D/X noise-floor core as the scalar affinity. Pharma cares about the
    # binding POSE, not just the affinity scalar.
    pose_keys = ("ligand_rmsd", "1-pocket_lddt")
    pose_labels = {"ligand_rmsd": "ligand-pose RMSD (Å)", "1-pocket_lddt": "1-pocket-lDDT"}
    pose_cross = {k: [] for k in pose_keys}
    pose_rf = {k: [] for k in pose_keys}
    pose_df = {k: [] for k in pose_keys}
    have_pose = False
    _ref_d = [Path(d) for d in args.ref_dirs]
    _dev_d = [Path(d) for d in args.dev_dirs]
    for da, db in itertools.product(_dev_d, _ref_d):
        m = _pose_metrics(da, db, args.target_id)
        if m:
            have_pose = True
            for k in pose_keys:
                pose_cross[k].append(m[k])
    for da, db in itertools.combinations(_ref_d, 2):
        m = _pose_metrics(da, db, args.target_id)
        if m:
            for k in pose_keys:
                pose_rf[k].append(m[k])
    for da, db in itertools.combinations(_dev_d, 2):
        m = _pose_metrics(da, db, args.target_id)
        if m:
            for k in pose_keys:
                pose_df[k].append(m[k])
    if have_pose:
        print()
        print("| metric | dev-vs-ref (X) | ref-floor (R) | dev-floor (D) | X/floor | within floor |")
        print("|---|---|---|---|---|---|")
        for k in pose_keys:
            if not pose_cross[k]:
                continue
            v = noise_floor_verdict(pose_cross[k], pose_rf[k], pose_df[k], k)
            report["metrics"][k] = v
            print(f"| {pose_labels[k]} | {v['cross']['mean']:.3f}+/-{v['cross']['std']:.3f} "
                  f"| {v['ref_floor']['mean']:.3f} "
                  f"| {v['dev_floor']['mean']:.3f} "
                  f"| {v['cross_over_floor']:.2f} "
                  f"| {'yes' if v['within_noise_floor'] else 'NO'} |")
        if args.paired and len(_dev_d) == len(_ref_d) and _dev_d:
            pose_paired = {k: [] for k in pose_keys}
            for da, db in zip(_dev_d, _ref_d):
                m = _pose_metrics(da, db, args.target_id)
                if m:
                    for k in pose_keys:
                        pose_paired[k].append(m[k])
            print()
            print("| metric | same-seed (X_diag, n=%d) | all-pairs (X, n=%d) | "
                  "diag == cross? |" % (len(_dev_d), len(pose_cross[pose_keys[0]])))
            print("|---|---|---|---|")
            for k in pose_keys:
                if not pose_paired[k] or not pose_cross[k]:
                    continue
                import statistics
                pm = sum(pose_paired[k]) / len(pose_paired[k])
                cm = sum(pose_cross[k]) / len(pose_cross[k])
                # seed-independent iff the diagonal mean is NOT markedly below the
                # all-pairs mean (a diagonal much smaller than cross means matching
                # the RNG stream collapses the residual -> RNG stochasticity; a
                # diagonal ~ cross means shared draws do NOT help -> systematic).
                seed_indep = pm >= 0.9 * cm
                verdict = "yes (systematic bf16)" if seed_indep else "no (RNG-stochastic)"
                print(f"| {pose_labels[k]} | {pm:.3f} | {cm:.3f} | {verdict} |")
                report["metrics"][k]["same_seed_diagonal"] = {
                    "n": len(pose_paired[k]), "mean": pm,
                    "all_pairs_mean": cm,
                    "seed_independent": seed_indep,
                }
    else:
        print("\n(ligand-pose metrics skipped: no structure CIFs with a matched ligand found — "
              "run with a fixture that carries structures/<id>.cif to enable P3)")

    print("\nInterpretation: affinity_pred_value is a scalar (log10 IC50), so the")
    print("parity distance is |device - reference|. X within max(R, D) means the")
    print("device-vs-reference affinity delta is no larger than the run-to-run")
    print("diffusion-sampling spread each implementation already shows with itself.")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
