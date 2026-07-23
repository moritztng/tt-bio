#!/usr/bin/env python3
"""Deterministic shared-draws, measured-bf16-envelope integration-parity scorer.

This is the CORRECTNESS core of the release parity gate (``scripts/full_parity_gate.py``).
It replaces the old R/D/X same-backend self-consistency floor, which compared INDEPENDENT
stochastic samples against a guessed self-spread floor and so could not tell a real backend
bug from ordinary sample-to-sample diffusion noise (see docs/implementation-parity.md §"why
the self-consistency floor is unsound").

THE TEST. A diffusion model is a deterministic function of its input noise; the only
stochasticity is the noise draw. So feed byte-identical noise (initial coords + every per-step
eps) to three CLOSED-LOOP runs and compare their FINAL structures:

    device_bf16     tt-bio on Tenstorrent (bf16)          -- the port under test
    reference_fp32  tt-bio on CPU, fp32                   -- ground truth
    reference_bf16  tt-bio on CPU, bf16 autocast          -- intrinsic bf16 trajectory cost

Because the reference is tt-bio's OWN torch path (``--accelerator cpu`` => use_tenstorrent=
False), all three are the same code with a backend/dtype toggle and draw their diffusion
``torch.randn`` on CPU MT19937 from the one ``--seed`` (boltz2.py:4092/4127; the worker seeds
once, worker.py:278-286). Shared draws therefore hold BY CONSTRUCTION — the only difference
between any two runs is arithmetic (fp32 vs bf16, torch vs TT), nothing stochastic.

Per leg, per metric ``d(.,.)`` (the natural per-leg structural distance, reused unchanged):

    numerator = d(device_bf16, reference_fp32)     -- how far the port drifts from fp32
    envelope  = d(reference_bf16, reference_fp32)  -- how far a bf16 recompute drifts from fp32
    PASS  iff  numerator <= envelope * (1 + margin) + abs_floor   for EVERY metric.

In words: the device may differ from the fp32 reference by no more than a bf16 recomputation
of the reference differs from itself (plus a small honest residual for TT-bf16 vs torch-bf16
accumulation differences, absorbed by ``margin``). Deterministic, per-leg, measurable — no
percentile, no hand-tuned scalar floor, no distribution estimation, no permutation test. If the
numerator blows WELL past the envelope, that is an unambiguous bug signal regardless of margin.

``abs_floor`` guards the degenerate case where a scalar's bf16 envelope rounds to ~0 (e.g. an
ensemble-mean affinity value that happens to be bf16-identical between fp32 and bf16): without
it a zero envelope would fail every nonzero device residual by construction. It is per-metric,
tiny, and justified from the observed metric quantization (see MARGIN / ABS_FLOOR below).
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse the exact per-leg distance primitives already vetted by the R/D/X scorers — only WHAT
# is compared changes (shared-draw dev-vs-ref, not independent-seed pairs), never the metric.
from boltz2_affinity_parity import _extract_dev, _pose_metrics, _kabsch_rmsd  # noqa: E402


# --- margin / abs_floor: justified from observed spread, NOT pulled from thin air --------------
# margin absorbs the honest residual that TT-bf16 arithmetic (fused ops / different accumulation)
# rounds slightly differently from torch-bf16, so the device may legitimately drift a little more
# than the torch-bf16 reference does. Recorded/justified in
# ~/.coworker/state/tt-bio-integration-parity-gate.md §4 from the measured device-vs-envelope
# ratio across all clean legs before the value is committed.
DEFAULT_MARGIN = 0.50
# abs_floor is the metric's quantization / numerical-noise that must not count as a failure when
# the envelope itself rounds to ~0. Per metric, tiny.
ABS_FLOOR = {
    "affinity_pred_value": 0.01,       # log10(IC50) units; ref self-spread R~0.01 (fixture meta)
    "affinity_probability_binary": 0.01,
    "kabsch_rmsd": 0.05,               # Angstrom
    "ligand_rmsd": 0.05,               # Angstrom
    "1-pocket_lddt": 0.005,
    "1-coord_pcc": 0.001,
}


def _extract_scalar(dev_dir: Path, target_id: str, keys) -> dict:
    """affinity_pred_value / probability from a tt-bio results.json (all three runs are tt-bio)."""
    row = _extract_dev(dev_dir, target_id)
    return {k: row[k] for k in keys if k in row}


def _ca_coords(cif: Path):
    import gemmi
    _AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
    st = gemmi.read_structure(str(cif))
    pts = {}
    for ch in st[0]:
        for res in ch:
            if res.name in _AA:
                for atom in res:
                    if atom.name == "CA":
                        pts[(ch.name, res.seqid.num)] = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
    return pts


def _ca_rmsd(dA: Path, dB: Path, tid: str):
    a = _ca_coords(dA / "structures" / f"{tid}.cif")
    b = _ca_coords(dB / "structures" / f"{tid}.cif")
    keys = [k for k in a if k in b]
    if len(keys) < 3:
        return None
    A = np.array([a[k] for k in keys])
    B = np.array([b[k] for k in keys])
    return _kabsch_rmsd(A, B)


def divergence(kind: str, dA: Path, dB: Path, target_id: str) -> dict:
    """The per-leg structural distance between two shared-draw runs A and B.

    Returns {metric: value}. ``kind`` selects which natural metrics apply:
      affinity  -> affinity_pred_value (primary), affinity_probability_binary, + pose if CIFs
      structure -> kabsch_rmsd (CA), + pose if a ligand is present
    """
    out: dict = {}
    if kind == "affinity":
        sa = _extract_scalar(dA, target_id, ("affinity_pred_value", "affinity_probability_binary"))
        sb = _extract_scalar(dB, target_id, ("affinity_pred_value", "affinity_probability_binary"))
        for k in sa:
            if k in sb:
                out[k] = abs(sa[k] - sb[k])
    if kind in ("affinity", "structure"):
        pm = _pose_metrics(dA, dB, target_id)
        if pm:
            out.update(pm)
    if kind == "structure":
        r = _ca_rmsd(dA, dB, target_id)
        if r is not None:
            out["kabsch_rmsd"] = r
    return out


def envelope_verdict(dev_dir, ref_fp32_dir, ref_bf16_dir, kind: str, target_id: str,
                     margin: float = DEFAULT_MARGIN) -> dict:
    """Score one leg. Returns a report dict with per-metric numerator/envelope/pass and a verdict.

    verdict is 'PASS' iff every measured metric satisfies
        numerator <= envelope * (1 + margin) + abs_floor
    otherwise 'GAP' (a real residual exceeding the measured bf16 envelope — to hunt, not excuse).
    """
    dev_dir, ref_fp32_dir, ref_bf16_dir = map(Path, (dev_dir, ref_fp32_dir, ref_bf16_dir))
    num = divergence(kind, dev_dir, ref_fp32_dir, target_id)      # device_bf16 vs ref_fp32
    env = divergence(kind, ref_bf16_dir, ref_fp32_dir, target_id) # ref_bf16   vs ref_fp32
    metrics = {}
    verdict = "PASS"
    for k in sorted(set(num) | set(env)):
        n = num.get(k)
        e = env.get(k)
        if n is None or e is None:
            continue
        floor = ABS_FLOOR.get(k, 0.0)
        bound = e * (1.0 + margin) + floor
        ok = n <= bound
        ratio = (n / e) if e > 1e-12 else float("inf") if n > floor else 0.0
        metrics[k] = {"numerator": n, "envelope": e, "bound": bound,
                      "ratio": ratio, "margin": margin, "abs_floor": floor, "pass": ok}
        if not ok:
            verdict = "GAP"
    return {"mode": "integration_envelope", "kind": kind, "target": target_id,
            "margin": margin, "verdict": verdict, "metrics": metrics}


def _print_report(rep: dict) -> None:
    print(f"### integration-parity envelope: {rep['target']} ({rep['kind']})  margin={rep['margin']}\n")
    print("| metric | numerator d(dev_bf16,ref_fp32) | envelope d(ref_bf16,ref_fp32) | bound | ratio | pass |")
    print("|---|---|---|---|---|---|")
    for k, m in rep["metrics"].items():
        print(f"| {k} | {m['numerator']:.5f} | {m['envelope']:.5f} | {m['bound']:.5f} "
              f"| {m['ratio']:.2f} | {'yes' if m['pass'] else 'NO'} |")
    print(f"\n**verdict: {rep['verdict']}**")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", required=True, help="device_bf16 output dir (contains results.json)")
    ap.add_argument("--ref-fp32", required=True, help="reference fp32 output dir")
    ap.add_argument("--ref-bf16", required=True, help="reference bf16 output dir")
    ap.add_argument("--kind", required=True, choices=("affinity", "structure"))
    ap.add_argument("--target-id", required=True)
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    def _inner(d):
        # tt-bio writes results into <out>/<model>_results_<id>/; accept either.
        p = Path(d)
        cand = list(p.rglob("results.json"))
        return cand[0].parent if cand else p

    rep = envelope_verdict(_inner(args.dev), _inner(args.ref_fp32), _inner(args.ref_bf16),
                           args.kind, args.target_id, args.margin)
    _print_report(rep)
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
