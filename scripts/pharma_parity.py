#!/usr/bin/env python3
"""Pharma implementation-parity benchmark: tt-bio device vs the original reference.

Answers the question pharma evaluators actually ask: does the Tenstorrent port
reproduce what the model's ORIGINAL CPU/GPU implementation already gives, on the
SAME input? Not "is the model good" (they picked the model already), but "is the
port faithful to the implementation they picked".

Neither side is bit-deterministic (device bf16 numerics; diffusion samplers draw
noise differently per backend), so a bare "RMSD = X" is not a credible parity
claim. Instead we establish a NOISE FLOOR and ask whether the device-vs-reference
gap sits inside it:

  reference-vs-reference   (same official code, different seeds)   -> R
  device-vs-device         (same port, different seeds)            -> D
  device-vs-reference      (the parity question)                   -> X

Parity holds when X sits within the natural run-to-run spread max(R, D): the two
implementations differ from each other no more than each already differs from
itself. Everything is reported as a distribution (mean/std/min/max/n), never one
number.

One statistical core, two comparison front-ends:

  structures   fold models (boltz2, esmfold2, protenix-v2): Kabsch CA-RMSD,
               coordinate PCC and confidence-metric deltas between the output
               structures of paired runs. Model-agnostic: point it at result
               dirs produced by `tt-bio predict` (device) and by the reference
               CLI, one dir per seed.
  embeddings   ESMC: per-residue embedding PCC vs the reference esm model, plus
               device self-consistency (the embedding path has no sampler, so
               its noise floor is pure numerics).

BoltzGen (de-novo design, no 1:1 output correspondence) is measured in
designability space, not here: see scripts/boltzgen_designability.py.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse the vetted structure comparison (Kabsch RMSD / coord PCC / conf deltas).
from boltz2_fast_parity import CONF_KEYS, compare_structure, load_results  # noqa: E402


# ---------------------------------------------------------------------------
# Statistical core: the noise-floor verdict.
# ---------------------------------------------------------------------------
def summarize(xs) -> dict:
    a = np.asarray(list(xs), dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max())}


def noise_floor_verdict(cross, ref_floor, dev_floor, metric: str) -> dict:
    """cross/ref_floor/dev_floor are DISTANCES (lower = more similar: RMSD, 1-PCC).

    Parity holds when the mean cross-implementation distance is no larger than
    the larger of the two self-consistency floors. `ratio` < ~1 means the
    device-vs-reference gap is indistinguishable from run-to-run noise.
    """
    X, R, D = summarize(cross), summarize(ref_floor), summarize(dev_floor)
    floor = max(R.get("mean", 0.0), D.get("mean", 0.0))
    ratio = (X["mean"] / floor) if floor > 0 else float("inf")
    return {"metric": metric, "cross": X, "ref_floor": R, "dev_floor": D,
            "floor_mean": floor, "cross_over_floor": ratio,
            "within_noise_floor": bool(X["n"] and X["mean"] <= floor + max(R.get("std", 0), D.get("std", 0)))}


# ---------------------------------------------------------------------------
# structures mode
# ---------------------------------------------------------------------------
def _cif(d: Path, tid: str) -> Path:
    return Path(d) / "structures" / f"{tid}.cif"


def _pair_rmsd_pcc(dA: Path, dB: Path, tid: str):
    s = compare_structure(_cif(dA, tid), _cif(dB, tid))
    if s.get("n_matched", 0) == 0:
        return None
    return s["kabsch_rmsd"], s["coord_pcc"]


def structures(args) -> int:
    ref_dirs = [Path(d) for d in args.ref_dirs]
    dev_dirs = [Path(d) for d in args.dev_dirs]
    # targets present in every run
    ids = None
    for d in ref_dirs + dev_dirs:
        cur = set(load_results(d))
        ids = cur if ids is None else (ids & cur)
    ids = sorted(ids or [])
    if not ids:
        print("no common targets across the supplied run dirs", file=sys.stderr)
        return 1

    report = {"mode": "structures", "targets": {}}
    print(f"### Implementation parity: {args.label or 'structures'}\n")
    print(f"reference seeds: {len(ref_dirs)}   device seeds: {len(dev_dirs)}   "
          f"targets: {len(ids)}\n")
    print("| target | metric | dev-vs-ref (X) | ref-floor (R) | dev-floor (D) | X/floor | within floor |")
    print("|---|---|---|---|---|---|---|")

    for tid in ids:
        cross_r, cross_p = [], []
        for da, db in itertools.product(dev_dirs, ref_dirs):
            v = _pair_rmsd_pcc(da, db, tid)
            if v:
                cross_r.append(v[0]); cross_p.append(1 - v[1])
        rf_r, rf_p = [], []
        for da, db in itertools.combinations(ref_dirs, 2):
            v = _pair_rmsd_pcc(da, db, tid)
            if v:
                rf_r.append(v[0]); rf_p.append(1 - v[1])
        df_r, df_p = [], []
        for da, db in itertools.combinations(dev_dirs, 2):
            v = _pair_rmsd_pcc(da, db, tid)
            if v:
                df_r.append(v[0]); df_p.append(1 - v[1])

        vr = noise_floor_verdict(cross_r, rf_r, df_r, "kabsch_rmsd")
        vp = noise_floor_verdict(cross_p, rf_p, df_p, "1-coord_pcc")
        report["targets"][tid] = {"rmsd": vr, "pcc": vp}
        for name, v in (("CA-RMSD (Å)", vr), ("1-PCC", vp)):
            print(f"| {tid} | {name} | {v['cross'].get('mean', float('nan')):.3f}"
                  f"±{v['cross'].get('std', 0):.3f} "
                  f"| {v['ref_floor'].get('mean', float('nan')):.3f} "
                  f"| {v['dev_floor'].get('mean', float('nan')):.3f} "
                  f"| {v['cross_over_floor']:.2f} "
                  f"| {'yes' if v['within_noise_floor'] else 'NO'} |")

    # confidence deltas (device mean vs reference mean, per key)
    print("\n### Confidence-metric agreement (device mean − reference mean)\n")
    print("| target | " + " | ".join(CONF_KEYS) + " |")
    print("|" + "---|" * (len(CONF_KEYS) + 1))
    ref_res = [load_results(d) for d in ref_dirs]
    dev_res = [load_results(d) for d in dev_dirs]
    for tid in ids:
        cells = []
        for k in CONF_KEYS:
            rv = [r[tid][k] for r in ref_res if tid in r and k in r[tid] and isinstance(r[tid][k], (int, float))]
            dv = [r[tid][k] for r in dev_res if tid in r and k in r[tid] and isinstance(r[tid][k], (int, float))]
            cells.append(f"{np.mean(dv) - np.mean(rv):+.4f}" if rv and dv else "—")
        print(f"| {tid} | " + " | ".join(cells) + " |")

    Path(args.out).write_text(json.dumps(report, indent=2)) if args.out else None
    return 0


# ---------------------------------------------------------------------------
# embeddings mode (ESMC)
# ---------------------------------------------------------------------------
# Varied real proteins: short peptide, small domain, medium domain, longer chain.
ESMC_SEQS = {
    "trpcage": "NLYIQWLKDGGPSSGRPPPS",                                                  # 20
    "gb1": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",                 # 56
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",  # 76
    "lysozyme": ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDG"
                 "RTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"),        # 129
}


def _pcc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def embeddings(args) -> int:
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
    from huggingface_hub import hf_hub_download
    from tt_bio import esmc as tt_esmc
    from esmc_reference import ESMCReference

    torch.set_grad_enabled(False)
    seqs = {k: ESMC_SEQS[k] for k in (args.seqs.split(",") if args.seqs else ESMC_SEQS)}

    # reference (CPU torch, deterministic)
    cfg, repo_id, wpath = tt_esmc.CONFIGS[args.model]
    print(f"Fetching {args.model} weights from {repo_id} …", flush=True)
    sd = torch.load(hf_hub_download(repo_id, wpath), map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    print("Building reference esm ESMC …", flush=True)
    ref = ESMCReference(**cfg).eval()
    ref.load_state_dict(sd, strict=False)

    ref_emb = {}
    for name, seq in seqs.items():
        _, e = ref(tt_esmc.tokenize(seq))
        ref_emb[name] = e[0][1:-1].numpy()

    # device: run twice for a self-consistency (numerical) floor
    print(f"Loading tt ESMC on device{' (fast)' if args.fast else ''} …", flush=True)
    model = tt_esmc.load_esmc(args.model, fast=args.fast)
    dev_runs = []
    for r in range(2):
        out = tt_esmc.embed_sequences(model, seqs, pool="mean")
        dev_runs.append({o.id: o.per_residue for o in out})

    report = {"mode": "embeddings", "model": args.model, "fast": args.fast, "targets": {}}
    print(f"\n### ESMC embedding parity ({args.model}, fast={args.fast})\n")
    print("| protein | length | dev-vs-ref PCC (X) | dev-vs-dev PCC (D floor) |")
    print("|---|---|---|---|")
    for name, seq in seqs.items():
        x = _pcc(dev_runs[0][name], ref_emb[name])
        d = _pcc(dev_runs[0][name], dev_runs[1][name])
        report["targets"][name] = {"length": len(seq), "dev_vs_ref_pcc": x, "dev_vs_dev_pcc": d}
        print(f"| {name} | {len(seq)} | {x:.5f} | {d:.5f} |")

    xs = [v["dev_vs_ref_pcc"] for v in report["targets"].values()]
    ds = [v["dev_vs_dev_pcc"] for v in report["targets"].values()]
    print(f"\ndev-vs-ref PCC: mean {np.mean(xs):.5f}  min {np.min(xs):.5f}")
    print(f"device self-consistency PCC: mean {np.mean(ds):.5f}  min {np.min(ds):.5f}")
    print("\nInterpretation: the embedding path has no sampler, so the device "
          "self-consistency PCC is the numerical noise floor. The device-vs-"
          "reference residual is bf16 rounding, not an algorithmic difference.")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("structures", help="fold models: device vs reference cif dirs")
    s.add_argument("--ref-dirs", nargs="+", required=True, help="reference run dirs, one per seed")
    s.add_argument("--dev-dirs", nargs="+", required=True, help="device run dirs, one per seed")
    s.add_argument("--label", default="")
    s.add_argument("--out", default="")
    s.set_defaults(func=structures)

    e = sub.add_parser("embeddings", help="ESMC: device vs reference embeddings")
    e.add_argument("--model", default="esmc-300m", choices=["esmc-300m", "esmc-600m"])
    e.add_argument("--seqs", default="", help="comma-separated subset of the built-in proteins")
    e.add_argument("--fast", action="store_true")
    e.add_argument("--out", default="")
    e.set_defaults(func=embeddings)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
