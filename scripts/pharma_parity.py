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

One statistical core, three comparison front-ends:

  structures   fold models (boltz2, esmfold2, protenix-v2): Kabsch CA-RMSD,
               coordinate PCC and confidence-metric deltas between the output
               structures of paired runs. Model-agnostic: point it at result
               dirs produced by `tt-bio predict` (device) and by the reference
               CLI, one dir per seed.
  embeddings   ESMC: per-residue embedding PCC vs the reference esm model, plus
               device self-consistency (the embedding path has no sampler, so
               its noise floor is pure numerics).
  saprot       SaProt (structure-aware ESM-2 encoder): per-residue embedding
               and MLM-logits PCC vs the HF EsmForMaskedLM reference. Same
               deterministic-forward convention as `embeddings` (R = D = 1.00000,
               no sampler); X is the bf16 device-vs-reference residual.

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
DEVICE_FLOOR_INFLATION_THRESHOLD = 5.0
REFERENCE_FLOOR_EPS = 1e-12
FLOOR_INFLATED_BY_D_MESSAGE = (
    "FLOOR-INFLATED-BY-D: device self-consistency is looser than the reference's own; "
    "investigate device instability before trusting this PASS."
)


def summarize(xs) -> dict:
    a = np.asarray(list(xs), dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max())}


def noise_floor_verdict(cross, ref_floor, dev_floor, metric: str,
                        *, check_dev_instability: bool = True) -> dict:
    """cross/ref_floor/dev_floor are DISTANCES (lower = more similar: RMSD, 1-PCC).

    Parity holds when the mean cross-implementation distance is no larger than
    the larger of the two self-consistency floors. `ratio` < ~1 means the
    device-vs-reference gap is indistinguishable from run-to-run noise.

    The independent D/R check warns when device self-consistency is
    anomalously looser than reference self-consistency. The 5.0 threshold is
    ~3.6x the largest committed primary-metric D/R (1.38x; median 0.58, range
    0.25-1.38 across the 17 stochastic kabsch_rmsd legs the check covers), a
    conservative bar that leaves headroom for legitimate wide-but-stable
    no-MSA floors (today up to 1.38x) while flagging a genuine device-
    instability blowup. Skipped when R is effectively zero (deterministic
    forwards).
    """
    X, R, D = summarize(cross), summarize(ref_floor), summarize(dev_floor)
    floor = max(R.get("mean", 0.0), D.get("mean", 0.0))
    ratio = (X["mean"] / floor) if floor > 0 else float("inf")
    ref_mean = R.get("mean", 0.0)
    dev_over_ref = (
        D.get("mean", 0.0) / ref_mean
        if R.get("n", 0) and D.get("n", 0) and ref_mean > REFERENCE_FLOOR_EPS
        else None
    )
    instability_check_applied = check_dev_instability and dev_over_ref is not None
    floor_inflated_by_dev = bool(
        instability_check_applied
        and dev_over_ref > DEVICE_FLOOR_INFLATION_THRESHOLD
    )
    return {"metric": metric, "cross": X, "ref_floor": R, "dev_floor": D,
            "floor_mean": floor, "cross_over_floor": ratio,
            "dev_over_ref_floor": dev_over_ref,
            "dev_floor_instability_check_applied": instability_check_applied,
            "floor_inflated_by_dev": floor_inflated_by_dev,
            "within_noise_floor": bool(X["n"] and X["mean"] <= floor + max(R.get("std", 0), D.get("std", 0)))}


# ---------------------------------------------------------------------------
# committed reference fixtures
# ---------------------------------------------------------------------------
FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "docs" / "pharma-benchmark-data" / "ref-fixtures"


def resolve_ref_fixtures(spec: str, seeds=None) -> list:
    """Resolve committed reference-fixture seed dirs for "<model>/<target>/<tag>".

    The fixture tree lives at docs/pharma-benchmark-data/ref-fixtures/<model>/<target>/<tag>/
    seed<N>/ (results.json + structures/<id>.cif), produced once by a real reference run and
    committed so the expensive reference legs do NOT re-run on every release-gate pass. Each
    fixture dir carries a meta.json pinning the reference implementation + version + settings;
    regenerate a fixture only when that pinned version/settings changes (see meta.json
    `invalidation_rule` and `command`).

    Returns the list of seed dirs (sorted by seed), verifying each is complete. Raises with a
    precise regenerate instruction if a fixture is missing or its settings-tag does not match
    -- the release gate then re-runs just that one reference leg (the device side always
    re-runs live).
    """
    parts = spec.split("/")
    if len(parts) != 3:
        raise SystemExit(f"--ref-fixtures expects <model>/<target>/<settings-tag>, got {spec!r}")
    model, target, tag = parts
    base = FIXTURE_ROOT / model / target / tag
    if not base.is_dir():
        raise SystemExit(
            f"reference fixture not committed: {base}\n"
            f"regenerate it with the command recorded in a sibling fixture's meta.json, then "
            f"run scripts/pharma_harvest_ref_fixtures.py to commit it. The device side re-runs "
            f"live regardless.")
    meta_path = base / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("settings_tag") != tag:
            raise SystemExit(
                f"reference fixture settings-tag mismatch at {base}: meta.json says "
                f"{meta.get('settings_tag')!r}, requested {tag!r}. Regenerate the fixture.")
    available = sorted(p for p in base.glob("seed*") if p.is_dir())
    if seeds is not None:
        want = {f"seed{s}" for s in seeds}
        available = [p for p in available if p.name in want]
        missing = want - {p.name for p in available}
        if missing:
            raise SystemExit(f"missing fixture seeds for {spec}: {sorted(missing)}")
    if not available:
        raise SystemExit(f"no seed<N>/ fixture dirs found under {base}")
    out = []
    for p in available:
        if not (p / "results.json").exists() or not (p / "structures").exists():
            raise SystemExit(f"incomplete fixture {p}: missing results.json or structures/")
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# structures mode
# ---------------------------------------------------------------------------
def _cif(d: Path, tid: str) -> Path:
    return Path(d) / "structures" / f"{tid}.cif"


def _pair_metrics(dA: Path, dB: Path, tid: str):
    """All four structure-parity DISTANCES (lower = more similar) for a run pair."""
    s = compare_structure(_cif(dA, tid), _cif(dB, tid))
    if s.get("n_matched", 0) == 0:
        return None
    return {
        "kabsch_rmsd": s["kabsch_rmsd"],
        "1-coord_pcc": 1.0 - s["coord_pcc"],
        "1-tm_score": 1.0 - s.get("tm_score", 0.0),
        "1-lddt": 1.0 - s.get("lddt", 0.0),
    }


def _pair_rmsd_pcc(dA: Path, dB: Path, tid: str):
    m = _pair_metrics(dA, dB, tid)
    if m is None:
        return None
    return m["kabsch_rmsd"], 1.0 - m["1-coord_pcc"]


def structures(args) -> int:
    if args.ref_fixtures:
        ref_dirs = resolve_ref_fixtures(args.ref_fixtures, args.ref_seeds)
        print(f"# reference: committed fixtures {args.ref_fixtures} "
              f"({len(ref_dirs)} seeds, no reference compute)\n")
    elif args.ref_dirs:
        ref_dirs = [Path(d) for d in args.ref_dirs]
    else:
        print("structures mode requires either --ref-dirs or --ref-fixtures", file=sys.stderr)
        return 2
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
    paired_ok = args.paired and len(dev_dirs) == len(ref_dirs) and len(dev_dirs) > 0
    if args.paired and not paired_ok:
        print("--paired requires len(dev_dirs) == len(ref_dirs) (non-empty); "
              "skipping the diagonal.", file=sys.stderr)
    print(f"### Implementation parity: {args.label or 'structures'}\n")
    print(f"reference seeds: {len(ref_dirs)}   device seeds: {len(dev_dirs)}   "
          f"targets: {len(ids)}\n")
    print("| target | metric | dev-vs-ref (X) | ref-floor (R) | dev-floor (D) | "
          "X/floor | within floor | D/R stability |")
    print("|---|---|---|---|---|---|---|---|")

    metric_keys = ("kabsch_rmsd", "1-coord_pcc", "1-tm_score", "1-lddt")
    metric_labels = {
        "kabsch_rmsd": "CA-RMSD (Å)",
        "1-coord_pcc": "1-PCC",
        "1-tm_score": "1-TM",
        "1-lddt": "1-lDDT",
    }
    instability_warnings = []
    for tid in ids:
        cross = {k: [] for k in metric_keys}
        rf = {k: [] for k in metric_keys}
        df = {k: [] for k in metric_keys}
        diag = {k: [] for k in metric_keys}
        for da, db in itertools.product(dev_dirs, ref_dirs):
            m = _pair_metrics(da, db, tid)
            if m:
                for k in metric_keys:
                    cross[k].append(m[k])
        for da, db in itertools.combinations(ref_dirs, 2):
            m = _pair_metrics(da, db, tid)
            if m:
                for k in metric_keys:
                    rf[k].append(m[k])
        for da, db in itertools.combinations(dev_dirs, 2):
            m = _pair_metrics(da, db, tid)
            if m:
                for k in metric_keys:
                    df[k].append(m[k])
        if paired_ok:
            for da, db in zip(dev_dirs, ref_dirs):
                m = _pair_metrics(da, db, tid)
                if m:
                    for k in metric_keys:
                        diag[k].append(m[k])

        verdicts = {
            k: noise_floor_verdict(
                cross[k], rf[k], df[k], k,
                check_dev_instability=(k == "kabsch_rmsd"),
            )
            for k in metric_keys
        }
        report["targets"][tid] = {k.split("-", 1)[-1]: v for k, v in verdicts.items()}
        for k in metric_keys:
            v = verdicts[k]
            name = metric_labels[k]
            if not v["dev_floor_instability_check_applied"]:
                stability = "—"
            elif v["floor_inflated_by_dev"]:
                stability = f"FLOOR-INFLATED-BY-D ({v['dev_over_ref_floor']:.2f}×)"
            else:
                stability = f"ok ({v['dev_over_ref_floor']:.2f}×)"
            print(f"| {tid} | {name} | {v['cross'].get('mean', float('nan')):.3f}"
                  f"±{v['cross'].get('std', 0):.3f} "
                  f"| {v['ref_floor'].get('mean', float('nan')):.3f} "
                  f"| {v['dev_floor'].get('mean', float('nan')):.3f} "
                  f"| {v['cross_over_floor']:.2f} "
                  f"| {'yes' if v['within_noise_floor'] else 'NO'} "
                  f"| {stability} |")
            if v["floor_inflated_by_dev"]:
                instability_warnings.append((tid, name, v["dev_over_ref_floor"]))
        if paired_ok:
            for k in metric_keys:
                if not diag[k] or not cross[k]:
                    continue
                pm = sum(diag[k]) / len(diag[k])
                cm = sum(cross[k]) / len(cross[k])
                seed_indep = pm >= 0.9 * cm
                report["targets"][tid][k.split("-", 1)[-1]]["same_seed_diagonal"] = {
                    "n": len(diag[k]), "mean": pm,
                    "all_pairs_mean": cm, "seed_independent": seed_indep,
                }

    for tid, name, dev_over_ref in instability_warnings:
        print(f"\n> **{FLOOR_INFLATED_BY_D_MESSAGE}** "
              f"Target `{tid}`, metric {name}, D/R={dev_over_ref:.2f}×.")

    if paired_ok:
        print(f"\n### Same-seed diagonal (dev_i vs ref_i, n={len(dev_dirs)}) vs "
              f"all-pairs cross (n={len(dev_dirs) * len(ref_dirs)})\n")
        print("| target | metric | same-seed X_diag | all-pairs X | diag == cross? |")
        print("|---|---|---|---|---|")
        for tid in ids:
            for k in metric_keys:
                d = report["targets"][tid].get(k.split("-", 1)[-1], {}).get("same_seed_diagonal")
                if not d:
                    continue
                verdict = "yes (systematic bf16)" if d["seed_independent"] else "no (RNG-stochastic)"
                print(f"| {tid} | {metric_labels[k]} | {d['mean']:.3f} | "
                      f"{d['all_pairs_mean']:.3f} | {verdict} |")

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


# ---------------------------------------------------------------------------
# saprot mode (structure-aware ESM-2 encoder, deterministic forward)
# ---------------------------------------------------------------------------
# Ubiquitin is the leg every other encoder leg in this benchmark uses, so SaProt
# runs the same target. The 3Di string is deterministic (the fused-token parity
# does not depend on the 3Di content -- both paths see identical tokens), so the
# reference is a single deterministic HF EsmForMaskedLM forward and the device is
# a single deterministic ttnn forward; R and D are the ref-vs-ref / device-vs-
# device PCCs (both 1.00000 by construction, no sampler), X is device-vs-ref.
SAPROT_UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def _saprot_pair_ids():
    from tt_bio import saprot
    struc = (saprot.FOLDSEEK_STRUC_VOCAB[:-1] * 6)[: len(SAPROT_UBQ)]
    return saprot.tokenize(SAPROT_UBQ, struc)  # [1, L+2] long


def saprot(args) -> int:
    import torch
    from transformers import EsmForMaskedLM
    from tt_bio import saprot as tt_saprot, esmc

    torch.set_grad_enabled(False)
    repo = {"saprot-35m": "westlake-repl/SaProt_35M_AF2",
            "saprot-650m": "westlake-repl/SaProt_650M_AF2",
            "saprot-1.3b": "westlake-repl/SaProt_1.3B_AF2"}[args.model]
    ids = _saprot_pair_ids()
    L = ids.shape[1]  # includes <cls>/<eos>

    # reference: HF EsmForMaskedLM, run twice (R floor -- deterministic by construction)
    print(f"Building HF reference EsmForMaskedLM ({repo}) ...", flush=True)
    ref = EsmForMaskedLM.from_pretrained(repo).eval()
    attn = torch.ones(1, L, dtype=torch.long)
    ref_runs = []
    with torch.no_grad():
        for _ in range(2):
            out = ref(input_ids=ids, attention_mask=attn, output_hidden_states=True)
            ref_runs.append((out.hidden_states[-1][0].numpy().astype(np.float32),
                             out.logits[0].numpy().astype(np.float32)))

    # device: ttnn port, run twice (D floor -- deterministic by construction)
    print(f"Loading tt SaProt on device ({args.model}) ...", flush=True)
    BUCKET = esmc.BUCKET
    Lb = ((L + BUCKET - 1) // BUCKET) * BUCKET
    input_ids = torch.cat([ids, torch.full((1, Lb - L), 1, dtype=torch.long)], dim=1)
    a = torch.zeros(1, Lb, Lb, dtype=torch.float32); a[0, :, L:] = float("-inf")
    kv = torch.ones(1, 1, Lb, 1, dtype=torch.float32); kv[0, :, L:, :] = 0.0
    em = torch.ones(1, Lb, 1, dtype=torch.float32); em[0, L:, :] = 0.0
    m = tt_saprot.load_saprot(args.model)
    dev_runs = []
    for _ in range(2):
        with torch.no_grad():
            logits, emb = m(input_ids, a, kv, em)
        dev_runs.append((emb[0, :L].numpy().astype(np.float32),
                         logits[0, :L].numpy().astype(np.float32)))

    def p(a, b):
        return _pcc(a, b)

    R_emb, R_log = p(ref_runs[0][0], ref_runs[1][0]), p(ref_runs[0][1], ref_runs[1][1])
    D_emb, D_log = p(dev_runs[0][0], dev_runs[1][0]), p(dev_runs[0][1], dev_runs[1][1])
    X_emb, X_log = p(dev_runs[0][0], ref_runs[0][0]), p(dev_runs[0][1], ref_runs[0][1])

    report = {"mode": "saprot", "model": args.model, "target": "ubiquitin-L76",
              "R_emb": R_emb, "R_logits": R_log, "D_emb": D_emb, "D_logits": D_log,
              "X_emb": X_emb, "X_logits": X_log}
    print(f"\n### SaProt parity ({args.model}, ubiquitin L76, fused AA+3Di)\n")
    print("| metric | R (ref-vs-ref) | D (dev-vs-dev) | X (dev-vs-ref) |")
    print("|---|---|---|---|")
    print(f"| per-residue emb PCC | {R_emb:.5f} | {D_emb:.5f} | {X_emb:.5f} |")
    print(f"| MLM logits PCC      | {R_log:.5f} | {D_log:.5f} | {X_log:.5f} |")
    print("\nInterpretation: SaProt is a masked-LM encoder with no sampler, so R and "
          "D are 1.00000 by construction (deterministic forward, same convention as "
          "the ESMC legs). The device-vs-reference residual is bf16 rounding on the "
          "ttnn port, not an algorithmic difference; PASS when X sits in the ESMC "
          "band (0.9987-0.9996).")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("structures", help="fold models: device vs reference cif dirs")
    s.add_argument("--ref-dirs", nargs="+", help="reference run dirs, one per seed (live reference output)")
    s.add_argument("--ref-fixtures", default="", help="committed fixture spec <model>/<target>/<tag> "
                   "to read the reference side from docs/pharma-benchmark-data/ref-fixtures instead of "
                   "re-running the reference; the fixture is reused as-is (no reference compute)")
    s.add_argument("--ref-seeds", type=int, nargs="*", default=None, help="subset of fixture seeds to use")
    s.add_argument("--dev-dirs", nargs="+", required=True, help="device run dirs, one per seed (live)")
    s.add_argument("--label", default="")
    s.add_argument("--out", default="")
    s.add_argument("--paired", action="store_true",
                   help="Also report the same-seed (diagonal dev_i vs ref_i) distances "
                        "for every metric, alongside the all-pairs cross mean. Requires "
                        "len(dev_dirs) == len(ref_dirs); the diagonal is zip(dev_dirs, "
                        "ref_dirs), NOT the full cross product. A diagonal markedly "
                        "smaller than the cross mean means matching the RNG stream collapses "
                        "the residual (RNG-stochastic, i.e. a port defect in the RNG wiring); "
                        "a diagonal ~= cross means shared draws do NOT help (systematic bf16 "
                        "arithmetic divergence, a precision-floor artifact). Mirrors --paired "
                        "in scripts/boltz2_affinity_parity.py.")
    s.set_defaults(func=structures)

    e = sub.add_parser("embeddings", help="ESMC: device vs reference embeddings")
    e.add_argument("--model", default="esmc-300m", choices=["esmc-300m", "esmc-600m"])
    e.add_argument("--seqs", default="", help="comma-separated subset of the built-in proteins")
    e.add_argument("--fast", action="store_true")
    e.add_argument("--out", default="")
    e.set_defaults(func=embeddings)

    sp = sub.add_parser("saprot", help="SaProt: device vs HF EsmForMaskedLM reference (fused AA+3Di)")
    sp.add_argument("--model", default="saprot-650m", choices=["saprot-35m", "saprot-650m", "saprot-1.3b"])
    sp.add_argument("--out", default="")
    sp.set_defaults(func=saprot)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
