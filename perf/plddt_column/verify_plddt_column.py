#!/usr/bin/env python3
"""Does the plDDT a model reports equal the mean of the B-factor column it writes?

`k10-p2` flagged that tt-bio's reported plDDT was 0.048 away from the mean B-factor of its own
CIF at 512 aa and 0.008 away with the opposite sign at 298 aa. This runs the comparison inside
one process, on folds short enough to run on CPU, so the two numbers come from one fold and the
answer is a measurement rather than a reading of two tables.

For every model the check is the same: `metrics` must carry a complex-level plDDT that equals the
mean of the per-atom B-factor column of the CIF that fold just wrote, to the writer's own
rounding. Boltz-2 writes one plDDT per residue, so its mean over all atoms is atom-count weighted
and only its mean over CA matches; Protenix-v2 / OpenFold3 / OpenBind-0 / OpenDDE write a genuine
per-atom column and their mean over all atoms must match. Both readings are reported for every
fold, no device is opened, and the B-factor parser is `perf/other512/cif_rmsd.py`'s.

    verify_plddt_column.py --out out/verify.json --models boltz2 --steps 10 --recycles 0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))
sys.path.insert(0, str(REPO / "perf" / "other512"))

from cif_rmsd import bfactor_plddt, plddt_column_check  # noqa: E402  -- one B-factor parser

FIX = REPO / "perf" / "size512" / "fixtures"


def sweep() -> dict:
    """Run the column check over every fold on disk that ships a plDDT and a structure.

    The models this cannot fold here are covered by what is already committed: upstream
    reference fixtures under docs/, tt-bio OpenFold3 device folds under perf/of3_4xpd, tt-bio RF3
    device folds under perf/fused_sdpa (matched to their recorded plDDT by CIF digest, since that
    harness keeps the CIFs in one directory and the metrics in per-arm JSONs).
    """
    import hashlib
    rows = []

    def add(kind, name, cif, reported):
        rows.append({"kind": kind, "name": name, "cif": str(cif.relative_to(REPO)),
                     **plddt_column_check(cif, round(float(reported), 6))})

    for f in sorted((REPO / "docs/implementation-parity-data/ref-fixtures").glob("*/*/*/*/results.json")):
        cifs = sorted((f.parent / "structures").glob("*.cif"))
        if not cifs:
            continue
        try:
            rs = json.loads(f.read_text())
        except ValueError:
            continue
        r = rs[0] if isinstance(rs, list) else rs
        rep = next((r[k] for k in ("plddt", "complex_plddt") if k in r), None)
        if rep is not None:
            add("upstream-ref", str(f.parent).split("ref-fixtures/")[1], cifs[0], rep)

    for js, cd, size in (("ab_298_H_qb1c1.json", "h298_cifs", 298),
                         ("ab_512_H_qb1c1.json", "h_cifs", 512),
                         ("ab_512_AH_qb1c1.json", "ah_cifs", 512)):
        j = REPO / "perf/of3_4xpd" / js
        if not j.exists():
            continue
        for fo in json.loads(j.read_text())["folds"]:
            tag = fo["tag"] + ("_memo" if fo["memo"] else "_plain")
            cif = REPO / "perf/of3_4xpd" / cd / f"{tag}_cdk2x2_{size}.cif"
            if cif.exists():
                add("tt-openfold3", f"{js}:{tag}", cif, fo["plddt"])

    digests = {}
    for j in sorted((REPO / "perf/fused_sdpa").rglob("fold.json")):
        for fo in json.loads(j.read_text()).get("folds", []):
            for v in (fo.get("cif_sha256") or {}).values():
                digests[v] = fo.get("plddt")
    for cif in sorted((REPO / "perf/fused_sdpa/cifs").glob("rf3_*.cif")):
        rep = digests.get(hashlib.sha256(cif.read_bytes()).hexdigest()[:16])
        if rep is not None:
            add("tt-rf3", cif.name, cif, rep)

    # The check has to be able to fail, or a green sweep proves nothing
    # (memory `negative-control-must-break-what-check-reads`). confidence_score is the quantity
    # the harnesses used to record under the name plddt, so feed it in deliberately: every one of
    # these must come back ok=false, and if any comes back true the check has gone blind.
    sens = []
    for j in sorted((REPO / "perf/plddt_column/out").glob("verify_*.json")):
        for r in json.loads(j.read_text())["runs"]:
            cif = REPO / "perf/plddt_column/cif" / f"{r['fixture']}_{r['model']}" / f"{r['fixture']}.cif"
            if not cif.exists():
                continue
            if r["reported_plddt"] is not None:
                add(f"tt-{r['model']}", f"{j.name}:{r['fixture']}", cif, r["reported_plddt"])
            if r.get("confidence_score_fallback") is not None:
                c = plddt_column_check(cif, round(float(r["confidence_score_fallback"]), 6))
                sens.append({"name": f"{j.stem}:{r['fixture']}", "fed": "confidence_score",
                             "rejected": c["ok"] is False, "gap": c["gap"]})

    bad = [r for r in rows if r["ok"] is False]
    return {"n": len(rows), "mismatches": bad,
            "sensitivity": sens, "blind": [x for x in sens if not x["rejected"]],
            "by_kind": {k: {"n": sum(1 for r in rows if r["kind"] == k),
                            "pass": sum(1 for r in rows if r["kind"] == k and r["ok"] is True),
                            "no_column": sum(1 for r in rows if r["kind"] == k and r["ok"] is None),
                            "worst_gap": max([abs(r["gap"]) for r in rows
                                              if r["kind"] == k and r["ok"] is True], default=None),
                            "readings": sorted({r["reading"] for r in rows
                                                if r["kind"] == k and r["ok"] is not None})}
                        for k in sorted({r["kind"] for r in rows})},
            "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default="boltz2")
    ap.add_argument("--fixtures", default="cdk2x2_298")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--recycles", type=int, default=0)
    ap.add_argument("--keep-cif", type=Path, default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="check every fold on disk that ships a plDDT and a structure, fold "
                         "nothing, and exit non-zero on any mismatch")
    args = ap.parse_args()

    if args.sweep:
        out = {"doc": __doc__, "sweep": sweep()}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))
        for k, v in out["sweep"]["by_kind"].items():
            print(f"  {k:16s} n={v['n']:3d} pass={v['pass']:3d} no_column={v['no_column']:3d} "
                  f"worst_gap={v['worst_gap']} readings={v['readings']}")
        bad, blind = out["sweep"]["mismatches"], out["sweep"]["blind"]
        for x in out["sweep"]["sensitivity"]:
            print(f"  negative control  {x['name']:22s} confidence_score rejected="
                  f"{x['rejected']} gap={x['gap']:+.6f}")
        print(f"\n  {out['sweep']['n']} folds checked, {len(bad)} mismatch(es) -> {args.out}")
        for r in bad:
            print(f"    MISMATCH {r['name']}: reported {r['reported']} vs {r['reading']} "
                  f"{r[r['reading']]}, gap {r['gap']:+.6f}")
        for x in blind:
            print(f"    BLIND {x['name']}: the check accepted confidence_score")
        return 1 if bad or blind else 0

    import torch
    torch.set_grad_enabled(False)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree -- the venv's installed package "
        "would score a different tree (memory parity-gate-scores-installed-package-not-checkout)")
    import ab_flag_levers as AB
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts

    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = args.steps, args.recycles
    out = {"doc": __doc__, "env": {
        "host": socket.gethostname(), "accelerator": "cpu", "torch": torch.__version__,
        "threads": torch.get_num_threads(), "tt_bio_file": _TB.__file__,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {"sampling_steps": args.steps, "recycling_steps": args.recycles,
                     "diffusion_samples": 1, "seed": 0},
    }, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))

    dump()
    work = Path(tempfile.mkdtemp(prefix="plddt-col-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for name in args.fixtures.split(","):
        AB._seed_msa(FIX / f"{name}.yaml", (FIX / f"{name}.a3m").read_text(), msa_dir)

    for model in args.models.split(","):
        cfg = AB.build_cfg(msa_dir, struct_dir)
        cfg["model"] = model
        cfg["conf_kwargs"] = {**cfg["conf_kwargs"], "use_tenstorrent": False, "use_kernels": False}
        _ensure_local_artifacts(cfg)
        state = _WorkerState("cpu")
        state.load_model(cfg)
        state.bind_run(f"plddt-col-{model}", cfg)
        for name in args.fixtures.split(","):
            for p in struct_dir.glob("*"):
                p.unlink() if p.is_file() else shutil.rmtree(p)
            t = time.perf_counter()
            metrics, _b, _f = state.predict_one(FIX / f"{name}.yaml", cfg)
            wall = round(time.perf_counter() - t, 1)
            cifs = sorted(struct_dir.glob("*.cif"))
            assert cifs, f"{model} wrote no CIF"
            if args.keep_cif:
                d = args.keep_cif / f"{name}_{model}"
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cifs[0], d / cifs[0].name)
            m = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            col = bfactor_plddt(cifs[0])
            # the complex-level plDDT this model reports, under whichever key it uses
            reported = next((m[k] for k in ("complex_plddt", "plddt", "mean_plddt") if k in m), None)
            r = {"model": model, "fixture": name, "fold_s": wall, "metrics": m,
                 "metrics_keys": sorted(metrics), "bfactor": col, "reported_plddt": reported,
                 "d_mean_all": None if reported is None else round(reported - col["mean_all"], 6),
                 "d_mean_ca": None if reported is None or col["mean_ca"] is None
                              else round(reported - col["mean_ca"], 6)}
            # what a reader that asks for metrics["plddt"] with a confidence_score fallback gets,
            # which is what perf/k10_p2/fold_cpu_ref.py did and where the flagged mismatch came from
            r["plddt_key_present"] = "plddt" in metrics
            r["confidence_score_fallback"] = m.get("confidence_score")
            out["runs"].append(r)
            print(f"  {model:12s} {name:14s} {wall:7.1f}s reported={reported} "
                  f"mean_all={col['mean_all']} mean_ca={col['mean_ca']} "
                  f"d_all={r['d_mean_all']} d_ca={r['d_mean_ca']} "
                  f"plddt_key={r['plddt_key_present']} conf={r['confidence_score_fallback']}",
                  flush=True)
            dump()
        del state
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
