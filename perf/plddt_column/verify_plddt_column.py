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

from cif_rmsd import atom_site_table  # noqa: E402  -- one B-factor parser for this lineage

FIX = REPO / "perf" / "size512" / "fixtures"


def bfactor_means(cif: Path) -> dict:
    """Mean B-factor over all atoms and over CA only, both on the 0..1 plDDT scale."""
    idx, rows = atom_site_table(cif)
    col, name = "_atom_site.B_iso_or_equiv", "_atom_site.label_atom_id"
    b = [float(f[idx[col]]) for f in rows]
    ca = [v for v, f in zip(b, rows) if f[idx[name]].strip('"') == "CA"]
    return {"n_atoms": len(b), "n_ca": len(ca),
            "mean_all": round(sum(b) / len(b) / 100.0, 6),
            "mean_ca": round(sum(ca) / len(ca) / 100.0, 6) if ca else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", default="boltz2")
    ap.add_argument("--fixtures", default="cdk2x2_298")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--recycles", type=int, default=0)
    ap.add_argument("--keep-cif", type=Path, default=None)
    args = ap.parse_args()

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
            col = bfactor_means(cifs[0])
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
