#!/usr/bin/env python3
"""Boltz-2 step/recycle quality curve: one target, one card, the whole grid in one process.

The campaign's other levers make each of the 200 diffusion steps cheaper. This one asks
whether 200 steps and 3 recycles are the right numbers at all. That question is an accuracy
question, so this harness is built for accuracy first and time second.

The design decision that makes the curve readable: **the seed floor**. Boltz-2's sampler is
stochastic (Karras EDM schedule with churn, gamma_0=0.8), and changing the step count changes
the noise draws from the first step on. So the RMSD between a 50-step fold and a 200-step fold
contains two things at once: the real quality loss, and the pure sampler chaos two different
seeds would have produced anyway. Every target therefore folds (200,3) at three seeds, and the
spread of those three is the floor any arm has to beat before its RMSD means anything. Without
it the whole table is unreadable (see `cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`,
`confidence-scalar-not-a-parity-bar`).

One process holds one card, loads the model once, and folds the whole grid, so the arms share
weights, the MSA cache and the device's kernel caches. Results are written after every fold:
a run that is cut off still lands what it measured.

Timing here is wall clock on a box that may be running ten of these at once, so it is recorded
but it is NOT the perf claim. The perf claim comes from `paired_time.py`, one card, arms
interleaved in one process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

OUT: dict = {"folds": []}
OUT_PATH: Path | None = None


def dump() -> None:
    if OUT_PATH is not None:
        tmp = OUT_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(OUT, indent=1))
        tmp.replace(OUT_PATH)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        for b in iter(lambda: fp.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


#: The panel. ``a3m`` targets are monomers whose alignment is committed next to the fixture and
#: installed into the MSA cache by hand; every other target carries an ``msa:`` path inside its
#: own yaml, which the normal resolve path reads straight off disk. Nothing here touches the
#: network, so a ColabFold outage cannot silently turn a panel member into a single-sequence fold.
PANEL: dict[str, dict] = {
    "cdk2x2_298": {"yaml": "perf/size512/fixtures/cdk2x2_298.yaml",
                   "a3m": "perf/size512/fixtures/cdk2x2_298.a3m",
                   "kind": "monomer", "note": "CDK2, one real domain -- the campaign's control"},
    "cdk2x2_512": {"yaml": "perf/size512/fixtures/cdk2x2_512.yaml",
                   "a3m": "perf/size512/fixtures/cdk2x2_512.a3m",
                   "kind": "monomer-chimera", "hinge": True,
                   "note": "THE CELL. CDK2 fused to a truncated copy of itself; the inter-domain "
                           "hinge saturates whole-structure RMSD, so score it per domain"},
    "cdk2x2_128": {"yaml": "perf/size512/fixtures/cdk2x2_128.yaml",
                   "a3m": "perf/size512/fixtures/cdk2x2_128.a3m",
                   "kind": "monomer", "note": "small end of the size ladder"},
    "affinity_fkg": {"yaml": "examples/affinity_fkg_msa.yaml",
                     "kind": "protein+ligand", "note": "FKBP + CCD SB3, affinity target"},
    "affinity_dhfr": {"yaml": "examples/affinity_dhfr_msa.yaml",
                      "kind": "protein+ligand", "note": "DHFR + ligand"},
    "affinity_tryp": {"yaml": "examples/affinity_tryp_msa.yaml",
                      "kind": "protein+ligand", "note": "trypsin + ligand"},
    "9bk6": {"yaml": "examples/9bk6.yaml",
             "kind": "complex", "note": "two-chain protein-protein complex, MSA per chain"},
    "8hel": {"yaml": "examples/8hel_msa.yaml", "kind": "monomer", "note": "OpenFold3 bench target"},
    "7xi5": {"yaml": "examples/7xi5_notmpl.yaml", "kind": "monomer",
             "note": "OpenFold3 bench target, templates off"},
    "hsa": {"yaml": "examples/hsa.yaml", "kind": "monomer", "single_sequence": True,
            "note": "human serum albumin, 585 aa -- the long end, folded single-sequence"},
}

#: The grid. Steps swept at production recycles; recycles swept at production steps; the four
#: corners that matter if both axes move together; and the seed floor. Written as an explicit
#: list rather than a product so the corners are visible and the floor is not an afterthought.
def build_grid(steps_axis, recycle_axis, corners, floor_seeds, base_steps, base_recycles):
    grid = [{"steps": base_steps, "recycles": base_recycles, "seed": 0, "role": "reference"}]
    grid += [{"steps": s, "recycles": base_recycles, "seed": 0, "role": "steps"}
             for s in steps_axis if s != base_steps]
    grid += [{"steps": base_steps, "recycles": r, "seed": 0, "role": "recycles"}
             for r in recycle_axis if r != base_recycles]
    grid += [{"steps": s, "recycles": r, "seed": 0, "role": "corner"} for s, r in corners]
    grid += [{"steps": base_steps, "recycles": base_recycles, "seed": sd, "role": "floor"}
             for sd in floor_seeds]
    return grid


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=sorted(PANEL))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--steps", default="200,150,100,75,50,25")
    ap.add_argument("--recycles", default="3,2,1,0")
    ap.add_argument("--corners", default="100:2,50:2,100:1,50:1")
    ap.add_argument("--floor-seeds", default="1,2")
    ap.add_argument("--base-steps", type=int, default=200)
    ap.add_argument("--base-recycles", type=int, default=3)
    args = ap.parse_args()
    OUT_PATH = args.out
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    args.cifdir.mkdir(parents=True, exist_ok=True)

    spec = PANEL[args.target]
    tgt = ROOT / spec["yaml"]

    import torch
    torch.set_grad_enabled(False)
    import tt_bio.tenstorrent as T
    import tt_baseline as B

    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"

    import importlib.metadata as im
    OUT["env"] = {
        "host": os.uname().nodename,
        "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "lease_cards": os.environ.get("TT_BIO_LEASE_CARDS"),
        "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "ttnn": im.version("ttnn"), "torch": torch.__version__,
        "target": args.target, "target_spec": spec,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    dump()

    # build_fold opens the card and loads the model. It seeds the MSA cache for the target it is
    # handed, which only works for a monomer with a committed a3m; for every other panel member
    # the alignment path lives in the yaml, so it is seeded with the a3m targets' fixture and the
    # real target is folded through state.predict_one directly, exactly as baseline_attrib's
    # control phase does.
    msa_dir = Path(__file__).resolve().parent / f".msa_{args.target}"
    seed_tgt = tgt if "a3m" in spec else ROOT / "perf/size512/fixtures/cdk2x2_128.yaml"
    seed_a3m = ROOT / spec["a3m"] if "a3m" in spec else ROOT / "perf/size512/fixtures/cdk2x2_128.a3m"
    _one_fold, meta, state = B.build_fold("boltz2", msa_dir, seed_tgt, seed_a3m)

    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "load_s", "card_type", "aiclk_mhz")
                       if k in meta})
    struct_dir = Path(meta["struct_dir"])
    base_cfg = dict(meta["job_cfg"])
    if spec.get("single_sequence"):
        base_cfg["single_sequence"] = True
        base_cfg["use_msa_server"] = False
    dump()

    grid = build_grid([int(x) for x in args.steps.split(",")],
                      [int(x) for x in args.recycles.split(",")],
                      [tuple(int(v) for v in c.split(":")) for c in args.corners.split(",") if c],
                      [int(x) for x in args.floor_seeds.split(",") if x],
                      args.base_steps, args.base_recycles)
    OUT["grid"] = grid
    dump()

    # Cold fold at the reference setting, discarded: it warms every kernel cache, and until it has
    # run a timing number is a compile time, not a fold time.
    def fold(steps: int, recycles: int, seed: int, tag: str):
        cfg = dict(base_cfg)
        cfg.update(sampling_steps=steps, recycling_steps=recycles, seed=seed,
                   struct_dir=str(struct_dir))
        for p in struct_dir.glob("*"):
            if p.is_file():
                p.unlink()
        t0 = time.perf_counter()
        metrics, _best, _feats = state.predict_one(tgt, cfg)
        dt = time.perf_counter() - t0
        cifs = {}
        for f in sorted(struct_dir.glob("*.cif")):
            dst = args.cifdir / f"{args.target}__{tag}__{f.name}"
            shutil.copyfile(f, dst)
            cifs[f.name] = {"path": str(dst), "sha256": sha256_file(dst)}
        return dt, metrics, cifs

    print(f"=== cold fold {args.target} ({args.base_steps}/{args.base_recycles}) ===", flush=True)
    cold_s, cold_m, _ = fold(args.base_steps, args.base_recycles, 0, "cold")
    assert cold_m.get("msa") or spec.get("single_sequence"), "fold ran without an MSA"
    OUT["cold"] = {"fold_s": round(cold_s, 3), "plddt": cold_m.get("plddt"),
                   "n_tokens": cold_m.get("n_tokens"), "n_residues": cold_m.get("n_residues"),
                   "msa": cold_m.get("msa")}
    print(f"  cold {cold_s:.2f}s plddt={cold_m.get('plddt')}", flush=True)
    dump()

    for g in grid:
        tag = f"s{g['steps']}_r{g['recycles']}_seed{g['seed']}"
        try:
            dt, m, cifs = fold(g["steps"], g["recycles"], g["seed"], tag)
        except Exception:
            import traceback
            OUT["folds"].append({**g, "tag": tag, "error": traceback.format_exc()})
            dump()
            print(f"  {tag:24s} FAILED", flush=True)
            continue
        rec = {**g, "tag": tag, "fold_s": round(dt, 3),
               "plddt": m.get("plddt", m.get("complex_plddt")), "ptm": m.get("ptm"),
               "n_tokens": m.get("n_tokens"), "n_residues": m.get("n_residues"),
               "cifs": cifs, "loadavg": open("/proc/loadavg").read().split()[:3]}
        OUT["folds"].append(rec)
        dump()
        print(f"  {tag:24s} {dt:7.2f}s plddt={rec['plddt']} "
              f"{list(cifs.values())[0]['sha256'][:16] if cifs else 'NO-CIF'}", flush=True)

    OUT["env"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dump()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
