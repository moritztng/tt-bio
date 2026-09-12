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
    # 9bk6 / 8hel / 7xi5 are not panel members: their committed alignments are OpenFold3-style
    # .sto hit files, which Boltz-2's MSA resolver cannot read. Folding them here would either
    # reach for ColabFold or quietly drop to single-sequence, so the preflight rejects them.
    "prot": {"yaml": "examples/prot.yaml",
             "a3m": "docs/implementation-parity-data/ref-fixtures/boltz2/prot/"
                    "msa-colabfold_200step_1sample_3recycle_bf16/msa.a3m",
             "kind": "monomer", "note": "PDB 7ROA, 117 aa -- the release gate's own fold target"},
    "ubq": {"yaml": "examples/ubq.yaml",
            "a3m": "docs/implementation-parity-data/ref-fixtures/boltz2/ubiquitin/"
                   "msa-colabfold_200step_1sample_3recycle_bf16_gpu/msa.a3m",
            "kind": "monomer", "note": "ubiquitin, 76 aa -- the small end"},
    "multimer": {"yaml": "examples/multimer.yaml", "kind": "complex", "single_sequence": True,
                 "note": "two-chain protein-protein complex, folded single-sequence (no "
                         "committed per-chain alignment exists for it)"},
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

    # build_fold's cfg carries no Boltz-2 hyperparameters; fold_ab_multi injects exactly the ones
    # tt_bio.main builds, which is what the published cell ran. Seeded at the reference setting so
    # the model loads as production, then moved per fold (see set_arm).
    B.RECYCLING_STEPS, B.SAMPLING_STEPS = args.base_recycles, args.base_steps
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

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
    # Boltz-2's feature prep is a partial bound in bind_run, so the MSA flags come from the BOUND
    # cfg, not from the cfg handed to predict_one. Rebind, or a single-sequence target quietly
    # folds with whatever the load-time cfg said.
    state.bind_run("b2z", base_cfg)
    state.pfn = lambda *a, **k: None

    # What alignment this target will actually read, checked before the first fold rather than
    # inferred from a metrics key that the boltz-2 path does not emit. Every panel member is
    # either seeded from a committed a3m, carries an ``msa:`` path in its own yaml, or is declared
    # single-sequence; anything else would reach for the network and is an error here.
    def preflight_msa() -> dict:
        import yaml as _yaml
        if spec.get("single_sequence"):
            return {"source": "single_sequence", "rows": 0}
        if "a3m" in spec:
            return {"source": str(ROOT / spec["a3m"]), "rows": meta["n_msa"]}
        doc = _yaml.safe_load(tgt.read_text()) or {}
        rows = {}
        for ent in doc.get("sequences", []):
            prot = ent.get("protein")
            if not prot:
                continue
            m = prot.get("msa")
            assert m and m != "empty", f"{tgt} chain {prot.get('id')} has no committed MSA"
            mp = ROOT / m
            cands = [mp] if mp.is_file() else sorted(mp.glob("*.a3m")) + sorted(mp.glob("*.csv"))
            assert cands, f"{mp} does not exist"
            rows[prot["id"]] = sum(c.read_text().count(">") or c.read_text().count(chr(10))
                                   for c in cands[:1])
        return {"source": "yaml msa: paths", "rows": rows}

    OUT["env"]["msa"] = preflight_msa()
    print(f"  msa: {OUT['env']['msa']}", flush=True)
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
    # Boltz2 reads its step and recycle counts off ``model.predict_args`` inside ``predict_step``
    # (boltz2.py:5893), not off the job cfg -- the cfg keys only reach the models whose worker
    # branch passes them down. An arm that only writes the cfg runs 200/3 every time and reports a
    # flat, wrong curve, so the arm is the dict on the loaded model and the cfg is set to match so
    # the two can never disagree.
    def set_arm(steps: int, recycles: int) -> None:
        pa = state.model.predict_args
        pa["sampling_steps"], pa["recycling_steps"] = steps, recycles

    # Count what the arm actually executed. A setting that silently does not reach the sampler
    # produces a perfectly flat curve that looks like a finding, so every row carries the number
    # of denoise calls and the number of trunk cycles it really ran, and the run asserts they
    # match what was asked for.
    from tt_bio.boltz2 import AtomDiffusion
    from tt_bio.tenstorrent import TrunkModule
    CNT = {"denoise": 0, "trunk_recycles": None}
    _pnf, _trunk_call = AtomDiffusion.preconditioned_network_forward, TrunkModule.__call__

    def _counted_pnf(self, *a, **k):
        CNT["denoise"] += 1
        return _pnf(self, *a, **k)

    def _counted_trunk(self, s_inputs, s_init, z_init, feats, recycling_steps, *a, **k):
        # The recycle loop runs resident on the card, so counting host-side iterations would
        # record 1 for every arm. Record the count the device loop was actually handed.
        CNT["trunk_recycles"] = int(recycling_steps)
        return _trunk_call(self, s_inputs, s_init, z_init, feats, recycling_steps, *a, **k)

    AtomDiffusion.preconditioned_network_forward = _counted_pnf
    TrunkModule.__call__ = _counted_trunk

    def fold(steps: int, recycles: int, seed: int, tag: str):
        set_arm(steps, recycles)
        CNT["denoise"], CNT["trunk_recycles"] = 0, None
        cfg = dict(base_cfg)
        cfg.update(sampling_steps=steps, recycling_steps=recycles, seed=seed,
                   struct_dir=str(struct_dir))
        for p in struct_dir.glob("*"):
            if p.is_file():
                p.unlink()
        t0 = time.perf_counter()
        metrics, _best, _feats = state.predict_one(tgt, cfg)
        dt = time.perf_counter() - t0
        try:                                   # the alignment the fold really saw, per fold
            metrics = dict(metrics, msa_depth=int(_feats["msa"].shape[-2]))
        except Exception:
            pass
        counts = dict(CNT)
        cifs = {}
        for f in sorted(struct_dir.glob("*.cif")):
            dst = args.cifdir / f"{args.target}__{tag}__{f.name}"
            shutil.copyfile(f, dst)
            cifs[f.name] = {"path": str(dst), "sha256": sha256_file(dst)}
        return dt, metrics, cifs, counts

    print(f"=== cold fold {args.target} ({args.base_steps}/{args.base_recycles}) ===", flush=True)
    cold_s, cold_m, _, cold_c = fold(args.base_steps, args.base_recycles, 0, "cold")
    OUT["cold"] = {"fold_s": round(cold_s, 3), "plddt": cold_m.get("plddt"),
                   "n_tokens": cold_m.get("n_tokens"), "n_residues": cold_m.get("n_residues"),
                   "msa": cold_m.get("msa"), "counts": cold_c}
    print(f"  cold {cold_s:.2f}s plddt={cold_m.get('plddt')}", flush=True)
    dump()

    for g in grid:
        tag = f"s{g['steps']}_r{g['recycles']}_seed{g['seed']}"
        try:
            dt, m, cifs, counts = fold(g["steps"], g["recycles"], g["seed"], tag)
        except Exception:
            import traceback
            OUT["folds"].append({**g, "tag": tag, "error": traceback.format_exc()})
            dump()
            print(f"  {tag:24s} FAILED", flush=True)
            continue
        rec = {**g, "tag": tag, "fold_s": round(dt, 3),
               "plddt": m.get("plddt", m.get("complex_plddt")), "ptm": m.get("ptm"),
               "n_tokens": m.get("n_tokens"), "n_residues": m.get("n_residues"),
               "msa_depth": m.get("msa_depth"), "cifs": cifs, "counts": counts,
               "loadavg": open("/proc/loadavg").read().split()[:3]}
        # Structure diffusion plus the confidence head's own rollout both go through the denoiser,
        # so the count is asserted to be a multiple of the requested step count rather than equal
        # to it; what matters is that it moved with the arm.
        assert counts["denoise"] % g["steps"] == 0, f"{tag}: {counts['denoise']} denoise calls"
        assert counts["trunk_recycles"] == g["recycles"], f"{tag}: trunk ran {counts['trunk_recycles']}"
        OUT["folds"].append(rec)
        dump()
        print(f"  {tag:24s} {dt:7.2f}s plddt={rec['plddt']} "
              f"{list(cifs.values())[0]['sha256'][:16] if cifs else 'NO-CIF'}", flush=True)

    OUT["env"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dump()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
