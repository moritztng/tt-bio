#!/usr/bin/env python3
"""Is the 11.69 A that the OPM output stage moved the 512 aa structure by a real degradation, or
is the fixture itself multi-modal on current main?

Pass 1 measured A (legacy = main) vs B (fast path) at 11.6870 A all-atom on the 512 aa cdk2x2
fold, against a 0.60 A kill bar and a 1.84 A seed floor that was measured on a tree 87 commits
back. Those two readings have opposite verdicts and the seed floor is the thing that decides:

  * if arm A alone, re-run at four seeds on THIS tree, already spreads ~10 A, the fixture is
    multi-modal here, the 1.84 A floor is stale, and the lever sits inside variation the campaign
    already accepts;
  * if A's own seed spread stays near 1.84 A, the lever moves the structure outside its own seed
    variation and the verdict is NO-GO regardless of the +0.1577 s.

Both arms are folded at every seed, in one process on one device open, arms interleaved and the
interior order reversed on odd seeds, so:

  A(s) vs A(s')   the fixture's own seed spread on current main -- the floor under test
  B(s) vs B(s')   the same spread for the fast path, which must look like A's if the lever is
                  only relabelling a basin
  A(s) vs B(s)    the lever's own deviation, now at four seeds instead of one

NO TIMING IS CLAIMED HERE. The clock is sampled and recorded but not forced, and the board pair is
not guarded, because an RMSD does not care what the sibling chip is doing. Every second in this
file is a progress note, not a measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TT_BIO_TRACE_REGION_SIZE", str(1 << 30))

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "c12_orchestrator" / "relayed" / "c12_kblock"))
sys.path.insert(0, str(REPO / "perf" / "other512"))
FIX = REPO / "perf" / "size512" / "fixtures"

RECYCLING_STEPS, SAMPLING_STEPS, DIFFUSION_SAMPLES = 3, 200, 1
OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def _seed_msa(target: Path, a3m_text: str, msa_dir: Path) -> None:
    from tt_bio.main import _read_bio_chains
    chains = _read_bio_chains(target)
    assert len(chains) == 1, f"{target} is not a monomer: {len(chains)} chains"
    seq = chains[0][1]
    assert a3m_text.split("\n")[1] == seq, "a3m query row does not match the target sequence"
    msa_dir.mkdir(parents=True, exist_ok=True)
    (msa_dir / f"{hashlib.sha256(seq.encode()).hexdigest()[:16]}.a3m").write_text(a3m_text)


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--seeds", default="0,1,2,3")
    args = ap.parse_args()
    OUT_PATH = args.out
    seeds = [int(s) for s in args.seeds.split(",")]

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import clk
    from cif_rmsd import kabsch_rmsd, read_atoms
    from tt_bio.tenstorrent import get_device
    from tt_bio import tenstorrent as _T
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)

    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    nodes = clk.nodes_open_by_this_process()

    import socket
    OUT["env"] = {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "lease_cards": os.environ.get("TT_BIO_LEASE_CARDS"),
        "umd_nodes_open": nodes,
        "grid": [g.y, g.x],
        "aiclk_forced": False,
        "timing_claimed": False,
        "seeds": seeds,
        "protocol": {"n_residues": 512, "recycling_steps": RECYCLING_STEPS,
                     "sampling_steps": SAMPLING_STEPS,
                     "diffusion_samples": DIFFUSION_SAMPLES,
                     "fixture": "perf/size512/fixtures/cdk2x2_512.yaml + a3m"},
    }
    OUT["flag_default_is_legacy"] = _T._opm_legacy_layout()
    dump()

    work = Path(tempfile.mkdtemp(prefix="opm-seedspread-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    target = FIX / "cdk2x2_512.yaml"
    _seed_msa(target, (FIX / "cdk2x2_512.a3m").read_text(), msa_dir)
    args.cifdir.mkdir(parents=True, exist_ok=True)

    cfg = dict(
        model="boltz2", fast=False, output_format="cif",
        recycling_steps=RECYCLING_STEPS, sampling_steps=SAMPLING_STEPS,
        diffusion_samples=DIFFUSION_SAMPLES, seed=0, trace=False,
        msa_dir=str(msa_dir), struct_dir=str(struct_dir),
        use_msa_server=False, msa_db_path=None, use_envdb=False, msa_endpoint=None,
        single_sequence=False, msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy", msa_server_username=None,
        msa_server_password=None, api_key_value=None, max_msa_seqs=8192,
        write_pae=False, write_pde=False, write_embeddings=False, method=None,
        conf_kwargs=dict(
            predict_args={"recycling_steps": RECYCLING_STEPS,
                          "sampling_steps": SAMPLING_STEPS,
                          "diffusion_samples": DIFFUSION_SAMPLES,
                          "max_parallel_samples": 5},
            diffusion_process_args={
                "step_scale": 1.5, "gamma_0": 0.8, "gamma_min": 1.0,
                "noise_scale": 1.003, "rho": 7, "sigma_min": 0.0001,
                "sigma_max": 160.0, "sigma_data": 16.0, "P_mean": -1.2,
                "P_std": 1.5, "coordinate_augmentation": True,
                "alignment_reverse_diff": True, "synchronize_sigmas": True},
            pairformer_args={"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True},
            msa_args={"subsample_msa": False, "num_subsampled_msa": 1024,
                      "use_paired_feature": True, "msa_s": 64, "msa_blocks": 4,
                      "msa_dropout": 0.15, "z_dropout": 0.25,
                      "pairwise_head_width": 32, "pairwise_num_heads": 4,
                      "activation_checkpointing": True},
            steering_args={"fk_steering": False, "physical_guidance_update": False,
                           "contact_guidance_update": True, "num_particles": 3,
                           "fk_lambda": 4.0, "fk_resampling_interval": 3,
                           "num_gd_steps": 20},
            use_kernels=True, use_tenstorrent=True, trace=False,
            diffusion_trace=True,
        ),
    )
    _ensure_local_artifacts(cfg)

    t_load = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("c14-opm-seedspread", cfg)
    state.pfn = lambda *a, **k: None
    state.model.progress_fn = lambda *a, **k: None
    OUT["model_load_s"] = round(time.perf_counter() - t_load, 3)
    dump()

    def fold(armname, seed):
        os.environ["TT_BIO_OPM_LEGACY_LAYOUT"] = "0" if armname == "B" else "1"
        cfg["seed"] = seed
        for p in struct_dir.glob("*"):
            p.unlink()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t0
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        keep = args.cifdir / f"512_{armname}_s{seed}"
        keep.mkdir(parents=True, exist_ok=True)
        for f in keep.glob("*.cif"):
            f.unlink()
        dst = keep / cifs[0].name
        dst.write_bytes(cifs[0].read_bytes())
        return {
            "arm": armname, "seed": seed,
            "fold_s_note_only": round(wall, 3),
            "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
            "cif_sha16": hashlib.sha256(dst.read_bytes()).hexdigest()[:16],
            "cif_path": str(dst),
        }

    runs = []
    sampler = clk.Sampler(nodes[0])
    print("[cold] discarded", flush=True)
    fold("A", seeds[0])
    for i, s in enumerate(seeds):
        for nm in (("A", "B") if i % 2 == 0 else ("B", "A")):
            r = fold(nm, s)
            runs.append(r)
            print("  seed %d %-2s  plddt %s  cif %s  (%.2f s, not a measurement)"
                  % (s, nm, r["plddt"], r["cif_sha16"], r["fold_s_note_only"]), flush=True)
            OUT["runs"] = runs; dump()
    OUT["clock_sampled_not_forced"] = sampler.stop()
    os.environ.pop("TT_BIO_OPM_LEGACY_LAYOUT", None)

    by = {(r["arm"], r["seed"]): r for r in runs}
    atoms = {k: read_atoms(Path(v["cif_path"])) for k, v in by.items()}
    ident = {tuple(a[0]) for a in atoms.values()}
    assert len(ident) == 1, "atom identity differs between folds -- not comparable atom-for-atom"

    def rms(k1, k2):
        return round(kabsch_rmsd(atoms[k1][1], atoms[k2][1]), 6)

    seed_spread = {}
    for arm in ("A", "B"):
        pairs = {f"s{a}_vs_s{b}": rms((arm, a), (arm, b))
                 for a, b in itertools.combinations(seeds, 2)}
        vals = list(pairs.values())
        seed_spread[arm] = {"pairs": pairs, "min": min(vals), "max": max(vals),
                            "median": round(st.median(vals), 6)}
    lever = {f"s{s}": rms(("A", s), ("B", s)) for s in seeds}
    lv = list(lever.values())

    OUT["n_atoms"] = len(atoms[("A", seeds[0])][1])
    OUT["seed_spread"] = seed_spread
    OUT["lever_A_vs_B_per_seed"] = {"per_seed": lever, "min": min(lv), "max": max(lv),
                                    "median": round(st.median(lv), 6)}
    OUT["plddt"] = {arm: {f"s{s}": by[(arm, s)]["plddt"] for s in seeds} for arm in ("A", "B")}
    OUT["bars"] = {"kill_bar_A": 0.60, "standing_seed_floor_A": 1.84}
    OUT["digests_distinct"] = len({r["cif_sha16"] for r in runs}) == len(runs)
    dump()
    print(json.dumps({k: OUT[k] for k in (
        "n_atoms", "seed_spread", "lever_A_vs_B_per_seed", "plddt", "bars",
        "clock_sampled_not_forced")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
