#!/usr/bin/env python3
"""Boltz-2 512 aa fold A/B for the OuterProductMean output stage (c14-opm-layout).

Three arms, interleaved rep by rep in ONE process on ONE device open, each fold through the
production ``_WorkerState.predict_one``:

  A    TT_BIO_OPM_LEGACY_LAYOUT=1 -- the pre-c14 stage: the 1/S mean applied to z (536,870,912 B)
       and proj_o issued against the token-row batch with the core grid pinned
  B    TT_BIO_OPM_LEGACY_LAYOUT=0 -- the branch default: scale folded into `a` (2,097,152 B),
       the row batch merged into M, program config left to the matmul
  A2   arm A again, as this session's own A/A floor -- both in time and in structure

`_opm_legacy_layout()` reads the environment per call, which is why one process can carry both
arms: no reload, no second compile, no cross-session ratio. Structure of the harness, the config
block and the per-fold CIF hash are `perf/b2x_trimul/fold_ab_trimul_512.py`'s, unchanged.

The delta under test lives in MSALayer, i.e. in the trunk. So the trunk boundary is marked and
the sampler is carried as a KNOWN-ANSWER CONTROL: an arm effect that shows up in the sampler
rather than the trunk is contention, not this lever, and the run says so instead of scoring it.
"""
from __future__ import annotations

import argparse
import hashlib
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

RECYCLING_STEPS, SAMPLING_STEPS, DIFFUSION_SAMPLES, SEED = 3, 200, 1, 0
MHZ = 1350
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
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--arms", default="A,B,A2")
    args = ap.parse_args()
    OUT_PATH = args.out

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

    # Clock first: every number below is read at a forced, during-sampled AICLK or not at all.
    held = clk.force(MHZ, clk.nodes_open_by_this_process())
    for _ in range(200):
        if all(clk.aiclk(n) >= MHZ - 5 for n in held):
            break
        time.sleep(0.05)

    import socket
    OUT["env"] = {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "lease_cards": os.environ.get("TT_BIO_LEASE_CARDS"),
        "umd_nodes_open": held,
        "grid": [g.y, g.x],
        "aiclk_forced_mhz": MHZ,
        "torch": torch.__version__,
        "protocol": {"n_residues": 512, "recycling_steps": RECYCLING_STEPS,
                     "sampling_steps": SAMPLING_STEPS,
                     "diffusion_samples": DIFFUSION_SAMPLES, "seed": SEED,
                     "fixture": "perf/size512/fixtures/cdk2x2_512.yaml + a3m"},
    }
    # The flag's shipped default, read before any arm has touched it: "B is the default" is part
    # of the claim, so it is recorded rather than assumed.
    OUT["flag_default_is_legacy"] = _T._opm_legacy_layout()
    dump()

    work = Path(tempfile.mkdtemp(prefix="opm-foldab-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    target = FIX / "cdk2x2_512.yaml"
    _seed_msa(target, (FIX / "cdk2x2_512.a3m").read_text(), msa_dir)
    args.cifdir.mkdir(parents=True, exist_ok=True)

    cfg = dict(
        model="boltz2", fast=False, output_format="cif",
        recycling_steps=RECYCLING_STEPS, sampling_steps=SAMPLING_STEPS,
        diffusion_samples=DIFFUSION_SAMPLES, seed=SEED, trace=False,
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
    state.bind_run("c14-opm-layout", cfg)
    OUT["model_load_s"] = round(time.perf_counter() - t_load, 3)
    dump()

    class Splitter:
        """Syncs only at the three stage transitions, all outside the 200-step loop."""
        def __init__(self):
            self.marks = []
            self.n_diff = 0

        def _mark(self, label):
            ttnn.synchronize_device(dev)
            self.marks.append((label, time.perf_counter()))

        def __call__(self, stage=None, step=0, total=0, *a, **k):
            if stage == "diffusion":
                self.n_diff += 1
                if self.n_diff == 1:
                    self._mark("diffusion_conditioning")
                elif self.n_diff == 2:
                    self._mark("sampler")
            elif stage == "confidence":
                self._mark("confidence")

        def close(self):
            self._mark("end")
            return {a: round(t1 - t0, 4)
                    for (a, t0), (_b, t1) in zip(self.marks, self.marks[1:])}

    def fold(armname, rep):
        os.environ["TT_BIO_OPM_LEGACY_LAYOUT"] = "0" if armname == "B" else "1"
        sp = Splitter()
        state.pfn = sp
        state.model.progress_fn = sp
        for p in struct_dir.glob("*"):
            p.unlink()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t0
        stages = sp.close()
        pre = sp.marks[0][1] - t0
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        keep = args.cifdir / f"512_{armname}_{rep}"
        keep.mkdir(parents=True, exist_ok=True)
        for f in keep.glob("*.cif"):
            f.unlink()
        (keep / cifs[0].name).write_bytes(cifs[0].read_bytes())
        return {
            "arm": armname, "rep": rep,
            "fold_s": round(wall, 4),
            "prepare_and_trunk_s": round(pre, 4),
            "stages_s": stages,
            "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
            "cif_sha16": hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16],
            "cif_path": str(keep / cifs[0].name),
        }

    ARMS = args.arms.split(",")
    runs = []
    sampler = clk.Sampler(held[0])
    print("[cold] discarded", flush=True)
    r = fold("A", -1); r["cold"] = True; runs.append(r)
    print("  cold %.3f s" % r["fold_s"], flush=True)
    OUT["runs"] = runs; dump()
    for i in range(args.reps):
        # Interior order reversed on odd reps so drift cannot land on one arm.
        for nm in (ARMS if i % 2 == 0 else list(reversed(ARMS))):
            r = fold(nm, i); r["cold"] = False
            runs.append(r)
            print("  rep %d %-2s %9.4f s  trunk %8.4f  plddt %s  cif %s"
                  % (i, nm, r["fold_s"], r["prepare_and_trunk_s"], r["plddt"], r["cif_sha16"]),
                  flush=True)
            OUT["runs"] = runs; dump()
    OUT["clock"] = sampler.stop()
    os.environ.pop("TT_BIO_OPM_LEGACY_LAYOUT", None)

    warm = [r for r in runs if not r["cold"]]

    def med(nm, key="fold_s"):
        return round(st.median([r[key] for r in warm if r["arm"] == nm]), 4)

    OUT["median_fold_s"] = {nm: med(nm) for nm in ARMS}
    OUT["median_trunk_s"] = {nm: med(nm, "prepare_and_trunk_s") for nm in ARMS}
    OUT["median_sampler_s"] = {
        nm: round(st.median([r["stages_s"].get("sampler", float("nan"))
                             for r in warm if r["arm"] == nm]), 4) for nm in ARMS}
    A, B = OUT["median_fold_s"]["A"], OUT["median_fold_s"]["B"]
    OUT["AA_floor_pct"] = round(100 * abs(OUT["median_fold_s"]["A2"] - A) / A, 4)
    OUT["delta_s"] = round(A - B, 4)
    OUT["x_vs_A"] = {nm: round(A / v, 4) for nm, v in OUT["median_fold_s"].items()}
    OUT["trunk_delta_s"] = round(OUT["median_trunk_s"]["A"] - OUT["median_trunk_s"]["B"], 4)
    OUT["sampler_delta_s"] = round(OUT["median_sampler_s"]["A"] - OUT["median_sampler_s"]["B"], 4)
    OUT["per_rep_delta_s"] = [
        round(a["fold_s"] - b["fold_s"], 4)
        for a, b in zip([r for r in warm if r["arm"] == "A"],
                        [r for r in warm if r["arm"] == "B"])]
    OUT["cif_sha16"] = {nm: sorted({r["cif_sha16"] for r in warm if r["arm"] == nm})
                        for nm in ARMS}
    OUT["plddt"] = {nm: sorted({r["plddt"] for r in warm if r["arm"] == nm}) for nm in ARMS}

    # Structure, in Angstrom, against the 0.60 A kill bar.
    #
    # PER PSEUDO-DOMAIN, not whole-structure. cdk2x2_512 is CDK2 followed by its own residues
    # 1-214 with no interface between the copies, so the hinge between them saturates any
    # whole-molecule RMSD for a reassociation that moved no atom inside either domain.
    # `perf/k10_anchor/FINDINGS.md` measures the upstream fp32 reference's own whole-structure
    # column swinging 1.9 - 11.9 A against itself at a different seed, and this harness read
    # 11.6870 A on 2026-09-19 and called a lever failed that scores 0.30 A per domain. The
    # whole-structure column is kept below, labelled as the hinge, for contrast only.
    #
    # The floor is measured here, not quoted: the 1.84 A constant this campaign scaled against
    # came from one seed pair on one stack and is retracted campaign-wide
    # (`perf/c12_orchestrator/landing/LANDING.md`). A/A is the arithmetic floor; the seed floor
    # belongs to whatever run varies the seed (`perf/c14_opm_layout/seed_spread.py`).
    sys.path.insert(0, str(REPO / "perf" / "b2z2_fusebias"))
    import score as _S
    pick = {nm: [r["cif_path"] for r in warm if r["arm"] == nm][0] for nm in ARMS}
    atoms = {nm: read_atoms(Path(p)) for nm, p in pick.items()}
    assert atoms["A"][0] == atoms["B"][0] == atoms["A2"][0], \
        "atom identity differs between arms -- not comparable atom-for-atom"
    S512 = {nm: _S.load(Path(p), 298) for nm, p in pick.items()}

    def _read(arm, ref):
        d = _S.pair(S512[arm], S512[ref], 298)
        return {"worst_domain_all_atom_A": round(max(d["domain1_all_atom_A"],
                                                     d["domain2_all_atom_A"]), 6),
                "hinge_free_all_atom_A": d["hinge_free_all_atom_A"],
                "lddt_ca": d["lddt_ca"],
                "hinge_deg_not_a_bar": d["hinge_deg"],
                "whole_all_atom_A_is_the_hinge": d["whole_all_atom_A"]}

    OUT["rmsd_A"] = {
        "n_atoms": len(atoms["A"][1]),
        "metric": "per-pseudo-domain all-atom Kabsch, split at label_seq_id 298",
        "A_vs_A2_arithmetic_floor": _read("A2", "A"),
        "A_vs_B": _read("B", "A"),
        "kill_bar": 0.60,
    }
    dump()
    print(json.dumps({k: OUT[k] for k in (
        "median_fold_s", "AA_floor_pct", "delta_s", "x_vs_A", "trunk_delta_s",
        "sampler_delta_s", "per_rep_delta_s", "rmsd_A", "clock", "plddt")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
