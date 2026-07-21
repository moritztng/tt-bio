#!/usr/bin/env python3
"""Harvest committed reference-fixture caches for the pharma parity benchmark.

The expensive reference legs of the parity benchmark (Protenix-v2 / Boltz-2 /
OpenDDE official CPU implementations) take minutes-to-hours per seed. For a
fixed (reference implementation + version, target, seed, settings) the reference
output is reproducible, so each leg is a ONE-TIME cost whose output is a durable
golden fixture until the reference version or settings change.

This script copies already-produced reference outputs (a real reference run's
harness-format `results.json` + `structures/<id>.cif`, plus the exact MSA where
the model uses one) into the committed fixture tree and writes the provenance
metadata that makes the fixture reproducible and machine-checkable:

  docs/implementation-parity-data/ref-fixtures/<model>/<target>/<settings-tag>/
      meta.json          reference impl + version + commit, exact command, settings, date
      msa.a3m            the exact MSA fed to the reference (only when the model uses one)
      seed<N>/
          results.json
          structures/<id>.cif
          meta.json      seed, source path harvested from, selected sample, conf values

Every fixture must come from a REAL reference run. This script never generates
structures; it only copies + records provenance. If a harvested run's provenance
cannot be established, re-run the reference leg instead of trusting it.

The fixture dirs are in the same harness format `scripts/pharma_parity.py
structures --ref-dirs` already consumes, so the release gate points
`--ref-dirs` (or `--ref-fixtures <model>/<target>/<tag>`) straight at the
committed `seed<N>/` dirs and skips the reference compute entirely.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "docs" / "implementation-parity-data" / "ref-fixtures"


@dataclass
class SeedSpec:
    seed: int
    src_dir: str          # harness-format dir on disk: results.json + structures/<id>.cif
    target_id: str        # the structure id inside structures/ (e.g. "prot", "trpcage")


@dataclass
class FixtureSpec:
    model: str
    target: str
    settings_tag: str
    reference_impl: str
    reference_version: str
    reference_commit: str
    command: str                 # exact (or reconstructed) command that produced the reference output
    settings: dict
    seeds: list
    msa_source: str = ""         # path to the exact MSA used; "" when the model uses no MSA
    msa_note: str = ""
    provenance_note: str = ""
    date: str = field(default_factory=lambda: date.today().isoformat())


SPECS = [
    FixtureSpec(
        model="protenix-v2",
        target="prot",
        settings_tag="msa-server_200step_5sample_10cycle_bf16",
        reference_impl="official ByteDance Protenix (torch, CPU)",
        reference_version="protenix 2.0.0 (model protenix-v2, 464M params)",
        reference_commit="bytedance/Protenix c3bfc365b3e1341a11935eddfe7bfdc308092147",
        command=(
            "refenv312/bin/python protenix_ref_predict.py <seed> <out_dir>  "
            "(calls runner.batch_inference.inference_jsons: use_msa=True(server), "
            "seeds=[<seed>], n_cycle=10, n_step=200, n_sample=5, dtype=bf16, "
            "model_name=protenix-v2, trimul_kernel=torch, triatt_kernel=torch, "
            "use_template=False; CUDA FusedLayerNorm stubbed by torch LayerNorm, "
            "triangle kernels forced to torch so it runs CPU-only)"
        ),
        settings={
            "use_msa": True, "msa_source": "https://protenix-server.com/api/msa",
            "recycling_cycles": 10, "diffusion_steps": 200, "diffusion_samples": 5,
            "selection": "confidence-selected best-of-5 by ranking_score",
            "dtype": "bf16", "trimul_kernel": "torch", "triatt_kernel": "torch",
            "target": "examples/prot.yaml (PDB 7ROA, 117 res, 900 atoms)",
        },
        seeds=[
            SeedSpec(0, "/home/ttuser/pharma_protenix_run/ref_seed0", "prot"),
            SeedSpec(1, "/home/ttuser/pharma_protenix_run/ref_seed1", "prot"),
            SeedSpec(2, "/home/ttuser/pharma_seedbump2/protenix/ref_seed2", "prot"),
            SeedSpec(3, "/home/ttuser/pharma_seedbump2/protenix/ref_seed3", "prot"),
            SeedSpec(4, "/home/ttuser/pharma_seedbump2/protenix/ref_seed4", "prot"),
        ],
        msa_source="/home/ttuser/pharma_protenix_run/ref_seed0/raw/prot/msa/0.a3m",
        msa_note=(
            "Protenix-server.com MSA, identical for seed0 and seed1 "
            "(diff of ref_seed{0,1}/raw/prot/msa/0.a3m is empty). 166 a3m entries; "
            "the reference trims to N_msa=157 for the forward pass."
        ),
        provenance_note=(
            "Harvested from the 2026-07-13 qb2 reference run (REF_PREDICT_DONE markers "
            "at ref_seed{0,1}). Per-seed ptm in results.json matches "
            "docs/implementation-parity-data/protenix-v2.json (seed0 ptm 0.91748, "
            "seed1 ptm 0.82158). Checkpoint /home/ttuser/checkpoint/protenix-v2.pt."
        ),
    ),
    FixtureSpec(
        model="protenix-v2",
        target="ubq",
        settings_tag="msa-server_200step_5sample_10cycle_bf16",
        reference_impl="official ByteDance Protenix (torch, CPU)",
        reference_version="protenix 2.0.0 (model protenix-v2, 464M params)",
        reference_commit="bytedance/Protenix c3bfc365b3e1341a11935eddfe7bfdc308092147",
        command=(
            "refenv312/bin/python protenix_ref_predict_ubq.py <seed> <out_dir>  "
            "(calls runner.batch_inference.inference_jsons: use_msa=True(server), "
            "seeds=[<seed>], n_cycle=10, n_step=200, n_sample=5, dtype=bf16, "
            "model_name=protenix-v2, trimul_kernel=torch, triatt_kernel=torch, "
            "use_template=False; CUDA FusedLayerNorm stubbed by torch LayerNorm, "
            "triangle kernels forced to torch so it runs CPU-only; "
            "json prot_ubq.json names the target ubq, sequence = human ubiquitin PDB 1UBQ 76 res)"
        ),
        settings={
            "use_msa": True, "msa_source": "https://protenix-server.com/api/msa",
            "recycling_cycles": 10, "diffusion_steps": 200, "diffusion_samples": 5,
            "selection": "confidence-selected best-of-5 by ranking_score",
            "dtype": "bf16", "trimul_kernel": "torch", "triatt_kernel": "torch",
            "target": "examples/ubq.yaml (PDB 1UBQ, human ubiquitin, 76 res, 602 atoms)",
            "rationale": ("second Protenix-v2 structure target: different length/fold than the "
                           "7ROA leg (L76 vs L117, ubiquitin alpha-beta grasp vs EntV136), "
                           "customer-relevant (ubiquitin-proteasome oncology pathway), and the "
                           "same target as the Boltz-2 ubiquitin leg for cross-model comparability. "
                           "Same production settings as the 7ROA protenix leg (MSA server, "
                           "n_cycle=10, n_step=200, n_sample=5, bf16) so the two protenix legs "
                           "differ only in target, not methodology."),
        },
        seeds=[
            SeedSpec(0, "/home/ttuser/pharma_protenix_run/ref_ubq_seed0", "ubq"),
            SeedSpec(1, "/home/ttuser/pharma_protenix_run/ref_ubq_seed1", "ubq"),
            SeedSpec(2, "/home/ttuser/pharma_protenix_ubq5_run/ref_seed2/boltz_results_ubq", "ubq"),
            SeedSpec(3, "/home/ttuser/pharma_protenix_ubq5_run/ref_seed3/boltz_results_ubq", "ubq"),
            SeedSpec(4, "/home/ttuser/pharma_protenix_ubq5_run/ref_seed4/boltz_results_ubq", "ubq"),
        ],
        msa_source="/home/ttuser/pharma_protenix_run/ref_ubq_seed0/raw/ubq/msa/0.a3m",
        msa_note=(
            "Protenix-server.com MSA, identical across all 5 reference seeds "
            "(diff of ref_ubq_seed{0,1}/raw/ubq/msa/0.a3m is empty; seeds 2-4 reuse the same "
            "committed msa.a3m). 20826 a3m entries; ubiquitin is deeply aligned in sequence "
            "databases. The device folds the SAME MSA (staged into dev_ubq_msa/<seq_hash>.a3m, "
            "seq_hash=233b4b0b8c461609) so X measures pure port fidelity with input MSA held "
            "identical."
        ),
        provenance_note=(
            "Seeds 0-1 harvested from a FRESH 2026-07-18 qb2 reference run (REF_PREDICT_DONE "
            "markers at ref_ubq_seed{0,1}, mtime 2026-07-18 20:45/20:49 UTC). Seeds 2-4 harvested "
            "from a FRESH 2026-07-19 qb1 reference run (pharma_protenix_ubq5_run/ref_seed{2,3,4}, "
            "protenix_ref_predict.py in protenix_ref_venv, same pinned protenix 2.0.0 / commit "
            "c3bfc365b3e1341a11935eddfe7bfdc308092147 and same n_cycle=10 / n_step=200 / n_sample=5 "
            "/ bf16 settings as seeds 0-1). Per-seed model-forward time 132.08s (seed0, cold) / "
            "162.50s (seed1) from ref_ubq_seed{0,1}.log -- NOT the ~3.5h/seed a stale memory note "
            "claimed (that note misread the 7ROA logs; the real on-disk cost is ~2.5 min/seed for "
            "ubiquitin, ~10-23 min/seed for 7ROA). Per-seed ptm 0.93154 / 0.93144 (seeds 0,1, both "
            "confidence-selected sample 0). 5 reference + 5 device seeds: X = 2.09 +- 0.40 A Kabsch "
            "CA-RMSD (n=25), within floor max(R, D) (R over 10 ref-seed pairs, D over 10 dev-seed "
            "pairs) -> PASS, recorded in docs/implementation-parity-data/protenix-v2-ubiquitin.json. "
            "Checkpoint /home/ttuser/protenix_ckpt/protenix-v2.pt."
        ),
    ),
    FixtureSpec(
        model="opendde",
        target="prot",
        settings_tag="nomsa_10cycle_200step_1sample_fp32_prod",
        reference_impl="official Aureka Research OpenDDE (torch, CPU)",
        reference_version="opendde_v1 (655.79M params), dtype fp32",
        reference_commit="aurekaresearch/OpenDDE a0d5134d88f85d5c6a94629d01252251930fe5f8",
        command=(
            "python -c 'from runner.batch_inference import opendde_cli; opendde_cli()' pred "
            "-i prot_nomsa.json -o ref_prod -s 0 1 2 -c 10 -p 200 -e 1 --use_msa false  "
            "&& scripts/opendde_ref_to_harness.py ref_prod prot <seed> ref_harness_seed<seed>"
        ),
        settings={
            "use_msa": False, "recycling_cycles": 10, "diffusion_steps": 200,
            "diffusion_samples": 1, "seeds": [0, 1, 2], "dtype": "fp32",
            "trimul_kernel": "torch", "triatt_kernel": "torch",
            "target": "PDB 7ROA (117 res, 900 atoms, single-sequence no-MSA)",
            "rationale": "sample=1 isolates convergence (cycles/steps) from best-of-N selection",
        },
        seeds=[
            SeedSpec(0, "/home/ttuser/opendde_parity_prod/ref_harness_seed0", "prot"),
            SeedSpec(1, "/home/ttuser/opendde_parity_prod/ref_harness_seed1", "prot"),
            SeedSpec(2, "/home/ttuser/opendde_parity_prod/ref_harness_seed2", "prot"),
        ],
        provenance_note=(
            "Harvested from the 2026-07-13 qb2 production reference run (REF_PROD_DONE at "
            "opendde_parity_prod). Per-seed model-forward time from ref_prod.log: "
            "seed0 1376.14s (cold), seed1 235.58s, seed2 236.11s. Matches "
            "docs/implementation-parity-data/opendde-prod-leg.json (X=5.68 A within floor 8.06 A)."
        ),
    ),
    FixtureSpec(
        model="opendde",
        target="prot",
        settings_tag="nomsa_4cycle_20step_1sample_fp32_reduced",
        reference_impl="official Aureka Research OpenDDE (torch, CPU)",
        reference_version="opendde_v1 (655.79M params), dtype fp32",
        reference_commit="aurekaresearch/OpenDDE a0d5134d88f85d5c6a94629d01252251930fe5f8",
        command=(
            "python -c 'from runner.batch_inference import opendde_cli; opendde_cli()' pred "
            "-i prot.json -o ref2_prot_s<seed> -s <seed> -c 4 -p 20 -e 1 --use_msa false  "
            "&& scripts/opendde_ref_to_harness.py ref2_prot_s<seed> prot <seed> refh2_prot_s<seed>"
        ),
        settings={
            "use_msa": False, "recycling_cycles": 4, "diffusion_steps": 20,
            "diffusion_samples": 1, "seeds": [0, 1, 2], "dtype": "fp32",
            "trimul_kernel": "torch", "triatt_kernel": "torch",
            "target": "PDB 7ROA (117 res, 900 atoms, single-sequence no-MSA)",
        },
        seeds=[
            SeedSpec(0, "/home/ttuser/pharma_opendde_run/refh2_prot_s0", "prot"),
            SeedSpec(1, "/home/ttuser/pharma_opendde_run/refh2_prot_s1", "prot"),
            SeedSpec(2, "/home/ttuser/pharma_opendde_run/refh2_prot_s2", "prot"),
        ],
        provenance_note=(
            "Harvested from the 2026-07-12 qb2 reduced-settings reference run "
            "(pharma_opendde_run/ref2_driver.sh, 4c/20s/sample=1). Matches the reduced prot "
            "leg in docs/implementation-parity-data/opendde.json (X=7.65 A, 2.85x over floor -- "
            "the tight-device-floor artifact resolved by the production leg above)."
        ),
    ),
    FixtureSpec(
        model="boltz2",
        target="prot",
        settings_tag="msa-colabfold_200step_1sample_3recycle_bf16",
        reference_impl="official Boltz-2 (torch + pytorch-lightning, CPU)",
        reference_version="boltz 2.2.1",
        reference_commit="boltz 2.2.1 (pip-installed in boltz_ref_venv; upstream jwohlwend/boltz)",
        command=(
            "boltz_ref_venv/bin/boltz predict prot.yaml --out_dir <out> --seed <N> "
            "--recycling_steps 3 --diffusion_steps 200 --diffusion_samples 1  "
            "(prot.yaml sets msa: prot_msa.a3m, the colabfold server MSA; "
            "command reconstructed from the committed settings in boltz2.json + ref_prot_s0.log; "
            "the exact argv was not logged, but the output confidence values verify the run)"
        ),
        settings={
            "use_msa": True, "msa_source": "colabfold server (api.colabfold.com), 93 sequences",
            "recycling_steps": 3, "diffusion_steps": 200, "diffusion_samples": 1,
            "seeds": [0, 1, 2, 3, 4], "dtype": "bf16 (pytorch-lightning AMP)",
            "target": "examples/prot.yaml (PDB 7ROA, 117 res)",
        },
        seeds=[
            SeedSpec(0, "/home/ttuser/pharma_boltz2_msa_run/ref_harness_s0", "prot"),
            SeedSpec(1, "/home/ttuser/pharma_boltz2_msa_run/ref_harness_s1", "prot"),
            SeedSpec(2, "/home/ttuser/pharma_seedbump2/boltz2_prot_msa/ref_harness_s2", "prot"),
            SeedSpec(3, "/home/ttuser/pharma_seedbump2/boltz2_prot_msa/ref_harness_s3", "prot"),
            SeedSpec(4, "/home/ttuser/pharma_seedbump2/boltz2_prot_msa/ref_harness_s4", "prot"),
        ],
        msa_source="/home/ttuser/pharma_boltz2_msa_run/prot_msa_clean.a3m",
        msa_note=(
            "ColabFold server MSA (93 sequences), identical on device and reference "
            "(header-set diff = 0, per boltz2.json). prot_msa_clean.a3m is the deduped form "
            "fed to the reference; the raw colabfold dump is prot_msa.a3m."
        ),
        provenance_note=(
            "Harvested from the 2026-07-13 qb2 reference run (pharma_boltz2_msa_run/ref_harness_s{0,1}, "
            "mtime 2026-07-13 05:31). Per-seed confidence in results.json matches "
            "docs/implementation-parity-data/boltz2.json prot_msa leg (ref confidence_score 0.8916)."
        ),
    ),
    FixtureSpec(
        model="opendde",
        target="trpcage",
        settings_tag="nomsa_4cycle_20step_1sample_fp32_reduced",
        reference_impl="official Aureka Research OpenDDE (torch, CPU)",
        reference_version="opendde_v1 (655.79M params), dtype fp32",
        reference_commit="aurekaresearch/OpenDDE a0d5134d88f85d5c6a94629d01252251930fe5f8",
        command=(
            "cd /home/ttuser/opendde-src && opendde-ref-venv/bin/python -c "
            "'from runner.batch_inference import opendde_cli; opendde_cli()' pred "
            "-i trpcage.json -o opendde_trpcage_s<seed> -s <seed> -c 4 -p 20 -e 1 --use_msa false "
            "&& opendde-ref-venv/bin/python scripts/opendde_ref_to_harness.py "
            "opendde_trpcage_s<seed> trpcage <seed> opendde_harness_trpcage_s<seed>  "
            "(trpcage.json: NLYIQWLKDGGPSSGRPPPS, single-sequence no-MSA; checkpoint "
            "/home/ttuser/.cache/opendde/checkpoint/opendde.pt; CUDA_VISIBLE_DEVICES='')"
        ),
        settings={
            "use_msa": False, "recycling_cycles": 4, "diffusion_steps": 20,
            "diffusion_samples": 1, "seeds": [0, 1, 2], "dtype": "fp32",
            "trimul_kernel": "torch", "triatt_kernel": "torch",
            "target": "trp-cage (PDB 1L2Y, 20 res, 154 atoms, single-sequence no-MSA)",
            "rationale": "sample=1 isolates convergence (cycles/steps) from best-of-N selection; "
                         "matches the reduced-settings prot leg so the two trp-cage/prot reads are "
                         "directly comparable at the same compute budget.",
        },
        seeds=[
            SeedSpec(0, "/home/ttuser/pharma_ref_fixture_run/opendde_harness_trpcage_s0", "trpcage"),
            SeedSpec(1, "/home/ttuser/pharma_ref_fixture_run/opendde_harness_trpcage_s1", "trpcage"),
            SeedSpec(2, "/home/ttuser/pharma_ref_fixture_run/opendde_harness_trpcage_s2", "trpcage"),
        ],
        provenance_note=(
            "Harvested from a FRESH 2026-07-13 qb2 reference run (pharma_ref_fixture_run/"
            "opendde_trpcage_s{0,1,2}, mtime 2026-07-13 15:47-15:49), generated for this fixture "
            "rather than copied from a prior raw output. Per-seed model-forward ~5s (warm); "
            "ranking_score 0.0911 on all three seeds. The fresh reference-vs-reference floor "
            "R=0.31 A (mean of 3 seed pairs: 0.41/0.38/0.15) reproduces the published R=0.31 "
            "in docs/implementation-parity.md within noise."
        ),
    ),
    FixtureSpec(
        model="boltz2",
        target="trpcage",
        settings_tag="nomsa_200step_1sample_3recycle_bf16",
        reference_impl="official Boltz-2 (torch + pytorch-lightning, CPU)",
        reference_version="boltz 2.2.1",
        reference_commit="boltz 2.2.1 (pip-installed in boltz_ref_venv; upstream jwohlwend/boltz)",
        command=(
            "boltz_ref_venv/bin/boltz predict examples/trpcage_no_msa.yaml "
            "--out_dir <out> --seed <N> --recycling_steps 3 --sampling_steps 200 "
            "--diffusion_samples 1 --accelerator cpu  "
            "&& boltz_ref_venv/bin/python scripts/boltz2_ref_layout.py <out>/boltz_results_trpcage_no_msa "
            "<harness_dir>  (trpcage_no_msa.yaml sets msa: empty so boltz runs single-sequence; "
            "the no-MSA flag is --sampling_steps, the diffusion-step count; --diffusion_steps "
            "is not a boltz 2.2.1 option)"
        ),
        settings={
            "use_msa": False, "recycling_steps": 3, "sampling_steps": 200,
            "diffusion_samples": 1, "seeds": [0, 1, 2, 3, 4], "dtype": "bf16 (pytorch-lightning AMP)",
            "target": "trp-cage (examples/trpcage_no_msa.yaml, PDB 1L2Y, 20 res, msa: empty)",
        },
        seeds=[
            SeedSpec(0, "/home/ttuser/pharma_ref_fixture_run/boltz_harness_trpcage_s0", "trpcage_no_msa"),
            SeedSpec(1, "/home/ttuser/pharma_ref_fixture_run/boltz_harness_trpcage_s1", "trpcage_no_msa"),
            SeedSpec(2, "/home/ttuser/pharma_seedbump2/boltz2_trpcage/ref_harness_s2", "trpcage_no_msa"),
            SeedSpec(3, "/home/ttuser/pharma_seedbump2/boltz2_trpcage/ref_harness_s3", "trpcage_no_msa"),
            SeedSpec(4, "/home/ttuser/pharma_seedbump2/boltz2_trpcage/ref_harness_s4", "trpcage_no_msa"),
        ],
        provenance_note=(
            "Harvested from a FRESH 2026-07-13 qb2 reference run (pharma_ref_fixture_run/"
            "boltz_trpcage_s{0,1}, mtime 2026-07-13 15:55), generated for this fixture. "
            "Boltz-2 CPU is bit-exact deterministic (a repeat seed-0 run gave RMSD=0.000 and "
            "identical confidence). The fresh reference-vs-reference floor R=0.81 A (1 seed pair) "
            "reproduces the published R=0.79 in docs/implementation-parity.md within noise. "
            "Per-seed confidence_score 0.854/0.847, ptm 0.445/0.420 (the 0.85 in the prior note was the confidence_score, not ptm). Seeds 2,3,4 were generated 2026-07-21 on qb2 CPU (pinned boltz 2.2.1, same 3 recycle / 200 sampling steps / 1 sample settings, msa: empty) for the 2+2 -> 5+5 pharma-meeting hardening pass; the 5+5 read is X 0.66 +- 0.22 A vs floor max(R 0.60, D 0.57) = 0.60 A (X/floor 1.10, within the floor+std band on CA-RMSD; 1-lDDT exceeds at 1.93), reproducing the 2+2 X 0.60 within noise."
        ),
    ),
    FixtureSpec(
        model="boltz2",
        target="prot",
        settings_tag="nomsa_200step_1sample_3recycle_bf16",
        reference_impl="official Boltz-2 (torch + pytorch-lightning, CPU)",
        reference_version="boltz 2.2.1",
        reference_commit="boltz 2.2.1 (pip-installed in boltz_ref_venv; upstream jwohlwend/boltz)",
        command=(
            "boltz_ref_venv/bin/boltz predict examples/prot_no_msa.yaml "
            "--out_dir <out> --seed <N> --recycling_steps 3 --sampling_steps 200 "
            "--diffusion_samples 1 --accelerator cpu  "
            "&& boltz_ref_venv/bin/python scripts/boltz2_ref_layout.py <out>/boltz_results_prot_no_msa "
            "<harness_dir>  (prot_no_msa.yaml sets msa: empty so boltz runs single-sequence)"
        ),
        settings={
            "use_msa": False, "recycling_steps": 3, "sampling_steps": 200,
            "diffusion_samples": 1, "seeds": [0, 1, 2, 3, 4], "dtype": "bf16 (pytorch-lightning AMP)",
            "target": "prot/7ROA (examples/prot_no_msa.yaml, 117 res, 899 atoms, msa: empty)",
        },
        seeds=[
            SeedSpec(0, "/home/ttuser/pharma_ref_fixture_run/boltz_harness_prot_no_msa_s0", "prot_no_msa"),
            SeedSpec(1, "/home/ttuser/pharma_ref_fixture_run/boltz_harness_prot_no_msa_s1", "prot_no_msa"),
            SeedSpec(2, "/home/ttuser/pharma_seedbump_prot_nomsa/ref_harness_s2", "prot_no_msa"),
            SeedSpec(3, "/home/ttuser/pharma_seedbump_prot_nomsa/ref_harness_s3", "prot_no_msa"),
            SeedSpec(4, "/home/ttuser/pharma_seedbump_prot_nomsa/ref_harness_s4", "prot_no_msa"),
        ],
        provenance_note=(
            "Seeds 0-1 harvested from a FRESH 2026-07-13 qb2 reference run "
            "(pharma_ref_fixture_run/boltz_prot_no_msa_s{0,1}, mtime 2026-07-13 15:46/15:53). "
            "Seeds 2-4 harvested from a FRESH 2026-07-21 qb2 reference run "
            "(pharma_seedbump_prot_nomsa/ref_harness_s{2,3,4}), same pinned boltz 2.2.1 and same "
            "3 recycle / 200 sampling-step / 1 sample / no-MSA settings as seeds 0-1. Boltz-2 CPU "
            "is bit-exact deterministic (a repeat seed-0 run gave RMSD=0.000 and identical "
            "confidence). DISCREPANCY (historical, retained for honesty): the 2-seed "
            "reference-vs-reference floor R=6.94 A (1 seed pair, deterministic) did NOT reproduce "
            "the previously-published R=3.37; the 3.37's source run was not on disk and was not "
            "reproducible from the documented settings on the pinned boltz 2.2.1 (the only "
            "on-disk prot no-MSA reference runs at the time used 2 recycle / 20 steps and gave "
            "R=2.60). The trp-cage no-MSA leg at the same 3/200/1 settings DOES reproduce "
            "(R=0.81 vs 0.79), so the settings interpretation is correct; the prot 3.37 was the "
            "anomaly and was withdrawn. 5+5 hardening (2026-07-21): with 5 reference + 5 device "
            "seeds R and D are each 10 pairwise distances (a real distribution) rather than 1; "
            "CA-RMSD X=4.21+-1.59 A vs floor max(R 4.98, D 3.34)=4.98 A (X/floor 0.84, within floor "
            "on RMSD, 1-PCC, 1-TM and 1-lDDT). The 5+5 read reproduces the 2+2 verdict within noise "
            "(X 4.21 vs 4.83 A; the floor shifts inward R 6.94->4.98 as the single-pair extreme "
            "regresses to the 10-pair mean, D 2.93->3.34, X stays inside it). Per-seed reference "
            "confidence_score 0.525/0.603/0.625/0.620/0.586, ptm "
            "0.439/0.540/0.585/0.583/0.512; device confidence_score "
            "0.686/0.686/0.684/0.608/0.642, ptm 0.664/0.664/0.692/0.548/0.628."
        ),
    ),
    FixtureSpec(
        model="boltz2",
        target="hsa",
        settings_tag="nomsa_200step_1sample_3recycle_bf16",
        reference_impl="official Boltz-2 (torch + pytorch-lightning, GPU via vast.ai RTX3090)",
        reference_version="boltz 2.2.1",
        reference_commit="boltz 2.2.1 (pip-installed in /root/boltz_venv on vast.ai; upstream jwohlwend/boltz)",
        command=(
            "boltz predict examples/hsa_no_msa.yaml --out_dir <out> --seed <N> "
            "--recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 "
            "--accelerator gpu --no_kernels --override  "
            "&& python scripts/boltz2_ref_layout.py <out>/boltz_results_hsa_no_msa <harness_dir>  "
            "(hsa_no_msa.yaml sets msa: empty so boltz runs single-sequence; --no_kernels forces the "
            "torch einsum triangle path, matching the qb1 CPU reference kernel; GPU execution only)"
        ),
        settings={
            "use_msa": False, "recycling_steps": 3, "sampling_steps": 200,
            "diffusion_samples": 1, "seeds": [0, 1, 2, 3, 4], "dtype": "bf16 (pytorch-lightning AMP)",
            "target": "hsa (examples/hsa_no_msa.yaml, PDB 1AO6, 585 res, 3-domain, msa: empty)",
            "rationale": "large pharma-realistic target (L585, multi-domain human serum albumin, "
                         "classic drug-binding carrier) extending Boltz-2 past L117 to the L300-800 "
                         "regime; same no-MSA single-sequence methodology as the trpcage/"
                         "prot legs so the no-MSA length ladder reads L20/L117/L585 are directly "
                         "comparable. Reference generated on vast.ai GPU (CPU infeasible at L585) "
                         "with --no_kernels (torch einsum, identical kernel to the qb1 CPU ref).",
        },
        seeds=[
            SeedSpec(0, "/home/ttuser/hsa_ref_boltz/ref_boltz_harness_s0", "hsa_no_msa"),
            SeedSpec(1, "/home/ttuser/hsa_ref_boltz/ref_boltz_harness_s1", "hsa_no_msa"),
            SeedSpec(2, "/home/ttuser/hsa_ref_boltz/ref_boltz_harness_s2", "hsa_no_msa"),
            SeedSpec(3, "/home/ttuser/hsa_ref_boltz/ref_boltz_harness_s3", "hsa_no_msa"),
            SeedSpec(4, "/home/ttuser/hsa_ref_boltz/ref_boltz_harness_s4", "hsa_no_msa"),
        ],
        provenance_note=(
            "Harvested from a 2026-07-19 vast.ai RTX3090 reference run (ref_boltz_seed{0-4}, "
            "BOLTZ_SEED{0-4}_DONE 18:58-19:06 UTC). Pinned boltz 2.2.1, torch 2.5.1+cu124, "
            "--accelerator gpu --no_kernels (torch einsum triangle path, the SAME kernel as the "
            "qb1 CPU reference for the other boltz2 legs -- only the execution device differs). "
            "Per-seed wall ~1.8 min on RTX3090 (vs multi-hour on CPU, which is why vast.ai GPU was "
            "used). Boltz-2 is deterministic per seed; GPU-vs-CPU of the same torch-einsum path "
            "differs only by floating-point non-determinism, within the R/D/X tolerance."
        ),
    ),
    FixtureSpec(
        model="protenix-v2",
        target="hsa",
        settings_tag="msa-server_200step_5sample_10cycle_bf16",
        reference_impl="official ByteDance Protenix (torch, GPU via vast.ai RTX3090)",
        reference_version="protenix 2.0.0 (model protenix-v2, 464M params)",
        reference_commit="bytedance/Protenix c3bfc365b3e1341a11935eddfe7bfdc308092147",
        command=(
            "protenix_venv/bin/python protenix_ref_predict_hsa.py <seed> <out_dir>  "
            "(calls runner.batch_inference.inference_jsons: use_msa=True(server), "
            "seeds=[<seed>], n_cycle=10, n_step=200, n_sample=5, dtype=bf16, "
            "model_name=protenix-v2, trimul_kernel=torch, triatt_kernel=torch, "
            "use_template=False; CUDA FusedLayerNorm stubbed by torch LayerNorm, "
            "triangle kernels forced to torch (the SAME torch kernels as the qb2 CPU "
            "reference for the other protenix legs -- only the execution device differs "
            "GPU vs CPU); json prot_hsa.json names the target hsa, sequence = human serum "
            "albumin PDB 1AO6 585 res)"
        ),
        settings={
            "use_msa": True, "msa_source": "https://protenix-server.com/api/msa",
            "recycling_cycles": 10, "diffusion_steps": 200, "diffusion_samples": 5,
            "selection": "confidence-selected best-of-5 by ranking_score",
            "dtype": "bf16", "trimul_kernel": "torch", "triatt_kernel": "torch",
            "target": "examples/hsa.yaml (PDB 1AO6, human serum albumin, 585 res, 3-domain)",
            "rationale": ("large pharma-realistic Protenix-v2 target (L585, multi-domain human "
                           "serum albumin, classic drug-binding carrier) extending Protenix-v2 "
                           "past L117 to the L300-800 regime, and the SAME target as the Boltz-2 "
                           "HSA leg for cross-model comparability. Same production settings as the "
                           "7ROA/ubiquitin protenix legs (MSA server, n_cycle=10, n_step=200, "
                           "n_sample=5, bf16) so the protenix legs differ only in target, not "
                           "methodology. Reference generated on vast.ai GPU (CPU infeasible at "
                           "L585) with the same pinned protenix 2.0.0 / commit and same torch "
                           "triangle kernels as the qb2 CPU reference."),
        },
        seeds=[
            SeedSpec(0, "/home/ttuser/hsa_ref_protenix/ref_protenix_seed0", "hsa"),
            SeedSpec(1, "/home/ttuser/hsa_ref_protenix/ref_protenix_seed1", "hsa"),
            SeedSpec(2, "/home/ttuser/hsa_ref_protenix/ref_protenix_seed2", "hsa"),
            SeedSpec(3, "/home/ttuser/hsa_ref_protenix/ref_protenix_seed3", "hsa"),
            SeedSpec(4, "/home/ttuser/hsa_ref_protenix/ref_protenix_seed4", "hsa"),
        ],
        msa_source="/home/ttuser/hsa_ref_protenix/hsa_ref_msa.a3m",
        msa_note=(
            "Protenix-server.com MSA, identical across all 5 reference seeds (the ref script "
            "copies ref_protenix_seed0/raw/.../0.a3m to hsa_ref_msa.a3m). The device folds the "
            "SAME MSA (staged into the dev run) so X measures pure port fidelity with input MSA "
            "held identical."
        ),
        provenance_note=(
            "Harvested from a 2026-07-19 vast.ai RTX3090 reference run (ref_protenix_seed{0-4}). "
            "Pinned protenix 2.0.0 / commit c3bfc365b3e1341a11935eddfe7bfdc308092147, torch "
            "2.6.0+cu124, n_cycle=10 / n_step=200 / n_sample=5 / bf16, use_msa=True (protenix-"
            "server.com), trimul_kernel=torch, triatt_kernel=torch, FusedLayerNorm stubbed by "
            "torch LayerNorm -- the SAME torch kernels as the qb2 CPU reference for the other "
            "protenix legs; only the execution device differs (GPU vs CPU), so the fixture stays "
            "valid under the existing invalidation rule (same commit, same settings, same "
            "kernel). The data cache (components.cif + rdkit_mol.pkl) and the protenix-v2 "
            "checkpoint were copied from qb2 (the same cache+checkpoint the ubiquitin/7ROA protenix "
            "legs used, Jul 13 vintage) so the CCD+checkpoint version matches the existing protenix "
            "legs exactly. CPU was infeasible at L585 (multi-hour/seed), hence vast.ai GPU."
        ),
    ),
]


def _load_results(seed_dir: Path) -> list:
    return json.loads((seed_dir / "results.json").read_text())


def _selected_record(results: list, target_id: str) -> dict:
    for r in results:
        if r.get("id") == target_id or (len(results) == 1):
            return r
    return results[0] if results else {}


def harvest(spec: FixtureSpec, skip_missing: bool = False) -> None:
    base = FIXTURE_ROOT / spec.model / spec.target / spec.settings_tag
    base.mkdir(parents=True, exist_ok=True)

    for ss in spec.seeds:
        src = Path(ss.src_dir)
        cif = src / "structures" / f"{ss.target_id}.cif"
        res = src / "results.json"
        if not cif.exists() or not res.exists():
            if skip_missing and (base / f"seed{ss.seed}").exists():
                # already committed on a previous harvest (source dir may live on another
                # build host); keep the committed seed dir as-is and continue.
                print(f"skip (already committed, src not on this host): "
                      f"{spec.model}/{spec.target}/{spec.settings_tag}/seed{ss.seed} <- {src}")
                continue
            raise FileNotFoundError(
                f"reference fixture source missing for {spec.model}/{spec.target}/"
                f"{spec.settings_tag}/seed{ss.seed}: expected {cif} and {res}")
        seed_dst = base / f"seed{ss.seed}"
        (seed_dst / "structures").mkdir(parents=True, exist_ok=True)
        shutil.copy2(cif, seed_dst / "structures" / f"{ss.target_id}.cif")
        shutil.copy2(res, seed_dst / "results.json")
        rec = _selected_record(_load_results(src), ss.target_id)
        seed_meta = {
            "seed": ss.seed,
            "target_id": ss.target_id,
            "harvested_from": str(src),
            "selected_record": rec,
            "note": "real reference output copied verbatim; not regenerated or edited",
        }
        (seed_dst / "meta.json").write_text(json.dumps(seed_meta, indent=2) + "\n")

    if spec.msa_source:
        msa_src = Path(spec.msa_source)
        if msa_src.exists():
            shutil.copy2(msa_src, base / "msa.a3m")
        else:
            print(f"WARNING: msa_source not found, skipped: {msa_src}")

    settings_meta = {
        "model": spec.model,
        "target": spec.target,
        "settings_tag": spec.settings_tag,
        "reference_impl": spec.reference_impl,
        "reference_version": spec.reference_version,
        "reference_commit": spec.reference_commit,
        "command": spec.command,
        "settings": spec.settings,
        "seeds": [s.seed for s in spec.seeds],
        "msa": spec.msa_note if spec.msa_source else "none (no-MSA leg)",
        "provenance": spec.provenance_note,
        "date": spec.date,
        "invalidation_rule": (
            "Regenerate this fixture ONLY when the pinned reference_commit/version changes "
            "or the settings above change. For any other change (device seeds, device code, "
            "release tag) the fixture is reused as-is and only the device side re-runs."
        ),
    }
    (base / "meta.json").write_text(json.dumps(settings_meta, indent=2) + "\n")
    print(f"harvested {spec.model}/{spec.target}/{spec.settings_tag}: "
          f"{len(spec.seeds)} seeds -> {base}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=[],
                    help="only harvest these fixtures, given as <model>/<target> (e.g. "
                         "boltz2/ubiquitin protenix-v2/ubq). Default: harvest every spec. Useful "
                         "when only the freshly re-run legs' source dirs exist on this host "
                         "(other specs' source dirs may live on a different build host).")
    ap.add_argument("--skip-missing", action="store_true",
                    help="skip seeds whose source dir is not on this host when the seed is "
                         "already committed (use when re-harvesting a fixture whose earlier "
                         "seeds were produced on a different build host).")
    args = ap.parse_args()
    want = {(o.split("/")[0], o.split("/")[1]) for o in args.only if "/" in o}
    specs = [s for s in SPECS if not want or (s.model, s.target) in want]
    if want and not specs:
        raise SystemExit(f"--only matched no specs; known: {sorted({(s.model, s.target) for s in SPECS})}")
    for spec in specs:
        harvest(spec, skip_missing=args.skip_missing)
    print(f"\nfixture root: {FIXTURE_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
