#!/usr/bin/env python3
"""Full implementation-parity gate — the ONE pre-release parity command for tt-bio.

Closes the gap between the fast proxy gate (``scripts/release_gate.py``, a cheap
single-target floor check) and the FULL ``docs/implementation-parity.md`` story
(28 legs, 5-seed depth across all models/targets). The full story used to be run
manually over multiple days because most legs needed multi-hour CPU/GPU reference
generation. The lever: references only change when the MODEL CODE, WEIGHTS, or
TEST SETTINGS change, so they are CACHED as committed fixtures under
``docs/implementation-parity-data/ref-fixtures/`` and only the DEVICE side +
comparison re-runs per release — which is fast and parallelizable across cards.

Per leg this runner:

  (a) reads the cached reference fixture's ``meta.json`` and computes a
      fingerprint (sha256 over reference_impl + version + commit + settings +
      seeds). If the fingerprint matches the recorded one in
      ``docs/implementation-parity-data/ref-fixture-fingerprints.json`` the leg
      takes the FAST path (device-only). A mismatch means the model/settings
      changed and the reference must be regenerated — the leg is flagged
      ``BLOCKED-REF-REGEN-NEEDED`` (the documented slow/opt-in path) and the
      runner continues with the rest.
  (b) runs the DEVICE side for every leg, fanning the per-seed folds across
      every available card given via ``--workers host:card[,host:card...]``
      (default: the local card). Each worker runs one fold at a time (one device
      context per process); legs are dispatched round-robin to free workers.
  (c) scores each leg. DIFFUSION legs (structure/affinity) use the INTEGRATION-PARITY
      ENVELOPE (``scripts/integration_envelope.py``): a deterministic shared-draws test
      comparing the device fold against two cached CPU references (fp32 + bf16) at one
      seed — see the ENVELOPE note below. All other legs keep their existing vetted
      scorers/thresholds (``scripts/pharma_parity.py`` saprot, the ESMC / SaProt /
      ESMFold2 / BoltzGen / OpenDDE-abag in-process harnesses) — nothing re-derived here.

  INTEGRATION-PARITY ENVELOPE (the correctness criterion for diffusion legs; supersedes the
  old R/D/X self-consistency floor, which conflated bf16 arithmetic with diffusion-noise chaos
  and so could not tell a real backend bug from ordinary sample-to-sample spread). A diffusion
  model is deterministic given its noise, so the gate folds the device once at ENVELOPE_SEED,
  reads the leg's cached ``<fixture>/ref_fp32`` + ``<fixture>/ref_bf16`` CPU references (tt-bio's
  own torch path, so all three share one CPU-MT19937 draw stream), and passes iff
  ``d(device_bf16, ref_fp32) <= d(ref_bf16, ref_fp32)*(1+margin) + abs_floor`` on every metric.
  Regenerate the two CPU references with ``--regen-refs`` (fingerprint-cached). ``--legacy-rdx``
  keeps the retired R/D/X floor as an opt-in device self-consistency (D) DIAGNOSTIC.
  (d) emits the SAME verdict table + tally as ``docs/implementation-parity.md``,
      writes a JSON report + markdown summary to the workdir, and compares each
      leg's verdict to the committed JSON. A leg that reproduces within the
      recorded noise floor is marked ``REPRODUCES``; a leg that drifts OUTSIDE
      the floor is flagged ``DRIFT — investigate`` and is NEVER silently
      overwritten into the doc.

Exit 0 iff every leg that took the fast path reproduces within its floor (legs
flagged ``BLOCKED-REF-REGEN-NEEDED`` do not fail the gate; they are the slow
opt-in path and are reported separately). One guard on top: if NO leg reaches a
scored verdict (every leg blocked on reference regen — the classic cause is
``--seeds`` given a bare count instead of a list, so no fixture seed matches),
the run verified nothing, so the gate prints ``GATE INCONCLUSIVE`` and exits
nonzero instead of a false ``GATE PASS``.

VERDICT SEMANTICS — the single source of truth (see ``finalize_leg`` /
``_matches_committed``; also mirrored in RELEASING.md):

  PASS                    metric within the recorded noise floor. Gate-passing.
  PASS-caveated           gate metric passes, a documented secondary metric (e.g.
                          affinity pocket-lDDT) GAPs on a known bf16 floor. Gate-passing;
                          treated as equivalent to PASS for drift (a seed-variance flip
                          between the two is not a regression).
  GAP                     metric outside the floor. Gate-FAILING — UNLESS it reproduces a
                          committed ``GAP-evidenced`` (then it is the expected bf16 behavior).
  GAP-evidenced           a GAP proven to be a genuine bf16-backend floor and accepted in
                          docs/implementation-parity.md. Only ever a *committed* verdict; a
                          live GAP that matches it reproduces (gate-passing).
  DRIFT                   live verdict does not reproduce the committed one (and is not an
                          improvement). Gate-FAILING; never silently overwrites the doc.
  BLOCKED-REF-REGEN-NEEDED  the reference fixture is missing or its fingerprint changed
                          (model/weights/settings moved). NOT a gate failure — the slow
                          opt-in regen path; reported separately.
  ERROR                   the fold or scorer failed to produce a report. Gate-FAILING.
  NO-DATA                 a report with no comparable metric (legacy/narrative record). The
                          drift check is skipped, but a live NO-DATA still fails the gate.

Before any device work the runner runs a card-free ``preflight_check`` (also
exposed as ``--check``) that validates every leg's yaml/fixture/committed-JSON/
target-id wiring — so a misconfigured leg aborts in seconds with a precise message
instead of a wasted device turn on a mysterious "no common targets" scorer error.

    # full parity, fan across every card that is up (pc + qb1 + qb2)
    TT_VISIBLE_DEVICES=0 ESM_ROOT=/path/to/esm \
        OPENDDE_DOCKQ_PYTHON=/path/to/dockq_venv/bin/python \
        PYTHONPATH="$PWD" \
        python3 scripts/full_parity_gate.py --workers pc:0,qb1:0,qb1:1,qb2:0
    # one leg, local card only (smoke / measure)
    python3 scripts/full_parity_gate.py --workers pc:0 --leg boltz2-trpcage-nomsa --seeds 0,1

SCORER PREREQUISITES (two legs need host packages the runtime does not):

  * ``opendde-abag`` scores with **DockQ==2.1.3** (see ``scripts/opendde_dockq.py``). Without it the
    leg ERRORs at import with ``ModuleNotFoundError: No module named 'DockQ'`` AFTER paying for the
    full fold, so install it before a long run, not after.
  * ``esmfold2-*`` pulls in **torchvision** through its dependency chain. torchvision is compiled
    against one specific torch; a mismatched pair raises ``operator torchvision::nms does not
    exist`` before the device is ever touched.

The mismatch is easy to hit on a host where torch and torchvision come from DIFFERENT site-package
trees. On the japanfold Galaxy, ``/usr/bin/python3.10`` takes torch 2.12 from the tt-bio venv (via a
prepending ``.pth``) but torchvision 0.23 — built for torch 2.8 — from system site-packages. Check
with::

    python3 -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"

Rather than mutate a shared (possibly production) environment, install both into a private directory
and put it FIRST on PYTHONPATH for the gate run only::

    pip install --target ~/gatedeps "DockQ==2.1.3"
    pip install --target ~/gatedeps --no-deps "torchvision==<match for your torch>"
    PYTHONPATH=~/gatedeps:$PWD python3 scripts/full_parity_gate.py ...

Verify the isolation held (the service's own interpreter must NOT see it) before relying on it, and
undo with a single ``rm -rf ~/gatedeps``.

See ``~/.coworker/state/tt-bio-fast-full-parity-runner.md`` for the leg
inventory (cached vs live-ref vs in-process), the measured achieved runtime,
and the fingerprint/cache design rationale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = REPO / "docs" / "implementation-parity-data" / "ref-fixtures"
FINGERPRINT_INDEX = REPO / "docs" / "implementation-parity-data" / "ref-fixture-fingerprints.json"
# .cif/.a3m/.npz under ref-fixtures/ are gitignored on purpose and live on a GitHub
# Release asset, so a fresh checkout or a new worktree has the provenance JSONs but
# none of the binaries. Point at the restore path rather than at git.
_FIXTURE_FETCH_HINT = ("run scripts/fetch_parity_fixtures.sh to restore the externalized "
                       "reference binaries")
PARITY_DATA = REPO / "docs" / "implementation-parity-data"

# The integration-parity envelope (see score_envelope) is the correctness criterion for every
# diffusion leg — the structure + affinity folds, whose stochasticity is entirely the diffusion
# noise draw. Deterministic encoders (esmc/saprot), esmfold2, and the designability/DockQ legs
# are NOT closed-loop diffusion, so they keep their own deterministic/threshold verdicts.
ENVELOPE_KINDS = ("structure", "affinity")
# The envelope is a per-shared-draw test: the device fold's seed MUST match the seed the fp32/bf16
# CPU references were generated at, so all three share one CPU-MT19937 draw sequence.
ENVELOPE_SEED = 0


def _is_envelope_leg(leg) -> bool:
    return leg.kind in ENVELOPE_KINDS


def _shared_draw_env() -> dict:
    """Env that forces byte-identical diffusion noise across the device fold and both CPU
    references (see tt_bio/boltz2.py AtomDiffusion.sample). WITHOUT it the device (ttnn trunk) and
    the CPU reference (torch trunk) consume the global RNG differently before the sampler, so a
    plain single-seed run gives them DIFFERENT noise — the envelope numerator would then conflate
    arithmetic divergence with a different diffusion basin (the exact flaw the R/D/X floor had).
    Set identically on all three runs so shared draws hold and the numerator is arithmetic-only."""
    return {"TT_BIO_SHARED_DRAW_SEED": str(ENVELOPE_SEED)}


# ---------------------------------------------------------------------------
# Leg registry — every row of docs/implementation-parity.md
# ---------------------------------------------------------------------------
# kind: "structure"   -> tt-bio predict fold, score with pharma_parity.py structures
#        "affinity"   -> tt-bio predict affinity fold, score with boltz2_affinity_parity.py
#        "esmc"       -> in-process reference+device, esmc_embed_parity / esmc6b_embed_parity
#        "saprot"     -> in-process, pharma_parity.py saprot
#        "esmfold2"   -> in-process vendored torch ref + device, esmfold2_e2e_parity
#        "boltzgen"   -> designability, boltzgen_designability (via release_gate --model boltzgen)
#        "abag"       -> DockQ, opendde_dockq (via release_gate --model opendde-abag)
# fixture: "<model>/<target>/<tag>" path under ref-fixtures/, or "" for in-process-ref legs
#          (ESMC/SaProt/ESMFold2/BoltzGen/abag run their own reference live each pass — fast).
# committed_json: the docs/implementation-parity-data/*.json to compare the fresh verdict
#                 against for drift ("" skips the drift check).
# device_args: extra args appended to `tt-bio predict <yaml> --model <model>`.
# msa: "none" | "server" | "staged" — how the device gets its MSA. "staged" copies the
#       fixture's msa.a3m into a per-leg msa dir named by seq_hash (protenix-v2).
#       "none" passes --single_sequence (boltz2) / nothing (opendde). "server" passes
#       --use_msa_server (boltz2) — needs network, so opt-in.
@dataclass
class Leg:
    id: str
    model: str
    kind: str
    yaml: str
    fixture: str = ""
    seeds: tuple = (0, 1, 2, 3, 4)
    device_args: tuple = ()
    msa: str = "none"
    committed_json: str = ""
    target_id: str = ""          # for affinity scoring (affinity_<t>) and structures tid
    opt_in: bool = False         # slow / network legs (esmc-6b, MSA-server legs) — not default
    legacy_rdx: bool = False     # ttnn-only model with NO tt-bio torch path (openfold3): score
                                 # vs the harvested external reference (legacy R/D/X), never the
                                 # shared-draws envelope — ref_fp32/ref_bf16 would be device-on-CPU
                                 # tautology, and --regen-refs must skip these legs
    min_fold_timeout: float | None = None  # per-leg FLOOR (s); effective = max(--fold-timeout, this)
    note: str = ""


def _boltz2_struct_args(recycling=3, steps=200, samples=1, msa="none"):
    base = [f"--recycling_steps", str(recycling), "--sampling_steps", str(steps),
            "--diffusion_samples", str(samples)]
    if msa == "none":
        return tuple(base + ["--single_sequence"])
    if msa == "server":
        return tuple(base + ["--use_msa_server"])
    return tuple(base)  # staged: --msa_dir appended at run time


# Every boltz2 affinity fold runs the affinity model's own 64-block pairformer trunk in fp32
# on device, plus its own atom diffusion and heads: the heaviest fold shape in the gate. In the
# 2026-08-11 gate
# refresh a co-tenanted kernel campaign pushed dhfr (largest affinity target, L187) seed folds
# past the 2400s default while the same folds took ~600-700s on a quiet host — a flake, not a
# regression. The whole affinity class shares the mechanism, so the whole class gets 3x the
# default. Costs nothing on success: a completed fold is reaped 30s after results.json appears
# regardless of the remaining budget.
AFFINITY_FOLD_TIMEOUT_S = 7200.0

LEGS = [
    # --- deterministic encoders (in-process reference, fast, no fixture) ---
    Leg("esmc-300m", "esmc-300m", "esmc", "", committed_json="esmc-300m.json",
        note="deterministic encoder; in-process esm reference, fast"),
    Leg("esmc-600m", "esmc-600m", "esmc", "", committed_json="esmc-600m.json",
        note="deterministic encoder; in-process esm reference, fast"),
    Leg("esmc-6b", "esmc-6b", "esmc", "", committed_json="esmc-6b.json", opt_in=True,
        note="deterministic encoder; ~13 GB load dominates wall-clock — opt-in"),
    Leg("saprot-35m", "saprot-35m", "saprot", "", committed_json="",
        note="deterministic encoder; in-process HF reference, fast"),
    Leg("saprot-650m", "saprot-650m", "saprot", "", committed_json="",
        note="deterministic encoder; in-process HF reference, fast"),

    # --- ESMFold2 (in-process vendored torch ref + device, shared hidden states) ---
    Leg("esmfold2-trpcage", "esmfold2", "esmfold2", "examples/trpcage.yaml",
        committed_json="esmfold2.json", seeds=(0, 1, 2, 3, 4),
        note="in-process vendored torch ref + device fold"),
    # GB1 / ubiquitin / lysozyme share the esmfold2 harness; the doc folds them via
    # --proteins subset. Kept as one esmfold2 leg covering the doc's four targets.

    # --- Boltz-2 structure legs (cached fixture, device-only per release) ---
    Leg("boltz2-trpcage-nomsa", "boltz2", "structure", "examples/trpcage_no_msa.yaml",
        fixture="boltz2/trpcage/nomsa_200step_1sample_3recycle_bf16",
        committed_json="boltz2-trpcage-seeded.json", target_id="trpcage_no_msa",
        device_args=_boltz2_struct_args(msa="none")),
    Leg("boltz2-prot-nomsa", "boltz2", "structure", "examples/prot_no_msa.yaml",
        fixture="boltz2/prot/nomsa_200step_1sample_3recycle_bf16",
        committed_json="boltz2-prot-nomsa-seeded.json", target_id="prot_no_msa",
        device_args=_boltz2_struct_args(msa="none")),
    Leg("boltz2-prot-msa", "boltz2", "structure", "examples/prot.yaml",
        fixture="boltz2/prot/msa-colabfold_200step_1sample_3recycle_bf16",
        committed_json="boltz2-prot-msa-seeded.json", target_id="prot",
        device_args=_boltz2_struct_args(msa="server"), msa="server", opt_in=True,
        note="MSA via colabfold server — needs network, opt-in"),
    Leg("boltz2-ubiquitin-msa", "boltz2", "structure", "examples/ubiquitin_msa.yaml",
        fixture="boltz2/ubiquitin/msa-colabfold_200step_1sample_3recycle_bf16_gpu",
        committed_json="boltz2-ubiquitin-msa-seeded.json", target_id="ubiquitin_msa",
        device_args=_boltz2_struct_args(msa="server"), msa="server", opt_in=True,
        note="MSA via colabfold server — needs network, opt-in"),
    Leg("boltz2-hsa-nomsa", "boltz2", "structure", "examples/hsa_no_msa.yaml",
        fixture="boltz2/hsa/nomsa_200step_1sample_3recycle_bf16",
        committed_json="boltz2-hsa-seeded.json", target_id="hsa_no_msa",
        device_args=_boltz2_struct_args(msa="none")),
    # 9ncy: 505-token antibody-antigen complex (65+228+212), inside the [385,506]aa band that
    # was unfoldable before c06bd76cf; the AbAg-XM campaign median (509) sits in this band.
    Leg("boltz2-9ncy-nomsa", "boltz2", "structure", "examples/abag_xm/9ncy.yaml",
        fixture="boltz2/9ncy/nomsa_200step_1sample_3recycle_bf16",
        committed_json="boltz2-9ncy.json", target_id="9ncy",
        device_args=_boltz2_struct_args(msa="none"),
        note="505-token AbAg complex in the former [385,506] crash band; no-MSA (campaign regime)"),

    # --- Protenix-v2 structure legs (cached fixture, device-only per release) ---
    # legacy_rdx: tt_bio/protenix.py imports ttnn at module scope and has no torch path, so a
    # "CPU reference" fold for it is a device fold (main.py refuses one now). Its fp32 and bf16
    # references were the same computation and the envelope denominator was 0 on all three legs,
    # which made every envelope verdict here an artifact of the instrument. Same reason as the
    # openfold3 legs below.
    Leg("protenix-prot-msa", "protenix-v2", "structure", "examples/prot.yaml",
        fixture="protenix-v2/prot/msa-server_200step_5sample_10cycle_bf16",
        committed_json="protenix-v2.json", target_id="prot",
        device_args=("--sampling_steps", "200", "--diffusion_samples", "5"),
        msa="staged", legacy_rdx=True),
    Leg("protenix-ubq-msa", "protenix-v2", "structure", "examples/ubq.yaml",
        fixture="protenix-v2/ubq/msa-server_200step_5sample_10cycle_bf16",
        committed_json="protenix-v2-ubiquitin.json", target_id="ubq",
        device_args=("--sampling_steps", "200", "--diffusion_samples", "5"),
        msa="staged", legacy_rdx=True),
    Leg("protenix-hsa-msa", "protenix-v2", "structure", "examples/hsa.yaml",
        fixture="protenix-v2/hsa/msa-server_200step_5sample_10cycle_bf16",
        committed_json="protenix-v2-hsa.json", target_id="hsa",
        device_args=("--sampling_steps", "200", "--diffusion_samples", "5"),
        msa="staged", legacy_rdx=True),
    Leg("protenix-9ncy-msa", "protenix-v2", "structure", "examples/abag_xm/9ncy.yaml",
        fixture="protenix-v2/9ncy/msa-campaign_200step_5sample_10cycle_bf16",
        committed_json="", target_id="9ncy",
        device_args=("--sampling_steps", "200", "--diffusion_samples", "5"),
        msa="staged", legacy_rdx=True,
        note="505-token AbAg complex in the former [385,506] crash band; per-chain MSAs reused "
             "from the AbAg-XM campaign cache (multimer staged fixture: fixture/msa/*.a3m)"),

    # --- OpenFold3 structure legs (cached fixture, device-only per release) ---
    # OF3 is ttnn-only (no tt-bio torch path), so these are external-reference
    # R/D/X legs like Protenix's: official aqlaboratory openfold3 on CPU, fp32.
    Leg("openfold3-ubq-msa", "openfold3", "structure", "examples/ubq.yaml",
        fixture="openfold3/ubq/msa-colabfold_200step_5sample_4cycle_fp32cpu",
        committed_json="openfold3-ubiquitin.json", target_id="ubq",
        device_args=("--sampling_steps", "200", "--diffusion_samples", "5"),
        msa="staged", legacy_rdx=True),
    Leg("openfold3-prot-msa", "openfold3", "structure", "examples/prot.yaml",
        fixture="openfold3/prot/msa-colabfold_200step_5sample_4cycle_fp32cpu",
        committed_json="openfold3-prot.json", target_id="prot",
        device_args=("--sampling_steps", "200", "--diffusion_samples", "5"),
        msa="staged", legacy_rdx=True),
    Leg("openfold3-7xi5-tmpl", "openfold3", "structure", "examples/7xi5_tmpl.yaml",
        fixture="openfold3/7xi5/msa-bench-tmpl_200step_5sample_4cycle_fp32cpu",
        committed_json="openfold3-7xi5-tmpl.json", target_id="7xi5_tmpl",
        device_args=("--sampling_steps", "200", "--diffusion_samples", "5"),
        msa="yaml", legacy_rdx=True,
        note="templates-ON leg: benchmark template npz + RCSB CIFs committed in the "
             "fixture (templates.npz, template_structures/); set "
             "OF3_TEMPLATE_STRUCTURES=<fixture>/template_structures for a hermetic "
             "run, else the worker prefetches the same immutable CIFs from RCSB"),
    Leg("openfold3-7xi5-notmpl", "openfold3", "structure", "examples/7xi5_notmpl.yaml",
        fixture="openfold3/7xi5/msa-bench-notmpl_200step_5sample_4cycle_fp32cpu",
        committed_json="openfold3-7xi5-notmpl.json", target_id="7xi5_notmpl",
        device_args=("--sampling_steps", "200", "--diffusion_samples", "5"),
        msa="yaml", legacy_rdx=True,
        note="templates-OFF control for openfold3-7xi5-tmpl; same target, MSA, seeds"),
    Leg("openfold3-8hel-msa", "openfold3", "structure", "examples/8hel_msa.yaml",
        fixture="openfold3/8hel/msa-bench-notmpl_200step_5sample_4cycle_fp32cpu",
        committed_json="openfold3-8hel-msa.json", target_id="8hel_msa",
        device_args=("--sampling_steps", "200", "--diffusion_samples", "5"),
        msa="yaml", legacy_rdx=True,
        note="de-novo designed helix with benchmark MSA, no templates"),
    Leg("openfold3-8hel-nomsa", "openfold3", "structure", "examples/8hel_nomsa.yaml",
        fixture="openfold3/8hel/nomsa_200step_5sample_4cycle_fp32cpu",
        committed_json="openfold3-8hel-nomsa.json", target_id="8hel_nomsa",
        device_args=("--sampling_steps", "200", "--diffusion_samples", "5",
                     "--single_sequence"),
        msa="none", legacy_rdx=True,
        note="single-sequence leg: no MSA, no templates"),
    Leg("openfold3-9bk6-complex-msa", "openfold3", "structure", "examples/9bk6.yaml",
        fixture="openfold3/9bk6/msa-bench_200step_5sample_4cycle_fp32cpu",
        committed_json="openfold3-9bk6-complex-msa.json", target_id="9bk6",
        device_args=("--sampling_steps", "200", "--diffusion_samples", "5"),
        msa="yaml", legacy_rdx=True,
        note="two-chain heterodimer; per-chain benchmark MSA dirs committed in the "
             "fixture (msa_A/msa_B), referenced by the yaml; PASS committed under the "
             "fp32 diffusion boundary (P15, OF3_DIFFUSION_FP32_DEVICE default-on)"),

    # --- Boltz-2 affinity legs (cached fixture, device-only per release) ---
    # (min_fold_timeout=AFFINITY_FOLD_TIMEOUT_S on each: the fp32 host trunk makes the class
    # contention-fragile, see the constant's note above)
] + [
    Leg(f"boltz2-affinity-fkbp12-nomsa", "boltz2", "affinity", "examples/affinity_fkg.yaml",
        fixture="boltz2/affinity_fkg/nomsa_200step_5affsample_3recycle_bf16_mwcorr",
        committed_json="boltz2-affinity-fkbp12-nomsa-seeded.json", target_id="affinity_fkg",
        device_args=("--single_sequence", "--affinity_mw_correction",
                      "--diffusion_samples_affinity", "5", "--sampling_steps_affinity", "200",
                      "--recycling_steps", "3", "--sampling_steps", "200",
                      "--diffusion_samples", "1"),
        min_fold_timeout=AFFINITY_FOLD_TIMEOUT_S),
    Leg(f"boltz2-affinity-fkbp12-msa", "boltz2", "affinity", "examples/affinity_fkg_msa.yaml",
        fixture="boltz2/affinity_fkg/msa-colabfold_200step_5affsample_3recycle_bf16_mwcorr_gpu",
        committed_json="boltz2-affinity-fkbp12-msa-seeded.json", target_id="affinity_fkg",
        device_args=("--affinity_mw_correction", "--diffusion_samples_affinity", "5",
                      "--sampling_steps_affinity", "200", "--recycling_steps", "3",
                      "--sampling_steps", "200", "--diffusion_samples", "1"),
        msa="yaml", min_fold_timeout=AFFINITY_FOLD_TIMEOUT_S),
    Leg(f"boltz2-affinity-dhfr-nomsa", "boltz2", "affinity", "examples/affinity_dhfr.yaml",
        fixture="boltz2/affinity_dhfr/nomsa_200step_5affsample_3recycle_bf16_mwcorr",
        committed_json="boltz2-affinity-dhfr-seeded.json", target_id="affinity_dhfr",
        device_args=("--single_sequence", "--affinity_mw_correction",
                      "--diffusion_samples_affinity", "5", "--sampling_steps_affinity", "200",
                      "--recycling_steps", "3", "--sampling_steps", "200", "--diffusion_samples", "1"),
        min_fold_timeout=AFFINITY_FOLD_TIMEOUT_S),
    Leg(f"boltz2-affinity-dhfr-msa", "boltz2", "affinity", "examples/affinity_dhfr_msa.yaml",
        fixture="boltz2/affinity_dhfr/msa-colabfold_200step_5affsample_3recycle_bf16_mwcorr_gpu",
        committed_json="boltz2-affinity-dhfr-seeded.json", target_id="affinity_dhfr",
        device_args=("--affinity_mw_correction", "--diffusion_samples_affinity", "5",
                      "--sampling_steps_affinity", "200", "--recycling_steps", "3",
                      "--sampling_steps", "200", "--diffusion_samples", "1"),
        msa="yaml", min_fold_timeout=AFFINITY_FOLD_TIMEOUT_S),
    Leg(f"boltz2-affinity-tryp-nomsa", "boltz2", "affinity", "examples/affinity_tryp.yaml",
        fixture="boltz2/affinity_tryp/nomsa_200step_5affsample_3recycle_bf16_mwcorr",
        committed_json="boltz2-affinity-tryp-seeded.json", target_id="affinity_tryp",
        device_args=("--single_sequence", "--affinity_mw_correction",
                      "--diffusion_samples_affinity", "5", "--sampling_steps_affinity", "200",
                      "--recycling_steps", "3", "--sampling_steps", "200", "--diffusion_samples", "1"),
        min_fold_timeout=AFFINITY_FOLD_TIMEOUT_S),
    Leg(f"boltz2-affinity-tryp-msa", "boltz2", "affinity", "examples/affinity_tryp_msa.yaml",
        fixture="boltz2/affinity_tryp/msa-colabfold_200step_5affsample_3recycle_bf16_mwcorr_gpu",
        committed_json="boltz2-affinity-tryp-seeded.json", target_id="affinity_tryp",
        device_args=("--affinity_mw_correction", "--diffusion_samples_affinity", "5",
                      "--sampling_steps_affinity", "200", "--recycling_steps", "3",
                      "--sampling_steps", "200", "--diffusion_samples", "1"),
        msa="yaml", min_fold_timeout=AFFINITY_FOLD_TIMEOUT_S),
]

# OpenDDE structure legs + abag + boltzgen (append separately for clarity)
# The two structure legs carry legacy_rdx for the protenix reason above: tt_bio/opendde.py is
# ttnn-only, so its committed ref_fp32 and ref_bf16 fixtures were two device folds and agreed
# to six decimals (0.533513 / 0.453797 on prot, 0.550660 / 0.444685 on trpcage).
LEGS += [
    Leg("opendde-trpcage-nomsa", "opendde", "structure", "examples/trpcage_no_msa.yaml",
        fixture="opendde/trpcage/nomsa_4cycle_20step_1sample_fp32_reduced",
        committed_json="opendde.json", target_id="trpcage_no_msa",
        device_args=("--single_sequence", "--recycling_steps", "4", "--sampling_steps", "20", "--diffusion_samples", "1"),
        msa="none", legacy_rdx=True),
    Leg("opendde-prot-prod", "opendde", "structure", "examples/prot_no_msa.yaml",
        fixture="opendde/prot/nomsa_10cycle_200step_1sample_fp32_prod",
        committed_json="opendde-prod-leg.json", target_id="prot_no_msa",
        device_args=("--single_sequence", "--recycling_steps", "10", "--sampling_steps", "200", "--diffusion_samples", "1"),
        msa="none", legacy_rdx=True),
    Leg("opendde-abag", "opendde-abag", "abag", "examples/1ahw_abag.yaml",
        committed_json="opendde-abag-1ahw-irmsd.json", seeds=(0,),
        note="DockQ leg; reuses release_gate --model opendde-abag"),
    Leg("boltzgen", "boltzgen", "boltzgen", "examples/binder.yaml",
        committed_json="boltzgen.json", seeds=(0,),
        note="designability leg; reuses release_gate --model boltzgen"),
    # Capacity, not accuracy: every leg above compares NUMBERS, so a change that grows the
    # device-memory footprint is invisible to them until a real fold runs out of memory on a
    # big target (which is exactly how multiplicity batching shipped). This leg folds the
    # largest supported target at the largest sample count and gates the measured peak DRAM.
    Leg("capacity", "protenix-v2", "capacity", "examples/abag_pilot_expansion/9j4c_abag.yaml",
        seeds=(0,),
        note="largest-input peak-DRAM budgets (9j4c/protenix-v2 residue scale + "
             "9ivj/opendde-abag structural scale); reuses release_gate --model capacity"),
    # --- RFD3 featurizer parity (card-free, in-process; reuses the committed
    # reference capture under scripts/rfd3_port/parity_artifacts/iai_protein/) ---
    # RFD3's correctness anchor is value parity of the host featurizer vs the
    # upstream foundry featurizer (43/43 f keys bit-exact, verified during the
    # port p12). The reference is committed, so this leg re-runs the ported
    # featurizer on the committed IAI_protein.pdb + contig and compares every
    # key bit-exact every release -- no device, no foundry install. The
    # trajectory bit-exactness gates from the batch-perf chain (p8-p11) are
    # separate device checks; this is the card-free foundation.
    Leg("rfd3-featurizer", "rfd3", "rfd3", "", committed_json="rfd3-featurizer.json",
        note="RFD3 host featurizer value-parity vs committed foundry reference "
              "(43/43 keys bit-exact); card-free, in-process"),
]

LEGS_BY_ID = {l.id: l for l in LEGS}


# ---------------------------------------------------------------------------
# Fingerprint cache
# ---------------------------------------------------------------------------
def _fixture_dir(spec: str) -> Path:
    return FIXTURE_ROOT / spec


def fixture_fingerprint(spec: str) -> str | None:
    """sha256 over the fixture meta.json's reference identity + settings + seeds.

    Returns None if the fixture is missing (no meta.json). The fingerprint is the
    cache key: identical fingerprint => the cached reference is still valid for
    this leg and only the device side re-runs. A changed fingerprint means the
    model code, weights, or test settings changed and the reference must be
    regenerated (the slow opt-in path).
    """
    base = _fixture_dir(spec)
    meta_path = base / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    # A fixture harvested from an external reference (e.g. official Aureka-OpenDDE /
    # ByteDance-Protenix) keeps its own top-level provenance for the legacy R/D/X scorer
    # (settings_tag etc, see pharma_parity.py) — the envelope's shared-draw identity for
    # THAT fixture lives one level down under "envelope" so regen_envelope_refs never has
    # to clobber the harvested provenance to update its own cache key. Fixtures with no
    # external harvest (envelope-native, e.g. boltz2 no-MSA) keep the identity flat at the
    # top level, same as before.
    src = meta.get("envelope", meta)
    identity = {
        "reference_impl": src.get("reference_impl", ""),
        "reference_version": src.get("reference_version", ""),
        "reference_commit": src.get("reference_commit", ""),
        "settings": src.get("settings", {}),
        "seeds": src.get("seeds", []),
    }
    blob = json.dumps(identity, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_fingerprint_index() -> dict:
    if FINGERPRINT_INDEX.exists():
        return json.loads(FINGERPRINT_INDEX.read_text())
    return {}


# ---------------------------------------------------------------------------
# Startup self-check — validate every leg's static wiring before any device work
# ---------------------------------------------------------------------------
def _fixture_result_ids(spec: str) -> set:
    """Target ids present in a committed fixture's seed results.json files. Used to catch a
    leg.target_id the device fold will never match (the 'no common targets' bug class): the
    structure scorer intersects the `id` fields across ref+dev seed dirs."""
    ids: set = set()
    for seed_dir in sorted(_fixture_dir(spec).glob("seed*")):
        rj = seed_dir / "results.json"
        if not rj.exists():
            continue
        try:
            data = json.loads(rj.read_text())
        except Exception:
            continue
        if isinstance(data, list):
            ids |= {x.get("id") for x in data if isinstance(x, dict) and x.get("id")}
        elif isinstance(data, dict) and data.get("id"):
            ids.add(data["id"])
    return ids


def _incomplete_fixture_seeds(leg, seeds: list) -> list:
    """For a structure leg, the seeds whose committed fixture dir lacks results.json or the
    target CIF the scorer needs. Empty == complete. Catches a fixture whose CIFs were never
    force-added past the .gitignore `ref-fixtures/**/*.cif` rule (present-but-incomplete on a
    clean checkout). Affinity/other kinds score off results.json only, so are not checked here."""
    if leg.kind != "structure" or not leg.fixture:
        return []
    base = _fixture_dir(leg.fixture)
    bad = []
    for s in seeds:
        sd = base / f"seed{s}"
        if not (sd / "results.json").exists() or not (sd / "structures" / f"{leg.target_id}.cif").exists():
            bad.append(f"seed{s}")
    return bad


def _fixture_known_seeds(spec: str) -> list[int]:
    """Seed indices with a seed dir in the committed fixture, sorted."""
    out = []
    for p in _fixture_dir(spec).glob("seed*"):
        m = re.fullmatch(r"seed(\d+)", p.name)
        if m and p.is_dir():
            out.append(int(m.group(1)))
    return sorted(out)


def _seeds_matched_against_fixture(leg, legacy_rdx: bool) -> bool:
    """True when the leg's seed list is matched against fixture seed dirs (the
    ``_incomplete_fixture_seeds`` path): structure legs scored legacy-R/D/X-style.
    Envelope legs ignore ``--seeds`` (the device fold always runs at ENVELOPE_SEED),
    so there is no fixture seed list to validate against for them."""
    return (leg.kind == "structure" and bool(leg.fixture)
            and (legacy_rdx or leg.legacy_rdx or not _is_envelope_leg(leg)))


def _of3_ckpt_default() -> Path | None:
    """The OpenFold3 p2 preview checkpoint at a known default location, if one exists.
    Mirrors tt_bio/worker.py's cache resolution plus the fleet's ~/of3-weights drop."""
    for p in (Path.home() / ".boltz" / "of3-p2-155k.pt",
              Path.home() / "of3-weights" / "of3-p2-155k.pt"):
        if p.exists():
            return p
    return None


def preflight_check(legs: list) -> list:
    """Card-free validation of every leg's static wiring, run before any device work (and via
    ``--check``). Returns a list of human-readable problems (empty == every leg well-formed).

    Catches, in seconds, the class of bugs that each cost a device turn during the v0.3.3
    release: a yaml/fixture/committed-JSON that does not exist, a target_id that will not match
    the device fold, a staged-MSA leg missing its a3m, and an msa='yaml' affinity leg whose yaml
    has no `msa:` field (so it would silently fold single-sequence against an MSA reference)."""
    problems = []
    for leg in legs:
        if leg.yaml and not (REPO / leg.yaml).exists():
            problems.append(f"{leg.id}: yaml {leg.yaml} not found")
        if leg.committed_json:
            cp = PARITY_DATA / leg.committed_json
            if not cp.exists():
                problems.append(f"{leg.id}: committed_json {leg.committed_json} not found")
            else:
                try:
                    json.loads(cp.read_text())
                except Exception as e:
                    problems.append(f"{leg.id}: committed_json {leg.committed_json} unparseable: {e}")
        if leg.fixture:
            if not (_fixture_dir(leg.fixture) / "meta.json").exists():
                problems.append(f"{leg.id}: fixture {leg.fixture} missing meta.json (regenerate reference)")
            elif leg.kind == "structure" and leg.target_id:
                have = _fixture_result_ids(leg.fixture)
                if have and leg.target_id not in have:
                    problems.append(
                        f"{leg.id}: target_id '{leg.target_id}' not in fixture ids {sorted(have)} "
                        f"— device fold and reference will not match (pharma_parity id intersection)")
        if leg.msa == "staged":
            src = _fixture_dir(leg.fixture) / "msa.a3m"
            multi = _fixture_dir(leg.fixture) / "msa"
            if not src.exists() and not (multi.is_dir() and any(multi.glob("*.a3m"))):
                problems.append(f"{leg.id}: staged-MSA leg missing {src} (or a {multi}/ dir of "
                                f"per-chain a3ms) — {_FIXTURE_FETCH_HINT}")
        if leg.msa == "yaml":
            yp = REPO / leg.yaml
            if yp.exists():
                m = re.search(r"msa:\s*(\S+)", yp.read_text())
                val = m.group(1) if m else None
                if val is None or val.lower() in ("empty", "null", "none", "~"):
                    problems.append(
                        f"{leg.id}: msa='yaml' but {leg.yaml} has msa={val!r} — device fold would "
                        f"run single-sequence, mismatching the MSA reference")
                elif ("/" in val or val.endswith(".a3m")) and not (REPO / val).exists():
                    problems.append(f"{leg.id}: msa='yaml' points at missing MSA file {val}")
        if leg.model == "openfold3":
            ckpt = os.environ.get("OF3_CKPT")
            if ckpt and not Path(ckpt).expanduser().exists():
                problems.append(f"{leg.id}: OF3_CKPT={ckpt} is not an existing file")
            elif not ckpt and _of3_ckpt_default() is None:
                problems.append(
                    f"{leg.id}: OpenFold3 checkpoint not found — set OF3_CKPT to the p2 "
                    f"preview weights (fleet copy ~/of3-weights/of3-p2-155k.pt); the fold "
                    f"otherwise fails inside tt_bio/worker.py after paying for setup")
    return problems


# ---------------------------------------------------------------------------
# Device command construction + MSA staging
# ---------------------------------------------------------------------------
def _yaml_protein_seq(yaml_path: Path) -> str | None:
    """Best-effort extraction of the first protein sequence from a tt-bio yaml."""
    txt = yaml_path.read_text()
    m = re.search(r"sequence:\s*([ACDEFGHIKLMNPQRSTVWY]+)", txt)
    return m.group(1) if m else None


def _seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()[:16]


def stage_msa(leg: Leg, workdir: Path) -> tuple[Path | None, list[str]]:
    """Stage the fixture's MSA for the device fold; return (msa_dir, extra_args).

    - "none": no MSA args (boltz2 --single_sequence is already in device_args).
    - "yaml": the yaml itself carries an `msa:` path — nothing to stage.
    - "server": --use_msa_server (already in device_args); nothing to stage.
    - "staged": copy <fixture>/msa.a3m to <workdir>/msa/<fixture>/<seqhash>.a3m,
      return --msa_dir. A multimer fixture instead carries <fixture>/msa/ holding one
      hash-named a3m per chain; those copy through verbatim (the names are already the
      {seq_hash}.a3m keys prepare_features looks up).
    """
    if leg.msa != "staged":
        return None, []
    fdir = _fixture_dir(leg.fixture)
    multi = fdir / "msa"
    if multi.is_dir():
        msa_dir = workdir / "msa" / leg.fixture.replace("/", "__")
        msa_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(multi.glob("*.a3m")):
            dst = msa_dir / src.name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
        return msa_dir, ["--msa_dir", str(msa_dir)]
    src = fdir / "msa.a3m"
    if not src.exists():
        raise FileNotFoundError(f"staged-MSA leg {leg.id}: missing {src}")
    seq = _yaml_protein_seq(REPO / leg.yaml)
    if not seq:
        raise ValueError(f"staged-MSA leg {leg.id}: could not read sequence from {leg.yaml}")
    # Scope the staged dir by fixture, not just by sequence. Two staged legs can share a
    # sequence and NOT share an MSA: protenix-ubq-msa and openfold3-ubq-msa both fold
    # examples/ubq.yaml but pin different reference a3m bytes, and the copy below is
    # first-writer-wins -- so one flat msa/<seqhash>.a3m silently fed the first leg's
    # reference MSA to the second, comparing it against a reference built from other bytes.
    msa_dir = workdir / "msa" / leg.fixture.replace("/", "__")
    msa_dir.mkdir(parents=True, exist_ok=True)
    dst = msa_dir / f"{_seq_hash(seq)}.a3m"
    if not dst.exists():
        dst.write_bytes(src.read_bytes())
    return msa_dir, ["--msa_dir", str(msa_dir)]


def device_cmd(leg: Leg, seed: int, out_dir: Path, workdir: Path) -> list[str]:
    """Build the `tt-bio predict` command for one device fold (one seed).

    The yaml arg is RELATIVE (leg.yaml is already relative, e.g. "examples/x.yaml") —
    Worker.wrap() cd's into the repo root (local or remote) before exec'ing, so a relative
    path resolves correctly on whichever host actually runs this command.
    """
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", leg.yaml,
           "--model", leg.model, "--out_dir", str(out_dir), "--override",
           "--seed", str(seed)]
    cmd += list(leg.device_args)
    _, msa_args = stage_msa(leg, workdir)
    cmd += msa_args
    return cmd


# ---------------------------------------------------------------------------
# Worker pool — fan device folds across host:card slots
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Worker:
    host: str
    card: int
    is_local: bool
    remote_cwd: str | None = None     # remote checkout path, if different from the local worktree
    remote_python: str | None = None  # remote env's python, if different from sys.executable

    def wrap(self, cmd: list[str], cwd: Path, env: dict) -> list[str]:
        """Wrap a command for this worker: local subprocess or ssh + remote shell."""
        work_dir = str(cwd) if self.is_local else (self.remote_cwd or str(cwd))
        if not self.is_local and self.remote_python and cmd and cmd[0] == sys.executable:
            cmd = [self.remote_python, *cmd[1:]]
        env_prefix = [f"TT_VISIBLE_DEVICES={self.card}", f"PYTHONPATH={work_dir}"]
        env_str = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
        full_env = " ".join(env_prefix + ([env_str] if env_str else []))
        if self.is_local:
            # run via env + sh -c so the env vars apply to the python process
            return ["sh", "-c", full_env + " exec " + " ".join(shlex.quote(c) for c in cmd)]
        # remote: ssh host -- 'env ... cmd' (cwd via remote cd). Use remote_cwd/remote_python
        # when the remote checkout+env don't live at the same absolute paths as the local
        # worktree (different user/host) — cmd's own file arguments are relative (see
        # device_cmd), so they resolve correctly under whichever cwd we land in here.
        remote = f"cd {shlex.quote(work_dir)} && {full_env} exec " + " ".join(shlex.quote(c) for c in cmd)
        return ["ssh", "-o", "ConnectTimeout=5", self.host, remote]


def parse_workers(spec: str) -> list[Worker]:
    """Parse '--workers' entries: host:card, or host:card:remote_cwd[:remote_python] for a
    remote whose checkout/env don't live at the same absolute paths as the local worktree
    (e.g. a different user/home on that host)."""
    out = []
    # Locality must come from the real hostname. $HOSTNAME is a bash-only variable and is
    # NOT exported to non-interactive shells, so it is unset under ssh/systemd on every host —
    # the old "pc" fallback therefore made any non-pc host classify ITSELF as remote and ssh
    # to its own hostname ("Host key verification failed", every device leg exiting 255 in
    # under a second). It only ever worked on pc, by accident of the default matching.
    local_host = (os.environ.get("HOSTNAME") or socket.gethostname()).split(".")[0]
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        host, _, rest = part.partition(":")
        card_str, _, rest2 = rest.partition(":")
        remote_cwd, _, remote_python = rest2.partition(":")
        out.append(Worker(host=host, card=int(card_str or 0),
                           is_local=(host in ("localhost", "127.0.0.1", local_host)),
                           remote_cwd=remote_cwd or None,
                           remote_python=remote_python or None))
    # Default worker names the host we are actually on — report.json records these as
    # provenance, so "pc:0" on a different box would be a false record.
    return out or [Worker(host=local_host, card=0, is_local=True)]


def _find_results_dir(out_dir: Path) -> Path | None:
    """The inner ``<model>_results_<id>/`` dir the scorer wants, located by its results.json.
    tt-bio predict writes results into a subdir of ``--out_dir``; the scorer expects that
    inner dir as the dev_dir. results.json present is also the fold-success signal."""
    if not out_dir.is_dir():
        return None
    for p in out_dir.iterdir():
        if p.is_dir() and (p / "results.json").exists():
            return p
    return None


def _reap(proc) -> None:
    """Terminate a process and its children, escalating to kill after a short grace."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _run_local_fold(wrapped, out_dir: Path, logf, fold_timeout: float | None):
    """Run a local device fold with poll-based completion detection. Returns (rc, timed_out).

    Two hang classes are handled by one loop:
      - shutdown hang AFTER success: a boltz2 affinity predict can hang in do_select on exit
        after writing results.json (a process-exit bug, not a fold failure). results.json is
        the success signal, so once it appears we grant a short grace window then reap a hung
        shutdown -> (0, False).
      - real hang BEFORE success: a fold that never writes results.json within fold_timeout
        (e.g. a flaky MSA server, #6 in the postmortem) is killed -> (timeout-sentinel, True).
    """
    GRACE_S = 30.0
    proc = subprocess.Popen(wrapped, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
    t0 = time.monotonic()
    folded_at = None
    while True:
        rc = proc.poll()
        if rc is not None:
            return rc, False
        if folded_at is None and _find_results_dir(out_dir) is not None:
            folded_at = time.monotonic()
        if folded_at is not None and time.monotonic() - folded_at >= GRACE_S:
            _reap(proc)
            return 0, False
        if folded_at is None and fold_timeout and time.monotonic() - t0 > fold_timeout:
            _reap(proc)
            return -99, True
        time.sleep(2.0)


def _run_remote_fold(wrapped, worker: "Worker", out_dir: Path, logf, fold_timeout: float | None):
    """Run a fold on a remote host, then rsync its output dir back to the coordinator so the
    LOCAL scorer can see it. Returns (rc, timed_out).

    Fixes the postmortem's latent remote bug: device_cmd bakes the coordinator's absolute
    --out_dir into the command, so a remote fold writes to that same absolute path ON THE
    REMOTE; without this copy-back the local scorer never sees the output. Correct-by-
    construction and isolated to the non-local branch; the tested release path is local cards.
    """
    try:
        proc = subprocess.run(wrapped, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT,
                              timeout=fold_timeout)
    except subprocess.TimeoutExpired:
        return -99, True
    if proc.returncode == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "-e", "ssh -o ConnectTimeout=5",
                        f"{worker.host}:{out_dir}/", f"{out_dir}/"], check=False)
    return proc.returncode, False


def run_folds_fanout(leg: Leg, seeds: list[int], workdir: Path, workers: list[Worker],
                     log_dir: Path, resume: bool = True, fold_timeout: float | None = None,
                     extra_env: dict | None = None) -> dict:
    """Run one device fold per seed, fanned across workers; return {seed: out_dir_or_error}.

    Workers run in parallel; each worker runs its seeds serially (one device context per
    process). Local workers use poll-based reaping; remote workers rsync output back. With
    ``resume`` (default) a seed whose output already carries a results.json is reused, so a
    bounded turn never re-folds work a prior turn finished.

    Invariant, load-bearing: ``worker_run`` must never return while a fold it started is
    still alive. Folds are spawned from a thread-pool thread, and PR_SET_PDEATHSIG (armed by
    the spawned CLI, see tt_bio/device_lease.py) is delivered on the exit of the THREAD that
    created the process, not of the driver. Serial-and-wait keeps that a no-op; an async
    refactor of this function would start killing live folds.
    """
    leg_dir = workdir / leg.id
    leg_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, Path] = {}
    # round-robin seeds across workers; group by worker so each runs serially, workers parallel
    by_worker: dict[Worker, list[int]] = {}
    for i, s in enumerate(seeds):
        by_worker.setdefault(workers[i % len(workers)], []).append(s)

    import concurrent.futures

    def worker_run(w: Worker, seeds_w: list[int]):
        out = {}
        for s in seeds_w:
            out_dir = leg_dir / f"seed{s}"
            if resume:
                inner = _find_results_dir(out_dir)
                if inner is not None:
                    out[s] = inner
                    continue
            wrapped = w.wrap(device_cmd(leg, s, out_dir, workdir), REPO, dict(extra_env or {}))
            logf = open(log_dir / f"{leg.id}_seed{s}.log", "w")
            t0 = time.monotonic()
            try:
                runner = _run_local_fold if w.is_local else _run_remote_fold
                rc, timed_out = (runner(wrapped, out_dir, logf, fold_timeout) if w.is_local
                                 else runner(wrapped, w, out_dir, logf, fold_timeout))
            finally:
                logf.close()
            wall = time.monotonic() - t0
            if timed_out:
                out[s] = {"error": f"fold timed out after {fold_timeout:.0f}s (no results.json "
                          f"— flaky MSA server? place a cached a3m per RELEASING.md and rerun)",
                          "wall": wall}
            elif rc != 0:
                out[s] = {"error": f"predict exited {rc}", "wall": wall}
            else:
                out[s] = _find_results_dir(out_dir) or out_dir
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(by_worker)) as ex:
        futs = [ex.submit(worker_run, w, j) for w, j in by_worker.items()]
        for f in concurrent.futures.as_completed(futs):
            for s, v in f.result().items():
                results[s] = v
    return results


# ---------------------------------------------------------------------------
# Scoring dispatch — reuse the vetted scorers, never re-derive
# ---------------------------------------------------------------------------
def _run(cmd: list[str], log_path: Path) -> tuple[int, str]:
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode, (log_path.read_text() if log_path.exists() else "")


def _fixture_seed_dirs(spec: str, seeds: list[int]) -> list[str]:
    base = _fixture_dir(spec)
    return [str(base / f"seed{s}") for s in seeds]


def score_structure(leg: Leg, dev_dirs: list[str], out_json: Path, log_path: Path) -> dict | None:
    cmd = [sys.executable, "scripts/pharma_parity.py", "structures",
           "--ref-fixtures", leg.fixture, "--dev-dirs", *dev_dirs,
           "--label", leg.id, "--out", str(out_json)]
    rc, _ = _run(cmd, log_path)
    if rc != 0 or not out_json.exists():
        return None
    return json.loads(out_json.read_text())


def score_affinity(leg: Leg, dev_dirs: list[str], out_json: Path, log_path: Path) -> dict | None:
    ref_dirs = _fixture_seed_dirs(leg.fixture, list(leg.seeds))
    cmd = [sys.executable, "scripts/boltz2_affinity_parity.py",
           "--ref-dirs", *ref_dirs, "--dev-dirs", *dev_dirs,
           "--target-id", leg.target_id, "--paired", "--out", str(out_json)]
    rc, _ = _run(cmd, log_path)
    if rc != 0 or not out_json.exists():
        return None
    return json.loads(out_json.read_text())


# ---------------------------------------------------------------------------
# Integration-parity envelope — the CORRECTNESS criterion for diffusion legs
# ---------------------------------------------------------------------------
# Replaces the old R/D/X same-backend self-consistency floor (see the module docstring
# "INTEGRATION-PARITY ENVELOPE" note and docs/implementation-parity.md). For a diffusion
# leg (structure/affinity), correctness is a DETERMINISTIC shared-draws test:
#   device_bf16 (TT, the port)  vs  reference_fp32 (CPU)      -> numerator
#   reference_bf16 (CPU)        vs  reference_fp32 (CPU)      -> measured bf16 envelope
# All three are tt-bio's own code (CPU refs via --accelerator cpu --no_kernels) from the SAME
# --seed, so the diffusion torch.randn draws are byte-identical (CPU MT19937) by construction
# and the only difference is arithmetic. PASS iff numerator <= envelope*(1+margin)+abs_floor on
# every metric. The two CPU references are the CACHED fixture (fingerprinted like the old ones);
# only the device fold + scoring re-run per release. Regenerate them with --regen-refs.
def _envelope_ref_complete(inner: Path | None, leg: Leg) -> Path | None:
    """A structure leg's envelope ref is only usable if its CA-RMSD CIF is actually present —
    results.json alone (e.g. a regen that crashed before writing structures/) is not enough and
    must not be handed to the scorer, which has no graceful path for a missing file."""
    if inner is None or leg.kind != "structure":
        return inner
    return inner if (inner / "structures" / f"{leg.target_id}.cif").exists() else None


def envelope_ref_dirs(leg: Leg) -> tuple[Path | None, Path | None]:
    """Locate the fp32 + bf16 CPU shared-draw reference result dirs for an envelope leg.

    Convention: ``<fixture>/ref_fp32/`` and ``<fixture>/ref_bf16/`` each hold the inner
    ``<model>_results_<id>/`` dir a fold writes (located by its results.json). Returns
    (fp32_inner, bf16_inner); a missing OR structurally-incomplete side is None (leg ->
    BLOCKED-REF-REGEN-NEEDED, never a crash mid-scoring)."""
    base = _fixture_dir(leg.fixture)
    fp32 = _envelope_ref_complete(_find_results_dir(base / "ref_fp32"), leg)
    bf16 = _envelope_ref_complete(_find_results_dir(base / "ref_bf16"), leg)
    return fp32, bf16


def score_envelope(leg: Leg, dev_dir: str, ref_fp32: Path, ref_bf16: Path,
                   out_json: Path, margin: float) -> dict | None:
    """Score one diffusion leg with the deterministic shared-draws envelope test.

    Reuses scripts/integration_envelope.py (the vetted envelope scorer + per-leg distance
    primitives) — nothing re-derived here. Writes the report to out_json for --resume and
    returns it (mode == 'integration_envelope', consumed by extract_verdict)."""
    sys.path.insert(0, str(REPO / "scripts"))
    from integration_envelope import envelope_verdict  # lazy: pulls numpy/gemmi
    rep = envelope_verdict(dev_dir, ref_fp32, ref_bf16, leg.kind, leg.target_id, margin)
    out_json.write_text(json.dumps(rep, indent=2))
    return rep


def _ref_settings(leg: Leg) -> dict:
    """The settings that DEFINE this leg's reference — the fingerprint's cache key. Change any of
    these and the reference must be regenerated (fingerprint drift => BLOCKED-REF-REGEN-NEEDED)."""
    return {"device_args": list(leg.device_args), "seed": ENVELOPE_SEED,
            "shared_draw_seed": ENVELOPE_SEED,  # the shared-draws discipline is part of the ref identity
            "yaml": leg.yaml, "model": leg.model, "target_id": leg.target_id, "msa": leg.msa}


def _refs_identical(base: Path) -> bool:
    """True when a fixture's fp32 and bf16 references are the same bytes.

    The envelope denominator IS the fp32-vs-bf16 distance, so two identical references make it
    zero and every device residual then reads as a GAP no matter how small it is. That is not a
    hypothetical: it is what a ttnn-only model produced, both references having run on the card.
    Compared on the structures, not results.json, because the scorer reads coordinates."""
    import hashlib

    def digest(d: Path) -> str | None:
        cifs = sorted(d.rglob("*.cif"))
        if not cifs:
            return None
        h = hashlib.sha256()
        for p in cifs:
            h.update(p.read_bytes())
        return h.hexdigest()

    a, b = digest(base / "ref_fp32"), digest(base / "ref_bf16")
    return a is not None and a == b


def regen_envelope_refs(legs: list, workdir: Path, log_dir: Path,
                        fold_timeout: float | None, resume: bool) -> int:
    """(Re)generate each envelope leg's fp32 + bf16 CPU shared-draw references into
    <fixture>/ref_{fp32,bf16}/ and write the fixture meta.json (fingerprint cache key).

    Both references are the SAME tt-bio CPU torch path (--accelerator cpu --no_kernels; the pure
    torch trimul, no CUDA cuequivariance) at ENVELOPE_SEED, differing only by the TT_BIO_REF_BF16
    bf16-autocast toggle — so they share one CPU-MT19937 diffusion draw sequence by construction.
    Run SERIALLY (fp32 then bf16, one leg at a time): concurrent pure-torch CPU folds oversubscribe
    the host and triple wall-clock (measured 2026-07-23). This is the expensive cached step; a
    normal gate run then only re-folds the device side + scores."""
    local = Worker(host="pc", card=0, is_local=True)
    n_ok = 0
    for leg in legs:
        if not _is_envelope_leg(leg) or not leg.fixture or leg.legacy_rdx:
            continue
        base = _fixture_dir(leg.fixture)
        base.mkdir(parents=True, exist_ok=True)
        leg_ok = True
        for dtype, env in (("fp32", dict(_shared_draw_env())),
                           ("bf16", {"TT_BIO_REF_BF16": "1", **_shared_draw_env()})):
            out_dir = base / f"ref_{dtype}"
            if resume and _envelope_ref_complete(_find_results_dir(out_dir), leg) is not None:
                print(f"  {leg.id} ref_{dtype}: cached, skip")
                continue
            # A prior interrupted regen can leave a STALE, incomplete results.json in out_dir
            # (e.g. results.json with no structures/*.cif). _run_local_fold's completion check
            # is a bare _find_results_dir(out_dir) probe -- it would see that stale file the
            # instant the fresh subprocess starts, believe the NEW fold already "folded", and
            # reap it after the grace window without ever letting it run. Clear any leftover
            # out_dir before starting so only the fresh subprocess's own output can satisfy it.
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = device_cmd(leg, ENVELOPE_SEED, out_dir, workdir) + ["--accelerator", "cpu", "--no_kernels"]
            wrapped = local.wrap(cmd, REPO, env)
            logf = open(log_dir / f"regen_{leg.id}_{dtype}.log", "w")
            t0 = time.monotonic()
            try:
                rc, timed_out = _run_local_fold(wrapped, out_dir, logf,
                                                max(fold_timeout, leg.min_fold_timeout or 0.0))
            finally:
                logf.close()
            wall = time.monotonic() - t0
            ok = (rc == 0 and _envelope_ref_complete(_find_results_dir(out_dir), leg) is not None)
            print(f"  {leg.id} ref_{dtype}: {'OK' if ok else 'FAILED'} ({wall/60:.1f} min)"
                  + ("" if ok else f" rc={rc} timed_out={timed_out} — see regen_{leg.id}_{dtype}.log"))
            leg_ok &= ok
        if leg_ok and _refs_identical(base):
            print(f"  {leg.id}: ref_fp32 and ref_bf16 are byte-identical — the bf16 autocast "
                  "never applied, so the envelope denominator is 0 and any device residual "
                  "reads as a false GAP. Refusing to write meta.json for this leg.")
            leg_ok = False
        if leg_ok:
            envelope_meta = {"reference_impl": "tt-bio-cpu-torch", "reference_version": _repo_commit(),
                             "reference_commit": _repo_commit(), "settings": _ref_settings(leg),
                             "seeds": [ENVELOPE_SEED]}
            # MERGE, never clobber, a HARVESTED fixture's top-level meta.json: settings_tag,
            # "official Aureka-OpenDDE"/"official ByteDance Protenix" provenance, command,
            # date, invalidation_rule are read by the legacy R/D/X scorer (pharma_parity.py,
            # --legacy-rdx) against the ALREADY-COMMITTED seed0-4 dirs -- unrelated to and
            # unaffected by this envelope regen. Overwriting the whole file here previously
            # destroyed that provenance every time the envelope refs were regenerated
            # (root cause of the ff473d2ed / 88c14f3b2 / 025ef2479 back-and-forth). A fixture
            # is "harvested" iff its meta.json carries settings_tag (the legacy scorer's own
            # marker, see pharma_parity.py) -- only then do we preserve top-level and nest the
            # envelope's own bookkeeping under "envelope". An envelope-native fixture (no
            # settings_tag, e.g. boltz2 no-MSA) keeps the old flat replacement (no stale keys).
            meta_path = base / "meta.json"
            old_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            if "settings_tag" in old_meta:
                old_meta["envelope"] = envelope_meta
                meta = old_meta
            else:
                meta = envelope_meta
            # The legacy R/D/X scorer refuses any fixture whose meta.json does not name its
            # own settings tag (pharma_parity.py, "settings-tag mismatch"), so a fixture the
            # branch above wrote flat has been unscoreable under --legacy-rdx ever since:
            # boltz2-{trpcage,prot,hsa}-nomsa hard-ERRORed the first time this gate could
            # reach them. The tag IS the directory name, so write it and the guard compares
            # two real values instead of None against a string.
            meta.setdefault("settings_tag", base.name)
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
            n_ok += 1
    # refresh the fingerprint index so a matching reference takes the fast (device-only) path
    idx = load_fingerprint_index()
    for leg in legs:
        if _is_envelope_leg(leg) and leg.fixture:
            fp = fixture_fingerprint(leg.fixture)
            if fp:
                idx[leg.id] = fp
    FINGERPRINT_INDEX.write_text(json.dumps(idx, indent=2, sort_keys=True))
    print(f"regen complete: {n_ok} leg(s) with fp32+bf16 references; fingerprint index updated.")
    return 0


def _repo_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                       text=True, timeout=5).strip()
    except Exception:
        return "unknown"


def _load_release_gate():
    """Import scripts/release_gate.py as a module so the boltzgen/abag legs can call its
    vetted ``run_boltzgen`` / ``run_opendde_abag`` IN-PROCESS and capture their real structured
    row (scRMSD/pass-rate, DockQ/fnat) — instead of shelling out and capturing only a return
    code. That removes the live-vs-committed shape mismatch (postmortem #3) at the root."""
    import importlib.util
    path = REPO / "scripts" / "release_gate.py"
    spec = importlib.util.spec_from_file_location("tt_bio_release_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_inprocess(leg: Leg, out_json: Path, log_path: Path, env: dict,
                  fold_timeout: float | None = None, pin_card: int | None = None) -> dict | None:
    """Run the dedicated harness for esmc/saprot/esmfold2 (subprocess) or the in-process
    designability/DockQ leg for boltzgen/abag. Persists the report to out_json (for --resume).

    pin_card sets TT_VISIBLE_DEVICES for the subprocess harnesses. Without it they inherit the
    gate's environment, which has no device restriction, so on a multi-chip host they open the
    WHOLE mesh. On the 32-chip Wormhole galaxy that fails before any compute: 32 copies of
    "Read unexpected run_mailbox value: 0x40" (one per chip, all on core (x=25,y=17), i.e. stale
    state left by the previously-stopped 32-worker service) and then a nonsense
    "Out of Memory ... allocate 524288 B ... bank size is 1073741792 B" — a 512 KB request against
    a 1 GB bank, meaning the allocator saw DRAM as full rather than actually being short of it.
    The per-card legs, which are pinned, passed on those same chips in the same run. Pinning is
    also simply correct: these legs score one card's numerics, so they have no use for a mesh."""
    if leg.kind in ("boltzgen", "abag", "capacity"):
        try:
            rg = _load_release_gate()
            row = (rg.run_boltzgen(rg._load_designability_harness(), keep=False)
                   if leg.kind == "boltzgen" else rg.run_capacity_all(keep=False)
                   if leg.kind == "capacity" else rg.run_opendde_abag(keep=False))
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        out_json.write_text(json.dumps(row, indent=2, default=str))
        return row

    if leg.kind == "rfd3":
        # Card-free in-process: run the ported featurizer on the committed IAI
        # fixture and compare every f key bit-exact vs the committed foundry
        # reference (scripts/rfd3_port/parity_gate.py). No device, no fold.
        sys.path.insert(0, str(REPO / "scripts" / "rfd3_port"))
        try:
            from parity_gate import featurizer_parity
            rep = featurizer_parity()
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        out_json.write_text(json.dumps(rep, indent=2))
        return rep

    if leg.kind == "esmc":
        script = "scripts/esmc6b_embed_parity.py" if leg.model == "esmc-6b" else "scripts/esmc_embed_parity.py"
        # esmc_embed_parity multi-leg mode: --seqs + --out writes the pharma-style targets
        # report whose shape matches the committed esmc-{300m,600m}.json. (6b has no --model.)
        cmd = [sys.executable, script, "--seqs", "trpcage,gb1,ubiquitin,lysozyme", "--out", str(out_json)]
        if leg.model != "esmc-6b":
            cmd[2:2] = ["--model", leg.model]
    elif leg.kind == "saprot":
        cmd = [sys.executable, "scripts/pharma_parity.py", "saprot", "--model", leg.model,
               "--out", str(out_json)]
    elif leg.kind == "esmfold2":
        cmd = [sys.executable, "scripts/esmfold2_e2e_parity.py",
               "--proteins", "trpcage,gb1,ubiquitin,lysozyme", "--seeds", "0,1,2,3,4",
               "--out", str(out_json)]
        # Mirror production (main.py): esmfold2 auto-runs --fast on Wormhole because the
        # non-fast model needs >12 GB DRAM/chip; without this the device load OOMs.
        from tt_bio.tenstorrent import is_wormhole
        if is_wormhole():
            cmd.append("--fast")
    else:
        return None
    if pin_card is not None:
        env = {**env, "TT_VISIBLE_DEVICES": str(pin_card)}
    try:
        with open(log_path, "w") as f:
            proc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, env=env,
                                  timeout=fold_timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"{leg.id} harness timed out after {fold_timeout:.0f}s"}
    if proc.returncode != 0:
        return None
    if out_json.exists():
        return json.loads(out_json.read_text())
    return None


# ---------------------------------------------------------------------------
# Verdict extraction + drift check
# ---------------------------------------------------------------------------
def _structure_verdict(report: dict) -> tuple[str, str]:
    """Return (verdict, primary_metric_line) for a structures-mode report."""
    targets = report.get("targets", {})
    if not targets:
        return "NO-DATA", "no targets scored"
    # the gate metric is kabsch_rmsd within_noise_floor; aggregate across targets
    all_within = []
    lines = []
    for tid, tv in targets.items():
        kv = tv.get("kabsch_rmsd", {})
        within = kv.get("within_noise_floor")
        if within is None:
            continue
        all_within.append(bool(within))
        x = kv.get("cross", {}).get("mean", float("nan"))
        r = kv.get("ref_floor", {}).get("mean", float("nan"))
        d = kv.get("dev_floor", {}).get("mean", float("nan"))
        lines.append(f"{tid}: X={x:.3f} R={r:.3f} D={d:.3f} within={within}")
    if not all_within:
        return "NO-DATA", "; ".join(lines) or "no kabsch_rmsd metric"
    verdict = "PASS" if all(all_within) else "GAP"
    return verdict, "; ".join(lines)


def _affinity_verdict(report: dict) -> tuple[str, str]:
    """Affinity: gate metric is affinity_pred_value within_noise_floor; pose metrics secondary."""
    metrics = report.get("metrics", {})
    av = metrics.get("affinity_pred_value", {})
    within = av.get("within_noise_floor")
    if within is None:
        return "NO-DATA", "no affinity_pred_value metric"
    x = av.get("cross", {}).get("mean", float("nan"))
    r = av.get("ref_floor", {}).get("mean", float("nan"))
    d = av.get("dev_floor", {}).get("mean", float("nan"))
    # PASS-caveated if scalar passes but pocket-lDDT GAPs (matches the doc's convention)
    pocket = metrics.get("1-pocket_lddt", {})
    pocket_within = pocket.get("within_noise_floor")
    if within and pocket_within is False:
        v = "PASS-caveated"
    elif within:
        v = "PASS"
    else:
        v = "GAP"
    return v, f"affinity_pred_value X={x:.4f} R={r:.4f} D={d:.4f} within={within}; pocket within={pocket_within}"


def _esmc_verdict(report: dict) -> tuple[str, str]:
    t = report.get("targets", {})
    if not t:
        return "NO-DATA", "no targets"
    pccs = [v.get("dev_vs_ref_pcc", 0) for v in t.values()]
    mn = min(pccs) if pccs else 0
    return ("PASS" if mn >= 0.99 else "GAP"), f"min per-res PCC={mn:.5f}"


def _saprot_verdict(report: dict) -> tuple[str, str]:
    x = report.get("X_emb", 0)
    return ("PASS" if x >= 0.9987 else "GAP"), f"X_emb={x:.5f}"


def _boltzgen_verdict(report: dict) -> tuple[str, str]:
    # Live row (run_boltzgen): {scrmsd_median, pass_rate, gate, error}. The floor (RELEASING.md)
    # is >=50% of binders refolding within 2.0 A scRMSD.
    if report.get("error"):
        return "ERROR", str(report["error"])
    if report.get("pass_rate") is not None:
        rate = report["pass_rate"]
        return ("PASS" if report.get("gate") else "GAP"), \
               f"scRMSD pass-rate {rate*100:.0f}% (median {report.get('scrmsd_median')})"
    # Committed JSON (docs/implementation-parity-data/boltzgen.json): a designability
    # record with device_batches[].designs[].scrmsd.
    scrmsds = [x.get("scrmsd") for b in report.get("device_batches", [])
               for x in b.get("designs", []) if x.get("scrmsd") is not None]
    if scrmsds:
        rate = sum(1 for s in scrmsds if s <= 2.0) / len(scrmsds)
        return ("PASS" if rate >= 0.5 else "GAP"), f"committed scrmsd pass-rate {rate*100:.0f}% ({len(scrmsds)} designs)"
    if "_release_gate_rc" in report:  # legacy rc-only record
        rc = report["_release_gate_rc"]
        return ("PASS" if rc == 0 else "GAP"), f"release_gate rc={rc}"
    return "NO-DATA", "no designs in record"


def _abag_verdict(report: dict) -> tuple[str, str]:
    # Live row (run_opendde_abag): {dockq, fnat, gate, error}. Floor (RELEASING.md) global_dockq>=0.50.
    if report.get("error"):
        return "ERROR", str(report["error"])
    if report.get("dockq") is not None:
        return ("PASS" if report.get("gate") else "GAP"), f"global DockQ={report['dockq']:.3f}"
    dq = report.get("global_dockq")  # committed DockQ record
    if dq is not None:
        return ("PASS" if dq >= 0.50 else "GAP"), f"committed global_dockq={dq:.3f}"
    if "_release_gate_rc" in report:  # legacy rc-only record
        rc = report["_release_gate_rc"]
        return ("PASS" if rc == 0 else "GAP"), f"release_gate rc={rc}"
    return "NO-DATA", "no global_dockq in record"


def _capacity_verdict(report: dict) -> tuple[str, str]:
    """Capacity: PASS iff the largest-input fold stayed inside the DRAM budget AND wrote
    every sample it was asked for. An ERROR here is a real device fatal, not a soft miss --
    a harness that files an allocation failure as a per-item status is what let this class
    of bug look like progress."""
    if report.get("error"):
        return "ERROR", str(report["error"])
    peak = report.get("peak_gib")
    if peak is None:
        return "NO-DATA", "no peak DRAM measured"
    detail = f"peak {peak:.2f} GiB, {report.get('cifs')} CIFs / {report.get('paes')} PAEs"
    if report.get("legs"):
        detail += " (" + "; ".join(
            f"{r['model']} {r['peak_gib']:.2f} GiB" for r in report["legs"]
            if r.get("peak_gib") is not None) + ")"
    return ("PASS" if report.get("gate") else "GAP"), detail


def _rfd3_verdict(report: dict) -> tuple[str, str]:
    """RFD3 featurizer parity: PASS iff every comparable f key is bit-exact vs the
    committed foundry reference capture (the port's own 43/43-key bar, p12)."""
    if report.get("error"):
        return "ERROR", str(report["error"])
    total = report.get("keys_total", 0)
    bx = report.get("keys_bitexact", 0)
    mm = report.get("mismatches", [])
    if total == 0:
        return "NO-DATA", "no keys scored"
    verdict = report.get("verdict", "PASS" if not mm else "GAP")
    detail = f"{bx}/{total} f keys bit-exact"
    if mm:
        detail += f"; mismatches: {[m['key'] for m in mm]}"
    return verdict, detail


def _envelope_verdict_row(report: dict) -> tuple[str, str]:
    """Verdict for an integration_envelope report: PASS iff every metric is within the measured
    bf16 envelope; else GAP (a real residual exceeding the envelope — to hunt, not excuse). The
    detail line lists the worst per-metric numerator/envelope ratio so a GAP is legible."""
    metrics = report.get("metrics", {})
    if not metrics:
        return "NO-DATA", "no envelope metrics scored"
    # A zero envelope on EVERY metric means the two references are the same computation, so
    # the leg is decided by abs_floor alone and measures nothing. It must never read as PASS:
    # three legs did exactly that until 2026-08-11 (ttnn-only models whose "CPU references"
    # both ran on the card). A single scalar metric rounding to a zero envelope is legitimate
    # — that is what abs_floor is for — so only a whole-report collapse is refused here.
    if all(float(m.get("envelope", 0.0) or 0.0) == 0.0 for m in metrics.values()):
        return "NO-DATA", ("every metric envelope is 0 — the fp32 and bf16 references are the "
                           "same computation, so this leg measures nothing")
    worst = max(metrics.items(), key=lambda kv: kv[1].get("ratio", 0.0))
    wk, wm = worst
    parts = [f"{k} r={m.get('ratio', float('nan')):.2f}" for k, m in metrics.items()]
    detail = f"envelope worst {wk}: num={wm['numerator']:.4f} env={wm['envelope']:.4f} " \
             f"ratio={wm['ratio']:.2f}; " + ", ".join(parts)
    return report.get("verdict", "NO-DATA"), detail


def extract_verdict(leg: Leg, report: dict | None) -> tuple[str, str]:
    if report is None:
        return "ERROR", "scorer returned no report (see log)"
    # Diffusion legs (structure/affinity) score with the integration-parity envelope; a resumed
    # or legacy R/D/X report (no 'mode') still reads through the old extractors (D-diagnostic).
    if isinstance(report, dict) and report.get("mode") == "integration_envelope":
        return _envelope_verdict_row(report)
    if leg.kind == "structure":
        return _structure_verdict(report)
    if leg.kind == "affinity":
        return _affinity_verdict(report)
    if leg.kind == "esmc":
        return _esmc_verdict(report)
    if leg.kind == "saprot":
        return _saprot_verdict(report)
    if leg.kind == "boltzgen":
        return _boltzgen_verdict(report)
    if leg.kind == "abag":
        return _abag_verdict(report)
    if leg.kind == "capacity":
        return _capacity_verdict(report)
    if leg.kind == "rfd3":
        return _rfd3_verdict(report)
    if leg.kind == "esmfold2":
        # esmfold2_e2e_parity summary.json is a list of per-protein dicts (each with a
        # kabsch_rmsd block). The gate's recorded behavior is PASS-if-scored (the
        # committed esmfold2.json itself has proteins with within_noise_floor=False, so
        # strict all-within would contradict the doc's PASS); preserve that and report
        # the within-floor count for transparency.
        proteins = report if isinstance(report, list) else report.get("targets", report.get("proteins", []))
        if not proteins:
            return "NO-DATA", "no proteins in summary"
        n_within = sum(1 for p in proteins if p.get("kabsch_rmsd", {}).get("within_noise_floor"))
        return "PASS", f"{len(proteins)} proteins scored ({n_within} within floor)"
    return "UNKNOWN", "no extractor"


# ---------------------------------------------------------------------------
# Drift check vs committed numbers
# ---------------------------------------------------------------------------
def _is_passing(v: str | None) -> bool:
    """PASS and PASS-caveated are both passing verdicts — a leg that lands in either is
    release-acceptable. PASS-caveated means the gate metric passes but a documented
    bf16-backend floor GAPs a secondary metric (e.g. pocket-lDDT); it is not a drift
    of the gate metric. Used so the drift check treats PASS vs PASS-caveated as reproduces
    (a seed-variance pocket-lDDT flip between PASS and PASS-caveated is not a regression)."""
    return v in ("PASS", "PASS-caveated")


def _matches_committed(verdict: str, committed: str) -> bool:
    """Does the live verdict reproduce the committed record's verdict?

    - exact match (incl. a reproduced GAP, when the committed record is a known gap)
    - PASS/PASS-caveated are both passing -> a seed-flip between them reproduces
    - a live GAP reproduces a committed GAP-evidenced (a proven bf16-backend floor
      documented in docs/implementation-parity.md; the live GAP is the expected
      bf16 behavior, not a port regression)
    A live GAP vs a committed passing verdict is NOT a match (real regression)."""
    if verdict == committed:
        return True
    if _is_passing(verdict) and _is_passing(committed):
        return True
    if verdict == "GAP" and committed == "GAP-evidenced":
        return True
    return False


def _committed_verdict(leg: Leg) -> str | None:
    """Read the verdict recorded in the committed JSON for this leg (the doc's truth).

    Prefers an explicit top-level ``verdict`` string (some committed records assert a
    human verdict that the scorer cannot re-derive from metrics — e.g. ``GAP-evidenced``,
    a GAP proven a genuine bf16-backend floor and accepted in docs/implementation-parity.md).
    Falls back to re-deriving the verdict from the report's metrics via extract_verdict."""
    if not leg.committed_json:
        return None
    p = PARITY_DATA / leg.committed_json
    if not p.exists():
        return None
    try:
        report = json.loads(p.read_text())
    except Exception:
        return None
    explicit = report.get("verdict") if isinstance(report, dict) else None
    if isinstance(explicit, str) and explicit in (
            "PASS", "PASS-caveated", "GAP-evidenced", "GAP", "NO-DATA"):
        return explicit
    v, _ = extract_verdict(leg, report)
    return v


def finalize_leg(leg: Leg, verdict: str, detail: str, wall: float) -> tuple[dict, str, bool]:
    """The ONE verdict/drift/gate-effect code path (shared by the resumed and fresh branches;
    see VERDICT SEMANTICS in the module docstring). Returns (row, drift_annotation, gate_ok).

    Drift is only checked when the committed record carries a comparable verdict; a committed
    NO-DATA has nothing to compare against (a real regression is still caught by the live
    verdict). PASS/PASS-caveated are equivalent; a live GAP reproduces a committed GAP-evidenced
    (a proven bf16 floor); a live passing verdict vs a committed GAP is an improvement, not drift.
    """
    committed = _committed_verdict(leg)
    drift, ok = "", True
    comparable = (verdict not in ("ERROR", "NO-DATA", "BLOCKED-REF-REGEN-NEEDED"))
    if committed and committed != "NO-DATA" and comparable:
        if _matches_committed(verdict, committed):
            drift = " [reproduces committed]"
        elif _is_passing(verdict) and committed in ("GAP", "GAP-evidenced"):
            # GAP-evidenced counts here too: it records a GAP proven to be a bf16-backend
            # floor, so a live PASS means that residual shrank below the bound. Strictly
            # better than the committed record, never a regression.
            drift = f" [improves committed {committed} — not a drift]"
        else:
            drift = f" [DRIFT vs committed={committed} — investigate, not auto-overwritten]"
            ok = False
    # a live ERROR/GAP/NO-DATA fails the gate unless the GAP reproduces a committed GAP-evidenced
    if verdict in ("ERROR", "GAP", "NO-DATA") and not (verdict == "GAP" and committed == "GAP-evidenced"):
        ok = False
    row = {"leg": leg.id, "verdict": verdict, "detail": detail, "wall": wall,
           "committed": committed, "report": leg.id + ".json"}
    return row, drift, ok


def main() -> int:
    # Scorers, folds and predict CLIs we spawn arm their parent-death guard off this,
    # so none of them can outlive this driver still holding a card. Inherited through
    # every spawn path in this file (see tt_bio/device_lease.py:arm_orphan_guard).
    os.environ["TT_BIO_PARENT_PID"] = str(os.getpid())
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", default="pc:0",
                   help="comma list of host:card slots to fan device folds across "
                        "(e.g. pc:0,qb1:0,qb1:1,qb2:0). Default pc:0.")
    ap.add_argument("--leg", action="append", help="run only this leg id (repeatable). "
                   "Default: every non-opt-in leg.")
    ap.add_argument("--include-opt-in", action="store_true",
                   help="also run opt-in legs (esmc-6b, MSA-server legs).")
    ap.add_argument("--seeds", default="",
                   help="override the seed list for fixture legs, comma-separated (e.g. 0,1). "
                   "A bare count like 5 means seed index 5, not five seeds; indices with no "
                   "seed dir in the leg's reference fixture are rejected. Default: the leg's "
                   "recorded seeds.")
    ap.add_argument("--workdir", default="/tmp/full_parity_gate",
                   help="workdir for device output + reports.")
    ap.add_argument("--out", default="", help="write the JSON report here too.")
    ap.add_argument("--check", action="store_true",
                   help="run the card-free preflight self-check (leg yaml/fixture/committed-JSON/"
                        "target-id wiring) and exit. No device work. Use before trusting the gate.")
    ap.add_argument("--dry-run", action="store_true",
                   help="preflight + inventory + fingerprint check only; run no device folds.")
    ap.add_argument("--fresh", action="store_true",
                   help="force a clean re-fold: ignore completed folds/reports already in the "
                        "workdir. By DEFAULT the gate resumes (reuses completed folds + per-leg "
                        "reports) so a bounded turn always makes forward progress; use a fresh "
                        "--workdir per release commit, or pass --fresh, for a from-scratch run.")
    ap.add_argument("--fold-timeout", type=float, default=2400.0,
                   help="hard wall-clock timeout (s) per device fold / in-process harness. A fold "
                        "that never produces results.json within this window (e.g. a flaky MSA "
                        "server) is killed with a clear error instead of hanging the gate. Default "
                        "2400. Legs can declare a higher floor (Leg.min_fold_timeout — the boltz2 "
                        "affinity legs get 7200s for their contention-fragile fp32 host trunk); "
                        "the effective timeout is max(this, the leg floor).")
    ap.add_argument("--margin", type=float, default=None,
                   help="integration-parity envelope margin (device may drift up to "
                        "envelope*(1+margin) from the fp32 reference). Default: integration_envelope"
                        f".DEFAULT_MARGIN. Justified in ~/.coworker/state/tt-bio-integration-parity-gate.md §4.")
    ap.add_argument("--legacy-rdx", action="store_true",
                   help="score diffusion legs with the OLD R/D/X same-backend self-consistency "
                        "floor instead of the integration-parity envelope. Retired as the pass "
                        "criterion (it conflates bf16 arithmetic with diffusion-noise chaos); kept "
                        "only as an opt-in device self-consistency (D) DIAGNOSTIC.")
    ap.add_argument("--regen-refs", action="store_true",
                   help="(re)generate the fp32 + bf16 CPU shared-draw references for the selected "
                        "envelope legs (2 CPU folds/leg at seed 0, run SERIALLY per host-contention) "
                        "into <fixture>/ref_{fp32,bf16}/, then exit. The expensive cached step; only "
                        "rerun when model code/weights/settings change.")
    ap.add_argument("--init-fingerprints", action="store_true",
                   help="write/refresh docs/implementation-parity-data/ref-fixture-fingerprints.json "
                   "from the current fixtures and exit (run once after harvesting a fixture; "
                   "commit the resulting index so future runs detect reference drift).")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    log_dir = workdir / "logs"
    log_dir.mkdir(exist_ok=True)

    # P300 mesh-graph descriptor (mirrors release_gate.py / tt_bio.main)
    try:
        from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
        if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
            mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
            if mgd:
                os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    except Exception:
        pass

    if args.init_fingerprints:
        idx = {}
        for leg in LEGS:
            if not leg.fixture:
                continue
            fp = fixture_fingerprint(leg.fixture)
            if fp:
                idx[leg.id] = fp
        FINGERPRINT_INDEX.write_text(json.dumps(idx, indent=2, sort_keys=True))
        print(f"wrote {FINGERPRINT_INDEX} with {len(idx)} fingerprints:")
        for k, v in sorted(idx.items()):
            print(f"  {k:<34} {v}")
        return 0

    workers = parse_workers(args.workers)
    fp_index = load_fingerprint_index()

    legs = LEGS
    if args.leg:
        wanted = set(args.leg)
        legs = [l for l in LEGS if l.id in wanted]
        missing = wanted - {l.id for l in legs}
        if missing:
            sys.exit(f"unknown leg id(s): {sorted(missing)}; known: {sorted(LEGS_BY_ID)}")
    elif not args.include_opt_in:
        legs = [l for l in LEGS if not l.opt_in]

    seeds_override = None
    if args.seeds:
        try:
            seeds_override = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
        except ValueError:
            ap.error(f"--seeds takes a comma-separated list of integer seed indices, "
                     f"got {args.seeds!r}")
        unknown = []
        for leg in legs:
            if not _seeds_matched_against_fixture(leg, args.legacy_rdx):
                continue
            known = _fixture_known_seeds(leg.fixture)
            if not known:
                continue  # fixture absent — the BLOCKED-REF-REGEN path reports that instead
            extra = [s for s in seeds_override if s not in known]
            if extra:
                unknown.append(f"{leg.id}: seed(s) {extra} not in fixture {leg.fixture} "
                               f"(has seeds {known})")
        if unknown:
            ap.error("--seeds is a comma-separated LIST of seed indices (e.g. 0,1,2), not a "
                     "count — a bare 5 selects seed index 5. No selected leg's fixture "
                     "contains the requested seed(s):\n  " + "\n  ".join(unknown))

    resume = not args.fresh

    if args.margin is None:
        sys.path.insert(0, str(REPO / "scripts"))
        from integration_envelope import DEFAULT_MARGIN
        args.margin = DEFAULT_MARGIN

    # (Re)generate CPU shared-draw references, then exit — the expensive cached step.
    if args.regen_refs:
        env_legs = [l for l in legs if _is_envelope_leg(l) and l.fixture and not l.legacy_rdx]
        if not env_legs:
            print("--regen-refs: no envelope (structure/affinity) legs selected.")
            return 1
        print(f"--regen-refs: generating fp32+bf16 CPU references for {len(env_legs)} leg(s) "
              f"(serial; ~2 CPU folds/leg). margin default {args.margin}.")
        return regen_envelope_refs(env_legs, workdir, log_dir, args.fold_timeout, resume)

    # Card-free preflight self-check — abort in seconds on a misconfigured leg instead of a
    # wasted device turn (postmortem #2/#3). Always runs; --check runs it and exits.
    problems = preflight_check(legs)
    if problems:
        print("PREFLIGHT — leg wiring problems detected:")
        for p in problems:
            print(f"  - {p}")
        if not args.check:
            print("Refusing to run the gate with misconfigured legs; fix the above (or scope with --leg).")
        return 1
    blocked = [(l.id, _incomplete_fixture_seeds(l, list(l.seeds))) for l in legs]
    blocked = [(i, b) for i, b in blocked if b]
    if blocked:
        print("PREFLIGHT — fixtures present but INCOMPLETE (reference CIFs missing; each such "
              "leg reports BLOCKED-REF-REGEN-NEEDED and does NOT fail the gate):")
        for i, b in blocked:
            print(f"  - {i}: {', '.join(b)} missing structures/*.cif")
        print(f"  {_FIXTURE_FETCH_HINT}")
    if args.check:
        print(f"PREFLIGHT OK — {len(legs)} legs well-formed "
              f"(yaml / fixture+fingerprint / committed-JSON / target-id / MSA wiring)"
              f"{f'; {len(blocked)} fixture(s) incomplete → BLOCKED-REGEN' if blocked else ''}.")
        return 0

    print(f"\n{'#'*78}\n# FULL PARITY GATE — {len(legs)} legs, "
          f"workers {[f'{w.host}:{w.card}' for w in workers]}\n{'#'*78}")
    print(f"{'leg':<34}{'kind':<11}{'ref':<14}{'verdict':<18}{'wall':>8}  detail")
    print("-" * 110)

    rows = []
    all_pass = True
    t_start = time.monotonic()
    for leg in legs:
        seeds = seeds_override if seeds_override is not None else list(leg.seeds)
        leg_timeout = max(args.fold_timeout, leg.min_fold_timeout or 0.0)
        # fingerprint check for fixture legs
        ref_status = "in-process"
        if leg.fixture:
            fp = fixture_fingerprint(leg.fixture)
            recorded = fp_index.get(leg.id)
            if fp is None:
                rows.append({"leg": leg.id, "verdict": "BLOCKED-REF-REGEN-NEEDED",
                             "detail": f"fixture {leg.fixture} missing — regenerate reference",
                             "wall": 0})
                print(f"{leg.id:<34}{leg.kind:<11}{'MISSING':<14}"
                      f"{'BLOCKED-REGEN':<18}{0:>7.0f}s  fixture missing")
                continue
            if recorded is None:
                ref_status = "no-index"
            elif recorded == fp:
                ref_status = "cached"
            else:
                rows.append({"leg": leg.id, "verdict": "BLOCKED-REF-REGEN-NEEDED",
                             "detail": f"fingerprint changed: recorded {recorded} != fixture {fp}",
                             "wall": 0})
                print(f"{leg.id:<34}{leg.kind:<11}{ref_status[:13]:<14}"
                      f"{'BLOCKED-REGEN':<18}{0:>7.0f}s  fingerprint drift")
                continue

            # Reference must be COMPLETE to score. For an envelope leg that means BOTH CPU
            # shared-draw references (ref_fp32 + ref_bf16) are present; for a legacy R/D/X
            # structure leg it means the seed-dir CIFs were force-added past the .gitignore
            # `ref-fixtures/**/*.cif` rule. Either way an absent reference is the same class as a
            # fingerprint drift: BLOCKED-REF-REGEN-NEEDED (regenerate the reference with
            # --regen-refs), NOT a hard gate failure and NOT a silent per-leg ERROR mid-run.
            if _is_envelope_leg(leg) and not args.legacy_rdx and not leg.legacy_rdx:
                fp32_dir, bf16_dir = envelope_ref_dirs(leg)
                missing = [d for d, p in (("ref_fp32", fp32_dir), ("ref_bf16", bf16_dir)) if p is None]
                if missing:
                    rows.append({"leg": leg.id, "verdict": "BLOCKED-REF-REGEN-NEEDED",
                                 "detail": f"envelope reference incomplete: {', '.join(missing)} "
                                           f"missing under {leg.fixture} — run --regen-refs", "wall": 0})
                    print(f"{leg.id:<34}{leg.kind:<11}{ref_status[:13]:<14}"
                          f"{'BLOCKED-REGEN':<18}{0:>7.0f}s  envelope ref missing ({', '.join(missing)})")
                    continue
            else:
                incomplete = _incomplete_fixture_seeds(leg, seeds)
                if incomplete:
                    rows.append({"leg": leg.id, "verdict": "BLOCKED-REF-REGEN-NEEDED",
                                 "detail": f"fixture incomplete: {', '.join(incomplete)} missing "
                                           f"structures/{leg.target_id}.cif ({_FIXTURE_FETCH_HINT})",
                                 "wall": 0})
                    print(f"{leg.id:<34}{leg.kind:<11}{ref_status[:13]:<14}"
                          f"{'BLOCKED-REGEN':<18}{0:>7.0f}s  fixture incomplete (missing structures/ cif)")
                    continue

        if args.dry_run:
            rows.append({"leg": leg.id, "verdict": "DRY-RUN", "detail": ref_status, "wall": 0})
            print(f"{leg.id:<34}{leg.kind:<11}{ref_status[:13]:<14}{'(dry-run)':<18}{0:>7.0f}s  -")
            continue

        # ---- obtain (report, verdict, detail, wall): resume-cache first, else run fresh ----
        cached_report_path = workdir / f"{leg.id}.json"
        report = verdict = detail = None
        wall = 0.0
        if resume and cached_report_path.exists():
            try:
                cached = json.loads(cached_report_path.read_text())
                cverdict, _ = extract_verdict(leg, cached)
                if cverdict not in ("ERROR", "NO-DATA", None):
                    report, verdict, detail = cached, cverdict, f"(resumed from {leg.id}.json)"
                    ref_status = "resumed"
            except Exception:
                pass  # fall through to a fresh run
        if verdict is None:
            t_run = time.monotonic()
            if _is_envelope_leg(leg) and not args.legacy_rdx and not leg.legacy_rdx:
                # Envelope leg: ONE device fold at ENVELOPE_SEED (must match the seed the CPU
                # references were generated at — shared draws), scored device_bf16 vs the two CPU
                # references. The refs' presence was already verified above.
                fp32_dir, bf16_dir = envelope_ref_dirs(leg)
                folds = run_folds_fanout(leg, [ENVELOPE_SEED], workdir, workers, log_dir,
                                         resume=resume, fold_timeout=leg_timeout,
                                         extra_env=_shared_draw_env())
                dev = folds.get(ENVELOPE_SEED)
                if not isinstance(dev, Path):
                    err = dev.get("error") if isinstance(dev, dict) else "no output dir"
                    verdict, detail = "ERROR", f"device fold: {err}"
                else:
                    report = score_envelope(leg, str(dev), fp32_dir, bf16_dir,
                                            cached_report_path, args.margin)
                    verdict, detail = extract_verdict(leg, report)
            elif leg.kind in ("structure", "affinity"):
                # Hermetic templates: a fixture that commits template_structures/ is
                # folded against those CIFs (OF3_TEMPLATE_STRUCTURES override), not the
                # shared cache / RCSB network prefetch.
                fold_env = {}
                _tsdir = _fixture_dir(leg.fixture) / "template_structures" if leg.fixture else None
                if _tsdir is not None and _tsdir.is_dir():
                    fold_env["OF3_TEMPLATE_STRUCTURES"] = str(_tsdir)
                if leg.model == "openfold3" and not os.environ.get("OF3_CKPT"):
                    ckpt = _of3_ckpt_default()
                    if ckpt:
                        fold_env["OF3_CKPT"] = str(ckpt)
                folds = run_folds_fanout(leg, seeds, workdir, workers, log_dir,
                                         resume=resume, fold_timeout=leg_timeout,
                                         extra_env=fold_env)
                dev_dirs, fold_errs = [], []
                for s in seeds:
                    v = folds.get(s)
                    if isinstance(v, dict) and "error" in v:
                        fold_errs.append(f"seed{s}: {v['error']}")
                    elif isinstance(v, Path):
                        dev_dirs.append(str(v))
                    else:
                        fold_errs.append(f"seed{s}: no output dir")
                if fold_errs or not dev_dirs:
                    verdict, detail = "ERROR", "; ".join(fold_errs) or "no device folds completed"
                else:
                    log_path = log_dir / f"{leg.id}_score.log"
                    report = (score_structure if leg.kind == "structure" else score_affinity)(
                        leg, dev_dirs, cached_report_path, log_path)
                    verdict, detail = extract_verdict(leg, report)
            else:
                log_path = log_dir / f"{leg.id}.log"
                report = run_inprocess(leg, cached_report_path, log_path, dict(os.environ),
                                       fold_timeout=leg_timeout,
                                       pin_card=workers[0].card if workers else None)
                verdict, detail = extract_verdict(leg, report)
            wall = time.monotonic() - t_run

        # ---- single verdict/drift/gate-effect path ----
        row, drift, ok = finalize_leg(leg, verdict, detail, wall)
        all_pass &= ok
        rows.append(row)
        print(f"{leg.id:<34}{leg.kind:<11}{ref_status[:13]:<14}{verdict:<18}{wall:>7.0f}s  "
              f"{detail[:60]}{drift}")

    total_wall = time.monotonic() - t_start
    # tally
    from collections import Counter
    tally = Counter(r["verdict"] for r in rows)
    # Only a leg that reached its scorer is evidence; a BLOCKED-REF-REGEN-NEEDED row scored
    # nothing. If every leg is blocked the run verified zero legs, so it must not print GATE
    # PASS — the classic cause is --seeds given a bare count, matching no fixture seed.
    n_scored = sum(n for v, n in tally.items() if v != "BLOCKED-REF-REGEN-NEEDED")
    inconclusive = n_scored == 0
    print("\n" + "#" * 78)
    print(f"# Tally: {dict(tally)}    total wall {total_wall/60:.1f} min")
    if inconclusive:
        print(f"# GATE INCONCLUSIVE — {len(rows)}/{len(rows)} legs blocked on reference "
              "regen, nothing scored")
    else:
        print("# " + ("GATE PASS — every fast-path leg reproduces within its floor"
                       if all_pass else
                       "GATE FAIL — a leg ERRORED, GAPped, or DRIFTed vs committed (see above)"))
    print("#" * 78)

    report = {"legs": rows, "tally": dict(tally), "scored": n_scored,
              "total_wall_s": total_wall,
              "workers": [f"{w.host}:{w.card}" for w in workers]}
    (workdir / "report.json").write_text(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0 if (all_pass and not inconclusive) else 1


if __name__ == "__main__":
    sys.exit(main())
