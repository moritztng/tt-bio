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
opt-in path and are reported separately).

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
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = REPO / "docs" / "implementation-parity-data" / "ref-fixtures"
FINGERPRINT_INDEX = REPO / "docs" / "implementation-parity-data" / "ref-fixture-fingerprints.json"
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
    note: str = ""


def _boltz2_struct_args(recycling=3, steps=200, samples=1, msa="none"):
    base = [f"--recycling_steps", str(recycling), "--sampling_steps", str(steps),
            "--diffusion_samples", str(samples)]
    if msa == "none":
        return tuple(base + ["--single_sequence"])
    if msa == "server":
        return tuple(base + ["--use_msa_server"])
    return tuple(base)  # staged: --msa_dir appended at run time


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

    # --- Protenix-v2 structure legs (cached fixture, device-only per release) ---
    Leg("protenix-prot-msa", "protenix-v2", "structure", "examples/prot.yaml",
        fixture="protenix-v2/prot/msa-server_200step_5sample_10cycle_bf16",
        committed_json="protenix-v2.json", target_id="prot",
        device_args=("--sampling_steps", "200", "--diffusion_samples", "5"),
        msa="staged"),
    Leg("protenix-ubq-msa", "protenix-v2", "structure", "examples/ubq.yaml",
        fixture="protenix-v2/ubq/msa-server_200step_5sample_10cycle_bf16",
        committed_json="protenix-v2-ubiquitin.json", target_id="ubq",
        device_args=("--sampling_steps", "200", "--diffusion_samples", "5"),
        msa="staged"),
    Leg("protenix-hsa-msa", "protenix-v2", "structure", "examples/hsa.yaml",
        fixture="protenix-v2/hsa/msa-server_200step_5sample_10cycle_bf16",
        committed_json="protenix-v2-hsa.json", target_id="hsa",
        device_args=("--sampling_steps", "200", "--diffusion_samples", "5"),
        msa="staged"),

    # --- Boltz-2 affinity legs (cached fixture, device-only per release) ---
] + [
    Leg(f"boltz2-affinity-fkbp12-nomsa", "boltz2", "affinity", "examples/affinity_fkg.yaml",
        fixture="boltz2/affinity_fkg/nomsa_200step_5affsample_3recycle_bf16_mwcorr",
        committed_json="boltz2-affinity-fkbp12-nomsa-seeded.json", target_id="affinity_fkg",
        device_args=("--single_sequence", "--affinity_mw_correction",
                      "--diffusion_samples_affinity", "5", "--sampling_steps_affinity", "200",
                      "--recycling_steps", "3", "--sampling_steps", "200",
                      "--diffusion_samples", "1")),
    Leg(f"boltz2-affinity-fkbp12-msa", "boltz2", "affinity", "examples/affinity_fkg_msa.yaml",
        fixture="boltz2/affinity_fkg/msa-colabfold_200step_5affsample_3recycle_bf16_mwcorr_gpu",
        committed_json="boltz2-affinity-fkbp12-msa-seeded.json", target_id="affinity_fkg",
        device_args=("--affinity_mw_correction", "--diffusion_samples_affinity", "5",
                      "--sampling_steps_affinity", "200", "--recycling_steps", "3",
                      "--sampling_steps", "200", "--diffusion_samples", "1"),
        msa="yaml"),
    Leg(f"boltz2-affinity-dhfr-nomsa", "boltz2", "affinity", "examples/affinity_dhfr.yaml",
        fixture="boltz2/affinity_dhfr/nomsa_200step_5affsample_3recycle_bf16_mwcorr",
        committed_json="boltz2-affinity-dhfr-seeded.json", target_id="affinity_dhfr",
        device_args=("--single_sequence", "--affinity_mw_correction",
                      "--diffusion_samples_affinity", "5", "--sampling_steps_affinity", "200",
                      "--recycling_steps", "3", "--sampling_steps", "200", "--diffusion_samples", "1")),
    Leg(f"boltz2-affinity-dhfr-msa", "boltz2", "affinity", "examples/affinity_dhfr_msa.yaml",
        fixture="boltz2/affinity_dhfr/msa-colabfold_200step_5affsample_3recycle_bf16_mwcorr_gpu",
        committed_json="boltz2-affinity-dhfr-seeded.json", target_id="affinity_dhfr",
        device_args=("--affinity_mw_correction", "--diffusion_samples_affinity", "5",
                      "--sampling_steps_affinity", "200", "--recycling_steps", "3",
                      "--sampling_steps", "200", "--diffusion_samples", "1"),
        msa="yaml"),
    Leg(f"boltz2-affinity-tryp-nomsa", "boltz2", "affinity", "examples/affinity_tryp.yaml",
        fixture="boltz2/affinity_tryp/nomsa_200step_5affsample_3recycle_bf16_mwcorr",
        committed_json="boltz2-affinity-tryp-seeded.json", target_id="affinity_tryp",
        device_args=("--single_sequence", "--affinity_mw_correction",
                      "--diffusion_samples_affinity", "5", "--sampling_steps_affinity", "200",
                      "--recycling_steps", "3", "--sampling_steps", "200", "--diffusion_samples", "1")),
    Leg(f"boltz2-affinity-tryp-msa", "boltz2", "affinity", "examples/affinity_tryp_msa.yaml",
        fixture="boltz2/affinity_tryp/msa-colabfold_200step_5affsample_3recycle_bf16_mwcorr_gpu",
        committed_json="boltz2-affinity-tryp-seeded.json", target_id="affinity_tryp",
        device_args=("--affinity_mw_correction", "--diffusion_samples_affinity", "5",
                      "--sampling_steps_affinity", "200", "--recycling_steps", "3",
                      "--sampling_steps", "200", "--diffusion_samples", "1"),
        msa="yaml"),
]

# OpenDDE structure legs + abag + boltzgen (append separately for clarity)
LEGS += [
    Leg("opendde-trpcage-nomsa", "opendde", "structure", "examples/trpcage_no_msa.yaml",
        fixture="opendde/trpcage/nomsa_4cycle_20step_1sample_fp32_reduced",
        committed_json="opendde.json", target_id="trpcage_no_msa",
        device_args=("--single_sequence", "--recycling_steps", "4", "--sampling_steps", "20", "--diffusion_samples", "1"),
        msa="none"),
    Leg("opendde-prot-prod", "opendde", "structure", "examples/prot_no_msa.yaml",
        fixture="opendde/prot/nomsa_10cycle_200step_1sample_fp32_prod",
        committed_json="opendde-prod-leg.json", target_id="prot_no_msa",
        device_args=("--single_sequence", "--recycling_steps", "10", "--sampling_steps", "200", "--diffusion_samples", "1"),
        msa="none"),
    Leg("opendde-abag", "opendde-abag", "abag", "examples/1ahw_abag.yaml",
        committed_json="opendde-abag-1ahw-irmsd.json", seeds=(0,),
        note="DockQ leg; reuses release_gate --model opendde-abag"),
    Leg("boltzgen", "boltzgen", "boltzgen", "examples/binder.yaml",
        committed_json="boltzgen.json", seeds=(0,),
        note="designability leg; reuses release_gate --model boltzgen"),
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
    identity = {
        "reference_impl": meta.get("reference_impl", ""),
        "reference_version": meta.get("reference_version", ""),
        "reference_commit": meta.get("reference_commit", ""),
        "settings": meta.get("settings", {}),
        "seeds": meta.get("seeds", []),
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
            if not src.exists():
                problems.append(f"{leg.id}: staged-MSA leg missing {src}")
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
    - "staged": copy <fixture>/msa.a3m to <workdir>/msa/<seqhash>.a3m, return --msa_dir.
    """
    if leg.msa != "staged":
        return None, []
    fdir = _fixture_dir(leg.fixture)
    src = fdir / "msa.a3m"
    if not src.exists():
        raise FileNotFoundError(f"staged-MSA leg {leg.id}: missing {src}")
    seq = _yaml_protein_seq(REPO / leg.yaml)
    if not seq:
        raise ValueError(f"staged-MSA leg {leg.id}: could not read sequence from {leg.yaml}")
    msa_dir = workdir / "msa"
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
    local_host = os.environ.get("HOSTNAME", "pc").split(".")[0]
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        host, _, rest = part.partition(":")
        card_str, _, rest2 = rest.partition(":")
        remote_cwd, _, remote_python = rest2.partition(":")
        out.append(Worker(host=host, card=int(card_str or 0),
                           is_local=(host == "pc" or host == local_host),
                           remote_cwd=remote_cwd or None,
                           remote_python=remote_python or None))
    return out or [Worker(host="pc", card=0, is_local=True)]


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
                     log_dir: Path, resume: bool = True, fold_timeout: float | None = None) -> dict:
    """Run one device fold per seed, fanned across workers; return {seed: out_dir_or_error}.

    Workers run in parallel; each worker runs its seeds serially (one device context per
    process). Local workers use poll-based reaping; remote workers rsync output back. With
    ``resume`` (default) a seed whose output already carries a results.json is reused, so a
    bounded turn never re-folds work a prior turn finished.
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
            wrapped = w.wrap(device_cmd(leg, s, out_dir, workdir), REPO, {})
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
def envelope_ref_dirs(leg: Leg) -> tuple[Path | None, Path | None]:
    """Locate the fp32 + bf16 CPU shared-draw reference result dirs for an envelope leg.

    Convention: ``<fixture>/ref_fp32/`` and ``<fixture>/ref_bf16/`` each hold the inner
    ``<model>_results_<id>/`` dir a fold writes (located by its results.json). Returns
    (fp32_inner, bf16_inner); a missing side is None (leg -> BLOCKED-REF-REGEN-NEEDED)."""
    base = _fixture_dir(leg.fixture)
    return _find_results_dir(base / "ref_fp32"), _find_results_dir(base / "ref_bf16")


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
            "yaml": leg.yaml, "model": leg.model, "target_id": leg.target_id, "msa": leg.msa}


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
        if not _is_envelope_leg(leg) or not leg.fixture:
            continue
        base = _fixture_dir(leg.fixture)
        base.mkdir(parents=True, exist_ok=True)
        leg_ok = True
        for dtype, env in (("fp32", {}), ("bf16", {"TT_BIO_REF_BF16": "1"})):
            out_dir = base / f"ref_{dtype}"
            if resume and _find_results_dir(out_dir) is not None:
                print(f"  {leg.id} ref_{dtype}: cached, skip")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = device_cmd(leg, ENVELOPE_SEED, out_dir, workdir) + ["--accelerator", "cpu", "--no_kernels"]
            wrapped = local.wrap(cmd, REPO, env)
            logf = open(log_dir / f"regen_{leg.id}_{dtype}.log", "w")
            t0 = time.monotonic()
            try:
                rc, timed_out = _run_local_fold(wrapped, out_dir, logf, fold_timeout)
            finally:
                logf.close()
            wall = time.monotonic() - t0
            ok = (rc == 0 and _find_results_dir(out_dir) is not None)
            print(f"  {leg.id} ref_{dtype}: {'OK' if ok else 'FAILED'} ({wall/60:.1f} min)"
                  + ("" if ok else f" rc={rc} timed_out={timed_out} — see regen_{leg.id}_{dtype}.log"))
            leg_ok &= ok
        if leg_ok:
            meta = {"reference_impl": "tt-bio-cpu-torch", "reference_version": _repo_commit(),
                    "reference_commit": _repo_commit(), "settings": _ref_settings(leg),
                    "seeds": [ENVELOPE_SEED]}
            (base / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
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
                  fold_timeout: float | None = None) -> dict | None:
    """Run the dedicated harness for esmc/saprot/esmfold2 (subprocess) or the in-process
    designability/DockQ leg for boltzgen/abag. Persists the report to out_json (for --resume)."""
    if leg.kind in ("boltzgen", "abag"):
        try:
            rg = _load_release_gate()
            row = (rg.run_boltzgen(rg._load_designability_harness(), keep=False)
                   if leg.kind == "boltzgen" else rg.run_opendde_abag(keep=False))
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        out_json.write_text(json.dumps(row, indent=2, default=str))
        return row

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
    else:
        return None
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


def _envelope_verdict_row(report: dict) -> tuple[str, str]:
    """Verdict for an integration_envelope report: PASS iff every metric is within the measured
    bf16 envelope; else GAP (a real residual exceeding the envelope — to hunt, not excuse). The
    detail line lists the worst per-metric numerator/envelope ratio so a GAP is legible."""
    metrics = report.get("metrics", {})
    if not metrics:
        return "NO-DATA", "no envelope metrics scored"
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
        elif _is_passing(verdict) and committed == "GAP":
            drift = " [improves committed GAP — not a drift]"
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
                   help="override seed list for fixture legs, comma-separated (e.g. 0,1). "
                   "Default: the leg's recorded 5 seeds.")
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
                        "server) is killed with a clear error instead of hanging the gate. Default 2400.")
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
        seeds_override = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    resume = not args.fresh

    if args.margin is None:
        sys.path.insert(0, str(REPO / "scripts"))
        from integration_envelope import DEFAULT_MARGIN
        args.margin = DEFAULT_MARGIN

    # (Re)generate CPU shared-draw references, then exit — the expensive cached step.
    if args.regen_refs:
        env_legs = [l for l in legs if _is_envelope_leg(l) and l.fixture]
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
    if args.check:
        blocked = [(l.id, _incomplete_fixture_seeds(l, list(l.seeds))) for l in legs]
        blocked = [(i, b) for i, b in blocked if b]
        if blocked:
            print("PREFLIGHT — fixtures present but INCOMPLETE (reference CIFs missing; each such "
                  "leg reports BLOCKED-REF-REGEN-NEEDED and does NOT fail the gate):")
            for i, b in blocked:
                print(f"  - {i}: {', '.join(b)} missing structures/*.cif")
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
            if _is_envelope_leg(leg) and not args.legacy_rdx:
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
                                           f"structures/{leg.target_id}.cif (CIFs gitignored — force-add or regen)",
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
            if _is_envelope_leg(leg) and not args.legacy_rdx:
                # Envelope leg: ONE device fold at ENVELOPE_SEED (must match the seed the CPU
                # references were generated at — shared draws), scored device_bf16 vs the two CPU
                # references. The refs' presence was already verified above.
                fp32_dir, bf16_dir = envelope_ref_dirs(leg)
                folds = run_folds_fanout(leg, [ENVELOPE_SEED], workdir, workers, log_dir,
                                         resume=resume, fold_timeout=args.fold_timeout)
                dev = folds.get(ENVELOPE_SEED)
                if not isinstance(dev, Path):
                    err = dev.get("error") if isinstance(dev, dict) else "no output dir"
                    verdict, detail = "ERROR", f"device fold: {err}"
                else:
                    report = score_envelope(leg, str(dev), fp32_dir, bf16_dir,
                                            cached_report_path, args.margin)
                    verdict, detail = extract_verdict(leg, report)
            elif leg.kind in ("structure", "affinity"):
                folds = run_folds_fanout(leg, seeds, workdir, workers, log_dir,
                                         resume=resume, fold_timeout=args.fold_timeout)
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
                                       fold_timeout=args.fold_timeout)
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
    print("\n" + "#" * 78)
    print(f"# Tally: {dict(tally)}    total wall {total_wall/60:.1f} min")
    print("# " + ("GATE PASS — every fast-path leg reproduces within its floor"
                   if all_pass else
                   "GATE FAIL — a leg ERRORED, GAPped, or DRIFTed vs committed (see above)"))
    print("#" * 78)

    report = {"legs": rows, "tally": dict(tally), "total_wall_s": total_wall,
              "workers": [f"{w.host}:{w.card}" for w in workers]}
    (workdir / "report.json").write_text(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
