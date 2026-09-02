"""What the UX gate would still catch, and the checks that had stopped catching anything.

`scripts/ux_regression.py` is the third release-gate leg of RELEASING.md: accuracy is
`release_gate.py`, speed is `perf_regression.py`, and this one asserts the user-facing
plumbing still works. The live progress view advances through every phase, the written
files parse, the CLI lists what it ships. Its own docstring names the incident it exists
for: nesso1, openbind and pxdesign all reached 0.7.0 with zero UX coverage because the
gated set was three hardcoded lists. It has guarded every release since with no test of
its own.

Two defects found writing these tests, both fixed in the script and pinned below.

`_assert_full_model_coverage` read its covered set from a second expression naming the
same tuples the runner table names, so a model added to that expression and to nothing
else counted as covered and never ran. That is the same silence the check exists to
break. It reads `ALL_LEGS` now, which is built from `RUNNERS`, so covered means a runner
runs it.

`_check_cif` called itself a strict parse and was not one. It wrapped the parse in
`simplefilter("error", PDBConstructionWarning)`, but `MMCIFParser(QUIET=True)` installs
an ignore filter for exactly that category inside its own `catch_warnings`, and the inner
filter wins, so the escalation had never once fired. Switching it on is not the fix
either: 2 of the 15 valid deposited structures in this repo warn "Chain X is
discontinuous", so warnings-as-errors would fail the Ab-Ag leg on correct output. The
dead wrapper is gone and the parse says what it does. What actually catches the historical
missing-`_atom_site.occupancy` bug is the KeyError, and that is pinned here.

Two checks hardened. `_check_progress` gated `total > 0` on the trunk phase and not on
diffusion, so a diffusion bar stuck at 0 steps read as a pass; every diffusion emitter in
the tree (boltz2, protenix, rf3, openfold3, esmfold2) passes a positive total, so the
symmetric check is now there. And `LEGS_EXEMPT` entries are linted for a structural
reason, so the dict cannot quietly become a parking lot for "the box was down in August"
(memory: transient-reason-in-structural-exemption-dict).

Hermetic: no card, no fold, no network, no weights. The tests that read the real tree on
purpose are named `live_`.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "ux_regression.py"


@pytest.fixture()
def mod():
    """A fresh module per test, so a monkeypatched global cannot leak between them."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("ux_regression", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


# ── fixtures: a progress stream the way tt_bio.progress.make_progress_fn writes one ──

def stage_ev(stage: str, step: int = 0, total: int = 0) -> dict:
    return {"dev": 0, "worker": "0", "event": "stage", "stage": stage,
            "step": step, "total": total}


def fold_events(*, trunk=3, diffusion=4, trunk_total=None, diffusion_total=None,
                status="ok", done=True) -> list[dict]:
    """A healthy fold's event stream: start, msa, trunk ticks, diffusion ticks, done."""
    tt = trunk if trunk_total is None else trunk_total
    dt = diffusion if diffusion_total is None else diffusion_total
    evs: list[dict] = [{"event": "start", "name": "trpcage"}, stage_ev("msa")]
    evs += [stage_ev("trunk", i, tt) for i in range(trunk)]
    evs += [stage_ev("diffusion", k, dt) for k in range(diffusion)]
    evs += [stage_ev("confidence"), stage_ev("saving")]
    if done:
        evs.append({"event": "done", "name": "trpcage", "status": status, "time": 12.0})
    return evs


def drop_stage(events: list[dict], name: str) -> list[dict]:
    return [e for e in events if e.get("stage") != name]


# A real tt-bio mmCIF, trimmed to one residue. Produced by
# MolecularComplex.to_mmcif() (tt_bio/_vendor/esm/.../molecular_complex.py), which is what
# `tt-bio predict --model esmfold2` writes, so the column set is the shipped one.
CIF_TEXT = """data_trpcage
#
loop_
_atom_site.group_PDB
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.B_iso_or_equiv
_atom_site.occupancy
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
_atom_site.id
ATOM N C  . ALA A 1 1 . 1 ALA A C  9000.0 1.0 0.0 0.0 0.0 1 1
ATOM C CA . ALA A 1 1 . 1 ALA A CA 9000.0 1.0 1.5 0.0 0.0 1 2
#
_entity.id                 1
_entity.type               polymer
_entity.pdbx_description   'Polymer entity 1 (protein)'
#
_struct_asym.id          A
_struct_asym.entity_id   1
#
"""


def cif_without_occupancy() -> str:
    """The 17aeab9e regression: biotite omitted the column entirely when no occupancy
    annotation was set, and every reader that wants it raises KeyError."""
    lines = [l for l in CIF_TEXT.splitlines() if l.strip() != "_atom_site.occupancy"]
    out = []
    for l in lines:
        if l.startswith("ATOM "):
            f = l.split()
            del f[15]          # the occupancy value, in the same position as its tag
            l = " ".join(f)
        out.append(l)
    return "\n".join(out) + "\n"


# ── leg 1: the live progress view (the must-never-recur jump class) ──────────

def test_a_healthy_fold_stream_passes(mod):
    assert mod._check_progress(fold_events()) == []


def test_no_events_at_all_is_caught(mod):
    assert "no progress events captured" in mod._check_progress([])[0]


def test_the_trunk_phase_missing_is_caught(mod):
    """0 -> diffusion: the headline bug the predict-progress-fix work closed."""
    problems = mod._check_progress(drop_stage(fold_events(), "trunk"))
    assert any("trunk phase MISSING" in p for p in problems)


def test_a_stream_that_opens_on_diffusion_is_caught(mod):
    """loading -> diffusion: the bar jumps to the last phase and never shows the rest."""
    events = drop_stage(drop_stage(fold_events(), "trunk"), "msa")
    problems = mod._check_progress(events)
    assert any("first stage event is 'diffusion'" in p for p in problems)


def test_trunk_emitted_after_diffusion_is_caught(mod):
    """A trunk event that arrives late does not stop the view jumping past the phase."""
    events = ([{"event": "start", "name": "trpcage"}]
              + [stage_ev("diffusion", k, 4) for k in range(4)]
              + [stage_ev("trunk", i, 3) for i in range(3)]
              + [{"event": "done", "name": "trpcage", "status": "ok"}])
    problems = mod._check_progress(events)
    assert any("trunk not before diffusion" in p for p in problems)


def test_trunk_total_zero_on_every_tick_is_caught(mod):
    problems = mod._check_progress(fold_events(trunk_total=0))
    assert any("0 trunk iterations" in p for p in problems)


def test_diffusion_total_zero_on_every_tick_is_caught(mod):
    """Same defect as a flat trunk bar, one phase along. Every diffusion emitter in the
    tree passes a positive total, so a zero here is a regression, not a config."""
    problems = mod._check_progress(fold_events(diffusion_total=0))
    assert any("diffusion" in p and "total=0" in p for p in problems)


def test_a_missing_done_event_is_caught(mod):
    problems = mod._check_progress(fold_events(done=False))
    assert any("no 'done' event" in p for p in problems)


def test_a_failed_done_event_is_caught(mod):
    problems = mod._check_progress(fold_events(status="failed"))
    assert any("status=ok" in p for p in problems)


def test_non_monotonic_ticks_are_caught(mod):
    """One end-of-phase event instead of per-iteration ticking leaves steps out of order."""
    events = fold_events()
    trunk_idx = [i for i, e in enumerate(events) if e.get("stage") == "trunk"]
    events[trunk_idx[0]], events[trunk_idx[-1]] = events[trunk_idx[-1]], events[trunk_idx[0]]
    problems = mod._check_progress(events)
    assert any("not monotonic" in p for p in problems)


def test_load_events_skips_blank_and_malformed_lines(mod, tmp_path):
    cap = tmp_path / "events.jsonl"
    cap.write_text('{"event": "start"}\n\n  \nnot json at all\n{"event": "done"}\n')
    assert mod._load_events(cap) == [{"event": "start"}, {"event": "done"}]


# ── leg 1, the other three shapes: design, boltzgen, scalar affinity ─────────

GEN_STDOUT = """
>>> [1/6] design
    trunk 1/4
    diff 1/8
<<< ✓

>>> [2/6] inverse_folding
    batch 1/1
<<< ✓

>>> [3/6] folding
    trunk 1/4
<<< ✓

>>> [4/6] design_folding
    trunk 1/4
<<< ✓

>>> [5/6] analysis
<<< ✓

>>> [6/6] filtering
<<< ✓
"""


def test_a_healthy_boltzgen_stage_stream_passes(mod):
    assert mod._check_gen_progress(GEN_STDOUT) == []


def test_boltzgen_with_no_stage_lines_is_caught(mod):
    problems = mod._check_gen_progress("designing...\ndone\n")
    assert any("no `>>> [i/N] <step>` stage-start lines" in p for p in problems)


def test_boltzgen_skipping_the_refold_stage_is_caught(mod):
    stdout = "\n".join(l for l in GEN_STDOUT.splitlines()
                       if "design_folding" not in l and "] folding" not in l)
    problems = mod._check_gen_progress(stdout)
    assert any("no refold stage" in p for p in problems)


def test_boltzgen_skipping_analysis_is_caught(mod):
    stdout = "\n".join(l for l in GEN_STDOUT.splitlines() if "analysis" not in l)
    problems = mod._check_gen_progress(stdout)
    assert any("'analysis' stage MISSING" in p for p in problems)


def test_boltzgen_with_no_sub_step_ticks_is_caught(mod):
    """Stages that start and finish with nothing in between: the bar never moves."""
    stdout = "\n".join(l for l in GEN_STDOUT.splitlines()
                       if not l.startswith("    "))
    problems = mod._check_gen_progress(stdout)
    assert any("no sub-step tick lines" in p for p in problems)


def test_boltzgen_with_no_stage_done_lines_is_caught(mod):
    stdout = "\n".join(l for l in GEN_STDOUT.splitlines() if not l.startswith("<<<"))
    problems = mod._check_gen_progress(stdout)
    assert any("no `<<< ✓` stage-done lines" in p for p in problems)


RFD3_LINE = r"^\s*\S+#\d+:\s+\S+\.cif\s+\(\d+ atoms\)"
DESIGN_STDOUT = ("Designing 1 spec(s) × 1 design(s) → /tmp/out (rfd3)\n"
                 "  iai_inputs#0: /tmp/out/iai_inputs_0.cif (612 atoms)\n"
                 "Done — 1 design(s) → /tmp/out\n")


def test_a_healthy_design_stdout_passes(mod):
    assert mod._check_design_progress(DESIGN_STDOUT, RFD3_LINE) == []


def test_design_without_a_per_design_line_is_caught(mod):
    """Start then finish with nothing in between is the design-shaped version of the
    bar jumping a phase."""
    stdout = "\n".join(l for l in DESIGN_STDOUT.splitlines() if "#0:" not in l)
    problems = mod._check_design_progress(stdout, RFD3_LINE)
    assert any("no per-design result line" in p for p in problems)


def test_design_without_a_done_line_is_caught(mod):
    stdout = "\n".join(l for l in DESIGN_STDOUT.splitlines() if not l.startswith("Done"))
    problems = mod._check_design_progress(stdout, RFD3_LINE)
    assert any("no 'Done" in p for p in problems)


def test_design_reporting_a_result_before_it_starts_is_caught(mod):
    lines = DESIGN_STDOUT.splitlines()
    stdout = "\n".join([lines[1], lines[0], lines[2]])
    problems = mod._check_design_progress(stdout, RFD3_LINE)
    assert any("before the 'Designing" in p for p in problems)


def test_the_pxdesign_per_design_line_matches_its_own_pattern(mod):
    """The two design legs differ only in this pattern, so each one has to be checked
    against the line its model actually prints."""
    px = mod.DESIGN_LEGS["pxdesign"]["design_line"]
    stdout = ("Designing 1 spec(s) × 1 design(s) → /tmp/out (pxdesign)\n"
              "  PDL1_0.cif: 196 residues, 1544 atoms; target fit 0.87 A over 116 "
              "conditioned tokens\n"
              "Done — 1 design(s) → /tmp/out\n")
    assert mod._check_design_progress(stdout, px) == []


AFFINITY_STDOUT = ("Loading nesso1 (tenstorrent, trunk bf16) …\n"
                   "  fkg: affinity -8.123 p(binder) 0.912 (256 tokens, 41.2s)\n"
                   "Done — 1/1 scored → /tmp/out/affinity.csv\n")


def test_a_healthy_scalar_affinity_stdout_passes(mod):
    assert mod._check_scalar_affinity_progress(AFFINITY_STDOUT) == []


def test_scalar_affinity_that_prints_nothing_until_the_end_is_caught(mod):
    problems = mod._check_scalar_affinity_progress(
        "Done — 1/1 scored → /tmp/out/affinity.csv\n")
    assert any("missing loading" in p for p in problems)


def test_scalar_affinity_phases_out_of_order_are_caught(mod):
    lines = AFFINITY_STDOUT.splitlines()
    problems = mod._check_scalar_affinity_progress("\n".join([lines[0], lines[2], lines[1]]))
    assert any("out of order" in p for p in problems)


# ── leg 2: the written files parse ──────────────────────────────────────────

def test_a_real_tt_bio_cif_parses(mod, tmp_path):
    cif = tmp_path / "trpcage_model_0.cif"
    cif.write_text(CIF_TEXT)
    assert mod._check_cif(cif) == []


def test_a_cif_without_occupancy_is_caught(mod, tmp_path):
    """The 17aeab9e class. Bio.PDB hard-requires _atom_site.occupancy, so the writer
    regression surfaces as a parse failure, not as a warning."""
    cif = tmp_path / "trpcage_model_0.cif"
    cif.write_text(cif_without_occupancy())
    problems = mod._check_cif(cif)
    assert problems and "occupancy" in problems[0]


def test_an_empty_cif_is_caught(mod, tmp_path):
    cif = tmp_path / "trpcage_model_0.cif"
    cif.write_text("data_empty\n#\n")
    problems = mod._check_cif(cif)
    assert problems and ("0 atoms" in problems[0] or "parse failed" in problems[0])


def test_a_truncated_cif_is_caught(mod, tmp_path):
    cif = tmp_path / "trpcage_model_0.cif"
    cif.write_text(CIF_TEXT[:len(CIF_TEXT) // 2])
    assert mod._check_cif(cif) != []


def test_live_a_construction_warning_does_not_fail_the_parse(mod):
    """Pinning the QUIET=True decision with a file that actually warns. A discontinuous
    chain is a PDBConstructionWarning, and 9dsg is one of the two deposited structures in
    examples/ that raises one, so escalating warnings would fail the gate on correct
    output. The control is the second half: if 9dsg ever stops warning this test stops
    testing anything, so it fails rather than passing vacuously."""
    import warnings as W

    from Bio.PDB import MMCIFParser
    from Bio.PDB.PDBExceptions import PDBConstructionWarning

    cif = REPO / "examples" / "ground_truth_structures" / "9dsg.cif"
    if not cif.exists():
        pytest.skip(f"{cif} is not in this tree")
    with W.catch_warnings(record=True) as caught:
        W.simplefilter("always")
        MMCIFParser(QUIET=False).get_structure("x", str(cif))
    assert any(issubclass(c.category, PDBConstructionWarning) for c in caught), \
        "9dsg no longer warns, so this test no longer pins anything"
    assert mod._check_cif(cif) == []


def _npz(tmp_path, seq="NLYIQWLKDGGPSSGRPPPS", **over):
    import numpy as np
    arrays = {"per_residue": np.zeros((len(seq), 8), dtype=np.float32),
              "pooled": np.zeros((8,), dtype=np.float32),
              "sequence": np.array(seq)}
    arrays.update(over)
    path = tmp_path / "tiny.npz"
    np.savez(path, **{k: v for k, v in arrays.items() if v is not None})
    return path


def test_a_healthy_npz_passes(mod, tmp_path):
    assert mod._check_npz(_npz(tmp_path), "NLYIQWLKDGGPSSGRPPPS") == []


def test_an_npz_missing_pooled_is_caught(mod, tmp_path):
    problems = mod._check_npz(_npz(tmp_path, pooled=None), "NLYIQWLKDGGPSSGRPPPS")
    assert problems and "missing arrays ['pooled']" in problems[0]


def test_an_npz_whose_length_disagrees_with_the_sequence_is_caught(mod, tmp_path):
    import numpy as np
    path = _npz(tmp_path, per_residue=np.zeros((3, 8), dtype=np.float32))
    problems = mod._check_npz(path, "NLYIQWLKDGGPSSGRPPPS")
    assert problems and "per_residue L=3" in problems[0]


def _results(tmp_path, rows) -> Path:
    path = tmp_path / "results.json"
    path.write_text(json.dumps(rows))
    return path


def test_a_healthy_results_json_passes(mod, tmp_path):
    assert mod._check_results_json(
        _results(tmp_path, [{"id": "trpcage", "status": "ok", "plddt": 0.91}])) == []


def test_results_json_with_no_ok_row_is_caught(mod, tmp_path):
    problems = mod._check_results_json(
        _results(tmp_path, [{"id": "trpcage", "status": "failed"}]))
    assert problems and "no ok row" in problems[0]


def test_results_json_with_no_confidence_metric_is_caught(mod, tmp_path):
    """The CLI summary and the platform both read a confidence number off this row."""
    problems = mod._check_results_json(_results(tmp_path, [{"id": "t", "status": "ok"}]))
    assert problems and "no confidence metric" in problems[0]


def test_results_json_that_is_not_a_list_is_caught(mod, tmp_path):
    problems = mod._check_results_json(_results(tmp_path, {"id": "t", "status": "ok"}))
    assert problems and "not a non-empty list" in problems[0]


def test_affinity_results_without_the_scalar_are_caught(mod, tmp_path):
    """affinity_pred_value is the whole point of affinity mode; a fold-shaped row that
    passes the generic check still has to carry it."""
    path = _results(tmp_path, [{"id": "fkg", "status": "ok", "plddt": 0.9}])
    assert mod._check_results_json(path) == []
    problems = mod._check_affinity_results(path)
    assert problems and "no affinity_pred_value" in problems[0]


def test_affinity_results_with_the_scalar_pass(mod, tmp_path):
    path = _results(tmp_path, [{"id": "fkg", "status": "ok", "plddt": 0.9,
                                "affinity_pred_value": -8.1}])
    assert mod._check_affinity_results(path) == []


def _manifest(tmp_path, **over) -> Path:
    m = {"model": "esmc-600m", "pool": "mean", "format": "npz", "d_model": 1152,
         "dtype": "float32", "sequences": [{"id": "tiny", "length": 20}]}
    m.update(over)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(m))
    return path


def test_a_healthy_manifest_passes(mod, tmp_path):
    assert mod._check_manifest(_manifest(tmp_path), "tiny", "N" * 20) == []


def test_a_manifest_missing_d_model_is_caught(mod, tmp_path):
    m = json.loads(_manifest(tmp_path).read_text())
    del m["d_model"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(m))
    problems = mod._check_manifest(path, "tiny", "N" * 20)
    assert problems and "missing keys ['d_model']" in problems[0]


def test_a_manifest_that_lost_the_sequence_is_caught(mod, tmp_path):
    path = _manifest(tmp_path, sequences=[])
    problems = mod._check_manifest(path, "tiny", "N" * 20)
    assert problems and "don't include tiny" in problems[0]


def _metrics_csv(tmp_path, cols, name="aggregate_metrics_all.csv") -> Path:
    d = tmp_path / "designs"
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    path.write_text(",".join(cols) + "\n" + ",".join("0" for _ in cols) + "\n")
    return path


def test_healthy_gen_metrics_pass(mod, tmp_path):
    _metrics_csv(tmp_path, ["name", "designfolding-bb_rmsd", "plddt"])
    assert mod._check_gen_metrics(tmp_path) == []


def test_gen_metrics_without_a_designability_column_are_caught(mod, tmp_path):
    _metrics_csv(tmp_path, ["name", "plddt"])
    problems = mod._check_gen_metrics(tmp_path)
    assert problems and "no designability RMSD column" in problems[0]


def test_missing_gen_metrics_are_caught(mod, tmp_path):
    problems = mod._check_gen_metrics(tmp_path)
    assert problems and "analysis did not run" in problems[0]


def _scalar_affinity_out(tmp_path, *, json_row=None, csv_row=None) -> Path:
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    row = {"id": "affinity_fkg", "n_tokens": 256, "seconds": 41.2,
           "affinity_pred_value": -8.123, "affinity_probability_binary": 0.912}
    if json_row is not None:
        row = json_row
    (out / "affinity_fkg_affinity.json").write_text(json.dumps(row))
    cols = ["id", "n_tokens", "seconds", "affinity_pred_value",
            "affinity_probability_binary", "error"]
    vals = csv_row or ["affinity_fkg", "256", "41.2", "-8.123", "0.912", ""]
    (out / "affinity.csv").write_text(",".join(cols) + "\n" + ",".join(vals) + "\n")
    return out


def test_healthy_scalar_affinity_files_pass(mod, tmp_path):
    out = _scalar_affinity_out(tmp_path)
    assert mod._check_scalar_affinity_results(out, "affinity_fkg") == []


def test_scalar_affinity_json_missing_the_scalar_is_caught(mod, tmp_path):
    out = _scalar_affinity_out(tmp_path, json_row={"id": "affinity_fkg", "n_tokens": 256,
                                                   "seconds": 41.2})
    problems = mod._check_scalar_affinity_results(out, "affinity_fkg")
    assert any("missing keys" in p and "affinity_pred_value" in p for p in problems)


def test_scalar_affinity_csv_row_that_errored_is_caught(mod, tmp_path):
    out = _scalar_affinity_out(
        tmp_path, csv_row=["affinity_fkg", "", "", "", "", "ligand parse failed"])
    problems = mod._check_scalar_affinity_results(out, "affinity_fkg")
    assert any("errored" in p for p in problems)
    assert any("no affinity_pred_value" in p for p in problems)


# ── leg 3: the CLI behaves ──────────────────────────────────────────────────

def _help(flags, choices=()) -> str:
    body = "usage: tt-bio ...\n\noptions:\n"
    body += "".join(f"  {f} VALUE\n" for f in flags)
    if choices:
        body += "  --model {%s}\n" % ",".join(choices)
    return body


def cli_table(m) -> dict:
    """The help output of a healthy CLI, built from the CLI's own tuples so the fixture
    cannot drift away from what the gate reads."""
    tb = m.tt_bio_main
    return {
        ("predict", "--help"): (0, _help(
            ["--model", "--sampling_steps", "--diffusion_samples", "--recycling_steps",
             "--single_sequence", "--out_dir", "--seed"], m.FOLD_MODELS), ""),
        ("embed", "--help"): (0, _help(
            ["--model", "--format", "--out_dir", "--pool"], tb.EMBED_MODELS), ""),
        ("saprot", "--help"): (0, _help(
            ["--model", "--format", "--out_dir", "--pool", "--structure"],
            tb.SAPROT_MODELS), ""),
        ("gen", "run", "--help"): (0, _help(
            ["--num_designs", "--protocol", "--output", "--devices", "--budget"]),
            "`tt-bio gen` is deprecated; use `tt-bio design --model boltzgen`\n"),
        ("design", "--help"): (0, _help(
            ["--model", "--out_dir", "--num_designs", "--devices", "--seed", "--from_pdb",
             "--num_timesteps", "--batch_size", "--checkpoint", "--protocol", "--steps",
             "--budget", "--n_step"], tb.DESIGN_MODELS), ""),
        ("affinity", "--help"): (0, _help(
            ["--model", "--out_dir", "--accelerator", "--trunk", "--recycling_steps",
             "--tokens_budget", "--devices", "--seed"], tb.AFFINITY_MODELS), ""),
        ("--help",): (0, "usage: tt-bio ...\n\ncommands:\n  predict\n  embed\n  saprot\n"
                         "  design\n  affinity\n  weights\n", ""),
    }


def install_run(m, monkeypatch, table):
    """Replace the module's subprocess helper with a lookup on `table`, keyed by the
    tt_bio.main argv tail. Everything the gate learns about the CLI comes through here."""
    def fake(cmd, *, env=None, timeout=None, cwd=None):
        if cmd[1:2] == ["-c"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        key = tuple(cmd[3:])
        rc, out, err = table[key]
        return subprocess.CompletedProcess(cmd, rc, out, err)
    monkeypatch.setattr(m, "_run", fake)


def test_a_healthy_cli_passes(mod, monkeypatch):
    install_run(mod, monkeypatch, cli_table(mod))
    assert mod._check_cli() == []


def test_a_fold_model_dropped_from_the_predict_choices_is_caught(mod, monkeypatch):
    """The other half of the nesso1/openbind/pxdesign incident: a model that ships but
    that no user can type."""
    table = cli_table(mod)
    dropped = mod.FOLD_MODELS[-1]
    kept = [x for x in mod.FOLD_MODELS if x != dropped]
    table[("predict", "--help")] = (0, _help(
        ["--model", "--sampling_steps", "--diffusion_samples", "--recycling_steps",
         "--single_sequence", "--out_dir", "--seed"], kept), "")
    install_run(mod, monkeypatch, table)
    problems = mod._check_cli()
    assert any(f"does not list --model choice {dropped}" in p for p in problems)


def test_a_dropped_predict_flag_is_caught(mod, monkeypatch):
    table = cli_table(mod)
    rc, out, err = table[("predict", "--help")]
    table[("predict", "--help")] = (rc, out.replace("  --single_sequence VALUE\n", ""), err)
    install_run(mod, monkeypatch, table)
    assert any("missing flag --single_sequence" in p for p in mod._check_cli())


def test_a_help_that_exits_nonzero_is_caught(mod, monkeypatch):
    table = cli_table(mod)
    table[("embed", "--help")] = (2, "", "boom")
    install_run(mod, monkeypatch, table)
    assert any("embed --help exited 2" in p for p in mod._check_cli())


def test_the_deprecated_gen_alias_must_still_warn(mod, monkeypatch):
    table = cli_table(mod)
    rc, out, _ = table[("gen", "run", "--help")]
    table[("gen", "run", "--help")] = (rc, out, "")
    install_run(mod, monkeypatch, table)
    assert any("no deprecation warning" in p for p in mod._check_cli())


def test_the_deprecated_gen_alias_must_stay_hidden(mod, monkeypatch):
    table = cli_table(mod)
    rc, out, err = table[("--help",)]
    table[("--help",)] = (rc, out + "  gen   deprecated\n", err)
    install_run(mod, monkeypatch, table)
    assert any("visible in tt-bio --help" in p for p in mod._check_cli())


def test_the_deprecated_golden_dir_flag_must_stay_hidden(mod, monkeypatch):
    table = cli_table(mod)
    rc, out, err = table[("design", "--help")]
    table[("design", "--help")] = (rc, out + "  --golden_dir VALUE\n", err)
    install_run(mod, monkeypatch, table)
    assert any("--golden_dir" in p for p in mod._check_cli())


def test_an_affinity_model_missing_from_its_own_verb_is_caught(mod, monkeypatch):
    """`tt-bio affinity` is its own verb, and its choice list had no check at all until
    nesso1 had already shipped without one."""
    table = cli_table(mod)
    table[("affinity", "--help")] = (0, _help(
        ["--model", "--out_dir", "--accelerator", "--trunk", "--recycling_steps",
         "--tokens_budget", "--devices", "--seed"]), "")
    install_run(mod, monkeypatch, table)
    problems = mod._check_cli()
    assert any("affinity --help does not list --model choice nesso1" in p for p in problems)


# ── coverage discovery: the check the whole gate rests on ───────────────────

def test_live_every_shipped_model_is_covered_or_exempt(mod):
    mod._assert_full_model_coverage()


def test_a_new_model_tuple_is_not_a_free_pass(mod, monkeypatch):
    """The nesso1/openbind/pxdesign regression test. A new CLI verb arrives with its own
    *_MODELS tuple; the gate must refuse to start rather than quietly not cover it."""
    monkeypatch.setattr(mod.tt_bio_main, "SCREEN_MODELS", ("brand-new-model",),
                        raising=False)
    with pytest.raises(SystemExit, match="brand-new-model"):
        mod._assert_full_model_coverage()


def test_a_model_dropped_from_a_leg_list_is_caught(mod, monkeypatch):
    """The other direction: the model still ships, but someone trimmed the list the gate
    iterates. Silence is what the check exists to prevent."""
    dropped = mod.FOLD_MODELS[-1]
    monkeypatch.setattr(mod, "FOLD_MODELS", [x for x in mod.FOLD_MODELS if x != dropped])
    monkeypatch.setattr(mod, "ALL_LEGS", [x for x in mod.ALL_LEGS if x != dropped])
    with pytest.raises(SystemExit, match=dropped):
        mod._assert_full_model_coverage()


def test_a_model_listed_as_covered_but_wired_to_no_runner_is_caught(mod, monkeypatch):
    """Covered used to mean "named in one of the leg lists", which is a second list to
    keep in sync with RUNNERS. It means "a runner runs it" now, so adding a name to a leg
    list and forgetting the runner table fails instead of shipping a model nothing runs."""
    monkeypatch.setattr(mod.tt_bio_main, "PREDICT_MODELS",
                        tuple(mod.tt_bio_main.PREDICT_MODELS) + ("ghost-model",))
    monkeypatch.setattr(mod, "FOLD_MODELS", list(mod.FOLD_MODELS) + ["ghost-model"])
    with pytest.raises(SystemExit, match="ghost-model"):
        mod._assert_full_model_coverage()


def test_live_every_leg_has_a_runner(mod):
    """ALL_LEGS is derived from RUNNERS, so this is the statement that every model the
    gate claims to cover is dispatched somewhere."""
    dispatched = {m for _, group in mod.RUNNERS for m in group}
    assert dispatched == set(mod.ALL_LEGS)
    shipped = set().union(*mod._shipped_models().values())
    assert shipped - set(mod.LEGS_EXEMPT) <= dispatched


def test_an_exemption_is_a_free_pass_only_with_a_reason(mod, monkeypatch):
    monkeypatch.setattr(mod.tt_bio_main, "SCREEN_MODELS", ("brand-new-model",),
                        raising=False)
    monkeypatch.setattr(mod, "LEGS_EXEMPT", {
        "brand-new-model": "scored through the predict verb, which has its own leg"})
    mod._assert_full_model_coverage()


# ── LEGS_EXEMPT: structural reasons only ────────────────────────────────────

def test_live_every_exemption_reason_is_structural(mod):
    assert mod._exemption_reason_problems(mod.LEGS_EXEMPT) == []


def test_a_dated_exemption_reason_is_rejected(mod):
    """A gate arm that goes red over a broken card is tracked, not exempted: nothing ever
    revisits an entry that carries a reason, so a transient reason makes the gap
    permanent (memory transient-reason-in-structural-exemption-dict)."""
    problems = mod._exemption_reason_problems(
        {"rf3": "qb2 was down for hardware replacement in September 2026"})
    assert problems and "rf3" in problems[0]


def test_a_shares_a_code_path_exemption_reason_is_rejected(mod):
    """The exact reasoning that let opendde-abag ship with zero perf coverage."""
    problems = mod._exemption_reason_problems(
        {"opendde-abag": "shares a code path with opendde, which is covered"})
    assert problems and "opendde-abag" in problems[0]


def test_an_empty_exemption_reason_is_rejected(mod):
    assert mod._exemption_reason_problems({"openbind": "  "}) != []


def test_a_structural_exemption_reason_is_accepted(mod):
    assert mod._exemption_reason_problems(
        {"esmc-6b": "no CLI verb of its own; the embed leg drives every checkpoint size"
                    " through one command"}) == []


def test_the_gate_refuses_to_start_on_a_transient_exemption(mod, monkeypatch):
    monkeypatch.setattr(mod, "LEGS_EXEMPT", {"rf3": "card 3 was wedged on 2026-09-01"})
    with pytest.raises(SystemExit, match="rf3"):
        mod._assert_full_model_coverage()


# ── row assembly and the verdict a failing leg produces ─────────────────────

def fake_predict(*, events=None, cif=CIF_TEXT, results=None, rc=0, stderr="",
                 timeout=False):
    """Stand in for a real `tt-bio predict` child: write the files and the event capture
    the runner goes looking for, then return its exit status."""
    from tt_bio.main import predict_results_dir_name

    def fake(cmd, *, env=None, **kw):
        if timeout:
            raise subprocess.TimeoutExpired(cmd, 1)
        data = Path(cmd[4])
        model = cmd[cmd.index("--model") + 1]
        out_dir = Path(cmd[cmd.index("--out_dir") + 1])
        rdir = out_dir / predict_results_dir_name(model, data.stem)
        (rdir / "structures").mkdir(parents=True, exist_ok=True)
        if cif is not None:
            (rdir / "structures" / f"{data.stem}_model_0.cif").write_text(cif)
        rows = results if results is not None else [
            {"id": data.stem, "status": "ok", "plddt": 0.91}]
        (rdir / "results.json").write_text(json.dumps(rows))
        cap = env.get("TT_BIO_PROGRESS_CAPTURE") if env else None
        if cap is not None:
            evs = fold_events() if events is None else events
            Path(cap).write_text("".join(json.dumps(e) + "\n" for e in evs))
        return subprocess.CompletedProcess(cmd, rc, "", stderr)
    return fake


def test_run_fold_passes_on_a_healthy_run(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_run", fake_predict())
    row = mod.run_fold("boltz2", tmp_path)
    assert row["gate"] is True
    assert (row["progress"], row["parse"], row["results"]) == (True, True, True)
    assert row["error"] is None


def test_run_fold_fails_when_the_bar_skips_the_trunk(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_run", fake_predict(events=drop_stage(fold_events(), "trunk")))
    row = mod.run_fold("boltz2", tmp_path)
    assert row["gate"] is False and row["progress"] is False
    assert "trunk phase MISSING" in row["error"]
    assert row["parse"] is True and row["results"] is True


def test_run_fold_fails_when_no_structure_was_written(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_run", fake_predict(cif=None))
    row = mod.run_fold("boltz2", tmp_path)
    assert row["gate"] is False and row["parse"] is False
    assert "wrote no CIF" in row["error"]


def test_run_fold_fails_when_results_json_lost_its_confidence(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_run",
                        fake_predict(results=[{"id": "trpcage", "status": "ok"}]))
    row = mod.run_fold("boltz2", tmp_path)
    assert row["gate"] is False and row["results"] is False
    assert "no confidence metric" in row["error"]


def test_run_fold_reports_a_nonzero_exit(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_run", fake_predict(rc=1, stderr="RuntimeError: no card"))
    row = mod.run_fold("boltz2", tmp_path)
    assert row["gate"] is False and "predict exited 1" in row["error"]
    assert "no card" in row["error"]


def test_run_fold_reports_a_timeout(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_run", fake_predict(timeout=True))
    row = mod.run_fold("boltz2", tmp_path)
    assert row["gate"] is False and "timed out" in row["error"]


def test_run_fold_uses_the_abag_fixture_for_the_abag_checkpoint(mod, monkeypatch, tmp_path):
    """opendde-abag folds the Ab-Ag complex, not trpcage. Everything downstream (the
    results dir name, the CIF glob) is keyed on that, so a leg pointed at the wrong
    fixture reports "wrote no CIF" rather than a real verdict."""
    seen = {}

    def spy(cmd, **kw):
        seen["data"] = cmd[4]
        seen["timeout"] = kw.get("timeout")
        return fake_predict()(cmd, **kw)

    monkeypatch.setattr(mod, "_run", spy)
    row = mod.run_fold("opendde-abag", tmp_path)
    assert Path(seen["data"]).name == "1ahw_abag.yaml"
    assert seen["timeout"] == mod.ABAG_MODEL_TIMEOUT_S
    assert row["gate"] is True


EMBED_STDOUT = ("Loading esmc-600m …\n"
                "Embedding 1 sequence(s) → /tmp/out\n"
                "Done — 1 sequence(s), d_model=1152 → /tmp/out (see manifest.json)\n")


def fake_embed(*, stdout=EMBED_STDOUT, npz=True, manifest=True, rc=0):
    import numpy as np

    def fake(cmd, **kw):
        out_dir = Path(cmd[cmd.index("--out_dir") + 1])
        seq = "NLYIQWLKDGGPSSGRPPPS"
        if npz:
            np.savez(out_dir / "tiny.npz",
                     per_residue=np.zeros((len(seq), 8), dtype=np.float32),
                     pooled=np.zeros((8,), dtype=np.float32),
                     sequence=np.array(seq))
        if manifest:
            (out_dir / "manifest.json").write_text(json.dumps(
                {"model": "esmc-600m", "pool": "mean", "format": "npz", "d_model": 8,
                 "dtype": "float32", "sequences": [{"id": "tiny", "length": len(seq)}]}))
        return subprocess.CompletedProcess(cmd, rc, stdout, "")
    return fake


def test_run_embed_passes_on_a_healthy_run(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_run", fake_embed())
    row = mod.run_embed("esmc-600m", tmp_path)
    assert row["gate"] is True and row["error"] is None


def test_run_embed_catches_a_silent_run(mod, monkeypatch, tmp_path):
    """No load, no embed, just the final line: the embed-shaped version of a bar that
    only moves once, at the end."""
    monkeypatch.setattr(mod, "_run", fake_embed(stdout="Done — 1 sequence(s)\n"))
    row = mod.run_embed("esmc-600m", tmp_path)
    assert row["gate"] is False and row["progress"] is False
    assert "missing load" in row["error"]


def test_run_embed_catches_a_missing_manifest(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_run", fake_embed(manifest=False))
    row = mod.run_embed("esmc-600m", tmp_path)
    assert row["gate"] is False and row["manifest"] is False


def test_saprot_runs_under_its_own_verb(mod, monkeypatch, tmp_path):
    """SaProt is not a `tt-bio embed` choice; sending it there would fail the leg for a
    reason that has nothing to do with SaProt."""
    seen = {}

    def spy(cmd, **kw):
        seen["verb"] = cmd[3]
        return fake_embed()(cmd, **kw)

    monkeypatch.setattr(mod, "_run", spy)
    mod.run_embed("saprot-650m", tmp_path)
    assert seen["verb"] == "saprot"
    assert mod._embed_subcommand("esmc-600m") == "embed"


# ── the verdict line and the exit code ──────────────────────────────────────

def test_a_failing_row_prints_its_reason(mod, capsys):
    mod._print_row({"model": "boltz2", "seconds": 61.0, "progress": False, "parse": True,
                    "results": True, "gate": False,
                    "error": "progress: trunk phase MISSING",
                    "checks": ["progress: FAIL", "  • trunk phase MISSING"]})
    out = capsys.readouterr().out
    assert "FAIL" in out and "trunk phase MISSING" in out and "progress=False" in out


def test_a_skipped_row_says_why(mod, capsys):
    mod._print_row({"model": "openbind", "skipped": True, "reason": "checkpoint absent"})
    out = capsys.readouterr().out
    assert "SKIP" in out and "checkpoint absent" in out


def _main_env(mod, monkeypatch, table=None, argv=("ux_regression.py",)):
    monkeypatch.setattr(mod.gate_guard, "declared_dependency_problems", lambda p: [])
    install_run(mod, monkeypatch, table if table is not None else cli_table(mod))
    monkeypatch.setattr(sys, "argv", list(argv))


def test_cli_only_exits_zero_on_a_healthy_cli(mod, monkeypatch, capsys):
    _main_env(mod, monkeypatch, argv=("ux_regression.py", "--cli-only"))
    assert mod.main() == 0


def test_cli_only_exits_one_on_a_cli_regression(mod, monkeypatch, capsys):
    table = cli_table(mod)
    rc, out, err = table[("predict", "--help")]
    table[("predict", "--help")] = (rc, out.replace("  --out_dir VALUE\n", ""), err)
    _main_env(mod, monkeypatch, table, argv=("ux_regression.py", "--cli-only"))
    assert mod.main() == 1
    assert "missing flag --out_dir" in capsys.readouterr().out


ONE_LEG_ARGV = ("ux_regression.py", "--model", "boltz2")


def _one_leg(mod, monkeypatch, row):
    """Point the runner table at a single stub leg. ALL_LEGS is left alone: the coverage
    assertion reads it, and `--model boltz2` is what narrows the run."""
    monkeypatch.setattr(mod, "RUNNERS", ((lambda model, base: dict(row, model=model),
                                          ["boltz2"]),))


def test_a_ux_regression_fails_the_gate_with_a_legible_reason(mod, monkeypatch, capsys):
    _main_env(mod, monkeypatch, argv=ONE_LEG_ARGV)
    _one_leg(mod, monkeypatch, {"seconds": 61.0, "progress": False, "parse": True,
                                "results": True, "gate": False,
                                "error": "progress: trunk phase MISSING",
                                "checks": ["progress: FAIL"]})
    assert mod.main() == 1
    out = capsys.readouterr().out
    assert "GATE FAIL" in out and "trunk phase MISSING" in out


def test_a_clean_run_passes_the_gate(mod, monkeypatch, capsys):
    _main_env(mod, monkeypatch, argv=ONE_LEG_ARGV)
    _one_leg(mod, monkeypatch, {"seconds": 61.0, "progress": True, "parse": True,
                                "results": True, "gate": True, "error": None,
                                "checks": []})
    assert mod.main() == 0
    assert "GATE PASS" in capsys.readouterr().out


def test_a_missing_optional_checkpoint_skips_loudly(mod, monkeypatch, capsys):
    """A skip is not coverage. It must survive into the verdict line, or a host that
    silently gated nothing reads the same as one that gated everything."""
    _main_env(mod, monkeypatch, argv=ONE_LEG_ARGV)
    _one_leg(mod, monkeypatch, {"seconds": 1.0, "gate": True, "error": None, "checks": []})
    monkeypatch.setattr(mod, "CKPT_POLICY",
                        {"boltz2": ("skip", "checkpoint absent at {path}")})
    monkeypatch.setattr(mod.weights, "resolve", lambda model: Path("/nowhere/ckpt.pt"))
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert "SKIP" in out and "NOT gated on this host" in out and "boltz2" in out


def test_a_missing_required_checkpoint_refuses_to_run(mod, monkeypatch):
    """rf3, pxdesign and openfold3 auto-resolve or are a release-host prerequisite, so a
    missing file there is a misconfigured gate host. Skipping it is how a model reaches a
    release with no coverage at all."""
    _main_env(mod, monkeypatch, argv=ONE_LEG_ARGV)
    _one_leg(mod, monkeypatch, {"seconds": 1.0, "gate": True, "error": None, "checks": []})
    monkeypatch.setattr(mod, "CKPT_POLICY",
                        {"boltz2": ("require", "fetch it with `tt-bio weights` or {path}")})
    monkeypatch.setattr(mod.weights, "resolve", lambda model: Path("/nowhere/ckpt.pt"))
    with pytest.raises(SystemExit, match="Refusing to skip"):
        mod.main()


def test_the_gate_refuses_an_interpreter_missing_declared_dependencies(mod, monkeypatch):
    """A leg that dies on a missing import reads as a product failure. rf3 reported FAIL
    on 2026-08-23 for a missing `toolz` and nothing said so."""
    monkeypatch.setattr(mod.gate_guard, "declared_dependency_problems",
                        lambda p: ["toolz is declared in pyproject.toml but not installed"])
    monkeypatch.setattr(sys, "argv", ["ux_regression.py", "--cli-only"])
    with pytest.raises(SystemExit, match="declared dependencies"):
        mod.main()


# ── the fixtures the legs are pointed at exist ──────────────────────────────

def test_live_every_fixture_a_leg_names_is_present(mod):
    """A leg whose fixture was renamed fails on the YAML having run no model at all,
    which is the shape packaging_smoke's --fold leg shipped with for a year."""
    missing = [str(p) for p in (mod.DATA, mod.ABAG_DATA, mod.GEN_SPEC, mod.AFFINITY_SPEC)
               if not p.exists()]
    missing += [str(leg["spec"]) for leg in mod.DESIGN_LEGS.values()
                if not leg["spec"].exists()]
    assert missing == []
