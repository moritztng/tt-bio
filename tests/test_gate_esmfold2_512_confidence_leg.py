"""The release gate's ESMFold2-Fast L=512 confidence leg.

Kabsch and the distance-matrix metric grade where the atoms went. On 2026-09-02 (05327d6ce)
ESMFold2-Fast placed cdk2x2_512 inside the seed floor while reading mean plDDT 0.021 below the
reference, and the esmfold2 rule, PASS-if-scored, could not see it. The leg's `max_plddt_delta`
grades that gap. The cases below are its negative controls, built from the two real records.
"""
import ast
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "full_parity_gate.py"
FIXTURE = (REPO / "docs/implementation-parity-data/ref-fixtures/esmfold2-fast/cdk2x2_512/"
           "nomsa_10cycle_100step_1sample_fp32_prod")
BAR = 0.005

# Production lever set on acbce1322, qb2 card 0, 2026-10-15.
TODAY = {"protein": "cdk2x2_512", "plddt_mean": {"ref_seed_spread": 0.00422,
                                                 "mean_abs_delta": 0.00140}}
# The record the gap was found in, 05327d6ce, 2026-09-02.
SEP02 = {"protein": "cdk2x2_512", "plddt_mean": {"ref_seed_spread": 0.00430,
                                                 "mean_abs_delta": 0.02096}}


def _rule():
    """The confidence check alone, without running the gate's module-level setup."""
    body = [n for n in ast.parse(GATE.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "_esmfold2_confidence_gap"]
    assert body, "the gate no longer defines _esmfold2_confidence_gap"
    ns = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(GATE), "exec"), ns)
    return ns["_esmfold2_confidence_gap"]


def test_todays_record_passes():
    assert _rule()([TODAY], BAR) == ""


def test_the_sep02_gap_is_reported():
    gap = _rule()([SEP02], BAR)
    assert "cdk2x2_512" in gap and "0.0210" in gap, gap


def test_a_record_without_a_plddt_block_is_not_a_pass():
    rec = copy.deepcopy(TODAY)
    rec.pop("plddt_mean")
    assert _rule()([rec], BAR)


# The Wormhole --fast record, .108 chip 23, 2026-10-17 (spd-gap-whesmf wpl, arm D): ws_lofi alone
# moves it 0.015, inside the charter's fast bar and far outside the normal one.
WH_FAST = {"protein": "cdk2x2_512", "fast": True,
           "plddt_mean": {"ref_seed_spread": 0.00422, "mean_abs_delta": 0.01674}}
FAST_BAR = 0.03


def test_a_fast_record_is_graded_at_the_fast_bar():
    assert _rule()([WH_FAST], BAR, FAST_BAR) == ""
    collapse = copy.deepcopy(WH_FAST)
    collapse["plddt_mean"]["mean_abs_delta"] = 0.0350
    assert "0.0350" in _rule()([collapse], BAR, FAST_BAR)


def test_a_record_without_the_fast_mark_keeps_the_normal_bar():
    rec = copy.deepcopy(WH_FAST)
    rec.pop("fast")
    assert "0.0167" in _rule()([rec], BAR, FAST_BAR)
    assert "0.0167" in _rule()([WH_FAST], BAR)


def test_a_leg_without_a_bar_is_ungraded():
    """The other esmfold2 legs keep PASS-if-scored."""
    assert _rule()([SEP02], None) == ""


def test_the_bar_sits_above_the_reference_seed_spread_and_below_the_old_gap():
    assert TODAY["plddt_mean"]["ref_seed_spread"] < BAR < SEP02["plddt_mean"]["mean_abs_delta"]


def test_the_leg_is_registered_against_its_committed_fixture():
    src = GATE.read_text()
    assert 'Leg("esmfold2-fast-512-confidence"' in src, "the leg is not in LEGS"
    leg = src.split('Leg("esmfold2-fast-512-confidence"', 1)[1].split("\n    Leg(", 1)[0]
    assert f"max_plddt_delta={BAR}" in leg and f"max_plddt_delta_fast={FAST_BAR}" in leg
    assert '"cdk2x2_512"' in leg and '"--loops", "10"' in leg and '"--steps", "100"' in leg
    assert "esmfold2-fast/cdk2x2_512/nomsa_10cycle_100step_1sample_fp32_prod" in leg
    meta = json.loads((FIXTURE / "meta.json").read_text())
    assert meta["settings"]["recycling_cycles"] == 10 and meta["settings"]["requested_steps"] == 100
    assert meta["envelope"]["settings"]["target_id"] == "cdk2x2_512"
    for s in (0, 1):
        assert (FIXTURE / "ref_fp32" / f"seed{s}.npz").exists(), f"seed{s} reference not committed"


def _checkpoint_identity():
    src = REPO / "scripts" / "esmfold2_e2e_parity.py"
    body = [n for n in ast.parse(src.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "checkpoint_identity"]
    ns = {"os": __import__("os")}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(src), "exec"), ns)
    return ns["checkpoint_identity"]


def test_checkpoint_sha_is_read_at_the_pinned_revision(monkeypatch, tmp_path):
    """qb2's cache had refs/main on a snapshot without weights: a lookup at main read no sha
    there and a real one on qb1, so the fixture recorded on qb2 failed the gate on qb1."""
    import sys
    import types
    blob = tmp_path / "60ca19f2"
    blob.write_bytes(b"")
    seen = {}

    def lookup(repo, filename, revision=None):
        seen["revision"] = revision
        return str(blob) if revision == "c6c7958d" else None

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(try_to_load_from_cache=lookup))
    config = types.SimpleNamespace(folding_trunk=types.SimpleNamespace(n_layers=24),
                                   msa_encoder=types.SimpleNamespace(enabled=False))
    ident = _checkpoint_identity()("biohub/ESMFold2-Fast", config, "c6c7958d")
    assert seen["revision"] == "c6c7958d"
    assert ident["sha256"] == "60ca19f2"


def test_checkpoint_sha_is_the_hubs_blob_name_in_a_deduplicated_cache(monkeypatch, tmp_path):
    """.108's cache links the repo blob on into a shared store (hub/blobs/fd/fd86c959...). The
    bytes hash to 60ca19f2, but following every link read fd86c959 and the scorer refused the
    fixture."""
    import sys
    import types
    store = tmp_path / "hub" / "blobs" / "fd" / "fd86c959"
    store.parent.mkdir(parents=True)
    store.write_bytes(b"")
    repo = tmp_path / "hub" / "models--biohub--ESMFold2-Fast"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "60ca19f2").symlink_to("../../blobs/fd/fd86c959")
    snap = repo / "snapshots" / "c6c7958d"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to("../../blobs/60ca19f2")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        try_to_load_from_cache=lambda repo_id, filename, revision=None:
            str(snap / "model.safetensors")))
    config = types.SimpleNamespace(folding_trunk=types.SimpleNamespace(n_layers=24),
                                   msa_encoder=types.SimpleNamespace(enabled=False))
    ident = _checkpoint_identity()("biohub/ESMFold2-Fast", config, "c6c7958d")
    assert ident["sha256"] == "60ca19f2"


def test_fixture_records_the_weight_sha():
    meta = json.loads((FIXTURE / "meta.json").read_text())
    assert "sha256:None" not in meta["settings"]["checkpoint"]
