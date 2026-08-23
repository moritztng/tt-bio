"""Device-free regression tests for scripts/full_parity_gate.py verdict logic.

Pins the false-pass found 2026-08-09/10: ``--seeds 5`` (a bare count, not a
comma-separated list) matched no fixture seed, so every leg reported
BLOCKED-REF-REGEN-NEEDED and the gate still printed GATE PASS with zero legs
scored. These tests run no device work: the seeds validation fires before any
fold, and the tally tests force the blocked path with a monkeypatched
``_incomplete_fixture_seeds``.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "full_parity_gate.py"


def _load():
    spec = importlib.util.spec_from_file_location("full_parity_gate", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # @dataclass resolves cls.__module__ through sys.modules during exec_module
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _quiet_host(monkeypatch):
    """Every test here sees an unloaded host.

    The gate refuses to start above 1.5x nproc. That guard runs before the leg loop, so on a
    busy box it fired instead of the logic under test and three of these tests went red for
    the host's load rather than for anything in the gate: green on an idle laptop at loadavg
    0.8, red on a QuietBox at 28.6. The ceiling itself is tested by overriding this again.
    """
    monkeypatch.setattr(os, "getloadavg", lambda: (0.5, 0.5, 0.5))


def _run_gate(args, tmp_path):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--workdir", str(tmp_path),
         "--workers", "localhost:0"],
        cwd=REPO, capture_output=True, text=True, timeout=180)


def test_seeds_bare_count_rejected_not_false_pass(tmp_path):
    """The exact reported misuse: `--seeds 5` meaning "5 seeds" errors out loudly
    instead of blocking every leg and printing GATE PASS."""
    proc = _run_gate(["--leg", "openfold3-8hel-nomsa", "--seeds", "5"], tmp_path)
    assert proc.returncode != 0
    assert "GATE PASS" not in proc.stdout
    assert "--seeds" in proc.stderr
    assert "not a" in proc.stderr  # names the list-not-count semantics
    assert "openfold3-8hel-nomsa" in proc.stderr  # names the leg + its real fixture seeds


def test_seeds_non_integer_rejected(tmp_path):
    proc = _run_gate(["--leg", "openfold3-8hel-nomsa", "--seeds", "abc"], tmp_path)
    assert proc.returncode != 0
    assert "GATE PASS" not in proc.stdout
    assert "--seeds" in proc.stderr


def test_all_blocked_run_is_inconclusive_not_pass(tmp_path, monkeypatch, capsys):
    """Every leg blocked on reference regen => nothing scored => INCONCLUSIVE,
    exit nonzero. Reproduces the all-blocked condition through the same
    _incomplete_fixture_seeds path the typo took, with an otherwise valid seed."""
    mod = _load()
    ckpt = tmp_path / "of3-p2-155k.pt"
    ckpt.write_bytes(b"x")  # preflight only checks existence
    monkeypatch.setenv("OF3_CKPT", str(ckpt))
    monkeypatch.setattr(mod, "_incomplete_fixture_seeds",
                        lambda leg, seeds: [f"seed{s}" for s in seeds])
    monkeypatch.setattr(sys, "argv", ["full_parity_gate.py",
                                      "--leg", "openfold3-8hel-nomsa", "--seeds", "0",
                                      "--workdir", str(tmp_path),
                                      "--workers", "localhost:0"])
    rc = mod.main()
    out = capsys.readouterr().out
    assert rc != 0
    assert "GATE INCONCLUSIVE" in out
    assert "1/1 legs blocked" in out
    assert "GATE PASS" not in out


def test_scored_leg_plus_blocked_leg_still_passes(tmp_path, monkeypatch, capsys):
    """Normal case unchanged: a blocked leg still does NOT fail the gate when a
    sibling leg reaches a real scored verdict (rfd3-featurizer is card-free and
    runs its committed bit-exact reference for real here)."""
    mod = _load()
    ckpt = tmp_path / "of3-p2-155k.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setenv("OF3_CKPT", str(ckpt))
    monkeypatch.setattr(mod, "_incomplete_fixture_seeds",
                        lambda leg, seeds: ["seed0"] if leg.id == "openfold3-8hel-nomsa" else [])
    monkeypatch.setattr(sys, "argv", ["full_parity_gate.py",
                                      "--leg", "rfd3-featurizer",
                                      "--leg", "openfold3-8hel-nomsa", "--seeds", "0",
                                      "--workdir", str(tmp_path),
                                      "--workers", "localhost:0"])
    rc = mod.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "GATE PASS" in out
    assert "GATE INCONCLUSIVE" not in out
    assert "'PASS': 1" in out
    assert "'BLOCKED-REF-REGEN-NEEDED': 1" in out


def test_of3_ckpt_preflight(tmp_path, monkeypatch):
    """OpenFold3 legs fail fast at preflight when no checkpoint resolves, instead of
    dying inside tt_bio/worker.py after paying for fold setup."""
    mod = _load()
    leg = mod.LEGS_BY_ID["openfold3-8hel-nomsa"]
    monkeypatch.delenv("OF3_CKPT", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.boltz or ~/of3-weights here
    problems = mod.preflight_check([leg])
    assert any("OF3_CKPT" in p and leg.id in p for p in problems)
    monkeypatch.setenv("OF3_CKPT", str(tmp_path / "nope.pt"))
    problems = mod.preflight_check([leg])
    assert any("OF3_CKPT" in p and "not an existing file" in p for p in problems)
    ckpt = tmp_path / "of3-p2-155k.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setenv("OF3_CKPT", str(ckpt))
    assert not any("OF3" in p for p in mod.preflight_check([leg]))


def test_staged_msa_legs_sharing_a_sequence_do_not_share_one_a3m(tmp_path, monkeypatch):
    """Two staged legs on the same sequence must consume their OWN reference MSA.

    ``protenix-ubq-msa`` and ``openfold3-ubq-msa`` both fold examples/ubq.yaml but pin
    different reference a3m bytes. Staging by sequence hash alone put both in one
    ``<workdir>/msa/<seqhash>.a3m`` and the copy is first-writer-wins, so the second leg
    silently folded against the first leg's MSA while being scored against a reference
    built from other bytes.
    """
    mod = _load()
    prot_leg = mod.LEGS_BY_ID["protenix-ubq-msa"]
    of3_leg = mod.LEGS_BY_ID["openfold3-ubq-msa"]
    assert prot_leg.yaml == of3_leg.yaml          # same sequence, hence the same seq hash
    assert prot_leg.fixture != of3_leg.fixture    # different reference MSA bytes

    fixtures = tmp_path / "fx"
    bytes_by_fixture = {prot_leg.fixture: b">q\nAAA\n>a\nAAC\n",
                        of3_leg.fixture: b">q\nAAA\n>b\nAAG\n"}
    for name, blob in bytes_by_fixture.items():
        d = fixtures / name
        d.mkdir(parents=True)
        (d / "msa.a3m").write_bytes(blob)
    monkeypatch.setattr(mod, "_fixture_dir", lambda spec: fixtures / spec)

    wd = tmp_path / "wd"
    staged = {}
    for leg in (prot_leg, of3_leg):
        msa_dir, args = mod.stage_msa(leg, wd)
        assert args[0] == "--msa_dir" and args[1] == str(msa_dir)
        a3m = list(msa_dir.glob("*.a3m"))
        assert len(a3m) == 1, f"{leg.id}: expected exactly one staged a3m, got {a3m}"
        staged[leg.id] = a3m[0]

    assert staged[prot_leg.id] != staged[of3_leg.id], "both legs staged to the same path"
    for leg in (prot_leg, of3_leg):
        assert staged[leg.id].read_bytes() == bytes_by_fixture[leg.fixture], (
            f"{leg.id} staged another fixture's MSA")

def test_every_committed_fixture_names_its_own_settings_tag():
    """The legacy R/D/X scorer refuses a fixture whose meta.json does not name its own
    settings tag, so a committed fixture without one is silently unscoreable under
    --legacy-rdx. boltz2-{trpcage,prot,hsa}-nomsa hard-ERRORed for exactly that reason the
    first time the gate could reach them: regen_envelope_refs wrote their meta.json flat,
    with no settings_tag, and nothing checked."""
    import json

    root = REPO / "docs" / "implementation-parity-data" / "ref-fixtures"
    bad = []
    for meta_path in sorted(root.glob("*/*/*/meta.json")):
        tag = json.loads(meta_path.read_text()).get("settings_tag")
        if tag != meta_path.parent.name:
            bad.append(f"{meta_path.parent.relative_to(root)}: meta.json says {tag!r}")
    assert not bad, "fixtures whose meta.json does not name their own directory:\n" + "\n".join(bad)


def test_scorer_env_names_the_driver(tmp_path, monkeypatch):
    """A spawned scorer must inherit TT_BIO_PARENT_PID so it can arm its parent-death
    guard; without it a scorer outliving a SIGKILLed driver held card 1 for 1 h 43 m.
    pin_card=None is the point: that is the branch that does not rebuild env."""
    mod = _load()
    leg = mod.Leg(id="esmc-300m", model="esmc-300m", kind="esmc", yaml="")
    captured = {}

    class _Proc:
        returncode = 0

    def fake_run(cmd, **kw):
        captured.update(kw.get("env") or {})
        return _Proc()

    monkeypatch.setattr(mod, "subprocess", types.SimpleNamespace(
        run=fake_run, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired))
    env = {**os.environ, "TT_BIO_PARENT_PID": str(os.getpid())}
    mod.run_inprocess(leg, tmp_path / "out.json", tmp_path / "log.txt", env, pin_card=None)
    assert captured.get("TT_BIO_PARENT_PID") == str(os.getpid())


def test_regen_envelope_meta_carries_the_settings_tag():
    """regen_envelope_refs must stamp settings_tag on the flat (envelope-native) path too,
    or every fixture it writes becomes unscoreable under --legacy-rdx."""
    src = SCRIPT.read_text()
    assert 'meta.setdefault("settings_tag", base.name)' in src, (
        "the envelope regen no longer stamps settings_tag onto the meta.json it writes"
    )


def test_legacy_rmsd_key_still_scores_and_no_leg_reads_no_data():
    """A committed record that files the metric under "rmsd" must still be comparable.

    ``_structure_verdict`` used to key only on "kabsch_rmsd", so protenix-v2-hsa.json (written
    before the rename, and carrying no explicit top-level verdict) read as NO-DATA and its
    drift check was skipped in silence. Every structure leg with a committed record on disk
    must now yield a comparable verdict.
    """
    mod = _load()
    legacy = {"mode": "structures", "targets": {"hsa": {"rmsd": {
        "metric": "kabsch_rmsd", "within_noise_floor": False,
        "cross": {"mean": 1.02}, "ref_floor": {"mean": 0.70}, "dev_floor": {"mean": 0.37}}}}}
    verdict, detail = mod._structure_verdict(legacy)
    assert verdict == "GAP", (verdict, detail)
    assert "X=1.020" in detail

    modern = {"mode": "structures", "targets": {"hsa": {"kabsch_rmsd": {
        "metric": "kabsch_rmsd", "within_noise_floor": True,
        "cross": {"mean": 0.69}, "ref_floor": {"mean": 0.70}, "dev_floor": {"mean": 0.58}}}}}
    assert mod._structure_verdict(modern)[0] == "PASS"

    unreadable = []
    for leg in mod.LEGS:
        if leg.kind != "structure" or not leg.committed_json:
            continue
        if not (mod.PARITY_DATA / leg.committed_json).exists():
            continue
        if mod._committed_verdict(leg) in (None, "NO-DATA"):
            unreadable.append((leg.id, leg.committed_json))
    assert not unreadable, f"committed records that read as NO-DATA (drift check skipped): {unreadable}"


# ---------------------------------------------------------------------------
# Card grant + host load (scripts/gate_guard.py)
# ---------------------------------------------------------------------------
# A fleet worker holds ONE card while a sibling worker holds another on the same box, so a
# gate that opens a card outside its own lease is a host-level overcommit: on 2026-08-21 qb1
# ran six jobs at loadavg 30-62 and then dropped off the network entirely. These pin the three
# guards. All card-free: the refusals fire before any device work, and the skip path is
# reached by declaring a leg needs more cards than the grant.
def _guard():
    spec = importlib.util.spec_from_file_location("gate_guard", REPO / "scripts" / "gate_guard.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_grant_is_ambient_visible_devices_and_unset_means_the_box():
    g = _guard()
    assert g.card_grant({}) is None                       # unpinned release run
    assert g.card_grant({"TT_VISIBLE_DEVICES": "  "}) is None
    assert g.card_grant({"TT_VISIBLE_DEVICES": "2"}) == {2}
    assert g.card_grant({"TT_VISIBLE_DEVICES": "1,3"}) == {1, 3}
    # A pin that resolves to nothing must read as unbounded, never as an empty grant that
    # would silently skip every leg.
    assert g.card_grant({"TT_VISIBLE_DEVICES": "nonsense"}) is None


def test_leg_needing_more_cards_than_granted_is_skipped_not_run():
    g = _guard()
    assert g.leg_grant_skip(1, {2}) is None               # fits
    assert g.leg_grant_skip(4, None) is None              # granted the box
    assert g.leg_grant_skip(4, {1, 2}) == "leg requires 4 cards, granted 2 ([1, 2]), not run"


def test_worker_pool_wider_than_the_grant_is_refused():
    g = _guard()
    assert g.worker_pool_problems([0, 1, 2, 3], None) == []       # unpinned: unchanged
    assert g.worker_pool_problems([2], {2}) == []
    problems = g.worker_pool_problems([0, 1, 2, 3], {2}, "qb1")
    assert len(problems) == 1
    assert "card(s) [0, 1, 3]" in problems[0] and "holds only card(s) [2]" in problems[0]


def test_load_ceiling_refuses_above_the_multiple_and_is_disablable(monkeypatch):
    g = _guard()
    monkeypatch.setattr(g.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(g.os, "getloadavg", lambda: (40.0, 30.0, 20.0))
    problem = g.load_ceiling_problem(1.5)
    assert problem and "loadavg 40.00" in problem and "1.5x nproc (8) = 12.0" in problem
    assert g.load_ceiling_problem(0) is None               # explicit override
    monkeypatch.setattr(g.os, "getloadavg", lambda: (2.0, 2.0, 2.0))
    assert g.load_ceiling_problem(1.5) is None


def test_delegated_leg_pin_lands_on_a_granted_card_not_card_zero():
    """boltzgen/abag/capacity run in-process and shell out from there, so the pin must be on
    THIS process's environment. Unpinned they design on card 0 whatever --workers said, and a
    fan-out with no pin selects every card on the box (measured [0, 1, 2, 3] on qb1)."""
    g = _guard()
    assert g.pin_target([2], {2}) == 2
    assert g.pin_target([], {3}, fallback=0) == 3          # no local slot: use the grant, not 0
    assert g.pin_target([1], None, fallback=1) == 1        # unpinned: honour --workers
    assert g.pin_target([], None, fallback=0) == 0         # unchanged from before the guard


def test_gate_refuses_a_worker_pool_it_was_not_granted(tmp_path, monkeypatch, capsys):
    """End to end through main(): granted card 0, asked for four. Refused in preflight, before
    a single fold. HEAD without this guard ran the legs and printed GATE PASS."""
    mod = _load()
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(sys, "argv", ["full_parity_gate.py", "--dry-run",
                                      "--leg", "rfd3-featurizer",
                                      "--workdir", str(tmp_path),
                                      "--workers", "localhost:0,localhost:1,localhost:2"])
    rc = mod.main()
    out = capsys.readouterr().out
    assert rc != 0
    assert "card(s) [1, 2]" in out and "holds only card(s) [0]" in out
    assert "GATE PASS" not in out


def test_skipped_leg_is_named_in_the_verdict_and_never_reads_as_coverage(
        tmp_path, monkeypatch, capsys):
    """A leg the grant cannot serve is SKIPPED-CARD-GRANT, listed under COVERAGE REDUCED, and
    excluded from the scored count -- so a narrowed gate cannot be read as a green full one."""
    mod = _load()
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "0")
    mesh_leg = mod.LEGS_BY_ID["rfd3-featurizer"]
    monkeypatch.setattr(mod, "LEGS", [replace(mesh_leg, id="mesh-only-leg", cards=4)])
    monkeypatch.setattr(mod, "LEGS_BY_ID", {l.id: l for l in mod.LEGS})
    monkeypatch.setattr(sys, "argv", ["full_parity_gate.py",
                                      "--leg", "mesh-only-leg",
                                      "--workdir", str(tmp_path),
                                      "--workers", "localhost:0"])
    rc = mod.main()
    out = capsys.readouterr().out
    assert "SKIPPED-GRANT" in out
    assert "leg requires 4 cards, granted 1 ([0]), not run" in out
    assert "COVERAGE REDUCED" in out and "mesh-only-leg" in out
    # Nothing scored, so the run is inconclusive rather than a pass: a skip is not evidence.
    assert "GATE INCONCLUSIVE" in out and "GATE PASS" not in out and rc != 0


def test_delegated_leg_runs_with_the_pin_in_its_own_environment(tmp_path, monkeypatch):
    """The card-identity half of the fix, card-free: run_inprocess must set
    TT_VISIBLE_DEVICES on THIS process before calling into release_gate, because the delegated
    legs shell out from here. Before the fix the recorded value was whatever the launcher had
    (None on a release host), which is how boltzgen designed on card 0 whatever --workers said.
    """
    mod = _load()
    seen = {}

    class _Stub:
        def _load_designability_harness(self):
            return object()

        def run_boltzgen(self, bg, keep):
            seen["visible"] = os.environ.get("TT_VISIBLE_DEVICES")
            return {"model": "boltzgen", "gate": True}

    monkeypatch.setattr(mod, "_load_release_gate", lambda: _Stub())
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "0")
    row = mod.run_inprocess(mod.LEGS_BY_ID["boltzgen"], tmp_path / "bg.json",
                            tmp_path / "bg.log", dict(os.environ), pin_card=3)
    assert row == {"model": "boltzgen", "gate": True}
    assert seen["visible"] == "3"
    # and the gate's own environment is put back, so one leg's pin cannot leak into the next
    assert os.environ["TT_VISIBLE_DEVICES"] == "0"


# --- remote worker probe: an unreachable / self-aliased / card-less worker aborts in seconds ---
#
# Pins the v0.6.7 gate-infra defect (2026-08-23): the gate ran ON qb2 with `--workers qb2:2,qb2:3`,
# `qb2` is an ssh alias that exists only in another user's config on another host, so locality
# matched by name classified it remote, ssh'd to an unresolvable name, and all 21 device legs
# exited 255 in 0s while the 9 in-process legs passed. 47 minutes, no model run, verdict FAIL.

def _workers(mod, spec):
    return mod.parse_workers(spec)


def test_unreachable_remote_worker_is_a_preflight_problem():
    mod = _load()
    ws = _workers(mod, "not-a-host:2,not-a-host:3")
    probs = mod.remote_worker_problems(
        ws, probe=lambda h: (255, "", "ssh: Could not resolve hostname not-a-host"))
    assert len(probs) == 1, probs
    assert "not-a-host" in probs[0] and "not reachable" in probs[0]
    # names the fix, and says what would otherwise happen
    assert "localhost:2" in probs[0] and "255" in probs[0]


def test_remote_worker_that_is_this_same_box_is_a_preflight_problem():
    mod = _load()
    import socket
    me = socket.gethostname().split(".")[0]
    ws = _workers(mod, "myalias:2")
    probs = mod.remote_worker_problems(ws, probe=lambda h: (0, f"{me}\n0\n1\n2\n3\n", ""))
    assert len(probs) == 1, probs
    assert "same box" in probs[0] and "localhost:2" in probs[0]


def test_remote_worker_missing_the_card_node_is_a_preflight_problem():
    mod = _load()
    ws = _workers(mod, "otherbox:3")
    probs = mod.remote_worker_problems(ws, probe=lambda h: (0, "otherbox\n0\n1\n", ""))
    assert len(probs) == 1, probs
    assert "/dev/tenstorrent/3" in probs[0]


def test_reachable_remote_worker_with_its_card_is_no_problem():
    mod = _load()
    ws = _workers(mod, "otherbox:2,otherbox:3")
    assert mod.remote_worker_problems(
        ws, probe=lambda h: (0, "otherbox\n0\n1\n2\n3\n", "")) == []


def test_local_workers_are_not_probed():
    mod = _load()
    ws = _workers(mod, "localhost:0")

    def boom(h):
        raise AssertionError(f"probed a local worker: {h}")

    assert mod.remote_worker_problems(ws, probe=boom) == []


def test_one_probe_per_host_not_per_worker():
    mod = _load()
    seen = []

    def probe(h):
        seen.append(h)
        return 0, "otherbox\n0\n1\n2\n3\n", ""

    mod.remote_worker_problems(_workers(mod, "otherbox:0,otherbox:1,otherbox:2"), probe=probe)
    assert seen == ["otherbox"]
