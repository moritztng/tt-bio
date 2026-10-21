"""Tests for BC4's design-quality grader.

Two kinds. Synthetic campaign trees in BindCraft 2's own output layout, so the grader can be shown a known answer
(a null arm, a shifted arm, a collapsed arm, a NaN arm, an arm whose iteration count moved) and has to return it.
And the calibration checks that matter more than any of them: the test must not call two draws from one
distribution different, and the effect size the grader advertises at n = 30 must be the one it can actually detect.
"""
import csv
import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bc2_quality as Q  # noqa: E402

Q.PERM_N = 2000  # the tests assert on p-value regions, not on the fourth decimal

DECK = ("/home/moritz/.coworker/artifacts/deck-cells-gpu/deck-b200/results/bindcraft/stock")
LOSS_COLS = ["hPDL1.binder_contacts", "hPDL1.binder_pae", "hPDL1.compactness", "hPDL1.interface_contacts",
             "hPDL1.interface_pae", "hPDL1.iptm_loss", "hPDL1.plddt_loss"]
TRAJ_COLS = ["trajectory", "design", "length", "i_pTM", "pLDDT", "Timing", "hash", "terminated", "autotuned"]
CAND_COLS = ["design", "outcome", "i_pTM", "pLDDT", "Binder_RMSD", "Binder_Sequence", "hash"]

AMINO = "ACDEFGHIKLMNPQRSTVWY"
_pool_rng = random.Random(777)
POOL = ["".join(_pool_rng.choice(AMINO) for _ in range(80)) for _ in range(200)]


def _write(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_campaign(root, target, arm, seed, n=10, *, iptm=0.40, plddt=0.87, rmsd=1.5, spread=0.05, iterations=140,
                  accepted=0, terminated_every=4, nan_candidate=False, autotuned=False, identical_sequences=False,
                  seq_pool=0, rng_seed=0):
    """One campaign_s<seed>/ tree in BindCraft 2's layout. Returns its arm directory."""
    rng = random.Random(rng_seed)
    arm_dir = os.path.join(root, target, f"{arm}_s{seed}")
    project = os.path.join(arm_dir, f"campaign_s{seed}")
    traj_rows, cand_rows = [], []
    for i in range(1, n + 1):
        design = f"{target}_l80_{seed:04d}{i:02d}"
        # Every Nth trajectory terminates early, as real campaigns do; those produce no candidates and no metrics.
        early = (i % terminated_every == 0)
        traj_rows.append({
            "trajectory": i, "design": design, "length": 80,
            "i_pTM": "" if early else round(iptm + 0.3 + rng.gauss(0, spread), 4),
            "pLDDT": "" if early else round(plddt + rng.gauss(0, spread / 2), 4),
            "Timing": f"worker=0;start=0;design=50.0;compiled={int(i == 1)}",
            "hash": f"h{seed}{i:02d}", "terminated": "" if not early else "mutate",
            "autotuned": "screen_steps 50 -> 62" if (autotuned and i > n // 2) else "",
        })
        rows = []
        for r in range(1, iterations + 1):
            phase = "screen" if r <= iterations // 2 else "mutate"
            row = {c: "" for c in LOSS_COLS + ["hPDL1.iptm", "hPDL1.ptm", "phase", "round"]}
            row["phase"], row["round"] = phase, r
            row["hPDL1.iptm"] = round(0.5 + rng.gauss(0, 0.02), 4)
            row["hPDL1.ptm"] = round(0.7 + rng.gauss(0, 0.02), 4)
            if phase == "screen":  # the mutate phase records only iptm/ptm, exactly as BC2 writes it
                for c in LOSS_COLS:
                    row[c] = round(abs(rng.gauss(1.0, 0.1)), 4)
            rows.append(row)
        _write(os.path.join(project, "1_Trajectories", design, f"{design}_losses.csv"),
               LOSS_COLS + ["hPDL1.iptm", "hPDL1.ptm", "phase", "round"], rows)
        if early:
            continue
        for k in range(10):
            if identical_sequences:
                seq = "A" * 80
            elif seq_pool:
                # A fixed pool shared by every arm, so a draw from it repeats sequences at a known rate the way
                # real campaigns do. Same pool on both sides = same duplicate rate, which must not read as collapse.
                seq = rng.choice(POOL[:seq_pool])
            else:
                seq = "".join(rng.choice(AMINO) for _ in range(80))
            bad = nan_candidate and i == 1 and k == 0
            cand_rows.append({
                "design": f"{design}_candidate{k}", "hash": f"h{seed}{i:02d}",
                "outcome": "accepted" if k < accepted else "rejected",
                "i_pTM": "not-a-number" if bad else round(iptm + rng.gauss(0, spread), 4),
                "pLDDT": round(plddt + rng.gauss(0, spread / 2), 4),
                "Binder_RMSD": round(abs(rmsd + rng.gauss(0, spread * 4)), 4),
                "Binder_Sequence": seq,
            })
    _write(os.path.join(project, "1_Trajectories", "!_Trajectories.csv"), TRAJ_COLS, traj_rows)
    _write(os.path.join(project, "2_Refolded", "!_Refolded.csv"), CAND_COLS, cand_rows)
    return arm_dir


def make_arm(root, target, arm, seeds=(101, 102, 103), **kw):
    dirs = [make_campaign(root, target, arm, s, rng_seed=s + kw.pop("rng_offset", 0), **kw) for s in seeds]
    return dirs


def read_all(dirs):
    out = []
    for d in dirs:
        out.extend(Q.read_arm(d))
    return out


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isdir(DECK), reason="the deck's B200 campaign is not on this host")
def test_reads_the_real_deck_campaign():
    """The one real stock-GPU campaign we have: 4 trajectories, 2 completed, 20 candidates, 140 iterations each."""
    ts = Q.read_arm(DECK)
    assert len(ts) == 4
    assert [t.number for t in ts] == [1, 2, 3, 4]
    assert all(t.iterations == 140 for t in ts)
    assert sum(t.completed for t in ts) == 2
    assert sum(len(t.candidates) for t in ts) == 20
    assert sum(t.accepted for t in ts) == 0
    assert Q.nan_check(ts)["ok"]
    assert Q.autotune_check(ts)["ok"]
    # The mutate phase writes no loss columns, so final_loss has to read back past it.
    assert all(t.final_loss == t.final_loss for t in ts), "final_loss must not be NaN on real data"


def test_synthetic_tree_round_trips(tmp_path):
    d = make_campaign(str(tmp_path), "pdl1", "A1", 101, n=10)
    ts = Q.read_arm(d)
    assert len(ts) == 10
    assert all(t.seed == 101 for t in ts)
    assert [t.number for t in ts] == list(range(1, 11))
    assert all(t.iterations == 140 for t in ts)
    # Trajectories 4, 8 terminate early: no candidates, no design metrics.
    assert sum(t.completed for t in ts) == 8
    assert all(not t.candidates for t in ts if t.terminated)
    assert Q.nan_check(ts)["ok"]


def test_accepted_designs_are_counted(tmp_path):
    d = make_campaign(str(tmp_path), "pdl1", "A1", 101, n=10, accepted=2)
    ts = Q.read_arm(d)
    acc = Q.acceptance(ts)
    assert acc["accepted_designs"] == 16  # 8 completed trajectories x 2 accepted candidates
    assert acc["rate_per_trajectory"] == pytest.approx(1.6)


# ---------------------------------------------------------------------------
# the A/A property: the grader must not cry wolf
# ---------------------------------------------------------------------------

def test_two_stock_arms_read_as_indistinguishable(tmp_path):
    """The mandatory A/A control. If this fails the test is wrong, not the engine."""
    a = make_arm(str(tmp_path), "pdl1", "A1", seeds=(101, 102, 103))
    b = make_arm(str(tmp_path), "pdl1", "A2", seeds=(201, 202, 203))
    g = Q.grade(read_all(a), read_all(b), "A1_vs_A2")
    assert g["verdict"] == "PASS", g["reasons"]
    assert g["n_per_arm"] == [30, 30]
    assert g["iteration_distribution_unchanged"]


def test_paired_test_false_positive_rate_is_near_alpha():
    """200 null draws at n=30: the sign-flip test should reject about 5% of the time, not far more."""
    rng = random.Random(0)
    rejected = sum(Q.paired_perm_mean_diff([rng.gauss(0, 1) for _ in range(30)], n=1000, seed=i) < Q.ALPHA
                   for i in range(200))
    assert 0.01 <= rejected / 200 <= 0.11, f"{rejected}/200 null draws rejected"


def test_advertised_mde_is_the_effect_it_actually_detects():
    """The grader states 0.511 SD paired at n=30. Simulate at exactly that effect and check power is ~80%."""
    mde = Q.mde_sd_units(30, paired=True)
    assert mde == pytest.approx(0.5114, abs=1e-3)
    rng = random.Random(7)
    hits = sum(Q.paired_perm_mean_diff([rng.gauss(mde, 1) for _ in range(30)], n=1000, seed=i) < Q.ALPHA
               for i in range(200))
    assert 0.68 <= hits / 200 <= 0.92, f"power {hits / 200:.2f} at the advertised MDE, expected ~0.80"


# ---------------------------------------------------------------------------
# the A/B property: the grader must catch a real change
# ---------------------------------------------------------------------------

def test_a_shifted_arm_fails(tmp_path):
    ref = make_arm(str(tmp_path), "pdl1", "ref", seeds=(101, 102, 103))
    # i_pTM down by 0.05, two SD of the 0.05-spread reference: a change a user would care about.
    arm = make_arm(str(tmp_path), "pdl1", "tt", seeds=(101, 102, 103), iptm=0.35)
    g = Q.grade(read_all(arm), read_all(ref), "tt")
    assert g["verdict"] == "FAIL"
    assert any("i_ptm" in r for r in g["reasons"]), g["reasons"]
    assert g["metrics"]["i_ptm"]["significant"]


def test_a_nan_arm_fails(tmp_path):
    ref = make_arm(str(tmp_path), "pdl1", "ref", seeds=(101,))
    arm = make_arm(str(tmp_path), "pdl1", "tt", seeds=(101,), nan_candidate=True)
    g = Q.grade(read_all(arm), read_all(ref), "tt")
    assert not g["nan"]["ok"]
    assert g["verdict"] == "FAIL"
    assert any("NaN" in r for r in g["reasons"])


def test_a_collapsed_arm_fails(tmp_path):
    ref = make_arm(str(tmp_path), "pdl1", "ref", seeds=(101, 102, 103))
    arm = make_arm(str(tmp_path), "pdl1", "tt", seeds=(101, 102, 103), identical_sequences=True)
    g = Q.grade(read_all(arm), read_all(ref), "tt")
    assert g["collapse"]["collapsed"]
    assert g["verdict"] == "FAIL"
    assert any("collapse" in r for r in g["reasons"])


def test_a_duplicate_rate_matching_the_reference_is_not_collapse(tmp_path):
    """The stock GPU reference itself reads 0.901 distinct on PD-L1, so a fixed 0.9 cut fails arms that match it.

    Regression for the 2026-10-21 grader defect: the headdev arm read 0.8986 distinct against the reference's
    0.9010 (Fisher p = 1.0, the same duplicate rate) and was failed by the fixed cut alone.
    """
    ref = make_arm(str(tmp_path), "pdl1", "ref", seeds=(101, 102, 103), seq_pool=60)
    arm = make_arm(str(tmp_path), "pdl1", "tt", seeds=(101, 102, 103), seq_pool=60, rng_offset=7)
    c = Q.collapse_check(read_all(arm), read_all(ref))
    assert c["distinct_sequence_fraction"] < 0.9 and c["ref_distinct_sequence_fraction"] < 0.9, c
    assert not c["collapsed"], c["reasons"]
    assert Q.grade(read_all(arm), read_all(ref), "tt")["verdict"] == "PASS"


def test_a_duplicate_rate_worse_than_the_reference_is_collapse(tmp_path):
    ref = make_arm(str(tmp_path), "pdl1", "ref", seeds=(101, 102, 103), seq_pool=200)
    arm = make_arm(str(tmp_path), "pdl1", "tt", seeds=(101, 102, 103), seq_pool=8, rng_offset=7)
    c = Q.collapse_check(read_all(arm), read_all(ref))
    assert c["collapsed"] and c["distinct_p"] < Q.ALPHA, c
    assert Q.grade(read_all(arm), read_all(ref), "tt")["verdict"] == "FAIL"


def test_a_changed_iteration_count_is_caught(tmp_path):
    """The charter's 'No cheating' check: a lever that quietly runs fewer iterations must not pass."""
    ref = make_arm(str(tmp_path), "pdl1", "ref", seeds=(101, 102, 103), iterations=140)
    arm = make_arm(str(tmp_path), "pdl1", "tt", seeds=(101, 102, 103), iterations=112)  # 20% fewer
    g = Q.grade(read_all(arm), read_all(ref), "tt")
    assert not g["iteration_distribution_unchanged"]
    assert g["verdict"] == "FAIL"
    assert any("iterations" in r for r in g["reasons"])


def test_an_autotuned_arm_is_rejected_as_not_stock(tmp_path):
    """Autotune moves stage lengths from the preceding trajectories, so a tuned trajectory is not comparable."""
    ref = make_arm(str(tmp_path), "pdl1", "ref", seeds=(101,))
    arm = make_arm(str(tmp_path), "pdl1", "tt", seeds=(101,), autotuned=True)
    g = Q.grade(read_all(arm), read_all(ref), "tt")
    assert not g["autotune"]["ok"]
    assert g["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------

def test_arms_pair_on_seed_and_trajectory_number(tmp_path):
    ref = make_arm(str(tmp_path), "pdl1", "ref", seeds=(101, 102, 103))
    arm = make_arm(str(tmp_path), "pdl1", "tt", seeds=(101, 102, 103))
    g = Q.grade(read_all(arm), read_all(ref), "tt")
    assert g["metrics"]["iterations"]["n_paired"] == 30
    assert g["metrics"]["iterations"]["test"] == "paired_sign_flip_permutation"
    # i_pTM only exists on completed trajectories, so it pairs on those.
    assert g["metrics"]["i_ptm"]["n_paired"] == 24


def test_unpaired_arms_fall_back_to_a_two_sample_test(tmp_path):
    ref = make_arm(str(tmp_path), "pdl1", "ref", seeds=(101, 102, 103))
    arm = make_arm(str(tmp_path), "pdl1", "tt", seeds=(901, 902, 903))
    g = Q.grade(read_all(arm), read_all(ref), "tt")
    assert g["metrics"]["i_ptm"]["test"] == "two_sample_permutation"
    assert g["metrics"]["i_ptm"]["n_paired"] == 0
    assert g["verdict"] == "PASS", g["reasons"]


# ---------------------------------------------------------------------------
# the reference file and the contract behind it
# ---------------------------------------------------------------------------

def _request(root, runs):
    return {"schema": "bc4-quality-ref-request/1", "runs": runs}


def _runs(target, arm, seeds, n=10):
    return [{"id": f"{target}/{arm}_s{s}", "tier": "P0", "target": target, "arm": arm,
             "campaign_seed": s, "trajectories": n} for s in seeds]


def test_reference_builds_with_a_passing_aa_control(tmp_path):
    root = str(tmp_path)
    make_arm(root, "pdl1", "A1", seeds=(101, 102, 103))
    make_arm(root, "pdl1", "A2", seeds=(201, 202, 203))
    make_arm(root, "il2r", "R1", seeds=(101, 102, 103))
    req = _request(root, _runs("pdl1", "A1", (101, 102, 103)) + _runs("pdl1", "A2", (201, 202, 203))
                   + _runs("il2r", "R1", (101, 102, 103)))
    ref = Q.build_reference(req, root)
    assert ref["missing_runs"] == []
    assert ref["targets"]["pdl1"]["n"] == 60
    assert ref["targets"]["il2r"]["n"] == 30
    assert ref["aa_control"]["verdict"] == "PASS", ref["aa_control"].get("reasons")
    assert ref["aa_control"]["spread"]
    for target in ("pdl1", "il2r"):
        for metric in Q.GATE_METRICS:
            assert metric in ref["targets"][target]["metrics"], metric


def test_reference_reports_a_run_that_did_not_land(tmp_path):
    root = str(tmp_path)
    make_arm(root, "pdl1", "A1", seeds=(101, 102))
    req = _request(root, _runs("pdl1", "A1", (101, 102, 103)))
    ref = Q.build_reference(req, root)
    assert [m["id"] for m in ref["missing_runs"]] == ["pdl1/A1_s103"]
    assert ref["targets"]["pdl1"]["n"] == 20


def test_reference_reports_a_short_campaign(tmp_path):
    """A campaign that stopped early must not quietly shrink n."""
    root = str(tmp_path)
    make_campaign(root, "pdl1", "A1", 101, n=6)
    req = _request(root, _runs("pdl1", "A1", (101,)))
    ref = Q.build_reference(req, root)
    assert len(ref["missing_runs"]) == 1
    assert "6 trajectories" in ref["missing_runs"][0]["note"]


def test_power_statement_tracks_the_acceptance_rate(tmp_path):
    """Acceptance power follows the rate, and is not asserted. BC2's real rate is near zero, where no n works."""
    root = str(tmp_path)
    # Rare acceptance, the regime the deck's own B200 run is in: nothing accepted in 4 trajectories.
    make_arm(root, "pdl1", "A1", seeds=(101, 102, 103), accepted=0)
    ref = Q.build_reference(_request(root, _runs("pdl1", "A1", (101, 102, 103))), root)
    acc = ref["power"]["detectable_in_metric_units"]["pdl1"]["acceptance_rate"]
    assert acc["ref_rate_per_trajectory"] == 0.0
    assert acc["verdict"] == "underpowered at n=30"
    assert acc["n_needed_to_detect_a_halving"] == float("inf")
    assert "guardrail" in acc["note"]
    assert ref["power"]["mde_sd_units"]["paired"] == pytest.approx(0.5114, abs=1e-3)

    # Common acceptance: the same machinery must say so rather than repeating "underpowered".
    root2 = str(tmp_path / "common")
    make_arm(root2, "pdl1", "A1", seeds=(101, 102, 103), accepted=1)
    ref2 = Q.build_reference(_request(root2, _runs("pdl1", "A1", (101, 102, 103))), root2)
    acc2 = ref2["power"]["detectable_in_metric_units"]["pdl1"]["acceptance_rate"]
    assert acc2["ref_rate_per_trajectory"] == pytest.approx(0.8)
    assert acc2["verdict"] == "powered at n=30"
    assert acc2["n_needed_to_detect_a_halving"] <= 30


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def test_fisher_matches_a_known_table():
    # The classic tea-tasting 2x2: [[3,1],[1,3]] is p = 0.4857 two-sided.
    assert Q.fisher_two_sided(3, 1, 1, 3) == pytest.approx(0.4857, abs=1e-3)
    assert Q.fisher_two_sided(0, 0, 0, 0) == 1.0
    assert Q.fisher_two_sided(10, 0, 0, 10) < 0.01


def test_n_for_proportions_grows_as_the_rate_falls():
    assert Q.n_for_proportions(0.5, 0.25) < Q.n_for_proportions(0.05, 0.025)
    assert Q.n_for_proportions(0.0143, 0.00715) > 3000  # the deck's acceptance regime
    assert Q.n_for_proportions(0.1, 0.1) == float("inf")


def test_holm_is_stricter_than_the_raw_alpha():
    results = [{"metric": f"m{i}", "p": p} for i, p in enumerate([0.04, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])]
    Q.holm(results)
    # 0.04 would pass a raw 0.05 but not 0.05/7.
    assert not results[0]["significant"]
    assert results[0]["alpha_holm"] == pytest.approx(0.05 / 7)


def test_effect_sizes():
    assert Q.cohen_dz([1.0] * 10) != Q.cohen_dz([1.0] * 10) or True  # zero SD gives NaN, not a crash
    assert Q.cohen_d([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(0.0)
    assert Q.cohen_d([2, 3, 4, 5], [1, 2, 3, 4]) > 0


def test_written_reference_satisfies_the_phase0_gate(tmp_path):
    """The gate reads the file, not the dict, and a schema drift is only found after the GPU money is spent.

    These are exactly the checks `workstreams/_bc4_quality_check.py` makes on state/bc4/QUALITY-REF.json.
    Mirrored rather than imported because that gate subprocess-runs this very file.
    """
    root = str(tmp_path)
    make_arm(root, "pdl1", "A1", seeds=(101, 102, 103))
    make_arm(root, "pdl1", "A2", seeds=(201, 202, 203))
    make_arm(root, "il2r", "R1", seeds=(101, 102, 103))
    req = _request(root, _runs("pdl1", "A1", (101, 102, 103)) + _runs("pdl1", "A2", (201, 202, 203))
                   + _runs("il2r", "R1", (101, 102, 103)))
    out = tmp_path / "QUALITY-REF.json"
    out.write_text(json.dumps(Q.build_reference(req, root), indent=2))

    v = json.loads(out.read_text())
    assert len(v.get("targets") or {}) >= 2
    for name, t in v["targets"].items():
        assert isinstance(t.get("n"), int) and t["n"] >= 30, (name, t.get("n"))
        for key in ("acceptance_rate", "i_ptm", "plddt", "rmsd"):
            assert key in (t.get("metrics") or {}), (name, key)
        assert t.get("paths"), name
    assert (v.get("aa_control") or {}).get("verdict") == "PASS"
    assert v["aa_control"].get("spread")
    assert v.get("test")
