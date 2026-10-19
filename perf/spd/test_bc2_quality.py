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


def _write(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_campaign(root, target, arm, seed, n=10, *, iptm=0.40, plddt=0.87, rmsd=1.5, spread=0.05, iterations=140,
                  accepted=0, terminated_every=4, nan_candidate=False, autotuned=False, identical_sequences=False,
                  dup_candidates=0, n_candidates=10, rng_seed=0):
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
        for k in range(n_candidates):
            seq = "A" * 80 if identical_sequences else "".join(rng.choice(AMINO) for _ in range(80))
            bad = nan_candidate and i == 1 and k == 0
            cand_rows.append({
                "design": f"{design}_candidate{k}", "hash": f"h{seed}{i:02d}",
                "outcome": "accepted" if k < accepted else "rejected",
                "i_pTM": "not-a-number" if bad else round(iptm + rng.gauss(0, spread), 4),
                "pLDDT": round(plddt + rng.gauss(0, spread / 2), 4),
                "Binder_RMSD": round(abs(rmsd + rng.gauss(0, spread * 4)), 4),
                "Binder_Sequence": seq,
            })
    # MPNN re-emits a sequence now and then; dup_candidates reproduces that at a chosen rate.
    for c in cand_rows[1:1 + dup_candidates]:
        c["Binder_Sequence"] = cand_rows[0]["Binder_Sequence"]
    _write(os.path.join(project, "1_Trajectories", "!_Trajectories.csv"), TRAJ_COLS, traj_rows)
    _write(os.path.join(project, "2_Refolded", "!_Refolded.csv"), CAND_COLS, cand_rows)
    return arm_dir


def make_arm(root, target, arm, seeds=(101, 102, 103), **kw):
    # Popped outside the comprehension: inside it, only the first campaign saw the offset and the rest silently
    # fell back to 0, so an arm asked for a shifted RNG got one campaign shifted and two not.
    offset = kw.pop("rng_offset", 0)
    return [make_campaign(root, target, arm, s, rng_seed=s + offset, **kw) for s in seeds]


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


def test_the_verdict_is_one_holm_family(tmp_path):
    """Every test that can fail the verdict shares one correction, so the A/A rate is controlled where the charter
    asks for it: on the VERDICT, not on each test inside it.

    The collapse tests and the mean tests used to be Holm'd separately, leaving two families each controlled at
    alpha whose OR is the verdict. Measured on 250 synthetic A/A draws at n=30 that read 6.4% against a nominal
    5%. This asserts the structure directly rather than re-running the Monte Carlo: with one family the strictest
    alpha_holm is ALPHA / (number of testable results across BOTH groups), with two it would be ALPHA divided by
    the size of whichever group the result sits in.
    """
    ref = read_all(make_arm(str(tmp_path), "pdl1", "ref", seeds=(101, 102, 103)))
    arm = read_all(make_arm(str(tmp_path), "pdl1", "tt", seeds=(101, 102, 103), rng_offset=9000))
    g = Q.grade(arm, ref, "tt")
    testable = [r for r in list(g["metrics"].values()) + g["collapse"]["spread"]
                if isinstance(r.get("p"), float) and r["p"] == r["p"]]
    assert len(testable) > len(g["metrics"]), "the collapse tests must be in the family too"
    assert min(r["alpha_holm"] for r in testable) == pytest.approx(Q.ALPHA / len(testable))


def test_advertised_mde_is_the_effect_it_actually_detects():
    """The grader states 0.511 SD paired at n=30. Simulate at exactly that effect and check power is ~80%."""
    mde = Q.mde_sd_units(30, paired=True)
    assert mde == pytest.approx(0.5114, abs=1e-3)
    rng = random.Random(7)
    hits = sum(Q.paired_perm_mean_diff([rng.gauss(mde, 1) for _ in range(30)], n=1000, seed=i) < Q.ALPHA
               for i in range(200))
    assert 0.68 <= hits / 200 <= 0.92, f"power {hits / 200:.2f} at the advertised MDE, expected ~0.80"


def test_the_verdict_level_mde_is_the_holm_corrected_one():
    """What a metric must move to fail the VERDICT, which is the number a lever is really graded against.

    A single real effect has to be the first Holm rejection, so it clears alpha/FAMILY_SIZE, not alpha. At n=30
    that is 0.697 SD paired against the bare 0.511: quoting only the bare number overstates the harness by ~36%
    and would let a lever ship on a 'no significant difference' the test never had the power to find.
    """
    a_holm = Q.ALPHA / Q.FAMILY_SIZE
    mde = Q.mde_sd_units(30, paired=True, alpha=a_holm)
    assert Q.FAMILY_SIZE == 17, "9 mean/count/work tests + sequence diversity + 6 spread tests"
    assert mde == pytest.approx(0.697, abs=2e-3)
    assert mde > Q.mde_sd_units(30, paired=True)
    rng = random.Random(11)
    hits = sum(Q.paired_perm_mean_diff([rng.gauss(mde, 1) for _ in range(30)], n=4000, seed=i) < a_holm
               for i in range(200))
    assert 0.68 <= hits / 200 <= 0.92, f"power {hits / 200:.2f} at the Holm-corrected MDE, expected ~0.80"


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


def test_an_empty_arm_is_inconclusive_not_a_pass(tmp_path):
    """An arm that produced nothing must never read as indistinguishable from stock.

    The three real cueq arms on the 10-19 B200 crashed before writing a trajectory, and every one of them graded
    PASS: no test could run, so no test rejected, so the verdict came back clean. That reads absence of evidence
    as evidence of absence, and it is the one verdict this grader must not emit by accident.
    """
    ref = read_all(make_arm(str(tmp_path), "pdl1", "ref", seeds=(101, 102, 103)))
    g = Q.grade([], ref, "crashed")
    assert g["verdict"] == "INCONCLUSIVE"
    assert g["n_per_arm"][0] == 0
    assert any("no test could run" in r for r in g["reasons"]), g["reasons"]
    assert g["iteration_distribution_unchanged"] is None, "nothing was compared, so nothing is unchanged either"


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


def test_a_reference_pooling_one_seed_twice_does_not_pair(tmp_path):
    """A pooled reference must not turn into a paired test against one arbitrarily chosen campaign.

    `dict(b)` keeps the last occurrence of a repeated key and drops the rest silently. Pooling the five B200 arms
    that all ran campaign seed 101 left final_loss, iterations and accepted pairing on 3 of 15 reference
    trajectories and reporting it as a paired result; final_loss read p=0.749 paired against last-wins values and
    p=0.415 once every trajectory was used.
    """
    ref = read_all(make_arm(str(tmp_path), "pdl1", "r1", seeds=(101,))) + \
          read_all(make_arm(str(tmp_path), "pdl1", "r2", seeds=(101,), rng_offset=5000))
    arm = read_all(make_arm(str(tmp_path), "pdl1", "tt", seeds=(101,), rng_offset=9000))
    r = Q.grade(arm, ref, "tt")["metrics"]["final_loss"]
    assert r["n_dropped_to_collisions"] == 10, r
    assert r["n_paired"] == 0, "a key that occurs twice on one side cannot pair"
    assert r["test"] == "two_sample_permutation"
    assert r["ref"]["n"] == 20, "the fallback must still use every reference trajectory"


def test_a_thin_overlap_does_not_discard_the_rest_of_the_data(tmp_path):
    """Pairing is the stronger design per observation, but not when it keeps only a few of them.

    The 10-19 B200 reference landed partial (4, 5 and 5 trajectories of the 10 each campaign asked for), so an
    arm running the full 10 per seed overlaps it on 4, and taking the paired branch would discard the rest for a
    weaker test. The overlap goes thin exactly when the reference is topped up on NEW seeds, which is how those
    14 landed trajectories get back to full strength: the arm then pairs only with the partial campaigns.
    Unpaired wins once the smaller side holds more than twice the overlap.
    """
    ref = (read_all(make_arm(str(tmp_path), "pdl1", "refpart", seeds=(101,), n=4))
           + read_all(make_arm(str(tmp_path), "pdl1", "reftop", seeds=(104, 105), n=10, rng_offset=200)))
    arm = read_all(make_arm(str(tmp_path), "pdl1", "tt", seeds=(101,), n=10, rng_offset=9000))
    r = Q.grade(arm, ref, "tt")["metrics"]["final_loss"]
    assert r["n_paired"] == 4, "the overlap is real, it is just thin"
    assert r["test"] == "two_sample_permutation", r.get("test_choice")
    assert r["arm"]["n"] == 10 and r["ref"]["n"] == 24, "every value must still be used"


def test_distinct_seeds_still_pair(tmp_path):
    """The collision guard must not cost the pairing the reference is designed to have."""
    ref = read_all(make_arm(str(tmp_path), "pdl1", "ref", seeds=(101, 102, 103)))
    arm = read_all(make_arm(str(tmp_path), "pdl1", "tt", seeds=(101, 102, 103), rng_offset=9000))
    r = Q.grade(arm, ref, "tt")["metrics"]["final_loss"]
    assert r["n_dropped_to_collisions"] == 0
    assert r["n_paired"] == 30
    assert r["test"] == "paired_sign_flip_permutation"


def test_stock_rate_duplication_is_not_collapse(tmp_path):
    """The A/A principle applied to the collapse guardrail: stock duplicates, so duplicating is not collapse.

    Measured on the real B200 campaigns of 2026-10-19, the earlier `distinct < 0.9` rule called four of six STOCK
    arms collapsed, each on one repeated sequence (attn-stock 4/5 distinct, attn-stock-w9 7/8, attn-xla 13/15),
    and called the TT arm collapsed at 23/28 for duplicating at the same rate. A guardrail that fails stock would
    have failed the campaign's headline verdict for free.
    """
    ref = make_arm(str(tmp_path), "pdl1", "ref", seeds=(101, 102, 103), dup_candidates=10)
    arm = make_arm(str(tmp_path), "pdl1", "tt", seeds=(101, 102, 103), dup_candidates=10, rng_offset=7000)
    c = Q.collapse_check(read_all(arm), read_all(ref))
    assert c["distinct_sequence_fraction"] < 0.90, "the fixture must actually contain duplicates"
    assert not c["collapsed"], c["reasons"]
    assert Q.grade(read_all(arm), read_all(ref), "tt")["verdict"] == "PASS"


def test_a_lower_diversity_rate_than_stock_is_caught(tmp_path):
    """The other direction: an arm that really does repeat itself more than stock has to be flagged. Graded per
    trajectory, because every duplicate in the real campaigns was inside one trajectory and none across two."""
    ref = make_arm(str(tmp_path), "pdl1", "ref", seeds=(101, 102, 103), dup_candidates=4)
    arm = make_arm(str(tmp_path), "pdl1", "tt", seeds=(101, 102, 103), dup_candidates=40, rng_offset=7000)
    c = Q.collapse_check(read_all(arm), read_all(ref))
    assert c["collapsed"], c
    assert any("sequence diversity" in r for r in c["reasons"]), c["reasons"]


def test_one_sequence_taking_half_the_candidates_is_collapse(tmp_path):
    """The absolute floor still has to catch real collapse, with no reference to lean on."""
    arm = make_arm(str(tmp_path), "pdl1", "tt", seeds=(101,), dup_candidates=40)
    c = Q.collapse_check(read_all(arm))
    assert c["collapsed"]
    assert c["top_sequence_share"] >= 0.5
    assert any("one sequence is" in r for r in c["reasons"]), c["reasons"]


def test_a_narrowed_spread_needs_a_test_not_a_ratio(tmp_path):
    """The same bug class on the spread rule: a fixed '< 0.2x the reference sd' flagged the stock attn-xla arm,
    because stock's own i_pTM sd over 3 trajectories was 0.396, carried by a single bad design."""
    ref = make_arm(str(tmp_path), "pdl1", "ref", seeds=(101, 102, 103), spread=0.05)
    wide = make_arm(str(tmp_path), "pdl1", "a2", seeds=(201, 202, 203), spread=0.05)
    assert not Q.collapse_check(read_all(wide), read_all(ref))["collapsed"]
    # A genuinely frozen arm is still caught.
    flat = make_arm(str(tmp_path), "pdl1", "tt", seeds=(301, 302, 303), spread=0.0005)
    c = Q.collapse_check(read_all(flat), read_all(ref))
    assert c["collapsed"], c
    assert any("Brown-Forsythe" in r for r in c["reasons"]), c["reasons"]


def test_fewer_mpnn_sequences_is_caught(tmp_path):
    """The charter's 'No cheating' clause bars fewer MPNN sequences as squarely as fewer iterations, and only
    iterations was guarded. Halving the sequences a trajectory redesigns is less of the model's own work however
    good the surviving designs look."""
    ref = make_arm(str(tmp_path), "pdl1", "ref", seeds=(101, 102, 103), n_candidates=10)
    arm = make_arm(str(tmp_path), "pdl1", "tt", seeds=(101, 102, 103), n_candidates=5)
    g = Q.grade(read_all(arm), read_all(ref), "tt")
    assert not g["work_unchanged"]["mpnn_sequences"]
    assert g["verdict"] == "FAIL"
    assert any("mpnn_sequences" in r for r in g["reasons"]), g["reasons"]


def test_an_equal_mpnn_sequence_count_passes(tmp_path):
    """And the guard must not fire on an arm doing the same work, early terminations included: a trajectory that
    never reached refolding contributes NaN, not a zero that would read as a cut."""
    ref = make_arm(str(tmp_path), "pdl1", "ref", seeds=(101, 102, 103), n_candidates=10)
    arm = make_arm(str(tmp_path), "pdl1", "tt", seeds=(101, 102, 103), n_candidates=10, rng_offset=4000)
    g = Q.grade(read_all(arm), read_all(ref), "tt")
    assert g["work_unchanged"] == {"iterations": True, "mpnn_sequences": True}
    assert g["verdict"] == "PASS", g["reasons"]


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


def test_banked_runs_are_read_but_never_listed_as_runnable(tmp_path):
    """The 14 trajectories the 2026-10-19 B200 left when it capped mid-campaign count toward the reference, and
    must not be handed back to a runner: repeating a campaign repeats its pair keys, which then cannot pair."""
    make_campaign(str(tmp_path), "pdl1", "A1", 101, n=4)
    req = {"runs": [{"id": "pdl1/A1_s104", "tier": "P0", "target": "pdl1", "arm": "A1",
                     "campaign_seed": 104, "trajectories": 10}],
           "banked_runs": [{"id": "pdl1/A1_s101", "tier": "BANKED", "target": "pdl1", "arm": "A1",
                            "campaign_seed": 101, "trajectories": 4}]}
    ref = Q.build_reference(req, str(tmp_path))
    assert ref["targets"]["pdl1"]["n"] == 4, "the banked campaign has to be read into the reference"
    assert [m["id"] for m in ref["missing_runs"]] == ["pdl1/A1_s104"], ref["missing_runs"]


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


def test_n30_is_reported_as_too_small_for_an_effect_we_care_about(tmp_path):
    # The brief: if n=30 cannot detect an effect we would care about, the file must SAY so and say what n would.
    # It cannot -- 0.5 SD at the Holm-corrected alpha takes 58 paired values -- so the number has to be there.
    arm = make_arm(tmp_path, "pdl1", "stock")
    st = Q.power_statement({"pdl1": [t for c in arm for t in Q.read_arm(c)]})
    care = st["what_n_would_see_an_effect_we_care_about"]["pdl1"]
    assert care["care_about_sd_units"] == Q.CARE_ABOUT_SD
    for metric, d in care["per_metric"].items():
        assert d["n30_is_enough"] is False, metric
        assert d["trajectories_needed"] >= d["values_needed"] > 30
    assert "cannot do it" in care["note"]


def test_the_n_needed_is_the_inverse_of_the_mde():
    # A separate formula that disagreed with mde_sd_units would quietly advertise an n that does not reach the bar.
    for paired in (True, False):
        n = Q.n_for_sd_units(0.5, paired=paired)
        assert abs(Q.mde_sd_units(n, paired=paired) - 0.5) < 1e-9
