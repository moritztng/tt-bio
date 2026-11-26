"""The curve driver contract, and the G3 curve grader shown failing on written logs.

``tests/test_training_parity_bars.py`` pins the bars and exercises the statistic on arrays.
This file grades the artifacts a real leg produces: three JSONL logs written through
:class:`tt_bio.train.curve.CurveLog`. Every refusal arm is here because the alternative to
refusing is a verdict that looks like a pass: two runs on different plans, a log missing its
seed-B control, a side that ran a different recycle depth.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tt_bio.train.curve import (CurveContractError, CurveLog, CurvePlan, grade_curve_files,
                                read_curve, summarise)

IDS = ("2kpz__1__1.A__1.B", "1a6b__1__1.A__1.C", "2jup__1__1.A__1.D")


def plan(steps=2000, **kw):
    kw.setdefault("crop", 256)
    kw.setdefault("warmup", min(100, steps // 4))
    return CurvePlan(steps=steps, example_ids=IDS, **kw)


def curve(n, seed=1, floor=1.0, noise=0.02):
    rng = np.random.default_rng(seed)
    s = np.arange(n)
    return floor + 2.0 * np.exp(-s / 300.0) + noise * rng.standard_normal(n)


def write(tmp_path, name, side, losses, p=None, **meta):
    p = p or plan(len(losses))
    path = tmp_path / name
    with CurveLog(path, p, side, meta=meta) as log:
        for i, v in enumerate(losses):
            log.record(i, v)
    return path


# ------------------------------------------------------------------------ the plan

def test_plan_fixes_the_data_order_and_the_seeds():
    p = plan(10, batch_size=2)
    assert [s.example_ids for s in (p.step(i) for i in range(3))] == [
        (IDS[0], IDS[1]), (IDS[2], IDS[0]), (IDS[1], IDS[2])]
    assert p.step(4).t_seed != p.step(4).noise_seed
    assert p.step(4).t_seed == plan(10, batch_size=2).step(4).t_seed  # reproducible, not random
    assert p.step(5).t_seed != p.step(4).t_seed


def test_recycle_depth_is_upstreams_own_seeded_draw():
    """handle_recycling with n_recycle_sync: Poisson(lambda) on 0x5EED0000 + global_step."""
    p = plan(50)
    for i in (0, 7, 49):
        rng = np.random.default_rng(0x5EED0000 + i)
        assert p.step(i).n_recycle == int(np.clip(rng.poisson(3.0), 1, 9))


def test_plan_digest_moves_with_every_field_that_changes_the_run():
    base = plan(2000)
    for change in (dict(steps=2001), dict(crop=384), dict(seed=7), dict(batch_size=2),
                   dict(exploration_k=2), dict(n_recycle_lambda=2.0), dict(warmup=200)):
        assert plan(**{**{"steps": 2000}, **change}).digest() != base.digest()
    assert plan(2000).digest() == base.digest()


def test_a_short_plan_is_legal_but_not_reportable():
    assert not plan(20).graded and plan(2000).graded


# ------------------------------------------------------------------------ the log

def test_log_round_trips_and_carries_provenance(tmp_path):
    p = plan(5)
    path = write(tmp_path, "ours.jsonl", "ours", curve(5), p=p, sha="deadbeef", aiclk_mhz=1350)
    header, losses = read_curve(path)
    assert header["plan_digest"] == p.digest() and header["sha"] == "deadbeef"
    assert header["aiclk_mhz"] == 1350 and header["host"] and header["side"] == "ours"
    assert len(losses) == 5
    row = json.loads(path.read_text().splitlines()[2])
    assert row["step"] == 1 and row["n_recycle"] == p.step(1).n_recycle
    assert row["example_ids"] == list(p.step(1).example_ids)


def test_log_refuses_a_step_out_of_order_and_a_mismatched_depth(tmp_path):
    p = plan(5)
    with CurveLog(tmp_path / "a.jsonl", p, "ours") as log:
        log.record(0, 1.0)
        with pytest.raises(CurveContractError):
            log.record(2, 1.0)
        with pytest.raises(CurveContractError):
            log.record(1, 1.0, n_recycle=p.step(1).n_recycle + 1)


def test_reader_refuses_a_gap_a_bad_header_and_an_edited_plan(tmp_path):
    p = plan(6)
    path = write(tmp_path, "ours.jsonl", "ours", curve(6), p=p)
    lines = path.read_text().splitlines()

    gapped = tmp_path / "gap.jsonl"
    gapped.write_text("\n".join(lines[:3] + lines[4:]) + "\n")
    with pytest.raises(CurveContractError):
        read_curve(gapped)

    headerless = tmp_path / "noheader.jsonl"
    headerless.write_text("\n".join(lines[1:]) + "\n")
    with pytest.raises(CurveContractError):
        read_curve(headerless)

    # The plan in the header is edited after the fact but the digest is left alone: the run that
    # produced these losses is no longer the run the file claims.
    h = json.loads(lines[0])
    h["plan"]["crop"] = 384
    edited = tmp_path / "edited.jsonl"
    edited.write_text("\n".join([json.dumps(h)] + lines[1:]) + "\n")
    with pytest.raises(CurveContractError):
        read_curve(edited)

    unknown = json.loads(lines[0])
    unknown["plan"]["lr"] = 1e-4
    unknown["plan_digest"] = "x"
    bad = tmp_path / "unknownfield.jsonl"
    bad.write_text("\n".join([json.dumps(unknown)] + lines[1:]) + "\n")
    with pytest.raises(CurveContractError):
        read_curve(bad)


# ------------------------------------------------------------------------ grading real logs

def test_grades_a_matched_leg_as_pass(tmp_path):
    n = 2000
    p = plan(n)
    paths = [write(tmp_path, f"{s}.jsonl", s, curve(n, seed=k), p=p)
             for k, s in enumerate(("ours", "gpu_a", "gpu_b"), start=3)]
    r = grade_curve_files(*paths)
    assert r.passed, str(r)
    assert "CURVE: 2000 steps PASS" in str(r)


def test_fails_a_three_percent_offset(tmp_path):
    """Converged 3 % above upstream: over the 2 % bar and over 1.5 x the seed gap."""
    n, p = 2000, plan(2000)
    paths = [write(tmp_path, "ours.jsonl", "ours", curve(n, seed=3, floor=1.09), p=p),
             write(tmp_path, "gpu_a.jsonl", "gpu_a", curve(n, seed=1), p=p),
             write(tmp_path, "gpu_b.jsonl", "gpu_b", curve(n, seed=2), p=p)]
    r = grade_curve_files(*paths)
    assert not r.passed and any("mean gap" in f for f in r.failures)


def test_fails_a_mid_run_excursion_that_ends_in_the_right_place(tmp_path):
    n, p = 2000, plan(2000)
    ours = curve(n, seed=3)
    ours[800:1000] *= 1.3
    paths = [write(tmp_path, "ours.jsonl", "ours", ours, p=p),
             write(tmp_path, "gpu_a.jsonl", "gpu_a", curve(n, seed=1), p=p),
             write(tmp_path, "gpu_b.jsonl", "gpu_b", curve(n, seed=2), p=p)]
    r = grade_curve_files(*paths)
    assert not r.passed and any("running mean" in f for f in r.failures)
    assert 800 <= r.worst_drift_step < 1100


def test_refuses_a_missing_seed_b_control(tmp_path):
    n, p = 2000, plan(2000)
    ours = write(tmp_path, "ours.jsonl", "ours", curve(n, seed=3), p=p)
    a = write(tmp_path, "gpu_a.jsonl", "gpu_a", curve(n, seed=1), p=p)
    with pytest.raises(CurveContractError, match="seed-B"):
        grade_curve_files(ours, a)          # not a PASS, and not a FAIL either


def test_refuses_logs_from_different_plans(tmp_path):
    n = 2000
    ours = write(tmp_path, "ours.jsonl", "ours", curve(n, seed=3), p=plan(n, crop=384))
    a = write(tmp_path, "gpu_a.jsonl", "gpu_a", curve(n, seed=1), p=plan(n))
    b = write(tmp_path, "gpu_b.jsonl", "gpu_b", curve(n, seed=2), p=plan(n))
    with pytest.raises(CurveContractError, match="different plans"):
        grade_curve_files(ours, a, b)


def test_refuses_a_mislabelled_side_and_a_log_short_of_its_plan(tmp_path):
    n, p = 2000, plan(2000)
    ours = write(tmp_path, "ours.jsonl", "ours", curve(n, seed=3), p=p)
    a = write(tmp_path, "gpu_a.jsonl", "gpu_a", curve(n, seed=1), p=p)
    b_as_a = write(tmp_path, "b.jsonl", "gpu_a", curve(n, seed=2), p=p)
    with pytest.raises(CurveContractError, match="logged as side"):
        grade_curve_files(ours, a, b_as_a)

    short = tmp_path / "short.jsonl"
    short.write_text("\n".join(b_as_a.read_text().splitlines()[:-5]) + "\n")
    with pytest.raises(CurveContractError):
        grade_curve_files(ours, a, short)


def test_a_twenty_step_plumbing_run_cannot_be_reported_as_a_curve(tmp_path):
    """The CPU smoke leg produces real logs; the grader still refuses to call it a CURVE."""
    n, p = 20, plan(20, warmup=5)
    paths = [write(tmp_path, f"{s}.jsonl", s, curve(n, seed=k), p=p)
             for k, s in enumerate(("ours", "gpu_a", "gpu_b"), start=3)]
    r = grade_curve_files(*paths)
    assert not r.passed and any("the bar needs >= 2000" in f for f in r.failures)


def test_summarise_reports_what_a_tail_shows():
    s = summarise(curve(300))
    assert s["steps"] == 300 and s["finite"] and s["mean_last_100"] < s["first"]


# ------------------------------------------------------------------------ the CLI verdict

def test_grader_cli_exits_0_1_and_2(tmp_path):
    """PASS 0, FAIL 1, refused 2: a refusal must never read as a pass to a calling script."""
    import subprocess
    import sys
    from pathlib import Path

    n, p = 2000, plan(2000)
    ours = write(tmp_path, "ours.jsonl", "ours", curve(n, seed=3), p=p)
    a = write(tmp_path, "gpu_a.jsonl", "gpu_a", curve(n, seed=1), p=p)
    b = write(tmp_path, "gpu_b.jsonl", "gpu_b", curve(n, seed=2), p=p)
    bad = write(tmp_path, "bad.jsonl", "ours", curve(n, seed=3, floor=1.09), p=p)
    cli = Path(__file__).resolve().parents[1] / "perf" / "r4t" / "grade_parity.py"

    def run(*args):
        return subprocess.run([sys.executable, str(cli), *map(str, args)],
                              capture_output=True, text=True)

    ok = run("--ours", ours, "--gpu-a", a, "--gpu-b", b)
    assert ok.returncode == 0 and "CURVE: 2000 steps PASS" in ok.stdout, ok.stdout + ok.stderr
    fail = run("--ours", bad, "--gpu-a", a, "--gpu-b", b)
    assert fail.returncode == 1 and "FAIL" in fail.stdout
    refused = run("--ours", ours, "--gpu-a", a)
    assert refused.returncode == 2 and "REFUSED" in refused.stdout and "seed-B" in refused.stdout
