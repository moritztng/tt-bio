#!/usr/bin/env python3
"""Offline tests for the part of rfd4_arms.py that decides what we publish. No GPU, no network.

The screen ranks every arm at one small batch, and on the B200 that batch leaves the card 7-50 % busy, so it
ranks host overhead as much as it ranks arms. These tests pin the two behaviours that follow from that:
an arm that loses the screen but scales better must win the size, and the arm the reference designs come from
must be the one that won after the climb -- not the screen's winner.

    python3 perf/kitcmp/rfd4_arms_test.py
"""
import json
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rfd4_arms as M  # noqa: E402

# rate(arm, size, batch) -> designs/s. "slow-start" loses at the screen batch and wins once the batch grows,
# which is the inversion the screen cannot see.
RATES = {
    "native-compile": lambda s, b: 0.5 * b ** 0.4,
    "compile-reduce-overhead": lambda s, b: 0.9 * b ** 0.2,
    "bioir": lambda s, b: 0.2 * b ** 0.9,
}
OOM_AT = 32  # every arm fails above this, so the ladder ends on the card pushing back, not on the ceiling


def fake_run(arm, size, batch, warm, overrides, tag=None, spec=None, extra=()):
    if arm not in RATES or batch > OOM_AT:
        return [{"kind": "warm", "status": "FAILED", "error": "CUDA out of memory", "wall_s": 0,
                 "designs": 0, "batch": batch}]
    r = RATES[arm](size, batch) * (128.0 / size)
    return [{"kind": "warm", "designs": batch, "wall_s": batch / r, "batch": batch, "sm_mhz": 1965}
            for _ in range(warm)]


def setup(tmp, arms=2):
    M.a = types.SimpleNamespace(out=Path(tmp), inputs=Path(tmp), deadline=2 ** 31 - 1,
                                screen_warm=5, final_warm=9, arms=arms, reference=0,
                                reference_arm=None, sweep_only=False)
    M.run = fake_run
    M.ARMS = {k: [] for k in RATES}
    return M.a


def test_climb_inverts_the_screen():
    with tempfile.TemporaryDirectory() as tmp:
        setup(tmp)
        screen = {a: M.summarise(fake_run(a, 256, M.SCREEN_BATCH[256], 5, [])) for a in RATES}
        screen_winner = max(screen, key=lambda k: screen[k]["designs_s"])
        assert screen_winner == "compile-reduce-overhead", screen_winner

        sweeps = {a: M.climb(a, 256, 3, f"t-{a}") for a in RATES}
        best = {}
        for a, sw in sweeps.items():
            ok = {b: v for b, v in sw.items() if v["status"] == "ok"}
            bb = max(ok, key=lambda b: ok[b]["designs_s"])
            best[a] = ok[bb]["designs_s"]
        assert max(best, key=lambda k: best[k]) == "bioir", best
        print(f"ok: screen picks {screen_winner}, the climb picks bioir ({best})")


def test_climb_stops_on_the_card_not_the_ladder():
    with tempfile.TemporaryDirectory() as tmp:
        setup(tmp)
        sw = M.climb("bioir", 256, 3, "t")
        rungs = sorted(sw)
        assert rungs == [1, 2, 4, 8, 16, 32, 64], rungs
        assert sw[64]["status"] == "FAILED", sw[64]
        assert max(b for b in sw if sw[b]["status"] == "ok") == OOM_AT
        print(f"ok: ladder ran {rungs} and stopped on the OOM rung, below the b{M.SWEEP_CEIL[256]} ceiling")


def test_extended_climb_reuses_every_arms_measured_rungs():
    """The extended climb must re-pay no rung this box already measured, for either climbed arm.

    ARMS.json records the winner's ladder twice (as `sweep` and inside `climbed`) and the runner-up's only
    inside `climbed`. Keying the reuse on the winner made the runner-up re-climb from b1: on 2026-10-21 that
    was 42 rungs across the three sizes, paid a second time on a rented B200.
    """
    rungs = {"1": {"status": "ok", "designs_s": 0.1, "n": 3},
             "2": {"status": "ok", "designs_s": 0.2, "n": 3},
             "4": {"status": "ok", "designs_s": 0.4, "n": 3}}
    prev = {"screen": {"256": {a: {"status": "ok", "designs_s": r} for a, r in
                               (("native-compile", 0.5), ("bioir", 0.4))}},
            "best": {"256": {"arm": "native-compile", "batch": 4, "sweep": rungs,
                             "climbed": {"native-compile": {"sweep": rungs}, "bioir": {"sweep": rungs}}}}}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "ARMS.json").write_text(json.dumps(prev))
        setup(tmp)
        M.a.sweep_only = True
        ran = []
        M.run = lambda arm, size, batch, warm, ov, **kw: (ran.append((arm, batch)) or
                                                          fake_run(arm, size, batch, warm, ov, **kw))
        argv = sys.argv
        sys.argv = ["rfd4_arms.py", "--inputs", tmp, "--out", tmp, "--deadline", str(2 ** 31 - 1),
                    "--sweep-only", "--arms", "2"]
        try:
            M.main()
        except SystemExit as e:
            assert e.code == 0, e.code
        finally:
            sys.argv = argv
        # Both arms were climbed, and neither re-ran a rung the seed already carried.
        assert {a for a, _ in ran} == {"native-compile", "bioir"}, ran
        repaid = [(a, b) for a, b in ran if str(b) in rungs]
        assert not repaid, repaid
    print(f"ok: the extended climb reused {len(rungs)} measured rungs for BOTH arms, re-paying none")


def test_chosen_arm_is_the_one_the_climb_ranked():
    best = {"128": {"arm": "bioir", "batch": 32}, "256": {"arm": "bioir", "batch": 32},
            "512": {"arm": "native-compile", "batch": 8}}
    final = {"512": {"designs_s": 0.1}}
    assert M.choose_arm(best, final) == "bioir"
    # a tie is broken by the largest size, because that is the size the campaign is gated on
    tie = {"256": {"arm": "native-compile", "batch": 8}, "512": {"arm": "bioir", "batch": 32}}
    assert M.choose_arm(tie, {"512": {"designs_s": 0.1}}) == "bioir"
    assert M.choose_arm({}, {}) is None
    print("ok: choose_arm votes by size and breaks ties on the largest size")


def test_partial_arms_json_names_a_supported_arm():
    """A run killed after 256 still writes best_arm, and it must be one the measured sizes support."""
    best = {"128": {"arm": "compile-reduce-overhead", "batch": 16}, "256": {"arm": "bioir", "batch": 32}}
    chosen = M.choose_arm(best, {})
    assert chosen in {v["arm"] for v in best.values()}, chosen
    print(f"ok: a partial ARMS.json names {chosen}, an arm its own sizes measured")


REF_INPUTS = Path("/home/moritz/.coworker/state/rfd4/inputs")


def test_reference_families_resolve():
    """Every reference family must name a spec that exists and tags upstream actually defines.

    A typo here is only discovered on a rented box, after the arms phase has already been paid for.
    """
    fams = json.loads((REF_INPUTS / "ref" / "FAMILIES.json").read_text())["families"]
    assert len(fams) >= 10, len(fams)
    with tempfile.TemporaryDirectory() as tmp:
        setup(tmp)
        M.a.inputs, M.a.src = REF_INPUTS, Path("/SRC")
        for fam, cfg in fams.items():
            spec = (REF_INPUTS / "ref" / cfg["spec"])
            assert spec.exists(), f"{fam}: missing spec {spec}"
            assert cfg["metrics_tags"], fam
            assert set(cfg["fallback_tags"]) <= set(cfg["metrics_tags"]), fam
            assert not any(t.endswith("_refolding") for t in cfg["fallback_tags"]), \
                f"{fam}: fallback still needs an oracle"
            p = M.materialise_spec(spec, fam)
            body = p.read_text()
            assert "${RFD4_SRC}" not in body, fam
            if p != spec.resolve():            # a materialised spec must point into the box's checkout
                assert "/SRC/" in body, fam
    print(f"ok: {len(fams)} reference families resolve, tags fall back without an oracle")


def test_binder_spec_keeps_its_relative_input():
    """The PD-L1 spec ships beside its own input, so it must be used in place, not copied elsewhere."""
    with tempfile.TemporaryDirectory() as tmp:
        setup(tmp)
        M.a.inputs, M.a.src = REF_INPUTS, Path("/SRC")
        spec = REF_INPUTS / "ref" / ".." / "pdl1-ref.yaml"
        assert M.materialise_spec(spec, "binder") == spec.resolve()
        assert (spec.resolve().parent / "5o45_cropped.cif.zst").exists()
    print("ok: the binder spec stays beside the input it names")


def test_published_prefers_the_extended_climb():
    """A worker sweep must measure the arm the evidence publishes, which is the extended climb's when it ran."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "ARMS.json").write_text(json.dumps({"best": {"128": {"arm": "bioir", "batch": 8},
                                                            "256": {"arm": "native-compile", "batch": 4}}}))
        (out / "ARMS-EXT.json").write_text(json.dumps({"128": {"arm": "compile-reduce-overhead", "batch": 64}}))
        pub = M.published(out)
        assert pub["128"] == ("compile-reduce-overhead", 64), pub
        assert pub["256"] == ("native-compile", 4), pub
    print("ok: the worker sweep reads the arm and batch the session publishes")


def test_workers_phase_picks_the_count_that_wins():
    """Host parallelism is the GPU's to use: the phase ranks num_workers at the published arm and batch."""
    seen = []

    def run_with_workers(arm, size, batch, warm, overrides, tag=None, spec=None, extra=(), seed=None, bucket=True):
        nw = next((int(o.split("=")[1]) for o in extra if o.startswith("num_workers=")), 4)
        seen.append((arm, size, batch, nw))
        # host-bound: the rate rises with loader workers and saturates at 16
        r = 1.0 * min(nw, 16) ** 0.5
        return [{"kind": "warm", "designs": batch, "wall_s": batch / r, "batch": batch, "sm_mhz": 1965}
                for _ in range(warm)]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "ARMS.json").write_text(json.dumps({"best": {"512": {"arm": "native-compile", "batch": 8}}}))
        setup(tmp)
        M.run = run_with_workers
        argv = sys.argv
        sys.argv = ["rfd4_arms.py", "--inputs", tmp, "--out", tmp, "--deadline", str(2 ** 31 - 1),
                    "--workers", "4,16,64"]
        try:
            M.main()
        except SystemExit as e:
            assert e.code == 0, e.code
        finally:
            sys.argv = argv
        w = json.loads((out / "ARMS-WORKERS.json").read_text())["512"]
        assert w["arm"] == "native-compile" and w["batch"] == 8, w
        assert w["best_workers"] == 16, w
        assert sorted(w["workers"]) == ["16", "4", "64"], w
        assert w["final"]["status"] == "ok", w
        assert {s[3] for s in seen} == {4, 16, 64}, seen
    print("ok: the worker sweep ranks num_workers at the published arm and keeps the best")


if __name__ == "__main__":
    test_published_prefers_the_extended_climb()
    test_workers_phase_picks_the_count_that_wins()
    test_reference_families_resolve()
    test_binder_spec_keeps_its_relative_input()
    test_climb_inverts_the_screen()
    test_climb_stops_on_the_card_not_the_ladder()
    test_extended_climb_reuses_every_arms_measured_rungs()
    test_chosen_arm_is_the_one_the_climb_ranked()
    test_partial_arms_json_names_a_supported_arm()
    print("all ok")
