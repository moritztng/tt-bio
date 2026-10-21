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


if __name__ == "__main__":
    test_reference_families_resolve()
    test_binder_spec_keeps_its_relative_input()
    test_climb_inverts_the_screen()
    test_climb_stops_on_the_card_not_the_ladder()
    test_chosen_arm_is_the_one_the_climb_ranked()
    test_partial_arms_json_names_a_supported_arm()
    print("all ok")
