#!/usr/bin/env python3
"""The two arms reduced to the four answers the campaign asked for.

Reads `step_exact_on_<tag>.json`, `step_exact_off_<tag>.json` and the shared profile JSONL,
and writes `ANSWER_<tag>.json`:

  STEP        the exactness-ON step, its six parts, and whether they sum to it
  OFFSTEP     the same step with `exact_training(False)`, cold and steady median
  EXACTPRICE  forward and backward SEPARATELY, each as ON minus OFF on that half -- a
              difference of two measurements on one board in one process, never a split of
              one number by a proxy
  PARTITION   the six parts, plus the memory phases off the profile

Two guards, because this campaign has been burned by both:

  * THE SUM. A six-part table that does not add to the step it claims to decompose is a
    scope mismatch. `residual_s` is published even when it is zero.
  * THE SILICON FLOOR. A ratio over ~8.5x compute / ~11x bandwidth is a scope mismatch every
    time, so `exact_ratio` carries `beats_silicon_floor` beside it. Here the ratio is
    EXPECTED to beat it, because the numerator is host float64 arithmetic and a PCIe round
    trip rather than device work -- and that is precisely why it must not be read as a
    chip-to-chip number. The flag says which.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "perf" / "of3t_stepqb2" / "out"
PARTS = ("trunk_s", "diffusion_s", "losses_s", "seed_upload_s", "backward_s", "optimizer_s")
# The halves the exactness is priced on separately. `seed_upload_s` is the cotangent upload
# and belongs to neither: it is the harness handing the backward its seeds.
FORWARD = ("trunk_s", "diffusion_s", "losses_s")
BACKWARD = ("backward_s",)


def rep(d):
    """The rep a step time should be read off: the steady median if there is one, else rep 0.

    A single rep is cold by definition, and in this harness that is not a compile penalty
    (`step_rekey_b_384.json`: cold 449.4 s against a warm median of 468.9 s), so a lone rep 0
    is quotable -- but it is labelled.
    """
    reps = d.get("reps") or []
    if not reps:
        return None, None
    key = "step_s_UNTAPED" if reps[0].get("step_s") is None else "step_s"
    vals = [(r[key], r) for r in reps if r.get(key) is not None]
    if len(vals) > 1:
        med = statistics.median([v for v, _ in vals[1:]])
        pick = min(vals[1:], key=lambda vr: abs(vr[0] - med))
        return pick[1], {"basis": "steady median", "n": len(vals) - 1,
                         "values_s": [v for v, _ in vals[1:]], "median_s": round(med, 3),
                         "cold_s": vals[0][0]}
    return vals[0][1], {"basis": "rep 0, cold, single repetition", "n": 1,
                        "values_s": [vals[0][0]]}


def parts(r):
    return {p: r.get(p) for p in PARTS if r.get(p) is not None}


def half(r, keys):
    return round(sum(r.get(k) or 0.0 for k in keys), 3)


def phases(jsonl: Path):
    """Dwell and RSS per phase off the 5 Hz profile. The exact softmax's float64 temporary is
    a transient, so the phase's MAX and its floor are different facts and both are kept."""
    seen: dict = {}
    for line in jsonl.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "phase" not in r or "rss_gib" not in r:
            continue
        e = seen.setdefault(r["phase"], {"samples": 0, "t_first": r["t"], "t_last": r["t"],
                                         "rss_min": r["rss_gib"], "rss_max": r["rss_gib"],
                                         "avail_min": r["avail_gib"]})
        e["samples"] += 1
        e["t_last"] = r["t"]
        e["rss_min"] = min(e["rss_min"], r["rss_gib"])
        e["rss_max"] = max(e["rss_max"], r["rss_gib"])
        e["avail_min"] = min(e["avail_min"], r["avail_gib"])
    for e in seen.values():
        e["dwell_s"] = round(e["t_last"] - e["t_first"], 2)
    return seen


def verbs(jsonl: Path):
    return [json.loads(l) for l in jsonl.read_text().splitlines()
            if '"verb"' in l]


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "384"
    on_p, off_p = OUT / f"step_exact_on_{tag}.json", OUT / f"step_exact_off_{tag}.json"
    jsonl = OUT / f"rss_step_{tag}.jsonl"
    on = json.loads(on_p.read_text())
    off = json.loads(off_p.read_text()) if off_p.is_file() else {}

    ans = {"row": "of3t-stepqb2", "tag": tag,
           "env": {k: on.get("env", {}).get(k) for k in
                   ("host", "commit", "branch", "arch", "aiclk_during", "aiclk_line",
                    "loadavg_start", "loadavg_end", "started_utc")},
           "host_facts": on.get("host_facts"), "config": on.get("config"),
           "scope": on.get("scope")}

    r_on, basis_on = rep(on)
    if r_on is None:
        # The arm did not publish a rep, so it died inside one. `fullstep.py` writes a rep only
        # after the optimizer; the per-verb records in the JSONL are written as each verb
        # RETURNS, so they survive a kill and are the only reading of a killed arm that exists.
        # Everything reconstructed here is a LOWER BOUND and the artifact says so in its own
        # field rather than leaving a reader to infer it.
        vs = verbs(jsonl)
        by = {}
        for v in vs:
            by.setdefault(v["verb"], []).append(v["s"])
        got = {"trunk_s": round(sum(by.get("trunk_forward", [])), 3),
               "diffusion_s": round(sum(by.get("diffusion", [])), 3),
               "losses_s": round(sum(by.get("losses", [])), 3),
               "backward_s": round(sum(by.get("backward", [])), 3),
               "optimizer_s": round(sum(by.get("optimizer_step", [])), 3)}
        breach = next((json.loads(l) for l in jsonl.read_text().splitlines()
                       if '"BREACH"' in l), None)
        last = [json.loads(l) for l in jsonl.read_text().splitlines()
                if '"phase"' in l and '"rss_gib"' in l][-1]
        unfinished = last["phase"]
        # The verb that was RUNNING when the guard fired never returned, so its seconds are the
        # wall between the last returned verb and the exit -- a floor under that verb, not it.
        t_done = max((v for v in vs), key=lambda v: v.get("s", 0) and 1) if vs else None
        ans["STEP"] = {
            "basis": {"basis": f"arm KILLED inside `{unfinished}`; reconstructed from the "
                               f"per-verb records, which are written as each verb returns",
                      "n": 0},
            "parts_s": {k: v for k, v in got.items() if v},
            "step_s": None,
            "step_s_lower_bound": round(sum(got.values()), 3),
            "unfinished_phase": unfinished,
            "peak_is_a_lower_bound": True,
            "breach": breach,
            "arm": on.get("arm", {}).get("name") or "exact_on",
            "exact_training": True,
            "aiclk_during_from_fullstep": on.get("env", {}).get("aiclk_during"),
            "note": "env.aiclk_during is absent because fullstep.py writes it on the last line "
                    "of main(); the clock for this arm is in the profile JSONL, sampled at 1 Hz "
                    "off the sysfs class node throughout",
        }
        ans["PARTITION"] = {"seconds": ans["STEP"]["parts_s"],
                            "memory_phases": phases(jsonl), "verbs": vs}
        if off.get("reps"):
            r_off, basis_off = rep(off)
            cold = off["reps"][0]
            ans["OFFSTEP"] = {"basis": basis_off, "parts_s": parts(r_off),
                              "step_s": r_off.get("step_s"),
                              "cold_parts_s": parts(cold), "cold_step_s": cold.get("step_s"),
                              "exact_training": off.get("arm", {}).get("exact_training"),
                              "host_quiet_pre": off.get("arm", {}).get("host_quiet_pre"),
                              "params_with_grad": r_off.get("params_with_grad")}
            fw_on = round(got["trunk_s"] + got["diffusion_s"] + got["losses_s"], 3)
            fw_cold = half(cold, FORWARD)
            fw_steady = half(r_off, FORWARD)
            ans["EXACTPRICE"] = {
                "method": "ON minus OFF on the same board and the same tree. The ON arm is ONE "
                          "COLD rep, so the like-for-like denominator is the OFF arm's rep 0, "
                          "also cold; the steady comparison is given beside it, not instead",
                "forward": {"on_s": fw_on, "off_cold_s": fw_cold, "off_steady_s": fw_steady,
                            "price_vs_cold_s": round(fw_on - fw_cold, 3),
                            "ratio_vs_cold": round(fw_on / fw_cold, 3),
                            "ratio_vs_steady": round(fw_on / fw_steady, 3),
                            "contains": list(FORWARD), "complete": True},
                "backward": {"on_s_lower_bound": got["backward_s"] or None,
                             "off_cold_s": cold.get("backward_s"),
                             "off_steady_s": r_off.get("backward_s"),
                             "complete": False,
                             "note": "the ON backward was killed by the memory floor guard "
                                     "before it returned, so only a floor under it exists"},
            }
        p2 = OUT / f"ANSWER_{tag}.json"
        p2.write_text(json.dumps(ans, indent=1, default=str))
        print(json.dumps({k: ans[k] for k in ("STEP", "OFFSTEP", "EXACTPRICE") if k in ans},
                         indent=1, default=str))
        print("WROTE", p2)
        return 0
    ans["STEP"] = {"basis": basis_on, "parts_s": parts(r_on) if r_on else None,
                   "step_s": r_on.get("step_s") if r_on else None,
                   "arm": on.get("arm", {}).get("name"),
                   "exact_training": on.get("arm", {}).get("exact_training"),
                   "host_quiet_pre": on.get("arm", {}).get("host_quiet_pre"),
                   "host_quiet_post": on.get("arm", {}).get("host_quiet_post"),
                   "params_with_grad": r_on.get("params_with_grad") if r_on else None,
                   "tape_nodes": r_on.get("tape_nodes") if r_on else None}
    if r_on:
        s = sum(r_on.get(p) or 0.0 for p in PARTS)
        ans["STEP"]["parts_sum_s"] = round(s, 3)
        ans["STEP"]["residual_s"] = round((r_on.get("step_s") or s) - s, 3)

    if off.get("reps"):
        r_off, basis_off = rep(off)
        ans["OFFSTEP"] = {"basis": basis_off, "parts_s": parts(r_off),
                          "step_s": r_off.get("step_s"),
                          "exact_training": off.get("arm", {}).get("exact_training"),
                          "host_quiet_pre": off.get("arm", {}).get("host_quiet_pre"),
                          "capture_reused_from_arm": off.get("capture_reused_from_arm"),
                          "params_with_grad": r_off.get("params_with_grad")}
        if r_on:
            fw_on, fw_off = half(r_on, FORWARD), half(r_off, FORWARD)
            bw_on, bw_off = half(r_on, BACKWARD), half(r_off, BACKWARD)
            st_on, st_off = r_on.get("step_s"), r_off.get("step_s")
            ratio = round(st_on / st_off, 3) if st_on and st_off else None
            ans["EXACTPRICE"] = {
                "method": "ON minus OFF on the same board, same process, same capture. "
                          "Two measurements differenced, never one number split",
                "forward": {"on_s": fw_on, "off_s": fw_off,
                            "price_s": round(fw_on - fw_off, 3),
                            "ratio": round(fw_on / fw_off, 3) if fw_off else None,
                            "contains": list(FORWARD)},
                "backward": {"on_s": bw_on, "off_s": bw_off,
                             "price_s": round(bw_on - bw_off, 3),
                             "ratio": round(bw_on / bw_off, 3) if bw_off else None,
                             "contains": list(BACKWARD)},
                "step": {"on_s": st_on, "off_s": st_off,
                         "price_s": round(st_on - st_off, 3) if st_on and st_off else None,
                         "ratio": ratio},
                "beats_silicon_floor": (None if ratio is None else ratio > 11.0),
                "why_that_is_expected": "the numerator is host float64 arithmetic and a PCIe "
                                        "round trip per softmax and per layer norm, not device "
                                        "work, so this is NOT a chip-to-chip ratio and the "
                                        "silicon floor does not apply to it",
            }

    ans["PARTITION"] = {"seconds": ans["STEP"]["parts_s"], "memory_phases": phases(jsonl),
                        "verbs": verbs(jsonl)}
    p = OUT / f"ANSWER_{tag}.json"
    p.write_text(json.dumps(ans, indent=1, default=str))
    print(json.dumps({k: ans[k] for k in ("STEP", "OFFSTEP", "EXACTPRICE") if k in ans},
                     indent=1, default=str))
    print("WROTE", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
