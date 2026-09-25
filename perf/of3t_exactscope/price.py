"""of3t-exactscope: what the narrowed scope actually saves, and what the exact layer norm costs.

Three arms of one ladder in one chain on one card: `noexact` (neither op exact), `softmax`
(EXACT_TRAINING_OPS = ("softmax",)) and `base` (both, the default). The accuracy ladder that
grades them is of3t-stackexact's, read by `scope.py`; this file is only the seconds.

The scope of each arm is asserted from the mechanism on BOTH legs, never from the argument
passed: `installed_inside_the_tape` is read off `exact_softmax_installed()` /
`exact_layer_norm_installed()` while the tape is open, and `host_roundtrips` is the exact
verbs' own call counters, differenced across the forward and across the backward separately.
An arm that silently kept both ops would read as "softmax alone is nearly free" and be the most
flattering possible wrong answer.

Writes PRICE.json. CPU only.
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).parent
ARMS = ("noexact", "softmax", "base")
SCOPE = {"noexact": "neither exact", "softmax": 'EXACT_TRAINING_OPS = ("softmax",)',
         "base": 'EXACT_TRAINING_OPS = ("softmax", "layer_norm") -- the default'}


def ok_scope(d, arm):
    """Did the arm run the scope it claims? Returns the reading, and whether it is consistent."""
    want = {"noexact": (False, False), "softmax": (True, False), "base": (True, True)}[arm]
    ins = d["env"].get("installed_inside_the_tape") or {}
    legs = {leg: d[leg]["host_roundtrips"] for leg in ("forward", "backward")}
    served = {op: {leg: sum(v for k, v in legs[leg][op].items() if k in ("verb", "raw", "bw"))
                   for leg in legs} for op in ("softmax", "layer_norm")}
    consistent = (
        bool(ins.get("softmax")) == want[0] and bool(ins.get("layer_norm")) == want[1]
        and all((served["softmax"][leg] > 0) == want[0] for leg in legs)
        and all((served["layer_norm"][leg] > 0) == want[1] for leg in legs))
    return {"expected_installed": {"softmax": want[0], "layer_norm": want[1]},
            "installed_inside_the_tape": ins,
            "ops_active": d["env"].get("exact_training_ops"),
            "calls_served_per_leg": served,
            "counters_per_leg": legs,
            "consistent": consistent}


def main() -> int:
    arms = {}
    for a in ARMS:
        p = HERE / "out" / f"hist_384_{a}.json"
        if not p.exists():
            print(f"!! missing {p}")
            continue
        d = json.loads(p.read_text())
        if "forward" not in d or "s" not in d.get("backward", {}):
            # bwprof dumps progressively, so a live arm's file is real but incomplete
            print(f"!! {a} still running or incomplete -- skipped")
            continue
        arms[a] = d

    out = {"what": __doc__.split("\n\n")[0], "row": "of3t-exactscope",
           "axis": "arm-to-arm, three arms back to back in one chain on one card, crop 384, "
                   "1 trunk cycle, taped; the accuracy that grades them is of3t-stackexact's "
                   "ladder, not re-measured here",
           "host": "pc", "card": 0, "board_class": "p150a",
           "board_serial": "000004033191410F",
           "why_board_matters": "qb1 p150a and qb2 p300c run the same 512 aa fold 17.39 s "
                                "against 14.59 s at the same 1350 MHz",
           "arms": {}}

    for a, d in arms.items():
        f, b = d["forward"], d["backward"]
        out["arms"][a] = {
            "scope": SCOPE[a],
            "scope_control": ok_scope(d, a),
            "forward_s": f["s"], "backward_s": b["s"],
            "step_s": round(f["s"] + b["s"], 2),
            "backward_ok": b.get("ok"),
            "tape_nodes": b.get("tape_nodes"),
            "forward_verb_calls": f["verb_calls"], "backward_verb_calls": b["verb_calls"],
            "params_with_grad": b.get("params_with_grad"),
            "params_declared": b.get("params_declared"),
            "aiclk_mhz_sampled_DURING": d["env"]["aiclk_during"],
            "loadavg_at_start": d["env"]["loadavg"],
            "commit": d["env"]["commit"][:9],
            # the same site under both regimes: the exact host closure and the shipped device
            # one fire the same number of times, so the ratio is the round trip's price at
            # one site on one board in one run -- not a cross-board quote.
            "layer_norm_bw_closure": {
                which: next(({"s": r["total_s"], "fired": r["fired"],
                              "ms_per_fire": r["ms_per_fire"], "verbs": r["verbs"]}
                             for r in d.get("by_closure", []) if tag in r["closure"]), None)
                for which, tag in (("exact_host", "_v_exact_layer_norm"),
                                   ("shipped_device", "_taped_layer_norm"))},
            "softmax_bw_closure": {
                which: next(({"s": r["total_s"], "fired": r["fired"],
                              "ms_per_fire": r["ms_per_fire"]}
                             for r in d.get("by_closure", []) if tag in r["closure"]), None)
                for which, tag in (("exact_host", "host_f64_softmax"),
                                   ("shipped_device", "_v_softmax"))},
        }

    def leg(a, k):
        return out["arms"][a][k]

    if {"base", "softmax", "noexact"} <= set(out["arms"]):
        out["the_route_is_the_same"] = {
            "tape_nodes": {a: leg(a, "tape_nodes") for a in ARMS},
            "params_with_grad": {a: leg(a, "params_with_grad") for a in ARMS},
            "identical": len({leg(a, "tape_nodes") for a in ARMS}) == 1
            and len({leg(a, "params_with_grad") for a in ARMS}) == 1}
        out["price"] = {}
        for k in ("forward_s", "backward_s", "step_s"):
            base, sm, ne = leg("base", k), leg("softmax", k), leg("noexact", k)
            out["price"][k] = {
                "base (softmax+LN)": base, "softmax only": sm, "noexact": ne,
                "the exact LAYER NORM costs (base - softmax)": round(base - sm, 2),
                "the exact SOFTMAX costs (softmax - noexact)": round(sm - ne, 2),
                "narrowing to softmax saves": round(base - sm, 2),
                "narrowing to softmax saves pct_of_base": round((base - sm) / base * 100, 1)
                if base else None,
                "x_base_over_softmax": round(base / sm, 3) if sm else None,
                "x_base_over_noexact": round(base / ne, 3) if ne else None}

    # same site, same fire count, one board, one run: what the host round trip costs
    site = {}
    for op, ex, sh in (("layer_norm", "base", "softmax"), ("softmax", "softmax", "noexact")):
        key = f"{op}_bw_closure" if op == "layer_norm" else "softmax_bw_closure"
        e = out["arms"].get(ex, {}).get(key, {}).get("exact_host")
        d_ = out["arms"].get(sh, {}).get(key, {}).get("shipped_device")
        if e and d_:
            site[op] = {"exact_host": e, "shipped_device": d_,
                        "same_fire_count": e["fired"] == d_["fired"],
                        "x_host_over_device": round(e["s"] / d_["s"], 1) if d_["s"] else None}
    out["backward_closure_at_one_site"] = site

    # HOST QUIET, and why the exact arms cannot make it green
    lf = HERE / "out" / "loadsample.tsv"
    if lf.exists():
        import statistics as stat
        rows = [l.split("\t") for l in lf.read_text().splitlines()[1:] if l.strip()]
        la = [float(r[2]) for r in rows if len(r) > 2]
        own = [float(x) for x in (r[3] for r in rows if len(r) > 3) if x not in ("", "0.0")]
        out["host_quiet"] = {
            "before_the_chain": "loadavg1 0.86, GREEN, and zero other processes held a "
                                "tenstorrent fd -- the chain was the box's only device tenant "
                                "from start to finish",
            "noexact_arm": "started GREEN at loadavg1 1.06",
            "exact_arms": "started RED, 2.17 over the 2.00 ceiling",
            "why": "the exact arms ARE host float64 work. The arm's own python3 was sampled at "
                   "204-208 % CPU, so an exact arm puts ~2.0 on loadavg1 by itself and cannot "
                   "read under a 2.00 ceiling no matter how idle the box is. The guard was built "
                   "for a device-bound timing read where host load is contamination; here the "
                   "host load is the subject being measured.",
            "foreign_load_measured": "NOT zero, and not explained away: openclaw-gateway "
                                     "(~140 % in bursts), tt-smi (~85-103 %) and other claude "
                                     "sessions appear in the samples. Foreign host CPU competes "
                                     "with exactly the host float64 threads the exact arms are "
                                     "made of, so base and softmax seconds are UPPER BOUNDS.",
            "what_survives_it": "the LAYER NORM PRICE is a difference of two arms contaminated "
                                "the same way and sampled minutes apart, so it is more robust "
                                "than either absolute. The counts -- tape nodes, parameters with "
                                "a gradient, call counters -- are load-insensitive.",
            "samples": len(la),
            "loadavg1_median_during_arms": round(stat.median(la), 2) if la else None,
            "loadavg1_max_during_arms": max(la) if la else None,
            "arm_own_cpu_pct_median": round(stat.median(own), 0) if own else None,
            "file": "out/loadsample.tsv",
            "caveat": "the sampler's `other_cpu_pct` column INCLUDES the arm's own python3; "
                      "read `other_procs` for the names.",
        }

    (HERE / "PRICE.json").write_text(json.dumps(out, indent=1))

    print(f"pc card 0, Blackhole {out['board_class']}, crop 384, 1 cycle, taped\n")
    print(f"{'arm':9s} {'scope':52s} {'fwd s':>8s} {'bwd s':>9s} {'step s':>9s} "
          f"{'AICLK med':>9s} {'scope ok':>9s}")
    for a in ARMS:
        if a not in out["arms"]:
            continue
        r = out["arms"][a]
        clk = r["aiclk_mhz_sampled_DURING"].get("0", {})
        print(f"{a:9s} {r['scope']:52s} {r['forward_s']:8.2f} {r['backward_s']:9.2f} "
              f"{r['step_s']:9.2f} {clk.get('median','?'):>9} "
              f"{str(r['scope_control']['consistent']):>9s}")
    if "price" in out:
        print()
        for k, v in out["price"].items():
            print(f"{k:12s}  exact LAYER NORM costs {v['the exact LAYER NORM costs (base - softmax)']:8.2f} s"
                  f"   exact SOFTMAX costs {v['the exact SOFTMAX costs (softmax - noexact)']:8.2f} s"
                  f"   narrowing saves {v['narrowing to softmax saves pct_of_base']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
