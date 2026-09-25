#!/usr/bin/env python3
"""The bf16:bfp8 pairing, read straight off the round artifacts. No device, pure arithmetic.

Round 1 is the only round both arms reach, because every bfp8 arm stops there. It is also the
round that pays the trace capture, so the capture is subtracted out: `wire.captures[0].capture_s`
happens inside the first `_taped` call, and the residual is the two taped calls the round makes.
Round 1s bf16 residual is checked against bf16s own steady-state rounds so the subtraction is
not taken on trust.
"""
import json, pathlib, statistics as st

D = pathlib.Path(__file__).parent
ARMS = {
    "bf16": ["round_trace_bf16_a_seed100", "round_trace_bf16_b_seed100", "round_finite_bf16_seed100"],
    "b8":   ["round_trace_b8_a_seed100",   "round_trace_b8_b_seed100",   "round_finite_b8_seed100"],
}

def load(n):
    return json.load(open(D / f"{n}.json"))

res = {}
for arm, names in ARMS.items():
    rows = []
    for n in names:
        d = load(n)
        r1 = [r for r in d["rounds"] if r["i"] == 1][0]
        cap = d["wire"]["captures"][0]["capture_s"]
        ai = r1["aiclk"]
        rows.append({
            "run": n, "capture_s": cap,
            "taped_resid": r1["trunk"]["taped"] - cap,
            "backward": r1["trunk"]["backward"],
            "aiclk_med": ai.get("median"), "aiclk_min": ai.get("min"), "aiclk_n": ai.get("n"),
            "load1": r1["load1"], "rounds_completed": max(x["i"] for x in d["rounds"]),
            "taped_calls": d["calls"]["taped"], "bwd_calls": d["calls"]["backward"],
            "commit": d["stamp"]["commit"][:9], "host": d["stamp"]["host"],
            "card": d["stamp"]["card"], "board": d["stamp"].get("subsystem_device"),
        })
    res[arm] = rows

for arm, rows in res.items():
    print(f"--- {arm} ---")
    for r in rows:
        print("  %-30s cap=%6.2f taped2=%.3f bwd=%.3f aiclk med=%s min=%s n=%s load=%.1f "
              "rounds=%d taped=%d bwd=%d" % (
              r["run"], r["capture_s"], r["taped_resid"], r["backward"], r["aiclk_med"],
              r["aiclk_min"], r["aiclk_n"], r["load1"], r["rounds_completed"],
              r["taped_calls"], r["bwd_calls"]))

print()
for k in ("taped_resid", "backward", "capture_s"):
    a = st.median([r[k] for r in res["bf16"]]); b = st.median([r[k] for r in res["b8"]])
    print("%-12s bf16 median %.3f s | bfp8 median %.3f s | bfp8/bf16 = %.3f" % (k, a, b, b / a))

# bf16 steady state, to show round 1 residual is a fair stand-in for a round
d = load("round_finite_bf16_seed100")
rr = sorted((r for r in d["rounds"] if r["i"] >= 1), key=lambda r: r["i"])
dt = [rr[i + 1]["trunk"]["taped"] - rr[i]["trunk"]["taped"] for i in range(len(rr) - 1)]
db = [rr[i + 1]["trunk"]["backward"] - rr[i]["trunk"]["backward"] for i in range(len(rr) - 1)]
print("\nbf16 steady state over rounds 2-11 of round_finite_bf16: taped2 median %.3f s, "
      "backward median %.3f s (n=%d)" % (st.median(dt), st.median(db), len(dt)))

print("\nlosses:")
for arm, n in (("bf16", "round_finite_bf16_seed100"), ("b8", "round_finite_b8_seed100")):
    print(" ", arm, [(l["round"], round(l["loss"], 4), l["finite"]) for l in load(n)["losses"]])
