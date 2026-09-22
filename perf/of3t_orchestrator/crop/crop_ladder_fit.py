"""Is crop 640 a near miss, or is it out of reach?

D14 records the backward DRAM ladder and says "the ladder stops between 384 and 640", with 640
failing at 34.215 GB of a 34.22 GB card. That reads like a 5 MB miss. But 34.215 GB is the
high-water REACHED BEFORE IT DIED -- the card ceiling -- not the requirement, and 640's live
allocation count (6016) is LOWER than 384's (6514) precisely because it never got to the peak.

Fit the passing rungs and extrapolate, so the gap is stated in GB rather than in adjectives.
"""
import json, math

# D14, of3t-l1 pass 51, qb2 p300c, AICLK 1350 during. Backward DRAM high-water.
LADDER = [(128, 2.522, 5581, "PASS"), (256, 7.384, 6249, "PASS"),
          (384, 17.983, 6514, "PASS"), (640, 34.215, 6016, "FAIL (card ceiling 34.22 GB)")]
CARD_GB = 34.22
PASSING = [(n, g) for n, g, _, v in LADDER if v == "PASS"]

def fit_affine_n2(p, q):
    """g = a + b*N^2 through two rungs."""
    (n1, g1), (n2, g2) = p, q
    b = (g2 - g1) / (n2**2 - n1**2)
    return g1 - b * n1**2, b

def power_exp(p, q):
    (n1, g1), (n2, g2) = p, q
    return math.log(g2 / g1) / math.log(n2 / n1)

out = {"ladder": [{"crop": n, "backward_dram_gb": g, "live_allocs": a, "verdict": v}
                  for n, g, a, v in LADDER], "card_gb": CARD_GB}

out["pairwise_power_exponent"] = {
    f"{PASSING[i][0]}->{PASSING[i+1][0]}": round(power_exp(PASSING[i], PASSING[i+1]), 4)
    for i in range(len(PASSING) - 1)}

a, b = fit_affine_n2(PASSING[1], PASSING[2])          # 256 -> 384, the two rungs nearest 640
out["fit_256_384"] = {"form": "a + b*N^2", "a_gb": round(a, 4), "b_gb_per_token2": b}
for n in (640, 768):
    need = a + b * n * n
    out[f"predicted_{n}"] = {"backward_dram_gb": round(need, 2),
                             "times_card": round(need / CARD_GB, 2),
                             "shortfall_gb": round(need - CARD_GB, 2)}
a2, b2 = fit_affine_n2(PASSING[0], PASSING[2])        # 128 -> 384, the widest passing span
out["fit_128_384"] = {"a_gb": round(a2, 4), "b_gb_per_token2": b2,
                      "predicted_640_gb": round(a2 + b2 * 640**2, 2),
                      "predicted_768_gb": round(a2 + b2 * 768**2, 2)}
out["caveat"] = ("Two-point extrapolations under an a + b*N^2 model. The 128->256 rung fits a "
                 "different exponent (1.55) from 256->384 (2.20), so this is an estimate with a "
                 "stated model, not a measurement. What is robust is the ORDER: 640 is not a "
                 "near miss. Checkpointing or allocator changes move the constant, not the N^2.")
json.dump(out, open("crop_ladder_fit.json", "w"), indent=1)

print(f"card = {CARD_GB} GB")
print("pairwise power-law exponents:", out["pairwise_power_exponent"])
print(f"fit 256->384:  g = {a:+.3f} + {b:.6e} * N^2")
for n in (640, 768):
    d = out[f"predicted_{n}"]
    print(f"  crop {n}: needs ~{d['backward_dram_gb']} GB = {d['times_card']}x the card, "
          f"short by {d['shortfall_gb']} GB")
print(f"fit 128->384 cross-check: 640 ~{out['fit_128_384']['predicted_640_gb']} GB, "
      f"768 ~{out['fit_128_384']['predicted_768_gb']} GB")
