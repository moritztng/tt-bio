#!/usr/bin/env python3
"""A44's mass field three ways from PERTENSOR.json (pertensor_draws.py):
per draw as score.py reads it, per tensor on the six-draw mean, and the tensors that lose on
most draws. Every draw's mass is normalised to 1 before averaging, so no draw outweighs another.

    drawmass.py PERTENSOR.json > DRAWMASS.json
"""
import json
import re
import sys

D = {int(k): v for k, v in json.load(open(sys.argv[1])).items()}
K = sorted(D)
names = sorted(set.intersection(*(set(D[k]) for k in K)))
frac = {k: {n: D[k][n][0] / sum(v[0] for v in D[k].values()) for n in D[k]} for k in K}

per_draw = {k: round(sum(frac[k][n] for n in D[k] if D[k][n][1] <= D[k][n][2]), 4) for k in K}
w = {n: sum(frac[k][n] for k in K) / len(K) for n in names}
mo = {n: sum(D[k][n][1] for k in K) / len(K) for n in names}
mb = {n: sum(D[k][n][2] for k in K) / len(K) for n in names}
losers = sorted((n for n in names if mo[n] > mb[n]), key=lambda n: -w[n])
nlose = {n: sum(D[k][n][1] > D[k][n][2] for k in K) for n in names}
fam = re.compile(r"diffusion_transformer\.blocks\.\d+\.attention_pair_bias\.layer_norm_a\.layer_norm_s\.weight")
famn = [n for n in names if fam.fullmatch(n.split("diffusion_module.")[-1]) or fam.search(n)]
out = {
    "per_draw_mass_at_or_better": per_draw,
    "mean_mass_at_or_better": round(1 - sum(w[n] for n in losers), 4),
    "mean_losers_top": [{"tensor": n, "mass": round(w[n], 5), "ours": round(mo[n], 4),
                         "bf16": round(mb[n], 4), "draws_lost": nlose[n]} for n in losers[:15]],
    "lost_on_5_or_6_draws_mass": round(sum(w[n] for n in names if nlose[n] >= 5), 5),
    "lost_on_5_or_6_draws_top": [{"tensor": n, "mass": round(w[n], 5), "ratio": round(mo[n] / mb[n], 3)}
                                 for n in sorted((n for n in names if nlose[n] >= 5), key=lambda n: -w[n])[:10]],
    "layer_norm_s_family": {"n": len(famn), "mass": round(sum(w[n] for n in famn), 4),
                            "ours_mean": round(sum(w[n] * mo[n] for n in famn) / sum(w[n] for n in famn), 4),
                            "bf16_mean": round(sum(w[n] * mb[n] for n in famn) / sum(w[n] for n in famn), 4),
                            "per_draw": {k: [round(sum(frac[k][n] for n in famn), 4),
                                             sum(D[k][n][1] > D[k][n][2] for n in famn)] for k in K}},
}
print(json.dumps(out, indent=1))
