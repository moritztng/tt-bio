#!/usr/bin/env python3
"""One line per result file: the shape, the endpoint, d_1 and the loop gain that produced it."""
import json
import sys

print(f"{'arm':9s} {'rho':>6s} {'exp':>7s} {'r2':>5s} {'share20':>8s} "
      f"{'rel_d2':>10s} {'rel_d20':>10s} {'med20':>10s} {'d1_ours':>10s} {'floor20':>9s}")
for p in sys.argv[1:]:
    d = json.load(open(p))
    g, s, rows = d["growth_k2_20"], d["feedback_share_of_drive_norm"], d["per_step"]
    r2, r20 = rows[1], rows[-1]
    print(f"{d['arm']:9s} {d['rho']:6.3f} {g['exponent']:+7.3f} {g['r2']:5.3f} "
          f"{s['20']:8.4f} {r2['rel_d']:10.3e} {r20['rel_d']:10.3e} "
          f"{r20['median_per_tensor']:10.3e} {d['d1']['ours_norm']:10.3e} "
          f"{r20['fp32_differencing_floor']:9.2e}")
