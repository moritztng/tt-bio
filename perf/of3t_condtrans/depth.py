import json, re
import numpy as np
from scipy.stats import spearmanr

d = json.load(open("perf/of3t_condtrans/MECHANISM.json"))["by_site"]
B = re.compile(r"^diffusion_transformer\.blocks\.(\d+)\.")
rows = []
for nm, e in d.items():
    m = B.match(nm)
    if m and e.get("cotangent_cos_dev_vs_ref") is not None:
        rows.append((int(m.group(1)), e))
rows.sort()
b = np.array([r[0] for r in rows])
cos = np.array([r[1]["cotangent_cos_dev_vs_ref"] for r in rows])
crel = np.array([r[1]["cotangent_rel_dev_vs_ref"] for r in rows])
nr = np.array([r[1]["cotangent_norm_ratio"] for r in rows])
ours = np.array([r[1]["OURS_vs_f64"] for r in rows])
print(f"{'blk':>3} {'cot rel':>8} {'cot cos':>8} {'cot r':>7} {'dW rel':>8}")
for (bb, e) in rows:
    print(f"{bb:3d} {e['cotangent_rel_dev_vs_ref']:8.4f} {e['cotangent_cos_dev_vs_ref']:+8.3f} "
          f"{e['cotangent_norm_ratio']:7.3f} {e['OURS_vs_f64']:8.4f}")
print()
print(f"spearman cot_cos vs block index  {spearmanr(cos, b).statistic:+.3f}  p={spearmanr(cos, b).pvalue:.2e}")
print(f"spearman cot_rel vs block index  {spearmanr(crel, b).statistic:+.3f}  p={spearmanr(crel, b).pvalue:.2e}")
print(f"spearman cot_r   vs block index  {spearmanr(nr, b).statistic:+.3f}  p={spearmanr(nr, b).pvalue:.2e}")
print(f"spearman dW_rel  vs block index  {spearmanr(ours, b).statistic:+.3f}  p={spearmanr(ours, b).pvalue:.2e}")
print(f"spearman dW_rel  vs cot_cos      {spearmanr(ours, cos).statistic:+.3f}  p={spearmanr(ours, cos).pvalue:.2e}")
last6 = cos[b >= 18]
mid = cos[(b >= 3) & (b <= 12)]
print(f"\ncos: blocks 18-23 mean {last6.mean():.3f}, blocks 3-12 mean {mid.mean():.3f}, "
      f"block 23 {cos[b==23][0]:.3f}, floor {cos.min():.3f} at block {b[cos.argmin()]}")
print(f"norm ratio range {nr.min():.3f}..{nr.max():.3f}")
