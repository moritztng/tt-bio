#!/usr/bin/env python3
"""Render PEROP_*.json as the per-op table. The table is BUILT from the artifact, never typed."""
import json
import sys

d = json.load(open(sys.argv[1]))
out = []
out.append("| op (site) | dtype | grad | rel L2 vs float64 | grade | backward alone | "
           "systematic mean | sigma | bias? | repeat bit-exact |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
for k, c in d["cases"].items():
    for dt, r in c["dtypes"].items():
        if "error" in r:
            out.append(f"| {k} | {dt} | - | ERROR {r['error'][:60]} | | | | | | |")
            continue
        for g, v in r["grads"].items():
            bw = v.get("rel_l2_bw_only_median")
            out.append(
                f"| `{k}` | {dt} | `{g}` | {v['rel_l2_median']:.3e} | **{v['grade']}** | "
                f"{('%.2e' % bw) if bw is not None else '-'} | {v['systematic_mean']:+.2e} | "
                f"{v['systematic_sigma']:.0f} | **{v['verdict_bias']}** | "
                f"{'yes' if v['bitexact_repeat'] else 'NO'} |")
print("\n".join(out))
print()
print("| case | fd rel err (bar 1e-6) | forward rel L2 | device row sum - 1 |")
print("|---|---|---|---|")
for k, c in d["cases"].items():
    for dt, r in c["dtypes"].items():
        if "error" in r:
            continue
        rs = r.get("device_row_sum", {}).get("mean_minus_one")
        print(f"| `{k}` {dt} | {r['fd']['rel_err']:.1e} "
              f"{'OK' if r['fd']['validated'] else '**FAIL**'} | "
              f"{r['forward_rel_l2_median']:.3e} | "
              f"{('%+.2e' % rs) if rs is not None else '-'} |")
