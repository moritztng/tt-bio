import json, sys, statistics as st
from pathlib import Path
D = Path("/home/ttuser/.coworker/wt/protenix-trunk--p3-l1-source-068/perf/p3l1s068")
arms = ["base", "base2", "base3", "x7", "pwa", "tpl", "all"]
w = {}
for a in arms:
    f = D / f"ops_{a}.json"
    if f.exists():
        d = json.load(open(f))
        w[a] = {"wall": d["wall"], "plddt": d["plddt"], "fold": d["median_fold_s"],
                "inst": d["instrumented_fold_s"], "refused": d["l1_out_refused"]}
KEYS = ["block:PairformerLayer", "body:TriangleMultiplication", "body:TriangleAttention",
        "body:AttentionPairBias", "body:PairWeightedAveraging", "residual_add_",
        "_pair_proj_linear|in0=[1, 298, 320, 256]|w=[256, 256]",
        "_pair_proj_linear|in0=[298, 320, 256]|w=[256, 256]",
        "_narrow_proj_linear|in0=[1, 298, 320, 256]|w=[256, 16]",
        "_narrow_proj_linear|in0=[298, 320, 256]|w=[256, 1]",
        "_narrow_proj_linear|in0=[1, 298, 320, 256]|w=[256, 64]",
        "_pair_proj_linear|in0=[1, 298, 320, 64]|w=[64, 64]",
        "_pair_proj_linear|in0=[298, 320, 64]|w=[64, 64]",
        "shared_layer_norm|in=[1, 298, 320, 256]",
        "shared_layer_norm|in=[298, 320, 256]"]
base = [b for b in ("base", "base2", "base3") if b in w]
print("baseline spread over", base)
hdr = f"{'class':62s} {'calls':>5} " + "".join(f"{b:>10}" for b in base) + f"{'spread%':>8}"
print(hdr)
ctrl = {}
for k in KEYS:
    vals = [w[b]["wall"].get(k, {}).get("wall_ms") for b in base]
    if any(v is None for v in vals):
        continue
    n = w[base[0]]["wall"][k]["calls"]
    m = st.mean(vals)
    ctrl[k] = m
    print(f"{k[:62]:62s} {n:>5} " + "".join(f"{v:>10.2f}" for v in vals)
          + f"{(max(vals)-min(vals))/m*100:>8.2f}")
print()
live = [a for a in ("x7", "pwa", "tpl", "all") if a in w]
print(f"{'class':62s} {'ctrl':>10} " + "".join(f"{a:>10}{'  d':>9}" for a in live))
tot = {a: 0.0 for a in live}
for k in KEYS:
    if k not in ctrl:
        continue
    row = f"{k[:62]:62s} {ctrl[k]:>10.2f} "
    for a in live:
        v = w[a]["wall"].get(k, {}).get("wall_ms")
        if v is None:
            row += f"{'-':>10}{'-':>9}"
            continue
        d = v - ctrl[k]
        if not k.startswith(("block:", "body:", "stage:")):
            tot[a] += d
        row += f"{v:>10.2f}{d:>+9.2f}"
    print(row)
print()
print(f"{'SUM of op-class walls (block/body rows excluded)':62s} {'':>10} "
      + "".join(f"{'':>10}{tot[a]:>+9.2f}" for a in live))
print()
for a in base + live:
    print(f"{a:6s} plddt={w[a]['plddt']}  fold={w[a]['fold']}  instrumented={w[a]['inst']}  "
          f"refused={w[a]['refused']}")
