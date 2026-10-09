"""Owner tables for state/spd/CENSUS.md from analyze.py's <prefix>_rows.json + <prefix>_summary.json.

Every op signature gets the SPD row that owns its cost (state/spd/PLAN.md "Rows"), or NOBODY. spd-overhead concluded
06:35Z: its swiglu went to spd-swiglu, its idle tail and layout ops to spd-hostlap. spd-pair takes the pair traffic
around the triangle ops (layer norms, residual adds, permutes, concats) from the subsystem rows. Idle gaps are spd-hostlap's.

usage: owners.py LABEL PREFIX UNPROFILED_WALL_S|- [TARGET_S|-] [ROW]   (prints markdown; - = none)
ROW gives every op to one row: another model's census, where that model's SPD row owns the whole fold.
"""
import collections, json, re, sys

LABEL, PREFIX, WALL = sys.argv[1], sys.argv[2], sys.argv[3]
TARGET = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
ROW = sys.argv[5] if len(sys.argv) > 5 else None
rows = json.load(open(f"{PREFIX}_rows.json")); summ = json.load(open(f"{PREFIX}_summary.json"))
ATTN_OPS = re.compile(r"scaled_dot_product_attention|softmax|sdpa")


PAIR_OPS = re.compile(r"^(layer_norm|add_?|permute|concat)$")
PAIR_CLS = ("triangle multiplication", "triangle attention", "pairformer attention", "transitions")


def owner(r):
    cls, ph, op, reg = r["cls"], r["phase"], r["op"].split(".")[-1], r["reg"].split("/")
    if ph not in ("confidence", "trunk: MSA module") and any(k in cls for k in PAIR_CLS) and PAIR_OPS.match(op) and "swiglu" not in r["site"]:
        return "spd-pair"
    if "triangle multiplication" in cls:
        return "spd-trimul"
    if "outer product mean" in cls or "pair-weighted" in cls or ph == "trunk: MSA module":
        return "spd-msa"
    if any(k in cls for k in ("triangle attention", "pairformer attention", "diffusion attention")):
        return "spd-attn"
    if "atom transformer" in cls:
        return "spd-attn" if ATTN_OPS.search(op) or "atom_tx_bias" in reg else (
            "spd-diffusion" if ph == "diffusion" else "spd-attn")
    if ph == "confidence":
        return "spd-attn" if ATTN_OPS.search(op) or "apb" in reg else "NOBODY"
    if "typecast/layout" in cls or "unhooked" in cls:
        return "spd-hostlap"
    if ph == "diffusion" or "dit" in reg or "sampler" in reg:
        return "spd-diffusion"
    if "transitions" in cls:
        return "spd-swiglu"
    return "NOBODY"


for r in rows:
    r["owner"] = ROW or owner(r)
dev, gap = summ["device_kernel_s"], summ["device_idle_gap_s"]
recon = dev + gap
# scale so device + idle sums to the unprofiled wall: the profiler adds a little per-program time
WALL = float(WALL) if WALL != "-" else None
k = WALL / recon if WALL and recon else 1.0
print(f"### {LABEL}\n")
vs = (f"against the unprofiled wall {WALL:.1f} s ({(recon / WALL - 1) * 100:+.1f} %)" if WALL
      else "(no unprofiled wall of this tree to reconcile against)")
print(f"Device kernel {dev:.1f} s + idle gaps {gap:.1f} s = {recon:.1f} s {vs}. AICLK {summ['aiclk']['median']} MHz median (min {summ['aiclk']['min']}, "
      f"n {summ['aiclk']['n']}). {summ['n_calls']:,} calls, {summ['n_sigs']:,} signatures, {summ['n_programs']:,} "
      f"programs, {summ['missing']} ids missing. Profiled wall {summ['wall_s']:.0f} s.\n")
by_owner = collections.defaultdict(lambda: [0.0, 0.0])
for r in rows:
    by_owner[r["owner"]][0] += r["s"]; by_owner[r["owner"]][1] += r["gap_s"]
print("| owner | device s | idle gap before its ops, s | share of fold % |\n|---|---|---|---|")
for o, (s, g) in sorted(by_owner.items(), key=lambda x: -sum(x[1])):
    print(f"| {o} | {s:.1f} | {g:.1f} | {(s + g) / recon * 100:.1f} |")
print(f"\nIdle gaps total {gap:.1f} s, all spd-hostlap's by PLAN (host idle, dispatch); the column above says which "
      f"lever row's ops they sit in front of.\n")
print("| phase | device s | share % |\n|---|---|---|")
for p, v in summ["phases"].items():
    print(f"| {p} | {v['s']:.1f} | {v['share'] * 100:.1f} |")
by_cls = collections.defaultdict(float)
for r in rows:
    by_cls[r["cls"]] += r["s"]
print("\n| class | device s | share % |\n|---|---|---|")
for c, v in sorted(by_cls.items(), key=lambda x: -x[1]):
    if v >= 0.5:
        print(f"| {c} | {v:.1f} | {v / recon * 100:.1f} |")
print("\n| # | owner | share % | s | op | class | call site | shapes | calls | us/call | roof frac | bound | idle before s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for i, r in enumerate(sorted(rows, key=lambda r: -r["s"])[:30], 1):
    sh = " x ".join(str(x) for x in r["shapes"][:2])
    print(f"| {i} | {r['owner']} | {r['s'] / recon * 100:.2f} | {r['s']:.2f} | {r['op'].replace('ttnn.', '')} | "
          f"{r['cls']} | {r['site']} | {sh} | {r['calls']:.0f} | {r['us']:.0f} | {r['roof']:.2f} | {r['bound']} | "
          f"{r['gap_s']:.2f} |")
nob = collections.defaultdict(lambda: [0.0, 0.0, None])
for r in rows:
    if r["owner"] == "NOBODY":
        key = (r["phase"], r["cls"]); nob[key][0] += r["s"]; nob[key][1] += r["gap_s"]
        if nob[key][2] is None or r["s"] > nob[key][2]["s"]:
            nob[key][2] = r
print("\nNOBODY (unowned device cost by phase x class; idle gaps are spd-hostlap's and not listed):\n")
print("| phase | class | device s | idle before s | biggest op |\n|---|---|---|---|---|")
for (p, c), (s, g, top) in sorted(nob.items(), key=lambda x: -x[1][0]):
    if s >= 0.5:
        print(f"| {p} | {c} | {s:.1f} | {g:.1f} | {top['op'].replace('ttnn.', '')} {top['site']} {top['s']:.1f} s |")
if TARGET:
    print(f"\nSeconds to target: {WALL:.1f} s now, target {TARGET:.0f} s, {WALL - TARGET:.1f} s must go "
          f"({(1 - TARGET / WALL) * 100:.0f} % of the fold).")
