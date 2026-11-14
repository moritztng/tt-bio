"""Summarise a bench.py sweep against the measured roofs.

  per class: baseline, best arm (and the best arm per format), speedup, AICLK window, A/A floor,
  achieved TFLOP/s and GB/s against roof.py's numbers (DRAM 240 GB/s; matmul 49 TF bf16, 111 TF bfp8 LoFi)
usage: analyze.py SWEEP_DIR PLAN.json [ROOF_JSONL]
"""
import json, sys
from collections import defaultdict
from pathlib import Path

d = Path(sys.argv[1])
rows = [json.loads(l) for l in open(d / "bench.jsonl")]
roof = {}
if len(sys.argv) > 3:
    for l in open(sys.argv[3]):
        r = json.loads(l)
        if r.get("kind") == "matmul":
            roof[(r["dtype"], r["fid"])] = r["tflops"]
        elif r.get("kind") == "dram_add":
            roof[("dram", r["dtype"])] = r["gbps"]
TILE = {"bf16": 2048, "bfp8": 1088, "bfp4": 576, "fp32": 4096}

def flops(shape):
    B, H, Sq, D = shape["q"]; Sk = shape["k"][2]
    return 4 * B * H * Sq * Sk * D

def nbytes(shape, dt, rereads):
    """q, k, v, out once each + the mask `rereads` times (the stock op re-reads a batch-broadcast mask per batch row)."""
    b = 0
    for t in ("q", "k", "v"):
        b += _tiles(shape[t]) * TILE[dt.get(t, "bf16")]
    b += _tiles(shape["q"]) * TILE["bf16"]
    b += _tiles(shape["mask"]) * TILE[dt.get("mask", "bf16")] * rereads
    return b

def _tiles(s):
    n = 1
    for x in s[:-2]:
        n *= x
    return n * -(-s[-2] // 32) * -(-s[-1] // 32)

groups = {g["name"]: g for g in json.load(open(sys.argv[2]))["groups"]}
refused = defaultdict(list)
for r in rows:
    if r.get("ev") == "refused":
        refused[r["group"].split(".")[0]].append(r["arm"])
arms = [r for r in rows if r.get("ev") == "arm"]
done = {r["group"]: r for r in rows if r.get("ev") == "group_done"}
by_cls = defaultdict(list)
for a in arms:
    by_cls[a["group"].split(".")[0]].append(a)
for cls, aa in by_cls.items():
    bases = [done[g]["base_ms"] for g in done if g.startswith(cls + ".")]
    floors = [done[g]["aa_floor_pct"] for g in done if g.startswith(cls + ".")]
    clks = [done[g]["aiclk"] for g in done if g.startswith(cls + ".")]
    ok = [a for a in aa if "ms" in a and a.get("check", {}).get("finite")]
    bad = [a for a in aa if a not in ok]
    print(f"== {cls}: {len(aa)} arms ({len(bad)} failed/non-finite), {len(bases)} groups; baseline "
          f"{min(bases):.3f}-{max(bases):.3f} ms; A/A floor max {max(floors):.2f} %; AICLK "
          f"{min(c['min'] for c in clks if c['n'])}-{max(c['max'] for c in clks if c['n'])} MHz "
          f"(n={sum(c['n'] for c in clks)}); refused {len(refused[cls])}")
    ok.sort(key=lambda a: a["ms"])
    seen = set()
    for a in ok[:8]:
        print(f"   {a['ms']:9.3f} ms  {a['speedup']:6.3f}x  spread {a.get('spread_pct', 0):5.2f} %  "
              f"rel_rms {a['check'].get('rel_rms', float('nan')):.3f}  {a['arm']}")
    for a in ok:
        key = a["arm"].split()[0] + " " + json.dumps(a["cfg"].get("dt", {}), sort_keys=True)
        if key not in seen:
            seen.add(key)
    shape = groups.get(aa[0]["group"], {}).get("shape")
    if shape:
        best = ok[0]
        f = flops(shape)
        print(f"   FLOPs/call {f:.3e}: baseline {f / min(bases) / 1e9:.1f} TF/s, best {f / best['ms'] / 1e9:.1f} TF/s"
              f" (roof bf16 {roof.get(('bf16', 'HiFi2'), 0):.0f}, bfp8 LoFi {roof.get(('bfp8', 'LoFi'), 0):.0f} TF/s)")
        if shape["q"][0] > 1 and shape["mask"][0] == 1:
            nb = nbytes(shape, best["cfg"].get("dt", {}), shape["q"][0])
            print(f"   stock mask re-read per batch row: best arm moves >= {nb / 1e9:.2f} GB = "
                  f"{nb / roof.get(('dram', 'bf16'), 240) / 1e6:.1f} ms at the measured DRAM roof")
    for a in bad[:5]:
        print(f"   FAILED {a['arm']}: {str(a.get('error', a.get('check')))[:160]}")
