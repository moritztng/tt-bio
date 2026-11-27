#!/usr/bin/env python3
"""r4t-ceiling roofline: price count_step.py's per-class FLOPs/bytes and an on-card Muon+Adam optimizer
against one BH p300c chip's peaks. Analytic only (step 1 of the row); measured efficiencies come in step 2.

Peaks (per chip, AICLK 1350 MHz, 120 Tensix: knowledgebase/knowledge/marketing/claims.md:233; DRAM 512 GB/s:
same line and knowledge/infra/machines.md:150). Matmul rate per Tensix = 2048 MAC/cycle at LoFi, derived from
the published p150a 774 FP8 TFLOPS / (140 Tensix x 1.35 GHz x 2) = 2048; HiFi2 = 1/2, HiFi4 = 1/4 (fidelity
phases). Activations priced at 2 B/element (bf16).
    roofline.py count.json [--fidelity 4]
"""
import json, sys, collections
CORES, CLK, BW = 120, 1.35e9, 512e9
LOFI = CORES * 2048 * 2 * CLK            # 663.6 TFLOP/s
def peak(fid): return LOFI / fid

def optimizer(shapes, ns_iters=5, fp32_passes=3):
    """Muon (Newton-Schulz quintic, fp32 = 3 split-bf16 products at HiFi4) on every >=2-D matrix, Adam elsewhere;
    elementwise state traffic fp32: p,g,m,v read + p,m,v write + EMA read/write (10 x 4 B/param) + clip-norm read."""
    ns_fl = 0; ns_by = 0; n = 0
    for shp, _name in shapes:
        k = 1
        for s in shp: k *= s
        n += k
        if len(shp) >= 2:
            a, b = shp[0], k // shp[0]; mm, nn = min(a, b), max(a, b)
            ns_fl += ns_iters * (4 * mm * mm * nn + 2 * mm ** 3)
            ns_by += ns_iters * (3 * mm * nn + 4 * mm * mm) * 4
    return n, ns_fl * fp32_passes, ns_by, n * 11 * 4

def main():
    d = json.load(open(sys.argv[1]))
    fid = int(sys.argv[sys.argv.index("--fidelity") + 1]) if "--fidelity" in sys.argv else 4
    P = peak(fid)
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for r in d["rows"]:
        k = (r["phase"], r["class"], r["size"]); a = agg[k]
        a[0] += r["flops"]; a[1] += r["elems"] * 2; a[2] += r["calls"]
    rows = []
    for (ph, c, s), (fl, by, calls) in sorted(agg.items()):
        tc, tb = fl / P, by / BW
        rows.append((ph, c, s, fl, by, calls, tc, tb, max(tc, tb)))
    n, nsf, nsb, elb = optimizer(d["param_shapes"])
    rows.append(("optimizer", "muon_ns_fp32", "matrix", nsf, nsb, 0, nsf / P, nsb / BW, max(nsf / P, nsb / BW)))
    rows.append(("optimizer", "elementwise_fp32", "param", n * 30, elb, 0, n * 30 / P, elb / BW, max(n * 30 / P, elb / BW)))
    print(f"# batch {d['batch']} depth {d['depth']} K {d['K']} params {d['params']} (opt sees {n}); peak {P/1e12:.1f} TFLOP/s "
          f"(HiFi{fid}) {BW/1e9:.0f} GB/s; ridge {P/BW:.0f} flop/B")
    print("| phase | class | size | GFLOP | GB | calls | t_compute ms | t_dram ms | roofline ms |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    tot = collections.defaultdict(float)
    for ph, c, s, fl, by, calls, tc, tb, t in rows:
        print(f"| {ph} | {c} | {s} | {fl/1e9:.1f} | {by/1e9:.2f} | {calls} | {tc*1e3:.2f} | {tb*1e3:.2f} | {t*1e3:.2f} |")
        tot[ph] += t
    for ph, t in tot.items(): print(f"TOTAL {ph} {t*1e3:.1f} ms")
    print(f"TOTAL step {sum(tot.values())*1e3:.1f} ms")

if __name__ == "__main__":
    main()
