#!/usr/bin/env python3
"""Turn a progtape.py capture into the program table and the class attribution.

Shares are shares of the TAPED module wall, because that is the only denominator the parts can
sum to: the tape's own syncs are inside it. The untaped module wall and the per-call sync cost
are both reported, and a sync-corrected column subtracts n_calls * that cost from every program,
which is the only correction the tape's bias admits (it scales with call count, not with time).
"""
import json
import sys
from pathlib import Path

LAYOUT_OPS = {"permute", "transpose", "reallocate", "chunk", "concat", "clone", "slice",
              "unsqueeze", "to_memory_config", "reshape", "typecast", "to_layout", "pad", "copy",
              "allocate_tensor_on_device"}
ARITH = {"projection", "contraction"}


def classify(p):
    op, sig = p["op"], p["sig"]
    site = next(iter(p["sites"]), "")
    if op == "generic_op":
        if "reblock_permute_gated" in sig:
            return "layout+gate", "gated channel move (E6 kernel)"
        if "reblock_permute_back" in sig:
            return "layout", "channel move back (kernel)"
        if "reblock_permute" in sig:
            return "layout", "channel move forward (kernel)"
        if "fused_tail" in site:
            return "projection", "out tail: p_out + g_out + gate, fused (F1)"
        if "in_proj" in site:
            return "projection", "in-projection (fused g|p, dual-NOC generic matmul)"
        return "other", "unclassified generic_op"
    if op == "minimal_matmul":
        return "projection", "in-projection"
    if op == "linear":
        return "projection", "out-projection"
    if op == "matmul":
        return "contraction", "the contraction"
    if op == "layer_norm":
        return "normalisation", "layer_norm"
    if op in ("multiply_", "multiply"):
        return ("gating", "sigmoid gate") if "SIGMOID" in sig else ("masking", "pair mask")
    if op in LAYOUT_OPS:
        return "layout", op
    return "other", op


def main():
    d = json.load(open(sys.argv[1]))
    folds = {f["tag"]: f for f in d["folds"]}
    mod = lambda t: folds[t]["module"]["body:TriangleMultiplication"]["s"]
    untaped = sorted(mod(t) for t in ("A", "C") if t in folds)
    D_un = sum(untaped) / len(untaped)
    D = mod("B")
    progs = [p for p in d["programs"] if p["n"]]
    calls = sum(p["n"] for p in progs)
    sync = (D - D_un) / calls
    print(f"# {Path(sys.argv[1]).name}  arm={d.get('arm')}  {d['size']} aa  grid={d['grid']}  "
          f"head={d['git_head'][:9]}")
    print(f"  flags {d['flags']}")
    for t, f in folds.items():
        c = f["clock"]
        print(f"  fold {t:5s} {f['fold_s']:8.3f}s  trimul {mod(t):7.4f}s/"
              f"{f['module']['body:TriangleMultiplication']['calls']} calls  "
              f"clk {c.get('aiclk_min')}-{c.get('aiclk_max')} (n={c.get('aiclk_n')})  "
              f"plddt {f['plddt']}  load {f['loadavg1']}")
    print(f"  untaped module wall {D_un:.4f}s (A/A {abs(untaped[0]-untaped[-1]):.4f}s = "
          f"{100*abs(untaped[0]-untaped[-1])/D_un:.3f}%), taped {D:.4f}s = {D/D_un:.4f}x")
    print(f"  {calls} taped dispatches -> sync cost {sync*1e6:.1f} us/call, "
          f"{calls*sync:.4f}s total")
    tot = sum(p["ms"] for p in progs) / 1e3
    rep = sum(p["ms_repeat"] for p in progs) / 1e3
    print(f"  taped sum {tot:.4f}s = {100*tot/D:.2f}% of the taped wall; repeat {rep:.4f}s "
          f"({100*abs(rep-tot)/tot:.2f}% apart); RESIDUAL {D-tot:.4f}s = {100*(D-tot)/D:.2f}%")
    print()
    print(f"{'ms':>9} {'%mod':>6} {'%corr':>6} {'n':>6} {'np':>3}  class            what / where")
    cls = {}
    clsc = {}
    for p in sorted(progs, key=lambda p: -p["ms"]):
        k, what = classify(p)
        cls[k] = cls.get(k, 0.0) + p["ms"] / 1e3
        corr = (p["ms"] / 1e3) - p["n"] * sync
        clsc[k] = clsc.get(k, 0.0) + corr
        site = next(iter(p["sites"]), "?")
        if p["ms"] / 1e3 / D < 0.002:
            continue
        print(f"{p['ms']:9.1f} {100*p['ms']/1e3/D:6.2f} {100*corr/D_un:6.2f} {p['n']:6d} "
              f"{p['new_programs']:3d}  {k:15s}  {what}")
        print(f"{'':32s}{site[:120]}")
        print(f"{'':32s}{p['sig'][:120]}")
    print()
    print("CLASS TOTALS")
    for k in sorted(cls, key=lambda k: -cls[k]):
        print(f"  {k:15s} {cls[k]:7.4f}s  {100*cls[k]/D:6.2f}% of taped wall   "
              f"{100*clsc[k]/D_un:6.2f}% sync-corrected")
    print(f"  {'RESIDUAL':15s} {D-tot:7.4f}s  {100*(D-tot)/D:6.2f}%   "
          f"{100*(D_un-sum(clsc.values()))/D_un:6.2f}% sync-corrected")
    scaf = sum(v for k, v in cls.items() if k not in ARITH)
    scafc = sum(v for k, v in clsc.items() if k not in ARITH)
    ar = sum(v for k, v in cls.items() if k in ARITH)
    arc = sum(v for k, v in clsc.items() if k in ARITH)
    print(f"  --> arithmetic (projection+contraction) {100*ar/D:6.2f}%   "
          f"{100*arc/D_un:6.2f}% sync-corrected")
    print(f"  --> SCAFFOLDING                         {100*scaf/D:6.2f}%   "
          f"{100*scafc/D_un:6.2f}% sync-corrected  (+ residual)")


main()
