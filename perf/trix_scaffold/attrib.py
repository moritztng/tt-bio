#!/usr/bin/env python3
"""progtape.py capture -> the program table and the class attribution.

Two denominators, both reported, because neither alone is honest. The TAPED module wall is the
only one the parts can sum to (the tape's own syncs are inside it). The UNTAPED wall is the number
the campaign's shares are shares of, and the parts reach it only after the tape's per-dispatch cost
is taken back out -- that cost scales with call count, not with time, so it is the one correction
the instrument's bias admits. `allocate_tensor_on_device`, `from_torch` and `to_torch` are listed
separately: they allocate or copy and compile no program, so their taped time IS the sync charge.
"""
import json
import sys
from pathlib import Path

HOST = {"allocate_tensor_on_device", "from_torch", "to_torch"}
LAYOUT_OPS = {"permute", "transpose", "reallocate", "chunk", "concat", "clone", "slice",
              "unsqueeze", "to_memory_config", "reshape", "typecast", "to_layout", "pad", "copy"}
ARITH = {"projection", "contraction"}


def classify(p):
    op, sig = p["op"], p["sig"]
    site = next(iter(p["sites"]), "")
    if op in HOST:
        return "host", op
    if op == "generic_op":
        if "reblock_permute_gated" in sig:
            return "layout+gate", "gated channel move (E6)"
        if "reblock_permute_back" in sig:
            return "layout", "channel move back"
        if "reblock_permute" in sig:
            return "layout", "channel move forward"
        if "fused_tail" in site:
            return "projection", "out tail p_out+g_out+gate (F1)"
        if "in_proj" in site:
            return "projection", "in-projection (dual-NOC)"
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

    def mod(t):
        return folds[t]["module"]["body:TriangleMultiplication"]["s"]

    untaped = sorted(mod(t) for t in ("A", "C") if t in folds)
    D_un = sum(untaped) / len(untaped)
    D = mod("B")
    progs = [p for p in d["programs"] if p["n"]]
    calls = sum(p["n"] for p in progs)
    sync = (D - D_un) / calls
    print(f"# {Path(sys.argv[1]).name}  arm={d.get('arm')}  {d['size']} aa  grid={d['grid']}  "
          f"head={d['git_head'][:9]}  protocol={d['protocol']}")
    print(f"  flags {d['flags']}")
    for t in ("cold", "A", "B", "B2", "C"):
        if t not in folds:
            continue
        f, c = folds[t], folds[t]["clock"]
        print(f"  fold {t:5s} {f['fold_s']:8.3f}s  trimul {mod(t):7.4f}s  "
              f"clk {c.get('aiclk_min')}-{c.get('aiclk_max')} n={c.get('aiclk_n')}  "
              f"plddt {f['plddt']}  load {f['loadavg1']}")
    aa = abs(untaped[0] - untaped[-1])
    print(f"  UNTAPED module wall {D_un:.4f}s over 1208 calls (A/A {aa:.4f}s = "
          f"{100*aa/D_un:.3f}%); TAPED {D:.4f}s = {D/D_un:.4f}x")
    print(f"  {calls} taped dispatches -> {sync*1e6:.1f} us/dispatch of tape cost, {calls*sync:.4f}s")
    tot = sum(p["ms"] for p in progs) / 1e3
    rep = sum(p.get("ms_repeat", 0.0) for p in progs) / 1e3
    if rep:
        print(f"  taped sum {tot:.4f}s, repeat fold {rep:.4f}s -> {100*abs(rep-tot)/tot:.2f}% apart")
    print(f"  taped sum {tot:.4f}s = {100*tot/D:.2f}% of the taped wall; "
          f"RESIDUAL {D-tot:.4f}s = {100*(D-tot)/D:.2f}%")
    print()
    hdr = f"{'ms':>8} {'%taped':>7} {'%corr':>6} {'calls':>6} {'rep%':>5}  {'class':<13} program"
    print(hdr)
    cls, clsc = {}, {}
    rows = []
    for p in sorted(progs, key=lambda p: -p["ms"]):
        k, what = classify(p)
        corr = (p["ms"] / 1e3) - p["n"] * sync
        cls[k] = cls.get(k, 0.0) + p["ms"] / 1e3
        clsc[k] = clsc.get(k, 0.0) + max(corr, 0.0)
        r = p.get("ms_repeat", 0.0)
        rows.append((p, k, what, corr, r))
    for p, k, what, corr, r in rows:
        if p["ms"] / 1e3 / D < 0.002:
            continue
        dr = f"{100*(r-p['ms'])/p['ms']:+.1f}" if r else "   -"
        print(f"{p['ms']:8.1f} {100*p['ms']/1e3/D:7.2f} {100*corr/D_un:6.2f} {p['n']:6d} {dr:>5}"
              f"  {k:<13} {what}")
        print(f"{'':16s}{next(iter(p['sites']), '?')[:118]}")
        print(f"{'':16s}{p['sig'][:118]}")
    print()
    print("CLASS TOTALS      taped s   % of taped wall   % of untaped wall, sync-corrected")
    for k in sorted(cls, key=lambda k: -cls[k]):
        print(f"  {k:<14} {cls[k]:8.4f}  {100*cls[k]/D:8.2f} %   {100*clsc[k]/D_un:8.2f} %")
    corr_sum = sum(clsc.values())
    print(f"  {'RESIDUAL':<14} {D-tot:8.4f}  {100*(D-tot)/D:8.2f} %   "
          f"{100*(D_un-corr_sum)/D_un:8.2f} %")
    scaf = sum(v for k, v in cls.items() if k not in ARITH and k != "host")
    scafc = sum(v for k, v in clsc.items() if k not in ARITH and k != "host")
    ar = sum(v for k, v in cls.items() if k in ARITH)
    arc = sum(v for k, v in clsc.items() if k in ARITH)
    print(f"  {'ARITHMETIC':<14} {ar:8.4f}  {100*ar/D:8.2f} %   {100*arc/D_un:8.2f} %")
    print(f"  {'SCAFFOLDING':<14} {scaf:8.4f}  {100*scaf/D:8.2f} %   {100*scafc/D_un:8.2f} %")
    print(f"  {'+ RESIDUAL':<14} {scaf+D-tot:8.4f}  {100*(scaf+D-tot)/D:8.2f} %   "
          f"{100*(scafc+D_un-corr_sum)/D_un:8.2f} %")
    print(f"  programs with calls: {len(progs)}; generic_op programs: "
          f"{sum(1 for p in progs if p['op'] == 'generic_op')}; "
          f"cache entries created in trimul on the cold fold: "
          f"{sum(p['new_programs'] for p in d['programs'])}")


main()
