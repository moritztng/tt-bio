#!/usr/bin/env python3
"""r4t-ceiling step 3: per-term table of the device-only step.

Reads the device-profiler ops csv of trunk_step (phase starts are Tracy signposts: fwd, heads_bwd, fold_bwd, opt,
rebuild, digest) and count_step.py's analytic json. For the chosen steady step it sums DEVICE KERNEL DURATION per
phase and per op code, prices each phase's analytic roofline, and assembles the ceiling:
  measured device time of what already runs on card (fwd incl. K-candidate pass + winner, tape bwd incl. card pair
  builder) + roofline-at-measured-efficiency of what must move on card (optimizer, fold VJP / grad accumulation,
  split rebuild) with the host feed overlapped (0).
    ceiling.py OPS.csv COUNT.json [--step N] [--shard 32] [--gpu 0.175]
"""
import csv, json, sys, collections
import roofline as R

def arg(k, d):
    return type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d

def phases(path):
    """[(step_index, phase, [rows])] in program order; a new step starts at each 'fwd' signpost."""
    out, cur, step = [], None, 0
    for r in csv.DictReader(open(path)):
        if r["OP TYPE"] == "signpost":
            name = r["OP CODE"].split("|")[-1]
            if name == "fwd":
                step += 1
            cur = [step, name, []]; out.append(cur); continue
        if cur is not None and r["DEVICE KERNEL DURATION [ns]"]:
            cur[2].append(r)
    return out

def main():
    ops, cnt = sys.argv[1], sys.argv[2]
    shard, gpu = arg("--shard", 32), arg("--gpu", 0.175)
    ph = phases(ops)
    steps = sorted({s for s, _, _ in ph})
    st = arg("--step", steps[-1])
    dev = collections.defaultdict(float); bycode = collections.defaultdict(lambda: collections.defaultdict(float))
    nops = collections.Counter()
    for s, name, rows in ph:
        if s != st: continue
        for r in rows:
            t = float(r["DEVICE KERNEL DURATION [ns]"]) / 1e9
            dev[name] += t; bycode[name][r["OP CODE"]] += t; nops[name] += 1
    d = json.load(open(cnt))
    P = R.peak(4)
    roof = collections.defaultdict(float)
    for r in d["rows"]:
        roof[r["phase"]] += max(r["flops"] / P, r["elems"] * 2 / R.BW)
    n, nsf, nsb, elb = R.optimizer(d["param_shapes"])
    fwd_roof = roof["cand_k"] + roof["win_fwd"]
    eff = dev["fwd"] / fwd_roof if fwd_roof else float("nan")        # measured device / roofline, the forward's factor
    opt_roof = max(nsf / P, nsb / R.BW) + elb / R.BW
    grad_acc = n * 4 * 3 / R.BW                                        # fp32 grad accumulate on card: read+read+write
    rebuild = n * (4 + 2) / R.BW                                       # bf16 shadow of fp32 masters: read 4 B, write 2 B
    terms = [("fwd (K-candidate pass + winner), measured device", dev["fwd"]),
             ("tape bwd incl. card pair builder (heads_bwd phase), measured device", dev["heads_bwd"]),
             ("fold VJP on card = fp32 grad accumulate, roofline x eff", grad_acc * eff),
             (f"optimizer Muon(fp32 NS)+Adam on card, roofline x eff, ZeRO-1 1/{shard}", opt_roof * eff / shard),
             ("split rebuild = bf16 shadow of masters on card, roofline x eff", rebuild * eff),
             ("host feed (batch upload), overlapped", 0.0)]
    T = sum(t for _, t in terms)
    ratio = 14.05 * gpu / T
    print(f"# step {st} of {steps}; device ms per phase (measured): " + ", ".join(f"{k} {v*1e3:.1f} ({nops[k]} ops)" for k, v in dev.items()))
    print(f"# forward roofline {fwd_roof*1e3:.1f} ms (cand {roof['cand_k']*1e3:.1f} + win {roof['win_fwd']*1e3:.1f}); bwd roofline {roof['win_bwd']*1e3:.1f} ms; eff factor (measured/roofline) {eff:.2f}")
    print(f"# optimizer roofline unsharded {opt_roof*1e3:.1f} ms; x eff {opt_roof*eff*1e3:.1f} ms")
    print("| term | s/step |\n|---|---:|")
    for k, t in terms: print(f"| {k} | {t:.4f} |")
    print(f"| **ceiling (device only, 1 chip, host overlapped)** | **{T:.4f}** |")
    print(f"RATIO {ratio:.2f}x per $ vs B200 {gpu} s (at DP eff 1.0; {ratio*0.9:.2f}x at 0.90)")
    for name in ("fwd", "heads_bwd"):
        print(f"## {name}: top op codes (device s)")
        for k, v in sorted(bycode[name].items(), key=lambda t: -t[1])[:10]: print(f"  {k:45s} {v:.4f}")
    print(f"CEIL_T {T:.4f} RATIO {ratio:.2f}")

if __name__ == "__main__":
    main()
