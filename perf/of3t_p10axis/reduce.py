#!/usr/bin/env python3
"""Reduce the axis arms to the four answers the row owes: the reproduction, the sample slope,
the gradient census and the memory ceiling."""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
GIB = 1024.0 ** 3


def load(tag="384"):
    arms = {}
    for p in sorted(OUT.glob(f"arm_*_{tag}.json")):
        name = p.name[len("arm_"):-len(f"_{tag}.json")]
        arms[name] = json.loads(p.read_text())
    return arms


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "384"
    arms = load(tag)
    rows = []
    for name, d in arms.items():
        cfg = d.get("config", {})
        reps = d.get("reps", [])
        steady = d.get("steady") or {}
        r0 = reps[0] if reps else {}
        rs = reps[1:] or reps
        def med(k):
            v = [r[k] for r in rs if r.get(k) is not None]
            return round(sorted(v)[len(v) // 2], 3) if v else None
        dram = None
        if r0.get("dram_after_diffusion") and r0.get("dram_after_trunk"):
            rr = rs[0] if rs else r0
            dram = round((rr["dram_after_diffusion"] - rr["dram_after_trunk"]) / 1e9, 3)
        rows.append({
            "arm": name, "cycles": cfg.get("cycles_pinned"),
            "samples": cfg.get("diffusion_samples"), "reps": len(reps),
            "cold_s": d.get("cold_s"), "steady_s": steady.get("median_s"),
            "trunk_s": med("trunk_s"), "diffusion_s": med("diffusion_s"),
            "losses_s": med("losses_s"), "backward_s": med("backward_s"),
            "optimizer_s": med("optimizer_s"), "tape_nodes": med("tape_nodes"),
            "grads_cold": r0.get("params_with_grad"),
            "grads_steady": (rs[0] if rs else {}).get("params_with_grad"),
            "diff_dram_gb": dram,
            "aiclk": (d.get("env", {}).get("aiclk_line") or "")[:70],
            "quiet_pre": (d.get("arm", {}).get("host_quiet_pre") or {}).get("green"),
            "error": (d.get("error") or "").strip().splitlines()[-1:] or None,
        })
    rows.sort(key=lambda r: (r["arm"] != "qb2repro", r["samples"] or 0))
    hdr = ["arm", "cycles", "samples", "cold_s", "steady_s", "trunk_s", "diffusion_s",
           "losses_s", "backward_s", "optimizer_s", "tape_nodes", "diff_dram_gb",
           "grads_cold", "grads_steady", "quiet_pre"]
    print(" | ".join(hdr))
    for r in rows:
        print(" | ".join(str(r.get(h)) for h in hdr))
    print()
    for r in rows:
        if r["error"]:
            print(f"ERROR {r['arm']}: {r['error']}")

    print("\n--- gradient census: who is missing, per rep ---")
    for name, d in arms.items():
        for c in d.get("grad_census", []):
            miss = c["missing"]
            pre = {}
            for m in miss:
                key = ".".join(m.split(".")[:3])
                pre[key] = pre.get(key, 0) + 1
            print(f"{name} rep{c['rep']}: {c['with_grad']}/{c['declared']} have a grad, "
                  f"{c['without_grad']} do not; top prefixes "
                  f"{sorted(pre.items(), key=lambda kv: -kv[1])[:6]}")

    print("\n--- rep0 vs rep1 delta (names that LOSE their gradient after the first step) ---")
    for name, d in arms.items():
        cs = d.get("grad_census", [])
        if len(cs) >= 2:
            lost = sorted(set(cs[1]["missing"]) - set(cs[0]["missing"]))
            print(f"{name}: {len(lost)} lost between rep0 and rep1")
            for n in lost[:12]:
                print("   ", n)
            if len(lost) > 12:
                print(f"    ... and {len(lost) - 12} more")


if __name__ == "__main__":
    main()
