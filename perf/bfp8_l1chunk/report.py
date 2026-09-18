#!/usr/bin/env python3
"""Read l1_surface.json and print the per-config CB budget table, bf16 against bfp8."""
import json, sys

d = json.load(open("perf/bfp8_l1chunk/l1_surface.json"))
L1, PR = d["L1_PER_CORE"], d["PROGRAM_RESERVE"]
print(f"L1_PER_CORE {L1}  PROGRAM_RESERVE {PR}  budget {L1 - PR}")
for seq in (sys.argv[1:] or ["512", "768", "1024", "1536"]):
    s = d["seqs"][seq]
    b16 = {(r["q_chunk"], r["k_chunk"]): r for r in s["rows_bf16"]}
    b8 = {(r["q_chunk"], r["k_chunk"]): r for r in s["rows_bfp8"]}
    print(f"\n=== seq {seq}   servable bf16 {s['n_servable_bf16']} -> bfp8 {s['n_servable_bfp8']}"
          f"   best bf16 {s['best_bf16']} bfp8 {s['best_bfp8']}"
          f"   newly fitting {s['newly_fitting']}")
    keys = sorted((k for k in b16 if b16[k]["preconds_ok"] or b8[k]["preconds_ok"]),
                  key=lambda k: (-k[0], -k[1]))
    print("   q_chunk k_chunk |   bf16 CB   fit |   bfp8 CB   fit | bfp8/bf16")
    for k in keys[:12]:
        r16, r8 = b16[k], b8[k]
        print(f"   {k[0]:7d} {k[1]:7d} | {r16['cb_bytes']:9d} {'FIT ' if r16['fits'] else 'OVER'} "
              f"| {r8['cb_bytes']:9d} {'FIT ' if r8['fits'] else 'OVER'} "
              f"| {r8['cb_bytes'] / r16['cb_bytes']:.4f}")
