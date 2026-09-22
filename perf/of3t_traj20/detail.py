#!/usr/bin/env python3
"""Every rung of one arm, plus the step logs both sides kept. The shape is the result and an
endpoint hides it, so this prints k = 1..20 and not a summary of it."""
import json
import sys

for p in sys.argv[1:]:
    d = json.load(open(p))
    print(f"\n### {d['arm']}  warmup={d['warmup_no_steps']} accum={d['accumulate_grad_batches']} "
          f"rho={d['rho']} scope={d['scope']['tensors']} tensors / {d['scope']['elements']} elements")
    print(f"  feedback share of drive norm: " +
          ", ".join(f"k={k}:{v:.4f}" for k, v in d["feedback_share_of_drive_norm"].items()))
    print(f"  growth k2..20 exponent {d['growth_k2_20']['exponent']:+.4f} "
          f"r2 {d['growth_k2_20']['r2']:.4f} "
          f"bit-identical rungs dropped {d['growth_k2_20']['rungs_dropped_bit_identical']}")
    print(f"{'k':>3} {'rel_d':>11} {'floor':>10} {'x floor':>9} {'median':>10} {'p90':>10} "
          f"{'worst':>10} {'rel_w':>10} {'bitid':>6} {'our_lr':>10} {'their_lr':>10}")
    for r, o, t in zip(d["per_step"], d["our_step_log"], d["their_step_log"]):
        print(f"{r['k']:3d} {r['rel_d']:11.4e} {r['fp32_differencing_floor']:10.3e} "
              f"{r['rel_over_floor']:9.3e} {r['median_per_tensor']:10.3e} "
              f"{r['p90_per_tensor']:10.3e} {r['worst_per_tensor']:10.3e} {r['rel_w']:10.3e} "
              f"{r['tensors_bit_identical']:6d} {o['lr']:10.4e} {t['lr']:10.4e}")
    w = max(d["per_step"][1:], key=lambda r: r["worst_per_tensor"])
    print(f"  worst tensor over k>=2: {w['worst_tensor']}  {w['worst_per_tensor']:.6e} at k={w['k']}")
    print(f"  d20 worst tensor: {d['per_step'][-1]['worst_tensor']} "
          f"{d['per_step'][-1]['worst_per_tensor']:.6e}")
    print(f"  tensors scored at k=20: {d['per_step'][-1]['tensors_scored']}, "
          f"zero-reference {d['per_step'][-1]['tensors_zero_ref']}")
    ps = d["their_step_log"][-1]["participation_spread"]
    print(f"  their participation spread at k=20: {ps}")
    oc = d["our_step_log"][-1]
    print(f"  our k=20 clip={oc['clip']} per_sample={oc['per_sample']} "
          f"grad_norm={oc['grad_norm']:.4e} per_sample_coefs={oc['per_sample_clip_coefs']}")
    print(f"  their k=20 per-sample clip coefs: "
          f"{[round(c, 6) for c in d['their_step_log'][-1]['per_sample_clip_coefs']]}")
