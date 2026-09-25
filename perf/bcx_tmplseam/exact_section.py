#!/usr/bin/env python3
"""Render the state doc's EXACT section from grade.json, so no number is typed by hand.

A digest whose numbers are carried across by a human is a transcription, and this row already
publishes one wrong figure it typed (the ON range "inside" the OFF range). Everything below is
read out of the artifact.
"""
import json
import pathlib
import sys

g = json.loads(pathlib.Path(sys.argv[1]).read_text())
L = []
w = L.append


def fmt(d, key="rel"):
    v = d.get(key)
    return "n/a" if v is None else (f"{v:.3e}" if abs(v) < 1e-2 else f"{v:.4f}")


w("EXACT: graded on the captured input of a real `sequence_gradients` call, per tensor, against a")
w("float64 `tt_bio.af2_reference` template pair stack -- never against another device arm. Artifact")
w(f"`perf/bcx_tmplseam/runs/grade_host/grade.json`, host {g.get('host')}, "
  f"card {g.get('card') or 'none (host-only leg)'}, seed {g.get('seed')},")
w(f"started {g.get('started_utc')}, finished {g.get('finished_utc')}, "
  f"loadavg {', '.join(f'{x:.1f}' for x in g.get('loadavg', []))}, OMP {g.get('omp')}.")
w("")
w("The stack's own output, relative L2 against the float64 arm and the cosine beside it:")
w("")
w("        arm        rel L2 vs f64        cosine        absmax diff")
for name, d in sorted(g.get("graded_vs_f64", {}).items()):
    w(f"        {name:<10} {fmt(d):<20} {fmt(d, 'cos'):<13} {fmt(d, 'absmax_diff')}")
w(f"        (f64 norm {g.get('f64_norm')}, act_out shape "
  f"{g.get('round_dropout0', {}).get('stack_shapes', {}).get('act_out')})")
w("")

env = g.get("envelope")
if env:
    w("ENVELOPE, the loop's own gradient `d(loss)/d(sequences)` from the same program with only the")
    w("template pair stack changed. The float64 arm is what makes the others readable:")
    w("")
    w("        arm        loss            |g|             rel L2 vs the f64 arm    seconds")
    for name in ("bc2_jax", "f64", "device"):
        d = env.get(name)
        if not d:
            continue
        w(f"        {name:<10} {d.get('loss'):<15.6g} {d.get('g_l2'):<15.6g} "
          f"{fmt(d.get('vs_f64', {})):<23} {d.get('seconds')}")
    r = env.get("ratio_device_over_jax")
    if r is not None:
        w("")
        w(f"        device distance / BindCraft 2's own distance from float64 = {r:.4f}")
        w("        Under 1 means the device arm is CLOSER to float64 than BindCraft 2's own JAX is.")
    else:
        w("")
        w("        The device row is absent: this leg ran without a card (`--skip-device`), so the")
        w("        device arm's own distance is the one number EXACT still owes.")
w("")
w("Dropout plumbing, as a check that the mask draw reproduces BindCraft 2's rather than a claim")
w(f"about speed: the two dropout arms take the same input ({g.get('input_is_same_in_both_dropout_arms')}) and")
w(f"produce outputs {fmt(g.get('output_differs_across_dropout_arms', {}))} apart in relative L2, so the masks are")
w("doing something and they are doing it to the same tensor.")
print("\n".join(L))
