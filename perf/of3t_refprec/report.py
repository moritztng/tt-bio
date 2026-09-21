#!/usr/bin/env python3
"""Read REFPREC.json and print the numbers the row reports, in the shapes it reports them.

Separate from compare_precision.py so the arms never have to be re-run to restate a figure, and
so every number in the state doc is transcribed by a program rather than by hand.
"""
import argparse
import json
from pathlib import Path

SETS = ["MODEL (all tensors)",
        "diffusion device-arm scope (diffusion_module minus diffusion_conditioning), "
        "FULL denominator per A20",
        "attention side of the diffusion arm",
        "rest of the diffusion arm",
        "the one leaf (blocks.N.attention_pair_bias.layer_norm_a.layer_norm_s.weight), "
        "device reads 10.6980",
        "diffusion device-arm scope minus the one leaf, device reads 0.2929"]


def f(x, spec=".6e"):
    return "n/a" if x is None else format(x, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", type=Path)
    args = ap.parse_args()
    d = json.loads(args.json.read_text())["arms"]

    print("=" * 100)
    print("HEADLINE, each arm against BUNDLE-MIN-043's float64 gradient (A23: mass-weighted "
          "first, median beside)")
    print("=" * 100)
    print(f"{'arm':28s} {'mass-wtd rel_l2':>16s} {'median/tensor':>14s} {'norm ratio r':>13s} "
          f"{'cos':>9s}  bar 2.0e-02")
    for label, rep in d.items():
        m = rep["sets"][0]
        print(f"{label:28s} {f(m['mass_weighted_rel_l2']):>16s} "
              f"{f(m['median_rel_l2_over_tensors']):>14s} "
              f"{f(m['mass_weighted_norm_ratio'], '.6f'):>13s} "
              f"{f(m['mass_weighted_cos'], '.6f'):>9s}  "
              f"{'INSIDE' if m['inside_mass_bar'] else 'OUTSIDE'}")

    for name in SETS[1:]:
        print()
        print("-" * 100)
        print(name)
        print(f"{'arm':28s} {'n':>5s} {'% of model':>11s} {'mass-wtd rel':>14s} "
              f"{'median':>13s} {'r':>10s} {'cos':>9s}")
        for label, rep in d.items():
            s = next(x for x in rep["sets"] if x["set"] == name)
            print(f"{label:28s} {s['n']:5d} {s['pct_of_model_mass']:11.4f} "
                  f"{f(s['mass_weighted_rel_l2']):>14s} {f(s['median_rel_l2_over_tensors']):>13s} "
                  f"{f(s['mass_weighted_norm_ratio'], '.4f'):>10s} "
                  f"{f(s['mass_weighted_cos'], '.5f'):>9s}")

    print()
    print("=" * 100)
    print("THE HEAVY TENSORS, INDIVIDUALLY (A23 rule 3). Twelve tensors, ordered by their share "
          "of the model.")
    print("=" * 100)
    first = next(iter(d.values()))
    for i, h in enumerate(first["heaviest_tensors_named_individually"]):
        print(f"\n[{h['pct_of_model_mass']:8.5f} % of model]  {h['param']}")
        print(f"    {'arm':28s} {'rel_l2':>14s} {'r':>12s} {'cos':>10s}")
        for label, rep in d.items():
            g = rep["heaviest_tensors_named_individually"][i]
            assert g["param"] == h["param"]
            print(f"    {label:28s} {f(g['rel_l2']):>14s} {f(g['r'], '.6f'):>12s} "
                  f"{f(g['cos'], '.6f'):>10s}")

    print()
    print("=" * 100)
    print("THE NAMED TENSOR, BESIDE WHAT THE DEVICE ARM READS ON IT")
    print("=" * 100)
    for nt in first.get("named_tensors_beside_the_device_reading", []):
        print(f"\n{nt['param']}")
        print(f"    {nt['pct_of_model_mass']:.5f} % of the model on its own")
        print(f"    device arm   rel_l2 {nt['device_rel_l2']}   ({nt['source']})")
        for label, rep in d.items():
            g = next(x for x in rep["named_tensors_beside_the_device_reading"]
                     if x["param"] == nt["param"])
            print(f"    {label:28s} rel_l2 {f(g['this_arm_rel_l2']):>14s} "
                  f"r {f(g['this_arm_r'], '.6f'):>12s} cos {f(g['this_arm_cos'], '.6f'):>10s}")

    print()
    print("=" * 100)
    print("SECTIONS (A23 rule 1: every set carries its share of the reference's squared norm)")
    print("=" * 100)
    secs = [s["set"] for s in first["sets"] if s["set"].startswith("section ")]
    print(f"{'section':45s} {'n':>5s} {'% model':>9s} " +
          " ".join(f"{l[:14]:>14s}" for l in d))
    for sec in secs:
        row = [next(x for x in rep["sets"] if x["set"] == sec) for rep in d.values()]
        s0 = row[0]
        print(f"{sec[8:]:45s} {s0['n']:5d} {s0['pct_of_model_mass']:9.4f} " +
              " ".join(f"{f(x['mass_weighted_rel_l2']):>14s}" for x in row))

    print()
    print("=" * 100)
    print("CONTROLS")
    print("=" * 100)
    for label, rep in d.items():
        lr = rep.get("loss_reproduction")
        if not lr:
            continue
        print(f"\n{label}")
        print(f"    loss {lr['arm_loss']!r} against the reference's {lr['reference_loss']!r}")
        print(f"    abs {lr['abs_diff']:.6e}   rel {lr['rel_diff']:.6e}   "
              f"bit-identical on RNG replay within the arm: "
              f"{lr['bit_identical_on_rng_replay_within_the_arm']}")
        print(f"    gradient global norm {lr['gradient_global_norm']!r} over "
              f"{lr['n_with_gradient']} tensors")
        print(f"    draws {lr['draws_replayed']} sha {str(lr['draws_sha256'])[:16]} "
              f"mismatches {lr['n_draw_mismatch']}")
        cp = lr.get("cast_policy") or {}
        print(f"    cast policy {cp.get('mode')}: their autocast contexts entered "
              f"{cp.get('n_autocast_contexts_entered')}, torch actually enabled "
              f"{cp.get('n_autocast_contexts_torch_actually_enabled')}; "
              f".float() calls {cp.get('n_tensor_float_calls')} of which "
              f"{cp.get('n_tensor_float_calls_that_changed_dtype')} changed dtype "
              f"{cp.get('tensor_float_source_dtypes')}")
    print()
    print(f"unmeasurable tensors (A14 floor, ref_norm < 1e-12): "
          f"{len(first['unmeasurable'])}, holding "
          f"{sum(u['pct_of_model_mass'] for u in first['unmeasurable']):.3e} % of the model")
    print(f"the top 8 tensors hold {first['mass_reached_by_the_top_8_tensors_pct']:.4f} % "
          f"of the model")


if __name__ == "__main__":
    main()
