"""Per-op FLOPs, DRAM bytes and floor time for one `_block` of `perf/hallgrad/e2e_distogram.py`.

The floor screen measured the whole block: `b(256) = 0.23204 s` forward+backward, `b_f = 0.08542 s`
forward only, at 6.9 % of an 18.65 TFLOP/s matmul roof taken in the same process on the same card
at the same clock. This prices every op in that block against the roof of its own class, which is
the thing that turns one unactionable percentage into a ranked list.

Two roofs, both measured on a Blackhole processor of a p300c, neither a datasheet number:

  matmul  18.6487 TFLOP/s   bf16 [65536,128] @ [128,128], `perf/hallgrad/p2_floor_screen.json`
  DRAM     390.7  GB/s      `ttnn.clone` read+write, `state/b2z2-redteam-v2.md` (the 444.9 that
                            campaign also quotes is a FITTED asymptote above every measured roof)

An op's floor is `max(flops/matmul_roof, bytes/dram_roof)`: it cannot beat the card's arithmetic
rate and it cannot beat the rate at which its operands arrive. Every op here materialises its
output to DRAM, which is the assumption fusion attacks, so the byte column is the fusion target
and the model prices the fused variant by dropping the intermediates.

    python3 perf/bcx_plan/block_cost_model.py [--n 256]

This is a MODEL. `bcx-census` measures the same block op by op on silicon; where they disagree the
census wins and the disagreement is the finding.
"""
import argparse
import json

MATMUL_ROOF = 18.6487e12      # FLOP/s, measured, n=256 arm of the floor screen
DRAM_ROOF = 390.7e9           # B/s, measured, BH p300c
BF16 = 2                      # bytes

# Measured anchors this model is checked against (same card, same clock, 1337-1350 MHz).
MEASURED = {
    "b_256_transition_on": 0.23204,
    "b_256_transition_off": 0.21106,
    "b_fwd_256": 0.08542,
    "b_bwd_256": 0.14662,     # b - b_fwd; the floor screen quotes the 1.715x ratio
    "b_128_transition_on": 0.06491,
}


def ops(n, c=128, heads=4, head_dim=32, hidden=128, factor=4):
    """Every op in `_block` + `_transition`, forward then backward, in execution order.

    (name, class, flops, bytes) with bytes counting DRAM reads plus the write. Weights are
    frozen in the floor screen (`Wtt` is built with the default `requires_grad=False`), so no
    linear carries a weight-gradient matmul and the backward is dgrad only.
    """
    P = n * n * c * BF16                    # one pair tensor, 16.78 MB at n=256
    H = n * n * hidden * BF16               # one trimul-hidden tensor, same size at hidden=c
    A = n * n * heads * head_dim * BF16     # per-head activation, = P at 4x32=128
    S = n * heads * n * n * BF16            # the attention score tensor, n x heads x n x n
    T = n * n * factor * c * BF16           # the transition hidden, 4 pair tensors
    fwd, bwd = [], []

    def f(name, cls, flops, byt):
        fwd.append((name, cls, flops, byt))

    def b(name, cls, flops, byt):
        bwd.append((name, cls, flops, byt))

    for tag in ("out", "in"):
        # --- triangle multiplication, outgoing then incoming
        f(f"tm_{tag}/layer_norm", "layernorm", 0, 2 * P)
        for proj in ("ag", "a", "bg", "b"):
            f(f"tm_{tag}/linear_{proj}", "matmul", 2 * n * n * c * hidden, P + H)
        f(f"tm_{tag}/sigmoid_ag", "eltwise", 0, 2 * H)
        f(f"tm_{tag}/mul_a", "eltwise", 0, 3 * H)
        f(f"tm_{tag}/sigmoid_bg", "eltwise", 0, 2 * H)
        f(f"tm_{tag}/mul_b", "eltwise", 0, 3 * H)
        f(f"tm_{tag}/pair_contract", "matmul", 2 * n ** 3 * hidden, 2 * H + H)
        f(f"tm_{tag}/layer_norm2", "layernorm", 0, 2 * H)
        f(f"tm_{tag}/linear_g", "matmul", 2 * n * n * c * c, P + P)
        f(f"tm_{tag}/sigmoid_g", "eltwise", 0, 2 * P)
        f(f"tm_{tag}/linear_z", "matmul", 2 * n * n * hidden * c, H + P)
        f(f"tm_{tag}/mul_gate", "eltwise", 0, 3 * P)
        f(f"tm_{tag}/add_residual", "eltwise", 0, 3 * P)

        # backward, frozen weights: one dgrad matmul per linear, the contraction twice
        b(f"tm_{tag}/add_residual^", "eltwise", 0, 2 * P)
        b(f"tm_{tag}/mul_gate^", "eltwise", 0, 4 * P)
        b(f"tm_{tag}/linear_z^", "matmul", 2 * n * n * c * hidden, P + H)
        b(f"tm_{tag}/sigmoid_g^", "eltwise", 0, 3 * P)
        b(f"tm_{tag}/linear_g^", "matmul", 2 * n * n * c * c, P + P)
        b(f"tm_{tag}/layer_norm2^", "layernorm", 0, 6 * H)
        b(f"tm_{tag}/pair_contract^", "matmul", 2 * (2 * n ** 3 * hidden), 3 * H + 2 * H)
        for proj in ("a", "b"):
            b(f"tm_{tag}/mul_{proj}^", "eltwise", 0, 4 * H)
            b(f"tm_{tag}/sigmoid_{proj}g^", "eltwise", 0, 3 * H)
        for proj in ("ag", "a", "bg", "b"):
            b(f"tm_{tag}/linear_{proj}^", "matmul", 2 * n * n * hidden * c, H + P)
        b(f"tm_{tag}/layer_norm^", "layernorm", 0, 6 * P)

    for tag in ("start", "end"):
        # --- triangle attention, starting then ending node
        if tag == "end":
            f("ta_end/permute_in", "movement", 0, 2 * P)
        f(f"ta_{tag}/layer_norm", "layernorm", 0, 2 * P)
        for proj in ("q", "k", "v", "g"):
            f(f"ta_{tag}/linear_{proj}", "matmul", 2 * n * n * c * heads * head_dim, P + A)
            f(f"ta_{tag}/reshape_permute_{proj}", "movement", 0, 2 * A)
        f(f"ta_{tag}/linear_bias", "matmul", 2 * n * n * c * heads, P + n * n * heads * BF16)
        f(f"ta_{tag}/permute_bias", "movement", 0, 2 * n * n * heads * BF16)
        f(f"ta_{tag}/attn_qk", "matmul", 2 * n * heads * n * n * head_dim, 2 * A + S)
        f(f"ta_{tag}/attn_softmax", "softmax", 0, 2 * S)
        f(f"ta_{tag}/attn_pv", "matmul", 2 * n * heads * n * n * head_dim, S + A + A)
        f(f"ta_{tag}/sigmoid_g", "eltwise", 0, 2 * A)
        f(f"ta_{tag}/mul_gate", "eltwise", 0, 3 * A)
        f(f"ta_{tag}/permute_reshape_o", "movement", 0, 2 * A)
        f(f"ta_{tag}/linear_o", "matmul", 2 * n * n * heads * head_dim * c, A + P)
        if tag == "end":
            f("ta_end/permute_out", "movement", 0, 2 * P)
        f(f"ta_{tag}/add_residual", "eltwise", 0, 3 * P)

        b(f"ta_{tag}/add_residual^", "eltwise", 0, 2 * P)
        b(f"ta_{tag}/linear_o^", "matmul", 2 * n * n * c * heads * head_dim, P + A)
        b(f"ta_{tag}/permute_reshape_o^", "movement", 0, 2 * A)
        b(f"ta_{tag}/mul_gate^", "eltwise", 0, 4 * A)
        b(f"ta_{tag}/sigmoid_g^", "eltwise", 0, 3 * A)
        # attention backward: recompute the scores, then dv, dp, dq, dk -- five contractions
        b(f"ta_{tag}/attn_recompute_qk", "matmul", 2 * n * heads * n * n * head_dim, 2 * A + S)
        b(f"ta_{tag}/attn_softmax^", "softmax", 0, 3 * S)
        b(f"ta_{tag}/attn_dv", "matmul", 2 * n * heads * n * n * head_dim, S + A + A)
        b(f"ta_{tag}/attn_dp", "matmul", 2 * n * heads * n * n * head_dim, 2 * A + S)
        b(f"ta_{tag}/attn_dq", "matmul", 2 * n * heads * n * n * head_dim, S + A + A)
        b(f"ta_{tag}/attn_dk", "matmul", 2 * n * heads * n * n * head_dim, S + A + A)
        for proj in ("q", "k", "v", "g"):
            b(f"ta_{tag}/reshape_permute_{proj}^", "movement", 0, 2 * A)
            b(f"ta_{tag}/linear_{proj}^", "matmul", 2 * n * n * heads * head_dim * c, A + P)
        b(f"ta_{tag}/linear_bias^", "matmul", 2 * n * n * heads * c, n * n * heads * BF16 + P)
        b(f"ta_{tag}/layer_norm^", "layernorm", 0, 6 * P)
        if tag == "end":
            b("ta_end/permute^", "movement", 0, 4 * P)

    # --- the transition. The harness runs a SwiGLU (three factor-4 matmuls, 24*N^2*c^2);
    # AF2's own pair transition is LayerNorm, linear, ReLU, linear (16*N^2*c^2). Both are
    # emitted so the difference is visible rather than argued about.
    f("tr/layer_norm", "layernorm", 0, 2 * P)
    f("tr/linear_a", "matmul", 2 * n * n * c * factor * c, P + T)
    f("tr/silu_mul", "eltwise", 0, 4 * T)
    f("tr/linear_b", "matmul", 2 * n * n * c * factor * c, P + T)
    f("tr/linear_o", "matmul", 2 * n * n * factor * c * c, T + P)
    f("tr/add_residual", "eltwise", 0, 3 * P)
    b("tr/add_residual^", "eltwise", 0, 2 * P)
    b("tr/linear_o^", "matmul", 2 * n * n * c * factor * c, P + T)
    b("tr/silu_mul^", "eltwise", 0, 5 * T)
    b("tr/linear_b^", "matmul", 2 * n * n * factor * c * c, T + P)
    b("tr/linear_a^", "matmul", 2 * n * n * factor * c * c, T + P)
    b("tr/layer_norm^", "layernorm", 0, 6 * P)
    return fwd, bwd


def floor(flops, byt):
    return max(flops / MATMUL_ROOF, byt / DRAM_ROOF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", default="perf/bcx_plan/block_cost_model.json")
    a = ap.parse_args()
    fwd, bwd = ops(a.n)
    rows = [(("fwd " + nm), cls, fl, by, floor(fl, by)) for nm, cls, fl, by in fwd]
    rows += [(("bwd " + nm), cls, fl, by, floor(fl, by)) for nm, cls, fl, by in bwd]

    by_class, by_phase = {}, {}
    for nm, cls, fl, by, t in rows:
        d = by_class.setdefault(cls, {"flops": 0, "bytes": 0, "floor": 0.0, "ops": 0})
        d["flops"] += fl; d["bytes"] += by; d["floor"] += t; d["ops"] += 1
        p = by_phase.setdefault(nm.split()[0], {"flops": 0, "bytes": 0, "floor": 0.0, "ops": 0})
        p["flops"] += fl; p["bytes"] += by; p["floor"] += t; p["ops"] += 1

    tot_floor = sum(r[4] for r in rows)
    meas = MEASURED["b_256_transition_on"] if a.n == 256 else MEASURED["b_128_transition_on"]
    print(f"n={a.n}: {len(rows)} ops, {sum(r[2] for r in rows)/1e9:.2f} GFLOP, "
          f"{sum(r[3] for r in rows)/1e6:.1f} MB moved")
    print(f"  floor {tot_floor*1e3:.3f} ms   measured b {meas*1e3:.3f} ms   "
          f"measured / floor = {meas/tot_floor:.2f}x\n")
    print(f"  {'class':10s} {'ops':>4s} {'GFLOP':>8s} {'MB':>8s} {'floor ms':>9s} "
          f"{'bound':>6s} {'% of floor':>10s}")
    for cls, d in sorted(by_class.items(), key=lambda kv: -kv[1]["floor"]):
        bound = "flops" if d["flops"] / MATMUL_ROOF > d["bytes"] / DRAM_ROOF else "bytes"
        print(f"  {cls:10s} {d['ops']:4d} {d['flops']/1e9:8.2f} {d['bytes']/1e6:8.1f} "
              f"{d['floor']*1e3:9.3f} {bound:>6s} {100*d['floor']/tot_floor:9.1f} %")
    print()
    for ph, d in by_phase.items():
        print(f"  {ph}: {d['ops']} ops, {d['flops']/1e9:.2f} GFLOP, {d['bytes']/1e6:.1f} MB, "
              f"floor {d['floor']*1e3:.3f} ms")
    print("\n  top 12 ops by floor time")
    for nm, cls, fl, by, t in sorted(rows, key=lambda r: -r[4])[:12]:
        print(f"    {t*1e3:7.3f} ms  {100*t/tot_floor:5.1f} %  {cls:10s} {nm}")

    fl = sum(r[2] for r in rows); by = sum(r[3] for r in rows)
    ridge = MATMUL_ROOF / DRAM_ROOF
    print(f"\n  roofline placement of the whole block")
    print(f"    arithmetic intensity {fl/by:7.2f} FLOP/byte   card ridge {ridge:.2f} FLOP/byte"
          f"  ->  {'DRAM' if fl/by < ridge else 'compute'}-bound by {ridge/(fl/by):.2f}x")
    print(f"    achieved  {fl/meas/1e12:6.3f} TFLOP/s = {100*fl/meas/MATMUL_ROOF:5.2f} % of the matmul roof")
    print(f"    achieved  {by/meas/1e9:6.1f} GB/s     = {100*by/meas/DRAM_ROOF:5.2f} % of the DRAM roof"
          f"   ({DRAM_ROOF*meas/by:.2f}x to reach it)")
    print(f"    bytes-only floor {by/DRAM_ROOF*1e3:.3f} ms; at bfp8 (1 B/elem) {by/2/DRAM_ROOF*1e3:.3f} ms")

    json.dump({"n": a.n, "intensity": fl/by, "ridge": ridge,
               "achieved_tflops": fl/meas/1e12, "achieved_gbs": by/meas/1e9, "matmul_roof": MATMUL_ROOF, "dram_roof": DRAM_ROOF,
               "measured": MEASURED, "total_floor_s": tot_floor,
               "by_class": by_class, "by_phase": by_phase,
               "rows": [{"op": nm, "class": cls, "flops": fl, "bytes": by, "floor_s": t}
                        for nm, cls, fl, by, t in rows]},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
