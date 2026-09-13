#!/usr/bin/env python3
"""`torch.equal` at the op for all three L1 placements this branch makes, plus negative controls.

A memory config decides which banks a tile lands in. It must not decide what is in the tile, and
the CIF digest at the fold is downstream evidence of that, not a direct test: a fold that never
reached a placement would pass it too. Each case here runs the SAME op twice, once with the
operand or the destination in DRAM and once in L1, and compares the two results element for
element. Each also runs a control that perturbs the input, which must then differ -- otherwise the
comparison is reading something that cannot fail.

Sizes sweep the token axis the fold buckets to, so the tile tail is covered as well as the round
case. Random operands, not constants: a constant input is equal to itself under any bug that drops
or duplicates a tile.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch                                                                   # noqa: E402
torch.set_grad_enabled(False)
import ttnn                                                                    # noqa: E402
import tt_bio.tenstorrent as TT                                                # noqa: E402
import tt_bio.triatt_qkv as TQ                                                 # noqa: E402

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
RESULTS: list = []


def to_dev(t, mc, dev):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=mc)


def case(name, S, run, perturb, gate):
    """`run(mc)` returns a torch result with the placement in `mc`; `perturb()` must change it.

    `gate()` is the production fit test for this placement. When it declines, the fold keeps the
    DRAM path and there is nothing to compare -- forcing L1 anyway measures a configuration that
    never ships, and at 768 aa it throws a circular-buffer clash. Record the refusal instead.
    """
    if not gate():
        row = {"case": name, "S": S, "gate_declined": True}
        RESULTS.append(row)
        print("  %-26s S=%-5d gate declines L1, DRAM path unchanged" % (name, S), flush=True)
        return row
    a = run(DRAM)
    b = run(L1)
    ctl = perturb()
    row = {"case": name, "S": S, "gate_declined": False, "equal": bool(torch.equal(a, b)),
           "max_abs_diff": float((a.float() - b.float()).abs().max()),
           "control_differs": ctl is None or not bool(torch.equal(a, ctl))}
    RESULTS.append(row)
    print("  %-26s S=%-5d equal=%-5s maxabs=%.3g control_differs=%s"
          % (name, S, row["equal"], row["max_abs_diff"], row["control_differs"]), flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes", default="128,298,512,768")
    args = ap.parse_args()
    dev = TT.get_device()
    CKC = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    C, HEADS, HD, CZ = 32, 4, 32, 128

    for S in [int(x) for x in args.sizes.split(",")]:
        torch.manual_seed(S)
        # --- site 1: the trimul's broadcast pair mask -------------------------------
        chunk = torch.randn(1, C, S, S).bfloat16()
        mask = (torch.rand(1, 1, S, S) > 0.3).float().bfloat16()

        def mul(mc, m=mask, c=chunk):
            a = to_dev(c, DRAM, dev)
            b = to_dev(m, mc, dev)
            out = ttnn.multiply_(a, b)
            r = ttnn.to_torch(out)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            return r

        def mul_ctl(m=mask.clone(), c=chunk):
            m = m.clone()
            m[0, 0, S - 1, S - 1] = 1.0 - m[0, 0, S - 1, S - 1]
            return mul(DRAM, m, c)
        case("trimul mask operand", S, mul, mul_ctl,
             lambda: TT._l1_fits(TT._padded_bytes((1, 1, S, S), 2), 1.0,
                                 TT._PAIR_L1_CONSUMER_RESERVE))

        # --- site 2a: the head-major output projection's destination -----------------
        gated = torch.randn(S, HEADS, S, HD).bfloat16()
        w = torch.randn(HEADS * HD, CZ).bfloat16()

        def proj(mc, g=gated, ww=w):
            a = to_dev(g, DRAM, dev)
            b = to_dev(ww, DRAM, dev)
            out = TQ.out_proj(a, b, CKC, ttnn.bfloat16, memory_config=mc)
            r = ttnn.to_torch(out)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            ttnn.deallocate(out)
            return r

        def proj_ctl(g=gated, ww=w):
            g = g.clone()
            g[0, 0, 0, 0] += 1.0
            return proj(DRAM, g, ww)
        case("triatt out_proj dest", S, proj, proj_ctl,
             lambda: TT._residual_update_memory_config((S, S, CZ), ttnn.bfloat16) is not None)

        # --- site 2b: the transition's row-block assembly ---------------------------
        parts = [torch.randn(1, 16, S, CZ).bfloat16() for _ in range(4)]

        def cat(mc, ps=parts):
            ts = [to_dev(p, DRAM, dev) for p in ps]
            out = TT._concat_to(ts, 1, None if mc is DRAM else mc)
            r = ttnn.to_torch(out)
            for t in ts:
                ttnn.deallocate(t)
            ttnn.deallocate(out)
            return r

        def cat_ctl(ps=parts):
            ps = [p.clone() for p in ps]
            ps[2][0, 0, 0, 0] += 1.0
            return cat(DRAM, ps)
        case("transition concat dest", S, cat, cat_ctl,
             lambda: TT._residual_update_memory_config((1, 64, S, CZ), ttnn.bfloat16)
             is not None)

    tested = [r for r in RESULTS if not r.get("gate_declined")]
    ok = bool(tested) and all(r["equal"] and r["control_differs"] for r in tested)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"all_equal_and_controls_fire": ok, "n_tested": len(tested),
         "n_gate_declined": len(RESULTS) - len(tested), "rows": RESULTS}, indent=1))
    print("\nALL torch.equal AND every control differs: %s" % ok, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
