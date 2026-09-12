#!/usr/bin/env python3
"""Does the grid rule pick the measured optimum away from the point it was fitted at?

`one-size-tuning-is-a-standing-defect-class` is a standing lesson on this campaign, and the
shipped `q_chunk` constant is exactly that defect. A rule fitted at one shape is the next one,
so this sweeps every valid `q_chunk` at five sequence lengths and two head counts, finds the
measured optimum at each, and prints what `_grid_q_chunk` would have picked. No fold and no
grabbed step: the question is about one op, and it is asked on synthetic operands of the shipped
shapes and dtypes. `torch.equal` against the shipped config is checked on every rung, because the
claim that q partitioning is free has to hold at every length, not just at 512.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "perf" / "b2z2_step_fusion", ROOT / "scripts" / "gpu_vs_tt"):
    sys.path.insert(0, str(p))

CASES = [(s, h, 64) for s in (320, 512, 768, 1024, 1536) for h in (8, 16)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import step_probe as SP

    SP.OUT_PATH = a.out
    out = SP.OUT
    out["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "card": os.environ.get("TT_VISIBLE_DEVICES"),
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip()}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    dev = T.get_device(trace_region_size=512 << 20)
    n_cores = T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]
    out["env"].update(arch=str(dev.arch()), n_cores=n_cores,
                      grid=list(T.COMPUTE_GRID_MAIN))
    fence = SP.make_fence(ttnn, dev)
    print(f"  {dev.arch()} {T.COMPUTE_GRID_MAIN} = {n_cores} cores", flush=True)

    def tt(x):
        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)

    rows = []
    for S, H, D in CASES:
        g = torch.Generator().manual_seed(0)
        q = tt(torch.randn(1, H, S, D, generator=g))
        k = tt(torch.randn(1, H, S, D, generator=g))
        v = tt(torch.randn(1, H, S, D, generator=g))
        bias = tt(torch.randn(1, H, S, S, generator=g))
        shipped = T._capped_sdpa_chunk_size(S)
        kc = shipped
        rule = T._grid_q_chunk(S, H, shipped, n_cores)

        def at(qc):
            return ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, is_causal=False, scale=D ** -0.5,
                program_config=T._sdpa_program_config(qc, kc))

        ref = ttnn.to_torch(at(shipped))
        sweep = {}
        for qc in range(32, min(T.SDPA_CHUNK_MAX, S) + 1, 32):
            if S % qc:
                continue
            try:
                got = ttnn.to_torch(at(qc))
                t = SP.timed_reps(ttnn, dev, lambda qc=qc: at(qc), (), {}, a.reps, fence, n_med=3)
            except Exception as exc:                       # noqa: BLE001
                sweep[qc] = {"refused": str(exc)[:120]}
                continue
            sweep[qc] = {"us": round(t["ms_per_call"] * 1e3, 2),
                         "units": H * (S // qc),
                         "bit_exact": bool(torch.equal(ref, got)),
                         "max_abs": float((ref - got).abs().max())}
        ok = {qc: r for qc, r in sweep.items() if "us" in r}
        best = min(ok, key=lambda qc: ok[qc]["us"])
        row = {"S": S, "heads": H, "shipped": shipped, "rule": rule, "measured_best": best,
               "rule_us": ok.get(rule, {}).get("us"), "best_us": ok[best]["us"],
               "shipped_us": ok[shipped]["us"],
               "rule_vs_shipped": round(ok[shipped]["us"] / ok[rule]["us"], 4) if rule in ok else None,
               "best_vs_shipped": round(ok[shipped]["us"] / ok[best]["us"], 4),
               "all_bit_exact": all(r["bit_exact"] for r in ok.values()),
               "sweep": sweep}
        rows.append(row)
        print(f"  S={S:5d} h={H:3d} shipped={shipped:4d} ({row['shipped_us']:7.2f} us)  "
              f"rule={rule:4d} ({row['rule_us']})  best={best:4d} ({row['best_us']:7.2f} us)  "
              f"rule/shipped={row['rule_vs_shipped']}  best/shipped={row['best_vs_shipped']}  "
              f"bit_exact_all={row['all_bit_exact']}", flush=True)
        for t_ in (q, k, v, bias):
            ttnn.deallocate(t_)
        out["ladder"] = rows
        SP.dump()

    miss = [r for r in rows if r["rule"] != r["measured_best"]]
    out["rule_misses"] = [{kk: r[kk] for kk in ("S", "heads", "rule", "measured_best",
                                                "rule_us", "best_us")} for r in miss]
    out["worst_rule_vs_shipped"] = min(r["rule_vs_shipped"] for r in rows
                                       if r["rule_vs_shipped"] is not None)
    SP.dump()
    print(f"\n  rule picks the measured optimum on {len(rows) - len(miss)}/{len(rows)} shapes; "
          f"worst rule/shipped {out['worst_rule_vs_shipped']}")
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
