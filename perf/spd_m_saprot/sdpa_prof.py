"""Where SaProt's page attention (shsdpa, B7 H20 S544 D64) spends its device time, per RISC, under the device profiler.

perf/shsdpa/bench.py (ls63, Galaxy chip at 1000 MHz) reads SaProt's config (HiFi4, fp32 dest) at 0.443 ms a call and the
same call with the exponential compiled out (ABLATE_EXP) at 0.437, so the SFPU exp is not what bounds it. This reads the
kernel duration of each RISC and the compute cluster's CB stall sums for each arm, the way mm_prof does for projections.

Needs the Tracy tt-metal build and TT_METAL_DEVICE_PROFILER=1 (environment as in mm_prof's jobs).

    python perf/spd_m_saprot/sdpa_prof.py --out prof.jsonl
"""
import argparse
import json

import torch
import ttnn

from mm_prof import KEYS, SUMS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bh", default="7,20")
    ap.add_argument("--lens", default="544,544,520,480,430,390,300")
    ap.add_argument("--calls", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)

    from tt_bio import tenstorrent as T, shsdpa as P
    dev = T.get_device()
    B, H = map(int, a.bh.split(","))
    S, D = 544, 64
    lens = ([int(x) for x in a.lens.split(",")] * B)[:B]
    gen = torch.Generator().manual_seed(544)
    q, k, v = (ttnn.from_torch(torch.randn(B, H, S, D, generator=gen).to(torch.bfloat16), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
               for _ in range(3))
    sc = D ** -0.5
    HIFI4F = (ttnn.MathFidelity.HiFi4, False, True, False)
    HIFI2F = (ttnn.MathFidelity.HiFi2, True, True, False)
    pads = {kc: P.key_pad(lens, S, dev, kc) for kc in (192, 288)}
    arms = {
        "ship": lambda: P.sdpa(q, k, v, sc, pads[288], ckc=HIFI4F, merged=True),
        "hifi2": lambda: P.sdpa(q, k, v, sc, pads[288], ckc=P.HIFI2, merged=True),
        "hifi2f": lambda: P.sdpa(q, k, v, sc, pads[288], ckc=HIFI2F, merged=True),
        "ablate_exp": lambda: P.sdpa(q, k, v, sc, pads[288], ckc=HIFI4F, merged=True, defines={"ABLATE_EXP": 1}),
        "k192": lambda: P.sdpa(q, k, v, sc, pads[192], k_chunk=192, ckc=HIFI4F, merged=True),
    }
    for qc in (64, 128, 288):
        arms[f"q{qc}"] = lambda qc=qc: P.sdpa(q, k, v, sc, pads[288], q_chunk=qc, split=None, ckc=HIFI4F, merged=True)
    out = open(a.out, "a")

    def flush():
        ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)
        return [p for progs in ttnn.get_latest_programs_perf_data().values() for p in progs]

    flush()
    print(json.dumps(dict(plan=P.plan(B, H, S, T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]))), flush=True)
    for arm, fn in arms.items():
        try:
            for _ in range(2):
                ttnn.deallocate(fn())
            flush()
            for _ in range(a.calls):
                ttnn.deallocate(fn())
            progs = flush()
        except Exception as e:  # noqa: BLE001 -- a refused program is a data point
            rec = dict(arm=arm, error=str(e).splitlines()[0][:200])
            print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()
            continue
        for p in progs:
            r = p.program_analyses_results
            cc = max(1, int(p.core_count))
            rec = dict(arm=arm, bh=a.bh, lens=a.lens, cores=cc,
                       **{s: round(r[k_].duration / 1e3, 2) for s, k_ in KEYS.items() if k_ in r},
                       **{s: round(r[k_].duration / cc / 1e3, 2) for s, k_ in SUMS.items() if k_ in r})
            print(json.dumps(rec), flush=True)
            out.write(json.dumps(rec) + "\n"); out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
