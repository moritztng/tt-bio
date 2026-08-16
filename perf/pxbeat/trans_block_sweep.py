#!/usr/bin/env python3
"""E8.1 / E9: find the block regime the fused swiglu needs at Transition's shape, or prove none wins.

E9 measured that `trimul_tail.eligible()` accepts Transition's chunk (`mt=256, kt=8, nt=32`) and that
F1's shipped `BLOCK = (4,8,1,4,1)` -- tuned for the trimul tail's `mt=8192, nt=8` -- does not perform
on it: at least ~3.5 s/call against a 0.417 ms chain. `BLOCK`'s `N=1` against `nt=32` is the suspect,
so this sweeps the block instead of guessing at it. `K` is pinned at 8 because that is `kt` and one K
block is the fusion's whole simplification.

The number every arm is scored against is `pair_only` = `layer_norm` + `fc1(silu)` + `fc2` +
`multiply_`, the leg a fused kernel replaces, MEASURED at 0.3697 ms/chunk in
`perf/pxbeat/trans_screen_512_c1.json`. Registered kill gate, written before the run (state doc E9):

    a fused arm must come in UNDER `pair_only` on this harness before any parity work is done.

Under it the fused kernel has spent the multiply's 0.806 s/fold before its epilogue can save it, and
the swiglu build is a NO-GO with a number. Timing here is per-call with a device sync per repeat, so
it prices ~2x high in absolute terms (`tt-bio-isolated-op-timing-oversync-inflates-cost`); both the
baseline and the arms carry the same inflation and the comparison is the point.
"""
import argparse, json, signal, statistics, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch                                                                     # noqa: E402
import ttnn                                                                      # noqa: E402
import tt_bio.tenstorrent as T                                                   # noqa: E402
from tt_bio import trimul_tail as TT                                             # noqa: E402
from tt_bio import mm_generic as MG                                              # noqa: E402

CHUNK_CALLS_PER_FOLD = 16768


class Timeout(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--c", type=int, default=256)
    ap.add_argument("--dff", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--budget", type=int, default=150, help="seconds per candidate before abandon")
    ap.add_argument("--only", default="", help="one M,K,N,sh,sw; the SIGALRM backstop cannot\n"
                    "preempt a ttnn call, so each candidate runs as its own process under an OS timeout")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    grid = tuple(T.COMPUTE_GRID_MAIN)
    cka = MG.ckc_args(ckc)
    torch.manual_seed(0)
    L1, DR = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG

    def dram(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                               memory_config=DR)

    x = dram(torch.randn(1, a.rows, a.w, a.c, dtype=torch.bfloat16))
    nw = dram(torch.randn(a.c, dtype=torch.bfloat16))
    nb = dram(torch.randn(a.c, dtype=torch.bfloat16))
    w1 = dram(torch.randn(a.c, a.dff, dtype=torch.bfloat16))
    w2 = dram(torch.randn(a.c, a.dff, dtype=torch.bfloat16))

    def ln(t):
        return ttnn.layer_norm(t, weight=nw, bias=nb, epsilon=1e-5, compute_kernel_config=ckc,
                               memory_config=L1)

    def pair_only():
        xn = ln(x)
        x1 = ttnn.linear(xn, w1, activation="silu", compute_kernel_config=ckc, memory_config=L1,
                         core_grid=T.CORE_GRID_MAIN)
        x2 = ttnn.linear(xn, w2, compute_kernel_config=ckc, memory_config=L1,
                         core_grid=T.CORE_GRID_MAIN)
        ttnn.deallocate(xn)
        p = ttnn.multiply_(x1, x2)
        ttnn.deallocate(x2)
        return p

    def fused():
        xn = ln(x)
        p = TT.fused_tail(xn, xn, w1, w2, cka, grid)
        ttnn.deallocate(xn)
        return p

    def timed(fn, reps):
        o = fn()
        if o is None:
            return None
        ttnn.synchronize_device(dev); ttnn.deallocate(o)
        ts = []
        for _ in range(reps):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = fn()
            ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t0)
            ttnn.deallocate(o)
        return statistics.median(ts)

    mt = a.rows * a.w // 32
    kt, nt = a.c // 32, a.dff // 32
    print(f"shape mt={mt} kt={kt} nt={nt}  grid={grid}  shipped BLOCK={TT.BLOCK}", flush=True)

    base = timed(pair_only, 5)
    print(f"pair_only (the leg a fusion replaces) {1e3*base:.4f} ms/call\n", flush=True)

    def divisors(n, cap):
        return [d for d in range(1, min(n, cap) + 1) if n % d == 0]

    cands = []
    for M in (2, 4, 8, 16, 64):
        if mt % M:
            continue
        for N in (1, 2, 4, 8, 16, 32):
            if nt % N:
                continue
            sh = max(d for d in divisors(M, 4))
            sw = max((d for d in divisors(N, 8 // sh)), default=1)
            if sh * sw > 8 or M % sh or N % sw:
                continue
            cands.append((M, kt, N, sh, sw))
    if a.only:
        cands = [tuple(int(v) for v in a.only.split(","))]
    print(f"{len(cands)} block candidates, K pinned at kt={kt}\n", flush=True)

    def handler(_s, _f):
        raise Timeout()
    signal.signal(signal.SIGALRM, handler)

    rows, shipped = [], tuple(TT.BLOCK)
    for cand in cands:
        TT.BLOCK = cand
        TT._CACHE.clear()
        row = {"block": list(cand), "is_shipped": cand == shipped}
        t0 = time.perf_counter()
        signal.alarm(a.budget)
        try:
            ms = timed(fused, a.reps)
            signal.alarm(0)
            if ms is None:
                row["declined"] = TT.eligible(x, x, w1, w2)
                print(f"  {str(cand):22s} DECLINED: {row['declined']}", flush=True)
            else:
                row["ms"] = round(1e3 * ms, 4)
                row["vs_pair_only"] = round(base / ms, 4)
                print(f"  {str(cand):22s} {1e3*ms:9.4f} ms  {base/ms:.4f}x vs pair_only",
                      flush=True)
        except Timeout:
            row["abandoned_s"] = a.budget
            print(f"  {str(cand):22s} ABANDONED after {a.budget}s", flush=True)
        except Exception as e:                                                   # noqa: BLE001
            signal.alarm(0)
            row["error"] = f"{type(e).__name__}: {str(e)[:140]}"
            print(f"  {str(cand):22s} {row['error']}", flush=True)
        row["wall_s"] = round(time.perf_counter() - t0, 1)
        rows.append(row)
    TT.BLOCK = shipped
    signal.alarm(0)

    ok = sorted([r for r in rows if "ms" in r], key=lambda r: r["ms"])
    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": "qb2", "card": 1, "grid": list(grid),
           "shape": [1, a.rows, a.w, a.c], "d_ff": a.dff, "m_tiles": mt, "k_tiles": kt,
           "n_tiles": nt, "reps": a.reps, "budget_s": a.budget,
           "loadavg": open("/proc/loadavg").read().split()[:3],
           "pair_only_ms": round(1e3 * base, 4), "shipped_block": list(shipped),
           "note": "perf only; the fused arm computes p*sigmoid(g), not silu(p)*g. No parity claim.",
           "arms": rows, "best": ok[:5]}
    res["best_ms"] = ok[0]["ms"] if ok else None
    res["verdict"] = ("GO: a block regime beats pair_only"
                      if ok and ok[0]["ms"] < 1e3 * base else "NO-GO: nothing beats pair_only")
    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\npair_only {1e3*base:.4f} ms   best fused "
          f"{res['best_ms'] if res['best_ms'] is not None else 'none ran'}   -> {res['verdict']}")
    print("wrote", a.out)


main()
