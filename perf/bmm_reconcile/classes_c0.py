#!/usr/bin/env python3
"""E7 — all 16 batched-matmul classes, every candidate chooser, one card, one process.

Arms:
  ttnn    the plain call (this is also what G1 does wherever it declines)
  G1      `wk/perfwar-batched-matmul-config` @ cac0eead, verbatim
  E1      `wk/perfwar-of3-matmul-sites` @ d8073b6b, verbatim
  R_occ   reconciled, occupancy-first per_core_M (E1's rule) + G1's exact CB model and safety rule
  R_cost  reconciled, G1's read-cost per_core_M + G1's exact CB model and safety rule

Both reconciled arms drop G1's `n_tiles > 4` gate and take E1's `in0_block_w` (2 only when Kt > 2
and even), which is the width measured bit-exact at Kt in {1, 2, 4, 10}. The two differ only in how
per_core_M is chosen, which is the one open question after the q@kT settling run.

Iteration count adapts to the op: a 0.05 ms op gets ~400 calls, not 4 (E1's lesson -- five
iterations of a 0.04 ms op produced a fake 0.75x). Device synced on both sides of every timed
region, arms round-robin within each rep.
"""
import argparse, json, math, time
import torch, ttnn
from tt_bio.tenstorrent import get_device
import tt_bio.tenstorrent as T

DRAM = ttnn.DRAM_MEMORY_CONFIG
F32, BF16 = ttnn.float32, ttnn.bfloat16
_TILE_READ_PER_TILE_MAC = 3.0


def _subblock_g1(per_core_M, n_tiles):
    sub_w = max(w for w in range(1, min(4, n_tiles) + 1) if n_tiles % w == 0)
    sub_h = max(h for h in range(1, min(4 // sub_w, per_core_M) + 1) if per_core_M % h == 0)
    return sub_h, sub_w


def _subblock_e1(per_core_M, per_core_N):
    best = (1, 1)
    for h in range(1, per_core_M + 1):
        if per_core_M % h:
            continue
        for w in range(1, per_core_N + 1):
            if per_core_N % w or h * w > 4:
                continue
            if h * w > best[0] * best[1]:
                best = (h, w)
    return best


def g1_config(batch, mt, kt, nt, eb, grid):
    gx, gy = grid
    cores = gx * gy
    if batch < 2 or nt > 4 or batch * mt < cores:
        return None
    bw = 2 if kt % 2 == 0 else 1
    l1 = int(ttnn.get_max_worker_l1_unreserved_size())
    tile, acc = 1024 * eb, 4096
    best = ()
    for p in range(1, mt + 1):
        if mt % p or (p != mt and batch * mt // p > cores):
            continue
        if 2 * (p + nt) * bw * tile + p * nt * (tile + acc) > l1:
            continue
        blocks = batch * mt // p
        reads = batch * mt * kt + blocks * kt * nt
        cost = max(_TILE_READ_PER_TILE_MAC * reads, -(-blocks // cores) * p * nt * kt)
        if not best or cost < best[0]:
            best = (cost, p)
    if not best:
        return None
    h, w = _subblock_g1(best[1], nt)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=bw, out_subblock_h=h, out_subblock_w=w,
        per_core_M=best[1], per_core_N=nt)


def e1_config(batch, mt, kt, nt, eb, grid):
    gx, gy = grid
    cores = gx * gy
    if batch < 2:
        return None
    p = mt
    for d in range(1, mt + 1):
        if mt % d == 0 and batch * (mt // d) <= cores:
            p = d
            break
    if p * nt * 6 * 1024 > 700 * 1024:
        return None
    h, w = _subblock_e1(p, nt)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=(2 if kt > 2 and kt % 2 == 0 else 1),
        out_subblock_h=h, out_subblock_w=w, per_core_M=p, per_core_N=nt)


def r_config(batch, mt, kt, nt, eb, grid, mode):
    """Reconciled: G1's safety rule + G1's exact CB footprint + E1's in0_block_w, no Nt gate."""
    gx, gy = grid
    cores = gx * gy
    if batch < 2 or batch * mt < cores:
        return None
    bw = 2 if kt > 2 and kt % 2 == 0 else 1
    l1 = int(ttnn.get_max_worker_l1_unreserved_size())
    tile, acc = 1024 * eb, 4096
    legal = []
    for p in range(1, mt + 1):
        if mt % p or (p != mt and batch * mt // p > cores):
            continue
        if 2 * (p + nt) * bw * tile + p * nt * (tile + acc) > l1:
            continue
        legal.append(p)
    if not legal:
        return None
    if mode == "occ":
        p = min(legal)  # most blocks = most engaged cores
    else:
        best = ()
        for q in legal:
            blocks = batch * mt // q
            reads = batch * mt * kt + blocks * kt * nt
            cost = max(_TILE_READ_PER_TILE_MAC * reads, -(-blocks // cores) * q * nt * kt)
            if not best or cost < best[0]:
                best = (cost, q)
        p = best[1]
    h, w = _subblock_g1(p, nt)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=bw, out_subblock_h=h, out_subblock_w=w,
        per_core_M=p, per_core_N=nt)


# (a, b, dtype, label, calls/fold, model)
CASES = [
    ((75, 4, 32, 128), (75, 4, 128, 32), F32, "atom q@kT", 1200, "protenix-v2"),
    ((75, 4, 32, 32), (75, 4, 32, 128), F32, "atom attn@v", 1200, "protenix-v2"),
    ((1, 16, 320, 320), (1, 16, 320, 64), F32, "DiT attn@v", 4800, "protenix-v2/of3"),
    ((1, 16, 320, 64), (1, 16, 64, 320), F32, "DiT q@kT", 4800, "protenix-v2/of3"),
    ((75, 4, 32, 128), (75, 4, 128, 32), BF16, "atom q@kT", 1200, "opendde"),
    ((75, 4, 32, 32), (75, 4, 32, 128), BF16, "atom attn@v", 1200, "opendde"),
    ((1, 16, 608, 608), (1, 16, 608, 64), BF16, "DiT attn@v", 0, "opendde"),
    ((1, 8, 608, 608), (1, 8, 608, 64), BF16, "DiT attn@v tail", 0, "opendde"),
    ((1, 16, 608, 64), (1, 16, 64, 608), BF16, "DiT q@kT 580aa", 0, "opendde"),
    ((298, 4, 298, 298), (298, 4, 298, 32), BF16, "tri-att attn@v", 1328, "openfold3 trunk"),
    ((298, 4, 298, 32), (298, 4, 32, 298), BF16, "tri-att q@kT", 1328, "openfold3 trunk"),
    ((1, 16, 298, 298), (1, 16, 298, 32), BF16, "AttnPairBias attn@v", 528, "openfold3"),
    ((1, 64, 298, 298), (1, 64, 298, 298), BF16, "trimul class", 2480, "openfold3"),
    ((1, 75, 4, 32, 32), (1, 75, 4, 32, 128), F32, "atom attn@v rank5", 1200, "openfold3"),
    ((1, 75, 4, 32, 128), (1, 75, 4, 128, 32), F32, "atom q@kT rank5", 1200, "openfold3"),
]


def med(xs):
    return sorted(xs)[len(xs) // 2]


def cfg_repr(c):
    if c is None:
        return "declined"
    return (f"pM={c.per_core_M} pN={c.per_core_N} bw={c.in0_block_w} "
            f"sub={c.out_subblock_h}x{c.out_subblock_w}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--target-ms", type=float, default=25.0)
    a = ap.parse_args()

    dev = get_device()
    T._configure_active_compute_grid(dev)
    grid = T.COMPUTE_GRID_MAIN
    cores = grid[0] * grid[1]
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    out = {"grid": list(grid), "cores": cores,
           "l1_unreserved": int(ttnn.get_max_worker_l1_unreserved_size()), "classes": []}
    print(f"grid {grid} = {cores} cores")

    for sa, sb, dt, label, calls, model in CASES:
        A = ttnn.from_torch(torch.randn(*sa) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=dt, memory_config=DRAM)
        B = ttnn.from_torch(torch.randn(*sb) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=dt, memory_config=DRAM)
        pa, pb = list(A.padded_shape), list(B.padded_shape)
        batch = 1
        for d in pa[:-2]:
            batch *= int(d)
        mt, kt, nt = int(pa[-2]) // 32, int(pa[-1]) // 32, int(pb[-1]) // 32
        eb = 4 if dt == F32 else 2
        arms = {
            "ttnn": None,
            "G1": g1_config(batch, mt, kt, nt, eb, grid),
            "E1": e1_config(batch, mt, kt, nt, eb, grid),
            "R_occ": r_config(batch, mt, kt, nt, eb, grid, "occ"),
            "R_cost": r_config(batch, mt, kt, nt, eb, grid, "cost"),
        }
        rec = {"label": label, "model": model, "a": list(sa), "b": list(sb),
               "dtype": "fp32" if dt == F32 else "bf16", "calls_per_fold": calls,
               "batch": batch, "Mt": mt, "Kt": kt, "Nt": nt,
               "configs": {k: cfg_repr(v) for k, v in arms.items()},
               "blocks": {k: (batch * mt // v.per_core_M) if v is not None else None
                          for k, v in arms.items()},
               "ms": {}, "bit_exact": {}}
        print(f"\n== {model} {label}  {sa}x{sb} {rec['dtype']}  "
              f"B={batch} Mt/Kt/Nt={mt}/{kt}/{nt}")

        def run(c):
            kw = {"compute_kernel_config": ckc}
            if c is not None:
                kw["program_config"] = c
            return ttnn.matmul(A, B, **kw)

        ref = ttnn.to_torch(run(None))
        for k, v in arms.items():
            if v is None:
                rec["bit_exact"][k] = None if k != "ttnn" else True
                continue
            got = ttnn.to_torch(run(v))
            rec["bit_exact"][k] = bool(torch.equal(ref, got))
            del got
        del ref

        live = [k for k, v in arms.items() if k == "ttnn" or v is not None]
        for k in live:
            for _ in range(2):
                ttnn.deallocate(run(arms[k]))
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(4):
            ttnn.deallocate(run(None))
        ttnn.synchronize_device(dev)
        est = (time.perf_counter() - t0) * 1e3 / 4
        pipe = max(4, min(400, int(math.ceil(a.target_ms / max(est, 1e-3)))))
        samples = {k: [] for k in live}
        for _ in range(a.reps):
            for k in live:
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                for _ in range(pipe):
                    ttnn.deallocate(run(arms[k]))
                ttnn.synchronize_device(dev)
                samples[k].append((time.perf_counter() - t0) * 1e3 / pipe)
        rec["iters"] = pipe * a.reps
        for k in live:
            rec["ms"][k] = round(med(samples[k]), 5)
            rec.setdefault("ms_all", {})[k] = [round(x, 5) for x in samples[k]]
        base = rec["ms"]["ttnn"]
        rec["speedup"] = {k: round(base / rec["ms"][k], 3) for k in live}
        for k in ["ttnn", "G1", "E1", "R_occ", "R_cost"]:
            if k in rec["ms"]:
                print(f"   {k:7s} {rec['ms'][k]:9.4f} ms {rec['speedup'][k]:6.2f}x  "
                      f"blocks={rec['blocks'][k]}  exact={rec['bit_exact'][k]}  "
                      f"{rec['configs'][k]}")
            else:
                print(f"   {k:7s} declined")
        print(f"   iters/arm={rec['iters']}")
        out["classes"].append(rec)
        ttnn.deallocate(A); ttnn.deallocate(B)

    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
