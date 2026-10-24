"""RFD4's triangle tail at 512 / 384 exactly as main runs it (T2CM stream16, g_out / z bfp8, p GEMM HiFi2, GROW, S1K,
ROT with x page-rotated as bmm_res writes it): median ms of warm synchronized calls, error vs float64, output sha,
then the same call under bench knobs. `abl=` T2_ABL (compute_ln.cpp: 1 LN, 2 p GEMM, 8 epilogue, 16 sigmoid,
64 the transposes), `dm=` T2_DMABL (reader 1 gather L1 reads, 2 stage DRAM reads; writer 4 g / z reads, 8 writes),
`grow=0`, `s1k=0`, `rot=0` turn main's levers off,
`lnblk=` T2_LNBLK, `wd=` T2_WDEPTH, `dmcfg=` T2_DMCFG, `xnoc=` T2_XNOC, `wstage=` T2_WSTAGE, `mcast=0` and `sbw=` the p GEMM's subblock width price compute levers. Knob sets are comma-joined, legs ';'-joined.

usage: TT_VISIBLE_DEVICES=<chip> timeout 900 python perf/rfd4/tail512/tail.py --ns 512,384 --legs "dm=15;abl=91"
"""
import argparse, hashlib, json, statistics, sys, time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="512,384")
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--legs", default="")
ap.add_argument("--rounds", type=int, default=2, help="base and every leg alternated this many times")
ap.add_argument("--out")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, mm_generic as _mm_generic
from tt_bio.rfd4.trunk import compute_kernel_config

dev = T.get_device()
ckc = _mm_generic.ckc_args(T.trunk_compute_kernel_config(compute_kernel_config()))
grid = tuple(T.COMPUTE_GRID_MAIN)
up = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                                 device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
KNOB = {"abl": "T2_ABL", "dm": "T2_DMABL", "grow": "T2_GROW", "s1k": "T2_S1K", "sbig": "T2_SBIG", "sstr": "T2_SSTR", "lnblk": "T2_LNBLK", "wd": "T2_WDEPTH", "dmcfg": "T2_DMCFG", "xnoc": "T2_XNOC", "wstage": "T2_WSTAGE"}
MAIN = {"T2_ABL": 0, "T2_DMABL": 0, "T2_GROW": True, "T2_S1K": True, "T2_SBIG": 0, "T2_SSTR": 0, "T2_LNBLK": 1, "T2_WDEPTH": 2, "T2_DMCFG": 0}
BLOCK, MCAST = dict(TT.T2_BLOCK), set(TT.T2_MCAST)


def aiclk():
    import glob
    return {f.split("/")[-2]: open(f).read().strip() for f in sorted(glob.glob("/sys/class/tenstorrent/*/tt_aiclk"))}


def emit(rec):
    line = json.dumps(rec)
    print(line, flush=True)
    if A.out:
        with open(A.out, "a") as f:
            f.write(line + "\n")


def setk(leg):
    """main's knobs, then the leg's; returns whether x is read rotated."""
    for k, v in MAIN.items():
        setattr(TT, k, v)
    TT.T2_BLOCK.clear(); TT.T2_BLOCK.update(BLOCK)
    TT.T2_MCAST = set(MCAST)
    rot = True
    for kv in filter(None, leg.split(",")):
        k, v = kv.split("=")
        if k == "rot":
            rot = bool(int(v))
        elif k == "mcast":    # 0: every core reads Wp, gamma, beta itself
            TT.T2_MCAST = MCAST if int(v) else set()
        elif k == "sbw":    # the p GEMM's output subblock width at RFD4's (8, 12)
            TT.T2_BLOCK[(8, 12)] = BLOCK[(8, 12)][:4] + (int(v),)
        else:
            setattr(TT, KNOB[k], int(v))
    return rot


def rotate(xc, n):
    """[1, K, n, n] -> channel c's tile (it, jt) at block slot (it * St + jt + c) % St^2, as bmm_res(rot=) writes x."""
    K, St = xc.shape[1], n // 32
    b = xc.reshape(K, St, 32, St, 32).permute(0, 1, 3, 2, 4).reshape(K, St * St, 32, 32)
    idx = (torch.arange(St * St)[None, :] - torch.arange(K)[:, None]) % (St * St)
    b = b.gather(1, idx[:, :, None, None].expand(K, St * St, 32, 32))
    return b.reshape(K, St, St, 32, 32).permute(0, 1, 3, 2, 4).reshape(1, K, n, n)


def timed(f, reps):
    f(); ttnn.synchronize_device(dev)
    ms, clk = [], []
    for i in range(reps):
        t0 = time.perf_counter()
        f(); ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3)
        if i % 5 == 0:
            clk.append(aiclk())
    return round(statistics.median(ms), 4), clk


def measure(n, K=256, N=384):
    from tt_bio.bmm_res import rot_pays
    gen = torch.Generator().manual_seed(n * 7 + K)
    h = {"x": torch.randn(1, n, n, K, generator=gen) * 2 + torch.randn(1, n, n, 1, generator=gen),
         "g": torch.randn(1, n, n, N, generator=gen), "z": torch.randn(1, n, n, N, generator=gen),
         "w": torch.randn(K, N, generator=gen) / K ** 0.5,
         "gamma": 1 + 0.1 * torch.randn(1, K, generator=gen), "beta": 0.1 * torch.randn(1, K, generator=gen)}
    h = {k: v.to(torch.bfloat16) for k, v in h.items()}
    d = {k: up(h[k]) for k in ("w", "gamma", "beta")}
    xc = h["x"].permute(0, 3, 1, 2).contiguous()
    d["x"] = up(xc)
    d["xr"] = up(rotate(xc, n))
    g = up(h["g"], ttnn.bfloat8_b)
    z0 = ttnn.to_torch(up(h["z"], ttnn.bfloat8_b)).reshape(h["z"].shape).double()
    g64 = ttnn.to_torch(g).reshape(h["g"].shape).double()
    x64 = {k: v.double() for k, v in h.items()}
    mu = x64["x"].mean(-1, keepdim=True)
    var = (x64["x"] - mu).pow(2).mean(-1, keepdim=True)
    xn = (x64["x"] - mu) / torch.sqrt(var + 1e-5) * x64["gamma"] + x64["beta"]
    ref = z0 + (xn @ x64["w"]) * torch.sigmoid(g64)
    TT.T2_HIFI2 = True
    legs = [""] + [l for l in A.legs.split(";") if l]
    for rnd in range(A.rounds):
        for leg in legs:
            rot = setk(leg) and rot_pays(dev, n)
            run = lambda z: TT.fused_tail_ln_cm(d["xr" if rot else "x"], d["gamma"], d["beta"], d["w"], g, z,  # noqa
                                                ckc, grid, stream16=True, rot=rot)
            rec = {"n": n, "leg": leg or "base", "round": rnd, "rot": rot}
            z = up(h["z"], ttnn.bfloat8_b)
            y = run(z)
            if y is None:
                rec["declined"] = {":".join(k): c for k, c in TT.REJECTS.items()}
                emit(rec); ttnn.deallocate(z); continue
            out = ttnn.to_torch(y).reshape(ref.shape)
            rec["sha"] = hashlib.sha256(out.to(torch.bfloat16).view(torch.int16).numpy().tobytes()).hexdigest()[:16]
            e = out.double() - ref
            rec["upd_rel_rms_f64"] = float(e.pow(2).mean().sqrt() / (ref - z0).pow(2).mean().sqrt())
            rec["ms"], rec["aiclk"] = timed(lambda: run(z), A.reps)
            emit(rec)
            ttnn.deallocate(z)
    setk("")


TT.T2 = TT.T2CM = True
emit({"start": True, "ckc": [str(c) for c in ckc], "grid": grid, "aiclk": aiclk()})
for n in [int(s) for s in A.ns.split(",") if s]:
    assert TT.t2cm_stream16_fits(n, grid[0] * grid[1]), n
    with T.lever_scope(frozenset({"pair_tail_cm", "pair_bf16z", "pair_bfp8z", "pair_tail", "trimul_t1g8",
                                  "trimul_tail_hifi2"})):
        measure(n)
