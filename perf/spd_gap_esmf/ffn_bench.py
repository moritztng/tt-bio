#!/usr/bin/env python3
"""ESMFold2's pair transition (`SwiGLUFFN.residual`, x + fc2(silu(x fc1a) * (x fc1b))) at a fold's shape, arms interleaved.

Census (c1, p150a, HiFi4): 11.5 ms per call at 512 aa, c_z 256, hidden 1024, 6.07 s of a 24.4 s device fold. The
roof is ~2 ms (412 GFLOP at HiFi4, 256 MiB of pair in + out), so this bench asks which path gets closest: the
row-blocked default, the pair Transition's block-sharded swiglu (`transition_shard`) and its fused fc12g kernel
(`swiglu_fc12g`, needs silu_f32). An arm is a lever set with optional `@rows=R` (row-block height) and `@bw=B`
(fc1's in0_block_w, pinned to 1 only because larger K blocks differ from it by one bf16 ulp). Time per call is the back-to-back slope (`--calls` calls, one
synchronize), warm program cache, arms round-robin per rep. Every arm records its latch deltas, so an arm that
silently declined to the default reads as such, and its error against a float64 reference of the same math.

    python perf/spd_gap_esmf/ffn_bench.py OUT --arms normal,normal@bw=8,normal@rows=64,normal+transition_shard+swiglu_fc12g
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--cz", type=int, default=256)
ap.add_argument("--hidden", type=int, default=1024)
ap.add_argument("--arms", default="normal,normal+transition_shard,normal+transition_shard+swiglu_fc12g")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--calls", type=int, default=4)
A = ap.parse_args()
OUT = Path(A.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "bench.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); s = json.dumps(kw, default=str)
    LOG.write(s + "\n"); LOG.flush(); print(s[:600], flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()  # a lone p300 chip is a CUSTOM fabric topology; set before ttnn opens it
import torch
import ttnn
import tt_bio.tenstorrent as T
import tt_bio.esmc as E

dev = T.get_device()
node = next((int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
             if os.path.realpath(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")), -1)


def aiclk():
    try:
        return int(Path(f"/sys/class/tenstorrent/tenstorrent!{node}/tt_aiclk").read_text().split()[0])
    except Exception:
        return None


g = torch.Generator().manual_seed(0)
sd = {"0.weight": 1 + 0.1 * torch.randn(A.cz, generator=g), "0.bias": 0.1 * torch.randn(A.cz, generator=g),
      "1.weight": torch.randn(2 * A.hidden, A.cz, generator=g) / A.cz ** 0.5,
      "3.weight": torch.randn(A.cz, A.hidden, generator=g) / A.hidden ** 0.5}
sd = {k: v.to(torch.bfloat16).float() for k, v in sd.items()}  # the device holds bf16; the reference uses those values
x_host = torch.randn(1, A.n, A.n, A.cz, generator=g).to(torch.bfloat16)


def reference(x):
    x = x.double(); w = {k: v.double() for k, v in sd.items()}
    h = torch.nn.functional.layer_norm(x, (A.cz,), w["0.weight"], w["0.bias"], 1e-5) @ w["1.weight"].t()
    a, b = h.chunk(2, dim=-1)
    return x + (torch.nn.functional.silu(a) * b) @ w["3.weight"].t()


ref = reference(x_host)
base = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
def parse(arm):
    spec, *kv = arm.split("@")
    return spec, tuple(sorted((k, int(v)) for k, v in (p.split("=") for p in kv)))


arms = [parse(a) for a in A.arms.split(",")]
mods = {}
for spec, rows in arms:  # build each arm's module under its own set, as the fold does (trunk config reads levers)
    with T.lever_scope(spec):
        ck = T.trunk_compute_kernel_config(base(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                fp32_dest_acc_en=True, packer_l1_acc=True))
        mods[(spec, rows)] = E.SwiGLUFFN(T.WeightScope(sd), ck, fuse_swiglu=True)
x = ttnn.from_torch(x_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
log(ev="open", node=node, arch=str(dev.arch()), grid=tuple(T.COMPUTE_GRID_MAIN), args=vars(A))


def call(spec, opts):
    o = dict(opts)
    prev_rows = E.set_pair_ffn_row_block(o["rows"]) if "rows" in o else None
    prev_bw, E._PAIR_FFN_FC1_BW = E._PAIR_FFN_FC1_BW, o.get("bw", E._PAIR_FFN_FC1_BW)
    try:
        with T.lever_scope(spec):
            return mods[(spec, opts)].residual(x)
    finally:
        E._PAIR_FFN_FC1_BW = prev_bw
        if prev_rows is not None:
            E.set_pair_ffn_row_block(prev_rows)


for spec, rows in arms:  # warm + accuracy + what fired
    before = {k: dict(v) for k, v in T.LATCH_STATS.items()}
    y = call(spec, rows); ttnn.deallocate(call(spec, rows))
    ttnn.synchronize_device(dev)
    yt = ttnn.to_torch(y).double()[..., :A.cz].reshape(ref.shape); ttnn.deallocate(y)
    err = yt - ref
    fired = {k: {f: v.get(f, 0) - before.get(k, {}).get(f, 0) for f in ("served", "refused", "declined")}
             for k, v in T.LATCH_STATS.items() if v != before.get(k)}
    log(ev="accuracy", arm=spec, opts=rows, finite=bool(torch.isfinite(yt).all()),
        rel_rms=float(err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()), max_abs=float(err.abs().max()),
        fired=fired, split=list(E.SPLIT_STATS), fused_resid=list(E.FUSED_RESID_STATS),
        shrunk={str(k): v for k, v in E._ROW_BLOCK_SHRUNK.items()})

times = {a: [] for a in arms}
for rep in range(A.reps):
    for arm in arms:
        ttnn.synchronize_device(dev); c0 = aiclk(); t0 = time.perf_counter()
        for _ in range(A.calls):
            ttnn.deallocate(call(*arm))
        ttnn.synchronize_device(dev); ms = (time.perf_counter() - t0) / A.calls * 1e3
        times[arm].append(ms)
        log(ev="rep", rep=rep, arm=arm[0], opts=arm[1], ms=round(ms, 4), aiclk=[c0, aiclk()])
summary = {s + "".join(f"@{k}={v}" for k, v in r): {"median_ms": statistics.median(v), "spread_ms": max(v) - min(v)} for (s, r), v in times.items()}
log(ev="summary", **summary)
(OUT / "result.json").write_text(json.dumps(summary, indent=1))
T.cleanup()
