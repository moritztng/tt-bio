"""GROW's cross-model check: every caller form of the moved T2 reader (T2CM) run with trimul_tail.T2_GROW off and on
in one process, same inputs, output sha256 of each. Equal shas = bit-identical; `grow` says whether the build took
the grouped gather at that key (where it does not fit, both arms compile the same program).
Forms: plain bf16 tail at c 128 (OpenFold3 / Boltz-2), 256 (Protenix-v2), 384 (OpenDDE, trimul_t2w); RFD4's
stream16 tail (K 256, N 384, g and z bfp8); AF2's GATE form (c 128, biases, mask, separate out), whole and with
prev (channels split at 64).

usage: TT_VISIBLE_DEVICES=<chip> timeout 1800 python perf/rfd4/g512/xtree.py --ns 256,1024 --out x.jsonl
"""
import argparse, hashlib, json, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
ap = argparse.ArgumentParser()
ap.add_argument("--ns", default="256,1024")
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
TOOK = []
_fits = TT._t2_grow_fits
TT._t2_grow_fits = lambda *a: TOOK.append(_fits(*a)) or TOOK[-1]


def emit(rec):
    line = json.dumps(rec)
    print(line, flush=True)
    if A.out:
        with open(A.out, "a") as f:
            f.write(line + "\n")


def sha(t):
    return hashlib.sha256(ttnn.to_torch(t).to(torch.bfloat16).view(torch.int16).numpy().tobytes()).hexdigest()[:16]


def case(form, n, K, N, levers, b8=False, af2=False, split=0):
    gen = torch.Generator().manual_seed(n + K + N + split)
    r = lambda *s: torch.randn(*s, generator=gen).to(torch.bfloat16)  # noqa: E731
    xcm = r(1, K, n, n) * 2
    lw, lb, wp = 1 + 0.1 * r(1, K), 0.1 * r(1, K), r(K, N) / K ** 0.5
    d = {"x": up(xcm[:, split:]), "lw": up(lw), "lb": up(lb), "wp": up(wp)}
    kw = {}
    if af2:
        kw = {"wg": up(r(K, N) / K ** 0.5), "bp": TT.bias_rows(up(0.1 * r(1, N))),
              "bg": TT.bias_rows(up(0.1 * r(1, N))), "mask": up((r(1, 1, n, N) > 0).to(torch.bfloat16))}
        if split:
            kw["prev"] = (up(torch.nn.functional.pad(xcm[:, :split].permute(0, 2, 3, 1), (0, K - split))), split)
        g = up(r(1, n, n, K))
    else:
        g = up(r(1, n, n, N), ttnn.bfloat8_b if b8 else ttnn.bfloat16)
    z0 = r(1, n, n, N)
    rec = {"form": form, "n": n, "K": K, "N": N}
    for grow in (False, True):
        TT.T2_GROW = grow
        TOOK.clear()
        z = up(z0, ttnn.bfloat8_b if b8 else ttnn.bfloat16)
        extra = {}
        if af2:
            extra["out"] = ttnn.allocate_tensor_on_device(z.shape, z.dtype, ttnn.TILE_LAYOUT, dev,
                                                          ttnn.DRAM_MEMORY_CONFIG)
        with T.lever_scope(frozenset(levers)):
            y = TT.fused_tail_ln_cm(d["x"], d["lw"], d["lb"], d["wp"], g, z, ckc, grid, stream16=b8,
                                    **kw, **extra)
        if y is None:
            rec["declined"] = {":".join(k): c for k, c in TT.REJECTS.items()}
            break
        rec["grow" if grow else "base"] = sha(y)
        if grow:
            rec["grow_taken"] = any(TOOK)
        ttnn.deallocate(z)
        if af2:
            ttnn.deallocate(extra["out"])
    rec["equal"] = rec.get("grow") is not None and rec.get("grow") == rec.get("base")
    emit(rec)
    TT.T2_GROW = True


TT.T2 = TT.T2CM = True
for n in [int(s) for s in A.ns.split(",") if s]:
    case("plain128", n, 128, 128, ())
    case("plain256", n, 256, 256, ())
    case("plain384_t2w", n, 384, 384, ("trimul_t2w",))
    case("rfd4_stream16", n, 256, 384, ("pair_tail_cm", "pair_bf16z", "pair_bfp8z", "pair_tail", "trimul_t1g8",
                                        "trimul_tail_hifi2"), b8=True)
    case("af2_gate", n, 128, 128, (), af2=True)
    case("af2_gate_prev64", n, 128, 128, (), af2=True, split=64)
