"""RFD4's triangle multiplications per arm: time per call, error against a float64 reference of the same
module (layer 0 weights, a layer-normed random pair, full mask), bits against the first arm, and which
channel-loop branch served the call.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/tri2/bench.py --weights <bundle> --out <json> \
        [--n 128] [--arms "default|lever:trimul_gin;set:tenstorrent.TRIMUL_GIN_L1=True"]

An arm is `default` or `;`-joined parts: `lever:a+b` (a lever set), `set:mod.ATTR=literal+mod.B=1`
(tt_bio module globals, restored after the arm), `mask:none` (no pair mask; the reference is unmasked
either way). Arms are separated by `|`.
"""
import argparse
import ast
import importlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402


def reference(z, w, ending):
    """openfold TriangleMultiplication{Outgoing,Incoming}, bias-free, full mask, float64."""
    ln = lambda x, g: torch.nn.functional.layer_norm(x, x.shape[-1:], g)  # noqa: E731
    zn = ln(z, w["layer_norm_in.weight"])
    a = torch.sigmoid(zn @ w["linear_a_g.weight"].t()) * (zn @ w["linear_a_p.weight"].t())
    b = torch.sigmoid(zn @ w["linear_b_g.weight"].t()) * (zn @ w["linear_b_p.weight"].t())
    x = torch.einsum("bkic,bkjc->bijc", a, b) if ending else torch.einsum("bikc,bjkc->bijc", a, b)
    x = ln(x, w["layer_norm_out.weight"]) @ w["linear_z.weight"].t()
    return torch.sigmoid(zn @ w["linear_g.weight"].t()) * x


def parse_arm(arm):
    lev, saved, mask = "", [], True
    for part in arm.split(";"):
        if part == "mask:none":
            mask = False
        elif part.startswith("lever:"):
            lev = part[6:].replace("+", ",")
        elif part.startswith("set:"):
            for kv in part[4:].split("+"):
                k, v = kv.split("=", 1)
                modname, attr = k.rsplit(".", 1)
                tgt = importlib.import_module(f"tt_bio.{modname}")
                saved.append((tgt, attr, getattr(tgt, attr)))
                setattr(tgt, attr, ast.literal_eval(v))
    return lev, saved, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--arms", default="default")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--layer", type=int, default=0)
    a = ap.parse_args()

    import ttnn
    from safetensors import safe_open

    from tt_bio import tenstorrent as T
    from tt_bio import trimul_tail as TT
    from tt_bio.rfd4.trunk import _trimul_weights, compute_kernel_config

    f = safe_open(str(Path(a.weights) / "model.safetensors"), "pt")
    pre = f"nn.pair_update_layers.{a.layer}."
    state = {k[len(pre):]: f.get_tensor(k).float() for k in f.keys() if k.startswith(pre)}
    dev = T.get_device()
    n = a.n
    z = torch.nn.functional.layer_norm(torch.randn(1, n, n, 384, generator=torch.Generator().manual_seed(0),
                                                   dtype=torch.float64), (384,)).to(torch.bfloat16)
    zt = ttnn.from_torch(z.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    mt = ttnn.from_torch(torch.ones(1, n, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    rows = []
    with clocksample.during() as clk:
        for ending, name in ((False, "tri_mult_out"), (True, "tri_mult_in")):
            w64 = {k[len(name) + 1:]: v.double() for k, v in state.items() if k.startswith(name + ".")}
            ref = reference(z.double(), w64, ending)
            base_out = None
            for arm in a.arms.split("|"):
                lev, saved, use_mask = parse_arm(arm)
                try:
                    with T.lever_scope(lev):
                        tri = T.TriangleMultiplication(ending, _trimul_weights(T.WeightScope(
                            state).child(name)), compute_kernel_config())
                        T.TRIMUL_MM_TRANSPOSE_STATS.clear()
                        g0 = list(TT.GIN_MOVE_STATS)
                        out = tri(zt, mt if use_mask else None)
                        got = ttnn.to_torch(out).double().reshape(ref.shape)
                        ttnn.deallocate(out)
                        branch = dict(T.TRIMUL_MM_TRANSPOSE_STATS)
                        gin = [x - y for x, y in zip(TT.GIN_MOVE_STATS, g0)]
                        ts = []
                        for _ in range(a.reps + 3):
                            ttnn.synchronize_device(dev)
                            t0 = time.perf_counter()
                            o = tri(zt, mt if use_mask else None)
                            ttnn.synchronize_device(dev)
                            ts.append(time.perf_counter() - t0)
                            ttnn.deallocate(o)
                finally:
                    for tgt, attr, v in saved:
                        setattr(tgt, attr, v)
                ts = sorted(ts[3:])
                if base_out is None:
                    base_out = got
                r = dict(n=n, module=name, arm=arm, ms=round(1e3 * ts[len(ts) // 2], 4),
                         p10_ms=round(1e3 * ts[len(ts) // 10], 4),
                         rel_rms_vs_f64=float(((got - ref).pow(2).mean() / ref.pow(2).mean()).sqrt()),
                         max_abs_vs_f64=float((got - ref).abs().max()),
                         max_abs_vs_first=float((got - base_out).abs().max()),
                         branch={f"{k[0]}/{k[1]}": v for k, v in branch.items()}, gin_move=gin,
                         t=time.strftime("%H:%M:%S"))
                rows.append(r)
                print(json.dumps(r), flush=True)
                del tri
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(rows=rows, aiclk=clk.summary(), script="perf/rfd4/tri2/bench.py"),
                                      indent=1))
    print(clk.line())


if __name__ == "__main__":
    main()
