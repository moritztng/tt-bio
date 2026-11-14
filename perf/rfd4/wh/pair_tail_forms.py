"""pair_tail's two residual forms on one chip, each graded against float64.

`tests/test_rfd4_pairblock.py::test_pair_tail_residual_in_place` covers only the float32-stream form (T2_RES32:
z float32, unpacked straight to DST). The default trunk carries `pair_bf16z`, where z is its own bf16 copy and the
tail takes the stream16 form instead. This runs both on the same inputs, so a failure can be placed on the form the
designs actually ran or on the one they did not.

    TT_VISIBLE_DEVICES=<chip> python perf/rfd4/wh/pair_tail_forms.py [--n 256]

Per form and direction: rel_rms of the update vs float64; for stream16 also the floor that form cannot beat (the
float64 update added to the bf16 stream and rounded to bf16 once). For res32 a least-squares fit of the update on
(reference, z) says what landed instead of the update.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402

from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio.rfd4 import trunk as TR  # noqa: E402
import rfd4_reference as REF  # noqa: E402


def rel(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


def bf16dev(t, dev):
    return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    dev = T.get_device()
    n = a.n
    w = REF.pair_update_weights(768, 384, 256, seed=5, prefix="pu")
    _, z, _, _ = REF.block_inputs(1, n, 768, 384, 256, seed=1)
    z32 = z.to(torch.float32)
    zb = REF.bf16(z32.double())
    out = {"arch": T.arch_name(), "n": n}
    for outgoing in (True, False):
        name = "tri_mult_out" if outgoing else "tri_mult_in"
        ref = REF.tri_mult({k: v.double() for k, v in w.items()}, f"pu.{name}", zb,
                           torch.ones(1, n, n, dtype=torch.float64), outgoing)
        rec = {"ref_rms": ref.pow(2).mean().sqrt().item(), "z_rms": zb.pow(2).mean().sqrt().item()}
        for form in ("res32", "stream16"):
            levers = {"trimul_t1", "pair_tail"} | ({"pair_bf16z"} if form == "stream16" else set())
            with T.lever_scope(frozenset(levers)):
                mod = TR.PairUpdate(T.WeightScope({k: v.to(torch.float32) for k, v in w.items()}).child("pu"),
                                    TR.compute_kernel_config())
                tri = mod.tri_out if outgoing else mod.tri_in
                tri(bf16dev(zb, dev), None)     # warmup, as the test does
                zbt = bf16dev(zb, dev)
                if form == "res32":
                    st = ttnn.from_torch(z32, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    base = z32.double()
                else:
                    st = zbt
                    base = zb.double()
                got = tri(zbt, None, stream32=st)
                took = bool(tri.took_residual) and got is st
                z_new = ttnn.to_torch(st).double().reshape(ref.shape)
            upd = z_new - base
            r = {"took_residual": took, "rel_rms": rel(upd, ref), "finite": bool(torch.isfinite(z_new).all())}
            if form == "stream16":
                r["bf16_floor_rel_rms"] = rel(REF.bf16(base + ref).double() - base, ref)
            else:
                X = torch.stack([ref.flatten(), base.flatten()], 1)
                coef = torch.linalg.lstsq(X, upd.flatten()[:, None]).solution.flatten()
                r["fit_upd_on_ref_and_z"] = [round(c, 4) for c in coef.tolist()]
                r["rel_rms_z_new_vs_ref_alone"] = rel(z_new, ref)
            rec[form] = r
        out[name] = rec
        print(json.dumps({name: rec}), flush=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
