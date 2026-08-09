#!/usr/bin/env python3
"""Per-op device time + shapes for ONE `Trunk._msa` and ONE `Trunk._template` call, at 298 aa.

The Pairformer stack has a per-op ledger (perf/ledger_298/); these two trunk stages have none. This
is the same instrument pointed at them: `perf/ledger_298/pf_block_ops.py`'s wrapper lets the model's
own call through untouched, then re-runs that exact call N more times inside one synchronised region,
so a per-call figure is device time with the host dispatch amortised.

The stage methods are the production ones -- `Trunk._msa` and `Trunk._template` are called as bound
methods on a `Trunk` built with only the MSA and template submodules, so the code path, the chunking
decisions and the weights are the fold's. The 48-block Pairformer is deliberately not built (it is
1.4 GB of device weights this rig never calls).

Shapes are the ones the fold runs, taken from a live 298 aa fold's feature dict:
N=298 tokens (pads to 320 on the tile grid), c_z=256, 4 templates at c=64, MSA depth 35 at c_m=128.

The unit here is a STAGE call, not a block: `_msa` runs 4 MSA blocks and `_template` runs 2 pair
blocks per template. One stage call happens per recycling cycle, so ms/fold = stage_ms x 10.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:... PYTHONPATH=$PWD \
      python3 perf/msa_template/mt_stage_ops.py --outdir perf/msa_template
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tt_bio.tenstorrent import (get_device, CORE_GRID_MAIN, OuterProductMean,  # noqa: E402
                                PairWeightedAveraging, PairformerLayer, Transition)
from tt_bio import protenix_weights as PW                                       # noqa: E402
from tt_bio.protenix import Trunk                                               # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "pf_block_ops", REPO / "perf" / "ledger_298" / "pf_block_ops.py")
PB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PB)


def load_sd(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ck = ck.get("model", ck)
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}


def build_trunk(sd, ckc, c_z=256):
    """A Trunk carrying only the MSA and template submodules, built exactly as Trunk.__init__ does."""
    t = Trunk.__new__(Trunk)
    t._w, t.compute_kernel_config, t.dev, t.C_Z = sd, ckc, get_device(), c_z
    t._wc, t._msa_update_first = {}, False
    n_tri_heads = c_z // Trunk.TRI_HEAD_DIM

    def sub(pp):
        return {k[len(pp):]: v for k, v in sd.items() if k.startswith(pp)}

    t.TPL = [PairformerLayer(32, 2, None, None, False,
                             PW.remap_msa_pair_stack(sub(f"template_embedder.pairformer_stack.blocks.{b}.")),
                             ckc) for b in range(2)]
    t.MSA = []
    for i in range(4):
        P = f"msa_module.blocks.{i}."
        opm = OuterProductMean(PW.remap_outer_product_mean(sub(P + "outer_product_mean_msa.")), ckc)
        pl = PairformerLayer(Trunk.TRI_HEAD_DIM, n_tri_heads, None, None, False,
                             PW.remap_msa_pair_stack(sub(P + "pair_stack.")), ckc)
        pwa = PairWeightedAveraging(8, 8, PW.remap_pair_weighted_averaging(
            sub(P + "msa_stack.msa_pair_weighted_averaging.")), ckc)
        tm = Transition(PW.remap_transition(sub(P + "msa_stack.transition_m.")), ckc)
        t.MSA.append((opm, pwa, tm, pl))
    t.PF = None
    return t


def stage_wall(fn, warm, reps, dev):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        o.append(time.perf_counter() - t0)
    return st.median(o), o


def record_stage(fn, name, n, c_z, wall_s, out_path, reps, small_us, dev):
    PB.RECORDS.clear()
    PB.STATE.update(dev=dev, reps=reps, small_us=small_us, on=False, idx=0)
    saved = PB.patch()
    PB.STATE["on"] = True
    fatal = None
    try:
        fn()
        ttnn.synchronize_device(dev)
    except Exception as e:                                       # noqa: BLE001
        import traceback
        fatal = f"{type(e).__name__}: {e}"[:400]
        traceback.print_exc()
    PB.STATE["on"] = False
    for ns, nm, f in saved:
        setattr(ns, nm, f)
    recs = list(PB.RECORDS)
    tot = sum(r["s"] for r in recs)
    print(f"[{name}] ops={len(recs)} sum={tot*1e3:.3f} ms stage_wall={wall_s*1e3:.3f} ms "
          f"coverage={100*tot/wall_s:.1f}%", flush=True)
    json.dump({"model": "protenix-v2", "n": n, "c_z": c_z, "block_wall_s": wall_s,
               "reps": reps, "n_ops": len(recs), "sum_s": tot, "fatal": fatal,
               "stage": name, "records": recs}, open(out_path, "w"), indent=1)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path.home() / ".boltz" / "protenix-v2.pt"))
    ap.add_argument("--n", type=int, default=298)
    ap.add_argument("--depth", type=int, default=35)
    ap.add_argument("--nt", type=int, default=4)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--wall-reps", type=int, default=5)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--small-us", type=float, default=60.0)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    sd = load_sd(args.ckpt)
    c_z = sd["template_embedder.linear_no_bias_z.weight"].shape[1]
    c_m = sd["msa_module.linear_no_bias_m.weight"].shape[0]
    N, D, nt = args.n, args.depth, args.nt
    print(f"N={N} depth={D} nt={nt} c_z={c_z} c_m={c_m} grid={CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}",
          flush=True)

    t = build_trunk(sd, ckc, c_z)
    torch.manual_seed(0)
    z3 = ttnn.from_torch(torch.randn(1, N, N, c_z) * 0.1, layout=ttnn.TILE_LAYOUT,
                         device=dev, dtype=ttnn.bfloat16)
    m_feat = ttnn.from_torch(torch.randn(1, D, N, c_m) * 0.1, layout=ttnn.TILE_LAYOUT,
                             device=dev, dtype=ttnn.bfloat16)
    # tpl_a: the per-template projection Trunk.__call__ hoists out of the recycling loop.
    tpl_a = [t._lin(t._up(torch.randn(1, N, N, 108) * 0.1),
                    "template_embedder.linear_no_bias_a.weight") for _ in range(nt)]

    res = {}
    for name, call in (("trunk_template", lambda: t._template(z3, tpl_a, N, nt)),
                       ("trunk_msa", lambda: t._msa(z3, m_feat))):
        def once(call=call):
            out = call()
            if isinstance(out, list):
                for o in out:
                    ttnn.deallocate(o)
            elif out is not None and out is not z3:
                ttnn.deallocate(out)
        wall, raw = stage_wall(once, args.warm, args.wall_reps, dev)
        print(f"[{name}] stage wall = {wall*1e3:.3f} ms  (x10 cycles = {wall*1e4:.1f} ms/fold)  "
              f"raw={[round(x*1e3, 2) for x in raw]}", flush=True)
        p = args.outdir / f"ops_{name}_{N}.json"
        record_stage(once, name, N, c_z, wall, p, args.reps, args.small_us, dev)
        res[name] = {"stage_ms": round(wall * 1e3, 3), "ms_per_fold_x10": round(wall * 1e4, 1),
                     "raw_ms": [round(x * 1e3, 3) for x in raw], "ops_json": str(p)}

    (args.outdir / f"stage_walls_{N}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
