#!/usr/bin/env python3
"""The size ladder crossed with MSA depth, one process per lever arm.

Step 2 of the root-cause pass. The three levers the pass-9 ladder exported are worth 2.107x
of a 512 aa fold at the shipped default, and `one-size-tuning-is-a-standing-defect-class`
forbids recommending a default flip screened at one size. The OPM lever is also depth-keyed
(`OPM_SMALL_DEPTH_MAX = 8`), so the screen needs both axes.

The levers are read at import time, so an arm is a process. Sizes and depths are cheap to
sweep inside one process because the checkpoint load dominates: load once, rebuild only the
host inputs per cell. Every cell writes its own line to the JSONL immediately, so a cell that
OOMs does not cost the cells before it.

`trunk_s_per_recycle` is a clean wall with no per-op syncs: a warm run, then two timed runs with
no accumulator attached. That is the number the arms are compared on, because the attributed
variant in msa_decompose.py pays one sync per wrapped op and the fused triangle attention
removes ops, so sync overhead is not constant across the arms.
"""
from __future__ import annotations

import argparse
import enum
import gc
import json
import os
import sys
import time
from pathlib import Path

if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self):
            return str(self.value)
    enum.StrEnum = StrEnum

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="128,256,512,768,1024")
    ap.add_argument("--depths", default="1,35")
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--n_recycles", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feat_cache", default="/home/ttuser/rf3_perf_work/featcache")
    ap.add_argument("--arm", required=True, help="label only; the levers come from the env")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    depths = [int(x) for x in args.depths.split(",")]

    from perf.rf3.featcache import featurized

    import ttnn
    from tt_bio import tenstorrent as tsr
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.rf3.host import HostInputs
    from tt_bio.tenstorrent import get_device
    from perf.rf3.tt_rf3_bench import net_config

    levers = {k: os.environ.get(k, "") for k in
              ("TT_BIO_TRIATT_FUSED_HIFI", "TT_BIO_RF3_GLN_ROW_FOLD",
               "TT_BIO_OPM_SMALL_DEPTH", "TT_BIO_OPM_SMALL_DEPTH_MAX")}
    print("[arm %s] levers %s" % (args.arm, levers), flush=True)

    cfg = net_config(args.ckpt)
    device = get_device()
    kcfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    tt = rf3_model.load(
        args.ckpt, kcfg,
        n_pairformer_blocks=cfg["recycler"]["n_pairformer_blocks"],
        n_msa_blocks=cfg["recycler"]["msa_module"]["n_block"],
        n_dit_blocks=cfg["diffusion_module"]["diffusion_transformer"]["n_block"],
        num_timesteps=50, with_confidence=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w")
    for aa in sizes:
        for depth in depths:
            stem = "rf3_%d" % aa if depth == 1 else "rf3_%d_msa35" % aa
            inp = str(REPO / ("perf/rf3/inputs/%s.json" % stem))
            tag = "%daa_d%d_%s" % (aa, depth, args.arm)
            host = None
            try:
                fo = featurized(inp, n_recycles=max(args.n_recycles, 2),
                                diffusion_batch_size=1, seed=args.seed,
                                cache_dir=args.feat_cache or None)
                f = fo["feats"]
                host = HostInputs.build(f, device)
                s_inputs, s_init, z_init = tt.feature_initializer(
                    host.single_in, host.pair_in, host.pair_v, host.keys_indexing,
                    host.atom_to_token_mean, host.window_mask, host.n_atom_padded,
                    host.token_feats, host.relpos_feat, host.bond_feat)
                tmpl = tt.recycler.template_embedder.embed_template_feats(host.template_feats)

                def run():
                    s = ttnn.mul(s_init, 0.0)
                    z = ttnn.mul(z_init, 0.0)
                    ttnn.synchronize_device(device)
                    t0 = time.perf_counter()
                    for i in range(args.n_recycles):
                        s, z = tt.recycler(host, tmpl,
                                           host.msa_stack[i % len(host.msa_stack)],
                                           s_inputs, s_init, z_init, s, z)
                    ttnn.synchronize_device(device)
                    return time.perf_counter() - t0

                run()
                tsr.TRIATT_FUSED_HIFI_STATS.update(served=0, declined=0, too_short=0)
                tsr.OPM_SMALL_DEPTH_STATS[0] = 0
                tsr.OPM_SMALL_DEPTH_STATS[1] = 0
                walls = [run() for _ in range(2)]
                rec = {
                    "tag": tag, "arm": args.arm, "aa": aa, "depth_req": depth,
                    "input": inp, "n_token": int(host.n_token),
                    "msa_feat_shape": [int(x) for x in host.msa_stack[0].shape],
                    "n_recycles": args.n_recycles,
                    "trunk_s_per_recycle": min(walls) / args.n_recycles,
                    "trunk_walls_s": walls,
                    "triatt_stats": dict(tsr.TRIATT_FUSED_HIFI_STATS),
                    "opm_stats": list(tsr.OPM_SMALL_DEPTH_STATS),
                    "levers": levers,
                }
            except Exception as e:  # a cell that dies must not cost the cells before it
                rec = {"tag": tag, "arm": args.arm, "aa": aa, "depth_req": depth,
                       "input": inp, "error": "%s: %s" % (type(e).__name__, e)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if "error" in rec:
                print("  %-24s FAILED %s" % (tag, rec["error"]), flush=True)
            else:
                print("  %-24s %8.4f s/recycle  msa %s  triatt %s  opm %s"
                      % (tag, rec["trunk_s_per_recycle"], rec["msa_feat_shape"],
                         rec["triatt_stats"], rec["opm_stats"]), flush=True)
            host = fo = f = None
            gc.collect()
    fh.close()
    print("wrote %s" % out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
