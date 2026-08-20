#!/usr/bin/env python3
"""Why triangle attention's ENDING variant costs more than its starting one, and how much.

The pass-7 trunk decomposition at three rungs reads `tri_att_end` at 1.23x `tri_att_start` at
512 aa, 1.54x at 768 and 1.47x at 1024 -- the only component whose share of the trunk grows
with N. The ending variant is the same module with `ending=True`, and the only things that
adds are two `_pair_transpose` calls on the full [S, S, c_z] pair tensor (one on the input,
one on the output) plus, when `transpose_bias=False`, one permute of the [1, H, S, S] bias.

This times those three against the whole call, on the real module off the real checkpoint, one
process, syncs around each piece. Attribution only -- the syncs inflate the total the same way
`trunk_decompose.py` documents.
"""
from __future__ import annotations

import argparse
import enum
import json
import statistics
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
    ap.add_argument("--aa", type=int, default=768)
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tt_bio.rf3.featurize import featurize
    fo = featurize(str(REPO / f"perf/rf3/inputs/rf3_{args.aa}.json"),
                   n_recycles=2, diffusion_batch_size=1, seed=args.seed)[0]
    f = fo["feats"]

    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.rf3.host import HostInputs
    from perf.rf3.tt_rf3_bench import net_config

    cfg = net_config(args.ckpt)
    device = T.get_device()
    kcfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    tt = rf3_model.load(
        args.ckpt, kcfg,
        n_pairformer_blocks=cfg["recycler"]["n_pairformer_blocks"],
        n_msa_blocks=cfg["recycler"]["msa_module"]["n_block"],
        n_dit_blocks=cfg["diffusion_module"]["diffusion_transformer"]["n_block"],
        num_timesteps=50, with_confidence=False)

    host = HostInputs.build(f, device)
    _s_inputs, _s_init, z_init = tt.feature_initializer(
        host.single_in, host.pair_in, host.pair_v, host.keys_indexing,
        host.atom_to_token_mean, host.window_mask, host.n_atom_padded,
        host.token_feats, host.relpos_feat, host.bond_feat)
    z = ttnn.clone(z_init)
    print("z", tuple(z.shape), z.dtype, flush=True)

    marks: dict[str, list] = {}

    def mark(name, dt, shape=None, mc=None):
        marks.setdefault(name, []).append(dt)
        if shape is not None:
            marks.setdefault(name + ".shape", []).append(shape)
        if mc is not None:
            marks.setdefault(name + ".dest", []).append(mc)

    orig_pt = T._pair_transpose
    orig_permute = ttnn.permute

    stage = {"n": 0}

    def timed_pair_transpose(t, memory_config):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        o = orig_pt(t, memory_config)
        ttnn.synchronize_device(device)
        stage["n"] += 1
        mark(f"pair_transpose.{stage['n']}", time.perf_counter() - t0,
             tuple(int(d) for d in t.shape), str(memory_config.buffer_type).split(".")[-1])
        return o

    def timed_permute(t, dims, **kw):
        # Only the 4-D bias permute inside triangle attention is of interest; the pair
        # transpose reaches ttnn.permute through _pair_transpose, which is timed above.
        if len(dims) == 4 and tuple(dims) == (0, 1, 3, 2):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            o = orig_permute(t, dims, **kw)
            ttnn.synchronize_device(device)
            mark("bias_permute", time.perf_counter() - t0, tuple(int(d) for d in t.shape))
            return o
        return orig_permute(t, dims, **kw)

    blk = tt.recycler.pairformer.blocks[0]
    rows = []
    for which in ("start", "end"):
        mod = getattr(blk, f"triangle_attention_{which}")
        # Warm: first call compiles kernels and picks program configs.
        out = mod(z)
        ttnn.deallocate(out)
        ttnn.synchronize_device(device)
        marks.clear()
        T._pair_transpose = timed_pair_transpose
        ttnn.permute = timed_permute
        totals = []
        for _ in range(args.reps):
            stage["n"] = 0
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            out = mod(z)
            ttnn.synchronize_device(device)
            totals.append(time.perf_counter() - t0)
            ttnn.deallocate(out)
        T._pair_transpose = orig_pt
        ttnn.permute = orig_permute
        row = {"which": which, "total_ms": statistics.median(totals) * 1e3,
               "totals_ms": [round(t * 1e3, 3) for t in totals]}
        for k, v in sorted(marks.items()):
            if k.endswith(".shape") or k.endswith(".dest"):
                row[k] = v[0]
            else:
                row[k + "_ms"] = round(statistics.median(v) * 1e3, 3)
        rows.append(row)
        print(json.dumps(row, indent=1), flush=True)

    st, en = rows[0], rows[1]
    named = sum(v for k, v in en.items()
                if k.endswith("_ms") and k not in ("total_ms", "totals_ms"))
    rep = {"aa": args.aa, "tokens": int(z.shape[1]), "reps": args.reps, "rows": rows,
           "excess_end_over_start_ms": en["total_ms"] - st["total_ms"],
           "transpose_and_bias_ms": named,
           "explained_frac": named / (en["total_ms"] - st["total_ms"])}
    print(f"\n{args.aa} aa: start {st['total_ms']:.3f} ms, end {en['total_ms']:.3f} ms, "
          f"excess {rep['excess_end_over_start_ms']:.3f} ms, of which the transposes and the "
          f"bias permute are {named:.3f} ms ({rep['explained_frac'] * 100:.1f} %)")
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
