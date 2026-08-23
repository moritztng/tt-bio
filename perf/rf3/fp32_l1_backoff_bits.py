#!/usr/bin/env python3
"""Is the L1-sharded fp32 softmax tail bit-identical to the interleaved one at 1024 aa?

The shard is a memory config and nothing else -- same ops, same dtypes, same reduction axis -- so
it should be. It has to be checked at 1024 aa specifically, and inside a real trunk, because that
is the only size where the sharded softmax refuses a 3-row block and the row-cap backoff picks a
2-row one instead: an isolated op cannot reproduce the refusal, since the term that causes it is
the rest of the model's L1 residency.

Three arms in one process on the same device tensors, so the comparison is per-element and the A/A
arm bounds the device's own repeatability:

  int_a   interleaved tail everywhere (per-core budget 0)
  int_b   the same again -- A/A control
  l1      the shipped budget, which refuses once at 3 rows and then shards 2 rows
"""
from __future__ import annotations

import argparse
import enum
import json
import sys
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
    ap.add_argument("--aa", type=int, default=1024)
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tt_bio.rf3.featurize import featurize
    fo = featurize(str(REPO / f"perf/rf3/inputs/rf3_{args.aa}.json"),
                   n_recycles=2, diffusion_batch_size=1, seed=args.seed)[0]
    f = fo["feats"]

    import ttnn
    from tt_bio import tenstorrent as T
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
    s_inputs, s_init, z_init = tt.feature_initializer(
        host.single_in, host.pair_in, host.pair_v, host.keys_indexing,
        host.atom_to_token_mean, host.window_mask, host.n_atom_padded,
        host.token_feats, host.relpos_feat, host.bond_feat)
    tmpl = tt.recycler.template_embedder.embed_template_feats(host.template_feats)

    shipped = T._FP32_SOFTMAX_L1_BYTES_PER_CORE
    res, out = {}, {}
    for arm, budget in (("int_a", 0), ("int_b", 0), ("l1", shipped)):
        T._FP32_SOFTMAX_L1_BYTES_PER_CORE = budget
        T._FP32_SOFTMAX_L1_ROW_CAP.clear()
        T._FP32_SOFTMAX_L1_FREE_ROW_CAP.clear()
        T.FP32_SOFTMAX_STATS.update({k: 0 for k in T.FP32_SOFTMAX_STATS})
        s = ttnn.mul(s_init, 0.0)
        z = ttnn.mul(z_init, 0.0)
        s, z = tt.recycler(host, tmpl, host.msa_stack[0], s_inputs, s_init, z_init, s, z)
        out[arm] = (ttnn.to_torch(s), ttnn.to_torch(z))
        ttnn.deallocate(s)
        ttnn.deallocate(z)
        res[f"{arm}_stats"] = dict(T.FP32_SOFTMAX_STATS)
        res[f"{arm}_row_caps"] = {str(k): v for k, v in T._FP32_SOFTMAX_L1_ROW_CAP.items()}
        print(f"[{arm}] budget={budget} {res[f'{arm}_stats']} caps={res[f'{arm}_row_caps']}",
              flush=True)
    T._FP32_SOFTMAX_L1_BYTES_PER_CORE = shipped

    for pair in (("int_a", "int_b"), ("int_a", "l1")):
        a, b = out[pair[0]], out[pair[1]]
        tag = f"{pair[0]}_vs_{pair[1]}"
        res[f"{tag}_s_equal"] = bool(torch.equal(a[0], b[0]))
        res[f"{tag}_z_equal"] = bool(torch.equal(a[1], b[1]))
        res[f"{tag}_s_maxabs"] = (a[0].float() - b[0].float()).abs().max().item()
        res[f"{tag}_z_maxabs"] = (a[1].float() - b[1].float()).abs().max().item()
        print(f"{tag}: s equal={res[f'{tag}_s_equal']} maxabs={res[f'{tag}_s_maxabs']:.6g}  "
              f"z equal={res[f'{tag}_z_equal']} maxabs={res[f'{tag}_z_maxabs']:.6g}", flush=True)

    res["aa"] = args.aa
    print("RESULT " + json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
