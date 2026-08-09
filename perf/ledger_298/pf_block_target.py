#!/usr/bin/env python3
"""Device-profiler target: ONE real trunk Pairformer block, warmed, for the 298 aa op ledger.

The Pairformer stack is 55% (protenix-v2) / 59% (opendde) of a warm 298 aa fold, so it is the
first thing the ledger has to attribute. Both models use the same `PairformerLayer` class; only the
pair-track width differs (c_z=256 protenix-v2, c_z=384 opendde), so one target covers both.

N is the PADDED token count: 298 aa -> 320, 117 aa -> 128 (PAIRFORMER_PAD_MULTIPLE=64).

The block is warmed --warm times before the profiled region so no JIT compile and no
program-cache miss lands inside it. --reps blocks are then run back to back, which is how they run
in the model, and the profiled region is closed with a synchronize so no device work escapes it.

    python3 -m tracy -r -o OUT --op-support-count 8000 -- \
        perf/ledger_298/pf_block_target.py --model protenix-v2 --n 320
"""
import argparse
import sys
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "stage_split_298"))

from tt_bio.tenstorrent import PairformerLayer, get_device  # noqa: E402
from tt_bio import protenix_weights as PW  # noqa: E402

TRI_HEAD_DIM = 32


def build(model, ckc):
    """Layer-0 trunk Pairformer block with real checkpoint weights.

    Values are irrelevant to op shapes and device time, but real weights remove every shape guess.
    """
    if model == "protenix-v2":
        path, prefix = "/home/ttuser/.boltz/protenix-v2.pt", "pairformer_stack.blocks.0."
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("aurekaresearch/OpenDDE", "opendde.pt")
        prefix = "pairformer_stack.blocks.0."
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ck = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
    blk = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not blk:
        pfx = sorted({k.split(".blocks.")[0] for k in sd if ".blocks." in k})
        raise SystemExit(f"no keys under {prefix!r}; candidate stacks: {pfx[:20]}")
    remapped = PW.remap_pairformer_block(blk)
    c_z = remapped["tri_mul_out.p_in.weight"].shape[1]
    return PairformerLayer(TRI_HEAD_DIM, c_z // TRI_HEAD_DIM, 384 // 16, 16, True, remapped, ckc), c_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["protenix-v2", "opendde"], default="protenix-v2")
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    layer, c_z = build(args.model, ckc)
    N = args.n
    torch.manual_seed(0)
    s = ttnn.from_torch(torch.randn(1, N, 384), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    print(f"model={args.model} c_z={c_z} tri_heads={c_z // TRI_HEAD_DIM} N={N}", flush=True)

    for _ in range(args.warm):
        s, z = layer(s, z)
    ttnn.synchronize_device(dev)

    print(f"PROFILE_MARK begin {args.model} N={N} reps={args.reps}", flush=True)
    for _ in range(args.reps):
        s, z = layer(s, z)
    ttnn.synchronize_device(dev)
    print("PROFILE_MARK end", flush=True)


if __name__ == "__main__":
    main()
