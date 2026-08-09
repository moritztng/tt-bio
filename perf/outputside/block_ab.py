#!/usr/bin/env python3
"""In-model A/B: one real Pairformer block, three arms.

  base        production: permute per chunk then one concat
  fused       the fused output op, one launch per channel chunk
  fused_dram  ditto, with every chunk given a DRAM result instead of an L1 one
  pair        two chunks per launch: 4 DRAM banks, half the launches, held chunk in DRAM

All arms in one process, same device, same allocator, same weights, alternating so allocator
drift cannot favour one. Parity is torch.equal on both block outputs after a whole block --
the fused op replaces a permute and a concat and does no arithmetic, so bit-exactness is the
bar. `pair` holds one chunk across the next triangle matmul, so this is also the L1-budget
test: it is the step W4's input-side leg died at.
"""
import argparse, json, statistics, sys, time
from pathlib import Path
import torch, ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio import tenstorrent as T                          # noqa: E402
from tt_bio.tenstorrent import PairformerLayer, get_device   # noqa: E402
from tt_bio import protenix_weights as PW                    # noqa: E402

TRI_HEAD_DIM = 32


def build(model, ckc):
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
    remapped = PW.remap_pairformer_block(blk)
    c_z = remapped["tri_mul_out.p_in.weight"].shape[1]
    return PairformerLayer(TRI_HEAD_DIM, c_z // TRI_HEAD_DIM, 384 // 16, 16, True,
                           remapped, ckc), c_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["protenix-v2", "opendde"], default="protenix-v2")
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="block_ab.json")
    a = ap.parse_args()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    layer, c_z = build(a.model, ckc)
    N = a.n
    dg = dev.compute_with_storage_grid_size()
    print(f"model={a.model} c_z={c_z} N={N} grid={dg.x}x{dg.y} "
          f"COMPUTE_GRID_MAIN={T.COMPUTE_GRID_MAIN}", flush=True)

    torch.manual_seed(0)
    s0 = torch.randn(1, N, 384)
    z0 = torch.randn(1, N, N, c_z)

    def fresh():
        return (ttnn.from_torch(s0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16),
                ttnn.from_torch(z0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16))

    ARMS = {"base": (False, False, False), "fused": (True, False, False),
            "fused_dram": (True, False, True), "pair": (True, True, False)}

    def select(arm):
        (T._TRIMUL_OUT_FUSED, T._TRIMUL_OUT_PAIR,
         T._TRIMUL_OUT_CHUNK_DRAM) = ARMS[arm]

    def one_block(arm):
        select(arm)
        s, z = fresh()
        so, zo = layer(s, z)
        return ttnn.to_torch(so), ttnn.to_torch(zo)

    # ---- parity first, on a clean block per arm ----
    res = {"model": a.model, "n": N, "c_z": c_z, "grid": f"{dg.x}x{dg.y}"}
    ref_s, ref_z = one_block("base")
    for arm in ("fused", "fused_dram", "pair"):
        got_s, got_z = one_block(arm)
        exact = bool(torch.equal(ref_s, got_s) and torch.equal(ref_z, got_z))
        res[f"bit_exact_block_{arm}"] = exact
        print(f"bit-exact whole block (s and z), {arm}: {exact}", flush=True)
        if not exact:
            dz = (ref_z.float() - got_z.float()).abs()
            res[f"z_max_abs_diff_{arm}"] = float(dz.max())
            res[f"z_frac_wrong_{arm}"] = float((dz > 0).float().mean())
            print(f"  z max|diff| {dz.max():.4g}  fraction wrong "
                  f"{(dz>0).float().mean():.5f}", flush=True)

    # ---- timing: alternate the arms so allocator state cannot favour one ----
    def wall(arm):
        select(arm)
        s, z = fresh()
        for _ in range(a.warm):
            s, z = layer(s, z)
        ttnn.synchronize_device(dev)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(3):
            s, z = layer(s, z)
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t0) * 1e3 / 3

    runs = {k: [] for k in ARMS}
    for _ in range(a.reps):
        for arm in ARMS:
            runs[arm].append(wall(arm))
    med = {k: statistics.median(v) for k, v in runs.items()}
    res["block_ms"] = med
    res["runs"] = runs
    b = med["base"]
    for arm in ARMS:
        res[f"speedup_{arm}"] = b / med[arm]
        res[f"ms_per_fold_{arm}"] = (b - med[arm]) * 480
        print(f"{arm:6s} {med[arm]:9.4f} ms/block  {b/med[arm]:7.4f}x  "
              f"{(b-med[arm])*480:8.1f} ms/fold   {[round(x,3) for x in runs[arm]]}",
              flush=True)
    dp = med["fused"] - med["pair"]
    res["ms_per_fold_pair_over_fused"] = dp * 480
    print(f"pair over fused: {dp:.4f} ms/block = {dp*480:.1f} ms/fold "
          f"({med['fused']/med['pair']:.4f}x)", flush=True)
    dd = med["fused"] - med["fused_dram"]
    res["ms_per_fold_chunk_dram_alone"] = dd * 480
    print(f"of which the DRAM chunk result alone: {dd:.4f} ms/block = {dd*480:.1f} ms/fold",
          flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


main()
