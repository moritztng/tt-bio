#!/usr/bin/env python3
"""In-model A/B: one real Pairformer block with and without the fused output op.

Both arms in one process, same device, same allocator, same weights, alternating so allocator
drift cannot favour one. Parity is torch.equal on both block outputs after a whole block --
the fused op replaces a permute and a concat and does no arithmetic, so bit-exactness is the
bar. This is the step W4's input-side leg died at, on the L1 budget.
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

    def one_block(fused):
        T._TRIMUL_OUT_FUSED = fused
        s, z = fresh()
        so, zo = layer(s, z)
        out = (ttnn.to_torch(so), ttnn.to_torch(zo))
        return out

    # ---- parity first, on a clean block each way ----
    ref_s, ref_z = one_block(False)
    got_s, got_z = one_block(True)
    exact = bool(torch.equal(ref_s, got_s) and torch.equal(ref_z, got_z))
    res = {"model": a.model, "n": N, "c_z": c_z, "grid": f"{dg.x}x{dg.y}",
           "bit_exact_block": exact}
    print(f"bit-exact whole block (s and z): {exact}", flush=True)
    if not exact:
        dz = (ref_z.float() - got_z.float()).abs()
        res["z_max_abs_diff"] = float(dz.max())
        res["z_frac_wrong"] = float((dz > 0).float().mean())
        print(f"  z max|diff| {dz.max():.4g}  fraction wrong {(dz>0).float().mean():.5f}",
              flush=True)

    # ---- timing: alternate the arms so allocator state cannot favour one ----
    def wall(fused):
        T._TRIMUL_OUT_FUSED = fused
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

    base, fus = [], []
    for _ in range(a.reps):
        base.append(wall(False))
        fus.append(wall(True))
    b, f = statistics.median(base), statistics.median(fus)
    res["block_ms_baseline"] = b
    res["block_ms_fused"] = f
    res["speedup"] = b / f
    res["ms_per_fold"] = (b - f) * 480
    print(f"block  baseline {b:.4f} ms   fused {f:.4f} ms   {b/f:.4f}x", flush=True)
    print(f"       saved {(b-f):.4f} ms/block  x480 = {(b-f)*480:.1f} ms/fold", flush=True)
    print(f"       baseline arms {[round(x,3) for x in base]}", flush=True)
    print(f"       fused arms    {[round(x,3) for x in fus]}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


main()
