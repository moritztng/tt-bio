#!/usr/bin/env python3
"""Depth ladder: how many DiT blocks does it take to produce the disagreement?

The synthetic ladder was refuted by our own fixture gate: random N(0,1) activations read
8.2e-01 at n=96 where the gate at 76 real tokens of 96 is bit-exact. operand_check.py then
showed the standalone call is sound -- our DiT on THEIR exact captured inputs reads 7.37e-02,
the same as in-module -- so the fault was the synthetic inputs, not the harness.

This ladder truncates the REAL captured activations instead of inventing them: the first n
tokens of their `a`, `s`, `z` and `mask`. Both sides get byte-identical truncated inputs, so
the comparison is valid at every rung and the activation statistics stay in distribution.

The fixture gate is bit-exact at 96 and this boundary is 7.4e-02 at 384, so an onset exists
between them. Finding it decides whether shipped folds of ordinary-sized targets are exposed,
which is a production question this campaign should not leave open.

No timing claim; tenancy stamped.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/home/ttuser/of3t_gradients/ref")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "of3t_gradients"))
import capture_trunk_boundary as CTB  # noqa: E402

CAP = "/home/ttuser/of3t_diffusion_cap"
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")


def rel(x, y):
    return float(torch.linalg.vector_norm(x.double() - y.double())
                 / (torch.linalg.vector_norm(y.double()) + 1e-300))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="1,2,4,8,16,24")
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--struct", type=int, default=0)
    ap.add_argument("--out", default="perf/of3t_diffusion/dit_depth_ladder.json")
    a = ap.parse_args()
    t0 = time.time()

    import bundle_min as BM
    import ttnn
    from tt_bio.tenstorrent import get_device, device_dtype_override
    from tt_bio.openfold3_diffusion_transformer import OF3DiffusionTransformer
    from tt_bio.openfold3_weights import _sub

    S = torch.load(f"{CAP}/sub_boundary.pt", map_location="cpu", weights_only=False)
    _, dit_kw = S["dit_in"]
    k = a.struct
    aa_full = dit_kw["a"][0, k].double()
    ss_full = dit_kw["s"][0, k].double() if dit_kw["s"].dim() >= 3 and \
        dit_kw["s"].shape[1] == dit_kw["a"].shape[1] else \
        dit_kw["s"].reshape(dit_kw["s"].shape[-2], -1).double()
    n_full = aa_full.shape[0]
    zz_full = dit_kw["z"].reshape(n_full, n_full, -1).double()
    mk_full = dit_kw["mask"].reshape(-1)[:n_full].double()

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(kk[6:] if kk.startswith("model.") else kk): v for kk, v in sd.items()}
    dmsd = _sub(sd, "diffusion_module")

    built = BM.build(torch.float64, 20260919, "cpu", num_recycles=0)
    model = built[1]
    sd64 = {kk: v.to(torch.float64) if torch.is_tensor(v) and v.is_floating_point() else v
            for kk, v in sd.items()}
    model.load_state_dict(sd64, strict=False)
    ref_dit = model.diffusion_module.diffusion_transformer

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    act = ttnn.float32
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=act)
    rows = {}
    n = a.n
    aa, ss, zz, mk = aa_full[:n], ss_full[:n], zz_full[:n, :n], mk_full[:n]
    full_blocks = list(ref_dit.blocks)
    print(f"[{time.time()-t0:.0f}s] DiT has {len(full_blocks)} blocks, n={n}", flush=True)
    for d in [int(x) for x in a.depths.split(",")]:
        if d > len(full_blocks):
            continue
        ref_dit.blocks = torch.nn.ModuleList(full_blocks[:d])
        with torch.no_grad(), BM.no_autocast():
            ref = ref_dit(a=aa.reshape(1, 1, n, -1), s=ss.reshape(1, 1, n, -1),
                          z=zz.reshape(1, 1, n, n, -1), mask=mk.reshape(1, 1, n),
                          _mask_trans=True, use_high_precision_attention=True)
        ref2 = ref.reshape(n, -1).double()
        with device_dtype_override(act):
            our_dit = OF3DiffusionTransformer(_sub(dmsd, "diffusion_transformer"), cfg,
                                              n_blocks=d)
            out = our_dit(ft(aa.reshape(1, n, -1)), ft(ss.reshape(1, n, -1)),
                          ft(zz.reshape(1, n, n, -1)), ft(mk.reshape(1, n)),
                          ft(mk.reshape(1, n, 1)))
        ours = ttnn.to_torch(out).reshape(n, -1).double()
        m = mk.bool()
        rows[d] = {"rel": rel(ours, ref2), "rel_real": rel(ours[m], ref2[m])}
        print(f"[{time.time()-t0:.0f}s] blocks={d:3d}  rel {rows[d]['rel']:.6e}  "
              f"real-rows {rows[d]['rel_real']:.6e}", flush=True)
    ref_dit.blocks = torch.nn.ModuleList(full_blocks)

    json.dump({"ladder": rows, "struct": k, "inputs": "real captured activations, DiT truncated to d blocks on BOTH sides",
               "dtype_device": str(act), "dtype_reference": "float64",
               "tenancy": subprocess.run(["bash", "-lc",
                                          "pgrep -af TT_VISIBLE_DEVICES | grep -v pgrep | wc -l"],
                                         capture_output=True, text=True).stdout.strip()},
              open(a.out, "w"), indent=1)
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
