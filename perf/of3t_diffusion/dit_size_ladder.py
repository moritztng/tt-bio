#!/usr/bin/env python3
"""Where does our diffusion transformer start disagreeing with upstream's? A size ladder.

The r=0 boundary says our DiT is a different function at 384 tokens: their DiT contracts a
2.63e-03 input perturbation to 1.15e-03, ours turns the same input into 7.35e-02, and forcing
every token real makes it worse (2.53e-01), so it is neither masking nor conditioning. The
shipped fixture gate runs 76 real tokens of 96 and is bit-exact. Those two facts can only both
be true if the defect has an ONSET in token count, and the onset decides whether shipped folds
of ordinary-sized targets are affected.

Both sides get byte-identical random inputs and the same checkpoint weights, every token real,
one sample. Upstream runs float64 on CPU; ours runs the shipped fp32 device path. The float64
side is the reference, so a small number at the bottom of the ladder is what "our port is
right here" looks like and a large one at the top is not precision.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.getcwd())
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="96,128,192,256,384")
    ap.add_argument("--out", default="perf/of3t_diffusion/dit_size_ladder.json")
    a = ap.parse_args()
    t0 = time.time()

    import ttnn
    from tt_bio.tenstorrent import get_device, device_dtype_override
    from tt_bio.openfold3_diffusion_transformer import OF3DiffusionTransformer
    from tt_bio.openfold3_weights import _sub
    sys.path.insert(0, "/home/ttuser/of3t_gradients/ref")
    import bundle_min as BM

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    dmsd = _sub(sd, "diffusion_module")

    built = BM.build(torch.float64, 20260919, "cpu", num_recycles=0)
    model = built[1]
    sd64 = {k: v.to(torch.float64) if torch.is_tensor(v) and v.is_floating_point() else v
            for k, v in sd.items()}
    model.load_state_dict(sd64, strict=False)
    ref_dit = model.diffusion_module.diffusion_transformer
    print(f"[{time.time()-t0:.0f}s] upstream DiT ready", flush=True)

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    act = ttnn.float32
    with device_dtype_override(act):
        our_dit = OF3DiffusionTransformer(_sub(dmsd, "diffusion_transformer"), cfg)
    print(f"[{time.time()-t0:.0f}s] device DiT built", flush=True)

    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=act)
    rows = {}
    gen = torch.Generator().manual_seed(20260919)
    for n in [int(x) for x in a.sizes.split(",")]:
        aa = torch.randn(1, 1, n, 768, generator=gen, dtype=torch.float64)
        ss = torch.randn(1, 1, n, 384, generator=gen, dtype=torch.float64)
        zz = torch.randn(1, 1, n, n, 128, generator=gen, dtype=torch.float64) * 0.1
        mk = torch.ones(1, 1, n, dtype=torch.float64)
        with torch.no_grad(), BM.no_autocast():
            ref = ref_dit(a=aa, s=ss, z=zz, mask=mk, _mask_trans=True,
                          use_high_precision_attention=True)
        ref2 = ref.reshape(n, 768).double()
        with device_dtype_override(act):
            out = our_dit(ft(aa.reshape(1, n, 768)), ft(ss.reshape(1, n, 384)),
                          ft(zz.reshape(1, n, n, 128)), ft(mk.reshape(1, n)),
                          ft(mk.reshape(1, n, 1)))
        ours = ttnn.to_torch(out).reshape(n, 768).double()
        r = float(torch.linalg.vector_norm(ours - ref2)
                  / (torch.linalg.vector_norm(ref2) + 1e-300))
        rows[n] = {"rel": r, "ref_norm": float(torch.linalg.vector_norm(ref2)),
                   "our_norm": float(torch.linalg.vector_norm(ours))}
        print(f"[{time.time()-t0:.0f}s] n={n:4d}  rel {r:.6e}  "
              f"|ref| {rows[n]['ref_norm']:.4e}  |ours| {rows[n]['our_norm']:.4e}", flush=True)

    json.dump({"ladder": rows, "dtype_device": str(act), "dtype_reference": "float64",
               "mask": "all tokens real", "samples": 1}, open(a.out, "w"), indent=1)
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
