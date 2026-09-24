#!/usr/bin/env python3
"""of3t-paez step 1: the reference's own ratio on the confidence head's inputs, 384 wide.

    ratio.py --dump conf_D384.pt --out RATIO.json [--stages-out STAGES.pt]

Upstream 0.4.3 runs the trunk on the 384 batch twice, float64 (`cast_policy("removed")`) and
upstream's bf16 recipe (`cast_policy("bf16")`, float32 parameters), then the confidence
Pairformer and the pae / pde heads, each at its own rolled-out structure (ref384c's
rollout_{f64,bf16}.pt; the device at the dump's repr_x), which is what each side of the scored
step differentiated. Against float64 on the real block: device (the CF384 tree's own dump,
devstep.py --conf-dump) and bf16, per tensor. `carried` is upstream's float64 head on each
side's trunk outputs at float64's structure: the part of the head's output its inputs carry.
The pair output after every trunk stage is kept for both CPU sides (step 3's curve).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "of3t_fullstep64"))
sys.path.insert(0, str(HERE.parent / "of3t_confpfe"))
import ref_step  # noqa: E402
from conf_split import heads, real, rel  # noqa: E402

bm = ref_step.bm
R = Path("/home/ttuser/of3t-campaign-refs")
REF = Path("/home/ttuser/of3t_confpfe/ref384c")
BATCH, BATCH_SHA = R / "bundle_min_043/batch_step003.pt", \
    "3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f"
CK = Path("/home/ttuser/of3-weights/of3-p2-155k.pt")
KEYS = ("s_input", "s_trunk", "z_trunk", "si_conf", "zij_conf", "pae_logits", "pde_logits")
PAIR = {"z_trunk", "zij_conf", "pae_logits", "pde_logits"}


def stage_hooks(model, rec):
    def grab(name):
        def hook(_m, _i, out):
            outs = out if isinstance(out, tuple) else (out,)
            z = [t for t in outs if torch.is_tensor(t) and t.dim() == 4 and t.shape[-1] == 128]
            if z:
                rec[name] = z[-1].detach().double().cpu()
        return hook
    hs = [getattr(model, n).register_forward_hook(grab(n))
          for n in ("input_embedder", "template_embedder", "msa_module")]
    hs += [b.register_forward_hook(grab(f"pf_{i:02d}"))
           for i, b in enumerate(model.pairformer_stack.blocks)]
    return hs


def run(dt, policy, repr_x):
    cfg, model, _d, _ck = ref_step.load(dt, CK, 20260919, None)
    batch = bm.move(torch.load(BATCH, weights_only=False), "cpu", dt)
    tok = batch["token_mask"]
    stages = {}
    hs = stage_hooks(model, stages)
    with torch.no_grad(), bm.cast_policy(policy, "cpu"):
        s_input, s, z = model.run_trunk(batch=batch, num_cycles=1, inplace_safe=False)
        for h in hs:
            h.remove()
        out = heads(model, s_input, s, z, repr_x.to(dt), tok, None)
    out.update(s_input=s_input, s_trunk=s, z_trunk=z)
    return model, batch, {k: v.double() for k, v in out.items()}, stages


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stages-out", type=Path)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if ref_step.sha256_file(BATCH) != BATCH_SHA:
        raise SystemExit("batch sha256 mismatch")
    bm.pin_deterministic_kernels(True)
    rx = {m: torch.load(REF / m / f"rollout_{m}.pt", weights_only=False)["repr_x"]
          for m in ("f64", "bf16")}
    W = rx["f64"].shape[0]
    cache = a.out.with_suffix(".cache.pt")
    if cache.exists():
        c = torch.load(cache, weights_only=False)
        bf, st_bf, f64, st_f64 = c["bf"], c["st_bf"], c["f64"], c["st_f64"]
        model = ref_step.load(torch.float64, CK, 20260919, None)[1]
        batch = bm.move(torch.load(BATCH, weights_only=False), "cpu", torch.float64)
    else:
        _mb, _bb, bf, st_bf = run(torch.float32, "bf16", rx["bf16"])
        del _mb
        model, batch, f64, st_f64 = run(torch.float64, "removed", rx["f64"])
        torch.save({"bf": bf, "st_bf": st_bf, "f64": f64, "st_f64": st_f64}, cache)
    tok = batch["token_mask"]
    n = int(tok.sum())
    import time
    while not a.dump.exists():      # the device dump may still be running
        time.sleep(30)
    time.sleep(10)
    d = torch.load(a.dump, weights_only=False)

    dev = {}
    for k, v in {**d["inputs"], **d["outputs"]}.items():
        if k not in KEYS:
            continue
        v = v.double()
        dev[k] = v.reshape(1, *v.shape[-3:])[:, :W, :W] if k in PAIR else \
            v.reshape(1, *v.shape[-2:])[:, :W]
    rec = {"batch": BATCH.name, "width": W, "real_tokens": n, "dump": str(a.dump),
           "device_input_dtypes": d.get("dtypes"), "table": {}, "carried": {}}
    for k in KEYS:
        p = k in PAIR
        ref = real(f64[k], n, p)
        rec["table"][k] = {"device": rel(real(dev[k], n, p), ref), "bf16": rel(real(bf[k], n, p), ref)}
        r = rec["table"][k]
        r["device_over_bf16"] = r["device"]["rel"] / r["bf16"]["rel"]
    with torch.no_grad():
        for side, src in (("device", dev), ("bf16", bf)):
            h = heads(model, src["s_input"], src["s_trunk"], src["z_trunk"], rx["f64"].double(), tok, None)
            rec["carried"][side] = {k: rel(real(h[k], n, k in PAIR), real(f64[k], n, k in PAIR))
                                    for k in ("si_conf", "zij_conf", "pae_logits", "pde_logits")}
    rec["stages_bf16_vs_f64"] = {k: rel(real(st_bf[k], n, True), real(st_f64[k], n, True))["rel"]
                                 for k in st_f64}
    json.dump(rec, open(a.out, "w"), indent=1)
    if a.stages_out:
        torch.save({"f64": {k: real(v, n, True) for k, v in st_f64.items()},
                    "bf16": {k: real(v, n, True) for k, v in st_bf.items()}}, a.stages_out)
    for k, r in rec["table"].items():
        print(f"{k:12s} device {r['device']['rel']:.4e}  bf16 {r['bf16']['rel']:.4e}  ratio {r['device_over_bf16']:.2f}")
    for s, v in rec["carried"].items():
        print("carried", s, {k: round(x["rel"], 5) for k, x in v.items()})
    print("stages bf16", {k: round(v, 5) for k, v in rec["stages_bf16_vs_f64"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
