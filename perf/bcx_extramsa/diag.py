#!/usr/bin/env python3
"""Where the device extra-MSA forward loses accuracy: per block, valid vs masked region.

Reads grade.py's capture (BindCraft 2's own pair_in, masks at n=275, dropout off) and runs the
four blocks three ways: float64 reference (MSA track real), host bf16 reference, and the device
block, the last both CHAINED (its own previous output) and ISOLATED (fed the float64 input of
that block). Distances are relative L2 against float64, over valid x valid pairs and over the
rest separately.
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_predictor"),
          str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np   # noqa: E402
import torch         # noqa: E402


def rel(ref, x, sel):
    a, b = np.asarray(ref, np.float64)[sel], np.asarray(x, np.float64)[sel]
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = dict(np.load(args.capture))
    pm = cap["mask_2d"]
    valid = pm > 0
    masked = ~valid

    import afgrad as A
    from splice import ExtraMsaOnDevice
    dm, ref = A.load_models(A.DEFAULT_PARAMS)
    dev = A.Dev(dm.to_device())
    extra = ExtraMsaOnDevice(dev, k_extra=1)

    def host_chain(model, dtype):
        emask = torch.from_numpy(cap["extra_msa_mask"]).to(dtype)
        pmask = torch.from_numpy(pm).to(dtype)
        msa = torch.from_numpy(cap["msa_in"]).to(dtype).reshape(emask.shape + (-1,))
        z = torch.from_numpy(cap["pair_in"]).to(dtype)
        outs = [z.double().numpy()]
        with torch.no_grad():
            for blk in model.extra_msa:
                msa, z = blk(msa, z, emask, pmask)
                outs.append(z.double().numpy())
        return outs

    f64 = host_chain(ref["f64"], torch.float64)
    bf16 = host_chain(ref["bf16"], torch.bfloat16)

    def dev_block(i, z_np):
        # ExtraMsaOnDevice runs blocks [0, k_extra); run block i alone by shifting the view.
        extra._block_offset = i
        return extra._primal(z_np.astype(np.float32), cap["extra_msa_mask"], pm)

    orig_block = ExtraMsaOnDevice._block

    def shifted(self, i, z, masks):
        return orig_block(self, i + getattr(self, "_block_offset", 0), z, masks)
    ExtraMsaOnDevice._block = shifted

    chained = [cap["pair_in"].astype(np.float64)]
    rows = []
    for i in range(4):
        iso = dev_block(i, f64[i])
        ch = dev_block(i, chained[-1])
        chained.append(ch.astype(np.float64))
        row = {"block": i,
               "norm_f64_valid": float(np.linalg.norm(f64[i + 1][valid])),
               "norm_f64_masked": float(np.linalg.norm(f64[i + 1][masked])),
               "absmax_f64": float(np.abs(f64[i + 1]).max())}
        for name, x in (("dev_isolated", iso), ("dev_chained", ch), ("bf16_chained", bf16[i + 1])):
            row[name] = {"valid": rel(f64[i + 1], x, valid), "masked": rel(f64[i + 1], x, masked),
                         "all": rel(f64[i + 1], x, slice(None))}
        # Error in the update alone, isolated: (out - in) against the float64 update.
        upd_ref = f64[i + 1] - f64[i]
        row["dev_isolated_update_rel"] = float(
            np.linalg.norm((iso - f64[i]) - upd_ref) / np.linalg.norm(upd_ref))
        rows.append(row)
        print(json.dumps(row), flush=True)
    rep = {"rows": rows,
           "jax_final": {"valid": rel(f64[4], cap["pair_out"], valid),
                         "masked": rel(f64[4], cap["pair_out"], masked)},
           "opm_constant_absmax": [float(c.detach().abs().max()) for c in dm.opm_constant],
           "opm_constant_norm": [float(c.detach().norm()) for c in dm.opm_constant]}
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}), flush=True)
    np.savez_compressed(out / "diag_arrays.npz", dev_chained=np.stack(chained[1:]).astype(np.float32),
                        f64=np.stack(f64[1:]).astype(np.float32))
    # The VJP, split the same way. A cotangent the device sends into a masked (i, j) with i valid
    # is not harmless: pair_in[i, j] carries left_single(i), so it would reach residue i.
    extra.k_extra, extra._block_offset = 4, 0
    _, token = extra._taped(cap["pair_in"], cap["extra_msa_mask"], pm)
    d_g = extra._backward(token, cap["g_pair_out"])
    dt = torch.float64
    emask = torch.from_numpy(cap["extra_msa_mask"]).to(dt)
    pmask = torch.from_numpy(pm).to(dt)
    msa = torch.from_numpy(cap["msa_in"]).to(dt).reshape(emask.shape + (-1,))
    pair = torch.from_numpy(cap["pair_in"]).to(dt).requires_grad_(True)
    z = pair
    for blk in ref["f64"].extra_msa:
        msa, z = blk(msa, z, emask, pmask)
    (g64,) = torch.autograd.grad(z, pair, torch.from_numpy(cap["g_pair_out"]).to(dt))
    g64 = g64.numpy()
    rep["vjp"] = {arm: {"valid": rel(g64, x, valid), "all": rel(g64, x, slice(None)),
                        "masked_l2": float(np.linalg.norm(np.asarray(x, np.float64)[masked]))}
                  for arm, x in (("device", d_g), ("bc2_jax", cap["g_pair_in"]))}
    rep["vjp"]["f64_masked_l2"] = float(np.linalg.norm(g64[masked]))
    rep["vjp"]["f64_valid_l2"] = float(np.linalg.norm(g64[valid]))
    print("vjp", json.dumps(rep["vjp"]), flush=True)
    (out / "diag.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
