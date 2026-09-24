#!/usr/bin/env python3
"""The captured call's mask path against float64: Evoformer block 0, the captured inputs and the
captured MSA mask (200 of 576 entries zero), forward and VJP, the fixed `_mask_biases` against
the one before the fix.

The all-ones `afgrad.py vjp --msa-mask` run cannot tell a right mask bias from a wrong one, since
both are zero on every logical element. This one masks for real, and writes the padding the
reshapes used to leave unwritten to one known value per arm (0, NaN, +-1e30, 50), so what a
given padding does to the result is measured rather than left to whatever the buffer held.
"""
import argparse, gc, hashlib, json, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)
import numpy as np, torch, ttnn
import afgrad as A, stack as S
from carried import NPZ
from tt_bio import af2
from tt_bio.af2 import MASK_LOGIT_BIAS

FIXED = af2.AF2EvoformerBlock._mask_biases


def padded_with(value):
    """`_mask_biases` as it was before e5cf74790, whose reshapes leave their tile padding holding
    whatever the buffer held, with that padding written to one known value instead."""
    def biases(self, msa_mask):
        if len(msa_mask.shape) == 3:
            msa_mask = ttnn.reshape(msa_mask, tuple(msa_mask.shape)[1:])
        rows, n = (int(d) for d in msa_mask.shape)
        flat = ttnn.multiply(ttnn.subtract(msa_mask, 1.0), MASK_LOGIT_BIAS)
        return (ttnn.fill_implicit_tile_padding(ttnn.reshape(flat, (rows, 1, 1, n)), value),
                ttnn.fill_implicit_tile_padding(
                    ttnn.reshape(ttnn.permute(flat, (1, 0)), (n, 1, 1, rows)), value))
    return biases


def crop(t, n, kind):
    return t[:, :n] if kind == "m" else t[:n, :n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "maskvjp.json"))
    args = ap.parse_args()
    torch.set_num_threads(4)
    z = np.load(NPZ)
    n = int(z["n"])
    m = A.bf(torch.from_numpy(z["msa_leaf"]))
    p_ = A.bf(torch.from_numpy(z["pair_leaf"]))
    mask = torch.from_numpy(z["mask"]).double()
    gm = torch.zeros(m.shape, dtype=torch.float64); gz = torch.zeros(p_.shape, dtype=torch.float64)
    gm[:, :n] = torch.from_numpy(z["cot_msa"]).double()
    gz[:n, :n] = torch.from_numpy(z["cot_pair"]).double()

    lv = S.Levers(); dm, ref = A.load_models(A.DEFAULT_PARAMS)
    dev = A.Dev(dm.to_device()); lv.arm("stack")
    i = args.block
    pair_mask = torch.ones(p_.shape[:2], dtype=torch.float64)
    t0 = time.time()
    arms = {}
    for name in ("f64", "bf16"):
        mod = ref[name]; dt = mod.trunk_dtype
        (dmr, dzr), (mo, zo) = A.ref_vjp(
            lambda a, b, mod=mod, dt=dt: mod.evoformer[i](a, b, mask.to(dt), pair_mask.to(dt)),
            [m.to(dt), p_.to(dt)], [gm, gz])
        arms[name] = {"m": mo.detach().double(), "z": zo.detach().double(),
                      "dm": dmr.double(), "dz": dzr.double()}
    print("REF", round(time.time() - t0, 1), "s", flush=True)
    r64 = arms["f64"]

    def grade(got):
        out = {}
        for k in ("m", "z", "dm", "dz"):
            kind = "m" if k.endswith("m") else "z"
            out[k] = {"full": A.cmp(got[k], r64[k]),
                      "n": A.cmp(crop(got[k], n, kind), crop(r64[k], n, kind)),
                      "nonfinite": int((~torch.isfinite(got[k])).sum())}
        return out

    rows = [{"arm": "bf16_torch", **grade(arms["bf16"])}]
    for r in range(args.reps):
        for arm, fill in (("fixed", 0.0), ("pad", float("inf")), ("pad", float("-inf")), ("pad", 3.3e38),
                          ("pad", -3.3e38)):
            af2.AF2EvoformerBlock._mask_biases = FIXED if arm == "fixed" else padded_with(fill)
            gc.collect()
            ml, zl = dev.leaf(m.float()), dev.leaf(p_.float())
            with dev.tt.tape():
                mo, zo = dev.evo(i, ml, zl, dev.up(mask.float()))
            dev.ag.backward([mo, zo], [dev.seed(gm.float(), mo), dev.seed(gz.float(), zo)])
            dev.sync()
            got = {"m": dev.down(mo.value, tuple(m.shape)).double(),
                   "z": dev.down(zo.value, tuple(p_.shape)).double(),
                   "dm": dev.grad(ml, tuple(m.shape)).double(),
                   "dz": dev.grad(zl, tuple(p_.shape)).double()}
            digest = hashlib.sha256(got["dm"].numpy().tobytes() + got["dz"].numpy().tobytes()).hexdigest()[:16]
            row = {"arm": arm, "pad": fill, "rep": r + 1, "digest": digest, **grade(got)}
            rows.append(row)
            del ml, zl, mo, zo
            print("ROW", arm, fill, r + 1, digest, " ".join(
                f"{k}={row[k]['n']['rel_l2']:.4g}/nf{row[k]['nonfinite']}" for k in ("m", "z", "dm", "dz")),
                flush=True)
    af2.AF2EvoformerBlock._mask_biases = FIXED
    pathlib.Path(args.out).write_text(json.dumps(
        {"stamp": A.stamp(2), "block": i, "n": n, "mask_zeros": int((mask == 0).sum()),
         "rows": rows}, indent=1, default=str))
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
