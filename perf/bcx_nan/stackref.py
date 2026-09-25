#!/usr/bin/env python3
"""Which of the two finite answers is right: the old clean phase or the fixed call?

After e5cf74790 the captured 48-block call reads d_msa absmax 0.289 on every replay; before it,
the clean phase read 0.364. Both are finite, so finiteness cannot say which gradient is right,
and a 48-block float64 reference at n=288 does not finish in a bounded pass on a loaded host.
So this hashes every block's forward output for each replay, finds the first block where the old
clean phase and the fixed call part, and grades both outputs of that block against
`af2_reference`'s block in float64 on their shared (bit-identical) input, with the captured MSA
mask and the all-ones pair mask the device call uses.

The old `_mask_biases` is replayed several times first so that at least one replay lands in the
clean phase of the period-2 alternation.
"""
import argparse, hashlib, json, os, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)
import numpy as np, torch
import afgrad as A, stack as S
from carried import NPZ

DEPTH = 48


def unfixed(self, msa_mask):
    """`_mask_biases` as it was before e5cf74790."""
    import ttnn
    from tt_bio.af2 import MASK_LOGIT_BIAS
    if len(msa_mask.shape) == 3:
        msa_mask = ttnn.reshape(msa_mask, tuple(msa_mask.shape)[1:])
    rows, n = (int(d) for d in msa_mask.shape)
    flat = ttnn.multiply(ttnn.subtract(msa_mask, 1.0), MASK_LOGIT_BIAS)
    return (ttnn.reshape(flat, (rows, 1, 1, n)),
            ttnn.reshape(ttnn.permute(flat, (1, 0)), (n, 1, 1, rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="unfixed,unfixed,unfixed,fixed")
    ap.add_argument("--out", default="stackref.json")
    args = ap.parse_args()
    from tt_bio import af2
    z = np.load(NPZ)
    n = int(z["n"])
    m = torch.from_numpy(z["msa_leaf"]).float(); p_ = torch.from_numpy(z["pair_leaf"]).float()
    mask = torch.from_numpy(z["mask"]).float()
    gm = torch.zeros(m.shape); gz = torch.zeros(p_.shape)
    gm[:, :n] = torch.from_numpy(z["cot_msa"]).float()
    gz[:n, :n] = torch.from_numpy(z["cot_pair"]).float()
    lv = S.Levers(); dm, ref = A.load_models(A.DEFAULT_PARAMS)
    dev = A.Dev(dm.to_device()); lv.arm("stack")
    fixed = af2.AF2EvoformerBlock._mask_biases
    dmask = dev.up(mask)
    reps = []
    for r, arm in enumerate(args.arms.split(",")):
        af2.AF2EvoformerBlock._mask_biases = fixed if arm == "fixed" else unfixed
        ml, zl = dev.leaf(m), dev.leaf(p_)
        outs, hs = [], []
        with dev.tt.tape():
            mo, zo = ml, zl
            for i in range(DEPTH):
                mo, zo = dev.stack(mo, zo, 0, 1, evo_first=i, ckpt=True, msa_mask=dmask)
                a = dev.down(mo.value, tuple(m.shape)); b = dev.down(zo.value, tuple(p_.shape))
                hs.append(hashlib.sha256(a.numpy().tobytes() + b.numpy().tobytes()).hexdigest()[:8])
                outs.append((a.to(torch.bfloat16), b.to(torch.bfloat16)))
        dev.ag.backward([mo, zo], [dev.seed(gm, mo), dev.seed(gz, zo)])
        dev.sync()
        ga = dev.grad(ml, tuple(m.shape)); gb = dev.grad(zl, tuple(p_.shape))
        dev.ag.release_pins()
        del ml, zl, mo, zo
        clean = bool(torch.isfinite(ga).all() and torch.isfinite(gb).all())
        rep = {"rep": r + 1, "arm": arm, "clean": clean,
               "digest": hashlib.sha256(ga.numpy().tobytes() + gb.numpy().tobytes()).hexdigest()[:16],
               "d_msa_absmax": float(ga[torch.isfinite(ga)].abs().max()),
               "fwd_nonfinite_first_block": next((i for i, (a, b) in enumerate(outs)
                                                  if not (torch.isfinite(a.float()).all() and torch.isfinite(b.float()).all())), None),
               "blocks": " ".join(hs)}
        print("REP", json.dumps(rep), flush=True)
        reps.append((rep, outs))
    fx = next((x for x in reps if x[0]["arm"] == "fixed"), None)
    old = next((x for x in reps if x[0]["arm"] == "unfixed" and x[0]["clean"]), None)
    out = {"depth": DEPTH, "n": n, "reps": [x[0] for x in reps]}
    if fx and old:
        fh, oh = fx[0]["blocks"].split(), old[0]["blocks"].split()
        k = next((i for i in range(DEPTH) if fh[i] != oh[i]), None)
        out["first_divergent_block"] = k
        if k is not None:
            mi, zi = (m, p_) if k == 0 else (fx[1][k - 1][0].float(), fx[1][k - 1][1].float())
            if k:
                assert torch.equal(fx[1][k - 1][0], old[1][k - 1][0]) and torch.equal(fx[1][k - 1][1], old[1][k - 1][1])
            mi = mi.to(torch.bfloat16).double(); zi = zi.to(torch.bfloat16).double()
            t0 = time.time()
            with torch.no_grad():
                rm, rz = ref["f64"].evoformer[k](mi, zi, mask.double(), torch.ones(zi.shape[:2], dtype=torch.float64))
            out["ref_s"] = round(time.time() - t0, 1)
            g = {}
            for name, (om, oz) in (("fixed", fx[1][k]), ("old_clean", old[1][k])):
                om, oz = om.double(), oz.double()
                g[name] = {"m": A.cmp(om[:, :n], rm[:, :n]), "z": A.cmp(oz[:n, :n], rz[:n, :n]),
                           "m_rows": [A.cmp(om[j, :n], rm[j, :n]) for j in range(om.shape[0])]}
            dm_ = (fx[1][k][0].double() - old[1][k][0].double()).abs()
            dz_ = (fx[1][k][1].double() - old[1][k][1].double()).abs()
            g["fixed_vs_old"] = {"m_maxabs": float(dm_.max()), "m_differs_at": int((dm_ > 0).sum()),
                                 "z_maxabs": float(dz_.max()), "z_differs_at": int((dz_ > 0).sum()),
                                 "m_differs_rows": sorted({int(j) for j in torch.nonzero(dm_ > 0)[:, 0]}),
                                 "m_differs_cols_min_max": [int(torch.nonzero(dm_ > 0)[:, 1].min()), int(torch.nonzero(dm_ > 0)[:, 1].max())] if (dm_ > 0).any() else None}
            out["graded_block"] = g
            print("GRADE", json.dumps(out.get("graded_block")), flush=True)
    out["stamp"] = A.stamp(int(os.environ.get("TT_VISIBLE_DEVICES", -1)))
    (HERE / args.out).write_text(json.dumps(out, indent=1, default=str))
    print("FIRST", out.get("first_divergent_block"), flush=True)


if __name__ == "__main__":
    main()
