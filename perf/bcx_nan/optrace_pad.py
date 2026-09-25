#!/usr/bin/env python3
"""Which op in Evoformer block 0's forward reads the state a depth-48 backward leaves on the chip?

Each rep is the depth-48 forward + backward that flips the state, with block 0 of that forward
traced: every ttnn verb the shim dispatches, its output downloaded and hashed, in call order.
Reps 2 and 3 (both past the first call's lazy cache builds) give the sequence in both phases;
the first op whose output differs while its inputs agree is the reader.
"""
import hashlib, json, os, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)
import numpy as np, torch, ttnn
import afgrad as A, stack as S
from carried import NPZ
from tt_bio import taped_ttnn as TT
from tt_bio import autograd as AG

TRACE = []
ON = [False]


def h(v):
    v = v.value if isinstance(v, AG.Tensor) else v
    if not isinstance(v, ttnn.Tensor):
        return None
    try:
        x = torch.Tensor(ttnn.to_torch(v)).float()
        tag = f"{tuple(x.shape)}:{hashlib.sha256(x.numpy().tobytes()).hexdigest()[:8]}"
        if v.layout == ttnn.TILE_LAYOUT:
            p = torch.Tensor(v.cpu().to_torch_with_padded_shape()).float()
            bad = int((~torch.isfinite(p)).sum()) - int((~torch.isfinite(x)).sum())
            tag += f":padnf={bad}"
        return tag
    except Exception as e:                                                  # noqa: BLE001
        return f"ERR:{type(e).__name__}"


_orig = TT._taped_verb


def traced(qual, shipped):
    inner = _orig(qual, shipped)
    def call(*args, **kwargs):
        if not ON[0] or qual == "deallocate":
            return inner(*args, **kwargs)
        ins = [h(a) for a in list(args) + list(kwargs.values())]
        out = inner(*args, **kwargs)
        outs = out if isinstance(out, (list, tuple)) else [out]
        TRACE.append({"op": qual, "in": [i for i in ins if i], "out": [h(o) for o in outs]})
        return out
    return call


TT._taped_verb = traced
for k in list(vars(TT._SHIM)):
    if k not in ("_real", "_prefix"):
        delattr(TT._SHIM, k)


def main():
    z = np.load(NPZ)
    lv = S.Levers(); dm, _ = A.load_models(A.DEFAULT_PARAMS)
    dev = A.Dev(dm.to_device()); lv.arm("stack")
    m = torch.from_numpy(z["msa_leaf"]).float(); p_ = torch.from_numpy(z["pair_leaf"]).float()
    mask = dev.up(torch.from_numpy(z["mask"]).float())
    n = int(z["n"])
    gm = torch.zeros(tuple(m.shape)); gz = torch.zeros(tuple(p_.shape))
    gm[:, :n] = torch.from_numpy(z["cot_msa"]).float()
    gz[:n, :n] = torch.from_numpy(z["cot_pair"]).float()
    traces, clean = [], []
    for r in range(3):
        TRACE.clear()
        ml, zl = dev.leaf(m), dev.leaf(p_)
        with dev.tt.tape():
            ON[0] = True
            mo, zo = dev.stack(ml, zl, 0, 1, ckpt=True, msa_mask=mask)
            ON[0] = False
            mo, zo = dev.stack(mo, zo, 0, 47, evo_first=1, ckpt=True, msa_mask=mask)
        traces.append(list(TRACE))
        dev.ag.backward([mo, zo], [dev.seed(gm, mo), dev.seed(gz, zo)])
        dev.sync()
        clean.append(bool(torch.isfinite(dev.grad(ml, tuple(m.shape))).all()))
        dev.ag.release_pins(); del ml, zl, mo, zo
        print("REP", r + 1, "ops", len(traces[-1]), "bwd48 clean", clean[-1], flush=True)
    traces = traces[1:]
    a, b = traces
    first = None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            first = i
            break
    print("FIRST DIFF", first, flush=True)
    if first is not None:
        for i in range(max(0, first - 3), min(len(a), first + 4)):
            print(i, json.dumps(a[i]), "|", json.dumps(b[i]["out"]) if i < len(b) else None, flush=True)
    (HERE / "optrace_pad.json").write_text(json.dumps({"first": first, "clean48": clean,
                                                    "a": a, "b": b}, indent=0))


if __name__ == "__main__":
    main()
