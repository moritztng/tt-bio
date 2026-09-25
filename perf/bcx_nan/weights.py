#!/usr/bin/env python3
"""Does a forward write into a buffer the model keeps? Hash every device tensor the Evoformer
blocks hold, run the forward, hash again.

`carried.py` found the forward itself alternating between two bit-exact outputs with no Python
attribute on the blocks changing identity, which leaves device memory the model owns: a weight or
cached constant that some op writes through. Forward only, checkpointed exactly as the design loop
runs it, so the backward is out of the picture.
"""
import argparse, hashlib, json, os, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)
import numpy as np, torch, ttnn
import afgrad as A, stack as S
from carried import NPZ


def tensors(dm, blocks):
    """(path, ttnn.Tensor) for every device tensor reachable from the chosen blocks."""
    out, seen = [], set()
    def walk(obj, path, depth):
        if id(obj) in seen or depth > 6:
            return
        seen.add(id(obj))
        if isinstance(obj, ttnn.Tensor):
            out.append((path, obj)); return
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}[{k!r}]", depth + 1)
            return
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]", depth + 1)
            return
        d = getattr(obj, "__dict__", None)
        if d:
            for k, v in d.items():
                walk(v, f"{path}.{k}", depth + 1)
    for i in blocks:
        walk(dm.device_evoformer[i], f"evo[{i}]", 0)
    return out


def digest(t):
    try:
        return hashlib.sha256(torch.Tensor(ttnn.to_torch(t)).float().numpy().tobytes()).hexdigest()[:12]
    except Exception as e:                                                  # noqa: BLE001
        return f"ERR {type(e).__name__}"


def globals_snapshot():
    """Scalars by value and containers by length, for every global of every tt_bio module and
    one level into each class-level dict, taken before a forward."""
    out = {}
    for name, mod in list(sys.modules.items()):
        if not name.startswith("tt_bio") or mod is None:
            continue
        for k, v in list(vars(mod).items()):
            if k.startswith("__"):
                continue
            key = f"{name}.{k}"
            if isinstance(v, (int, float, bool, str, type(None))):
                out[key] = repr(v)
            elif isinstance(v, (list, dict, set, tuple)):
                out[key] = f"len {len(v)}"
                if isinstance(v, dict) and len(v) < 200:
                    for kk, vv in list(v.items()):
                        if isinstance(vv, (int, float, bool, str, type(None))):
                            out[f"{key}[{kk!r}]"] = repr(vv)
            elif isinstance(v, type):
                for kk, vv in list(vars(v).items()):
                    if isinstance(vv, (int, float, bool, str, type(None))) and not kk.startswith("__"):
                        out[f"{key}.{kk}"] = repr(vv)
                    elif isinstance(vv, (dict, list, set)):
                        out[f"{key}.{kk}"] = f"len {len(vv)}"
            else:
                out[key] = f"id {id(v)}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--depth", type=int, default=48)
    ap.add_argument("--out", default="weights.json")
    ap.add_argument("--bwd", action="store_true", help="run the backward too, as the design loop does")
    args = ap.parse_args()
    z = np.load(NPZ)
    lv = S.Levers(); dm, _ = A.load_models(A.DEFAULT_PARAMS)
    dev = A.Dev(dm.to_device()); lv.arm("stack")
    m = torch.from_numpy(z["msa_leaf"]).float(); p_ = torch.from_numpy(z["pair_leaf"]).float()
    mask = dev.up(torch.from_numpy(z["mask"]).float())
    n = int(z["n"])
    gm = torch.zeros(tuple(m.shape)); gz = torch.zeros(tuple(p_.shape))
    gm[:, :n] = torch.from_numpy(z["cot_msa"]).float()
    gz[:n, :n] = torch.from_numpy(z["cot_pair"]).float()
    blocks = list(range(args.depth))
    ts = tensors(dm, blocks)
    print("device tensors held by the blocks:", len(ts), flush=True)
    prev = {p: digest(t) for p, t in ts}
    mask_d = digest(mask)
    rows = []
    gprev = None
    for r in range(args.reps):
        g = globals_snapshot()
        gdiff = sorted(f"{k}: {gprev.get(k)} -> {g.get(k)}" for k in set(g) | set(gprev)
                       if g.get(k) != gprev.get(k)) if gprev is not None else None
        gprev = g
        ml, zl = dev.leaf(m), dev.leaf(p_)
        with dev.tt.tape():
            mo, zo = dev.stack(ml, zl, 0, args.depth, ckpt=True, msa_mask=mask)
        dev.sync()
        fh = hashlib.sha256(dev.down(mo.value, tuple(m.shape)).numpy().tobytes()).hexdigest()[:12]
        clean = None
        if args.bwd:
            dev.ag.backward([mo, zo], [dev.seed(gm, mo), dev.seed(gz, zo)])
            dev.sync()
            a = dev.grad(ml, tuple(m.shape))
            clean = bool(torch.isfinite(a).all())
        dev.ag.release_pins()
        del ml, zl, mo, zo
        ts = tensors(dm, blocks)       # re-walk: a replaced handle is a new tensor
        now = {p: digest(t) for p, t in ts}
        moved = sorted(k for k in set(now) | set(prev) if now.get(k) != prev.get(k))
        row = {"rep": r + 1, "fwd": fh, "bwd_clean": clean, "globals_before_fwd": gdiff, "moved": len(moved), "which": moved[:60],
               "mask_moved": digest(mask) != mask_d}
        rows.append(row)
        print("REP", json.dumps(row), flush=True)
        prev = now
    (HERE / args.out).write_text(json.dumps({"depth": args.depth, "rows": rows,
                                             "stamp": A.stamp(int(os.environ.get("TT_VISIBLE_DEVICES", -1)))},
                                            indent=1))


if __name__ == "__main__":
    main()
