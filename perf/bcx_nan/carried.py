#!/usr/bin/env python3
"""Which state does one AF2 backward hand the next? Controls on the captured failing call.

`bcx-predictor` replayed one captured backward eight times in one process and got clean, NaN,
clean, NaN: strict period 2, each outcome bit-identical to itself. This runs the same call
`--reps` times and applies one control between calls, so the control that turns the
alternation into clean-every-call names the state that carries.

  none        nothing between calls (the symptom)
  noprog      program cache disabled for the whole run: every op compiles fresh
  clearprog   program cache cleared between calls
  pad         one extra DRAM tensor allocated and held before every other call, which moves
              the allocator layout without touching any Python state
  gc          gc.collect() between calls
  poison-nan  fill free DRAM with NaN between calls (allocate until refused, then free): a kernel
              that reads memory it never wrote turns every call bad
  poison-zero the same with zeros
  forget      autograd.forget_wrappers() right after each backward: the raw-handle map a
              backwards recomputes fill is otherwise only cleared when the NEXT forwards
              tape closes

Each call prints a digest of both gradients, so "bit-identical to rep 1" is a string compare.
"""
import argparse, gc, hashlib, json, os, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)
import numpy as np, torch
import afgrad as A, stack as S

NPZ = ROOT / "perf" / "bcx_predictor" / "nancap" / "nan_backward_inputs.npz"


def ttnn_bytes(dev, t):
    import ttnn
    return torch.Tensor(ttnn.to_torch(t.value)).float().numpy().tobytes()


def poison(dev, value, chunk_mb=256, cap_gb=30):
    """Overwrite every free DRAM page with , then give them all back."""
    import ttnn
    rows = chunk_mb * 1024 * 1024 // 2 // 1024
    held = []
    try:
        while len(held) * chunk_mb < cap_gb * 1024:
            held.append(ttnn.full((1, rows, 1024), value, dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=dev.device,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG))
    except Exception:                                                       # noqa: BLE001
        pass
    dev.sync()
    n = len(held)
    for t in held:
        ttnn.deallocate(t)
    print("POISON", value, "chunks", n, flush=True)


def snapshot(dm):
    """id() and a scalar summary of every attribute reachable from the device blocks, two levels
    deep. A Python-side cache that one call sets and the next consumes shows up as an attribute
    whose identity changes between calls."""
    out = {}
    def walk(obj, path, depth):
        d = getattr(obj, "__dict__", None)
        if d is None or depth > 3:
            return
        for k, v in d.items():
            if k.startswith("__"):
                continue
            key = f"{path}.{k}"
            if isinstance(v, (int, float, bool, str, type(None))):
                out[key] = repr(v)
            else:
                out[key] = id(v)
                if isinstance(v, (list, tuple)):
                    for i, e in enumerate(v[:64]):
                        out[f"{key}[{i}]"] = id(e) if not isinstance(e, (int, float, bool, str, type(None))) else repr(e)
                        walk(e, f"{key}[{i}]", depth + 1)
                elif isinstance(v, dict):
                    out[f"{key}#len"] = len(v)
                else:
                    walk(v, key, depth + 1)
    for i, b in enumerate(dm.device_evoformer):
        walk(b, f"evo[{i}]", 0)
    return out


def diff(a, b):
    ks = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return {"n": len(ks), "keys": ks[:40]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(NPZ))
    ap.add_argument("--mode", default="none",
                    choices=["none", "noprog", "clearprog", "pad", "gc", "forget", "poison-nan", "poison-zero"])
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--depth", type=int, default=48)
    ap.add_argument("--first", type=int, default=0, help="first Evoformer block")
    ap.add_argument("--nomask", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--trace-blocks", action="store_true",
                    help="hash every block output of the forward, to find where two phases part")
    args = ap.parse_args()
    z = np.load(args.npz)
    lv = S.Levers(); dm, _ = A.load_models(A.DEFAULT_PARAMS)
    dev = A.Dev(dm.to_device()); lv.arm("stack")
    if args.mode == "noprog":
        dev.device.disable_and_clear_program_cache()
    m = torch.from_numpy(z["msa_leaf"]).float()
    p_ = torch.from_numpy(z["pair_leaf"]).float()
    mask = None if args.nomask else dev.up(torch.from_numpy(z["mask"]).float())
    n = int(z["n"])
    gm = torch.zeros(tuple(m.shape)); gz = torch.zeros(tuple(p_.shape))
    gm[:, :n] = torch.from_numpy(z["cot_msa"]).float()
    gz[:n, :n] = torch.from_numpy(z["cot_pair"]).float()

    held = []
    snap_prev = None
    rows = []
    for r in range(args.reps):
        if r:
            if args.mode == "clearprog":
                dev.device.clear_program_cache()
            elif args.mode.startswith("poison"):
                poison(dev, float("nan") if args.mode == "poison-nan" else 0.0)
            elif args.mode == "gc":
                gc.collect()
            elif args.mode == "pad" and r % 2:
                held.append(dev.up(torch.zeros(1, 32 * 37, 32)))
        print("START", r + 1, flush=True)
        t0 = time.time()
        ml, zl = dev.leaf(m), dev.leaf(p_)
        with dev.tt.tape():
            if args.trace_blocks:
                mo, zo, bh = ml, zl, []
                for i in range(args.first, args.first + args.depth):
                    mo, zo = dev.stack(mo, zo, 0, 1, evo_first=i, ckpt=True, msa_mask=mask)
                    bh.append(hashlib.sha256(ttnn_bytes(dev, mo) + ttnn_bytes(dev, zo)).hexdigest()[:8])
                print("BLOCKS", r + 1, " ".join(bh), flush=True)
            else:
                mo, zo = dev.stack(ml, zl, 0, args.depth, evo_first=args.first, ckpt=True,
                                   msa_mask=mask)
        dev.sync()
        fh = hashlib.sha256(dev.down(mo.value, tuple(m.shape)).numpy().tobytes()
                            + dev.down(zo.value, tuple(p_.shape)).numpy().tobytes()).hexdigest()[:16]
        snap_before = snapshot(dev.dm)
        dev.ag.backward([mo, zo], [dev.seed(gm, mo), dev.seed(gz, zo)])
        dev.sync()
        a = dev.grad(ml, tuple(m.shape)); b = dev.grad(zl, tuple(p_.shape))
        wrapped = len(dev.ag._WRAPPED)
        changed = diff(snap_prev, snapshot(dev.dm)) if snap_prev is not None else None
        snap_prev = snapshot(dev.dm)
        dev.ag.release_pins()
        if args.mode == "forget":
            dev.ag.forget_wrappers()
        del ml, zl, mo, zo
        fa, fb = torch.isfinite(a), torch.isfinite(b)
        h = hashlib.sha256(a.numpy().tobytes() + b.numpy().tobytes()).hexdigest()[:16]
        row = {"rep": r + 1, "digest": h, "clean": bool(fa.all() and fb.all()),
               "d_msa_nonfinite": int((~fa).sum()), "d_pair_nonfinite": int((~fb).sum()),
               "d_msa_absmax": float(a[fa].abs().max()) if fa.any() else None,
               "d_pair_absmax": float(b[fb].abs().max()) if fb.any() else None,
               "fwd_digest": fh, "changed_attrs": changed, "wrapped_after_bwd": wrapped, "progcache": int(dev.device.num_program_cache_entries()),
               "s": round(time.time() - t0, 1)}
        rows.append(row)
        print("REP", json.dumps(row), flush=True)
    out = {"mode": args.mode, "depth": args.depth, "first": args.first, "nomask": args.nomask,
           "reps": rows, "clean": sum(r["clean"] for r in rows),
           "distinct_digests": sorted({r["digest"] for r in rows}),
           "stamp": A.stamp(int(os.environ.get("TT_VISIBLE_DEVICES", -1)))}
    print("SUMMARY", json.dumps({k: out[k] for k in ("mode", "depth", "clean", "distinct_digests")}),
          flush=True)
    if args.out:
        (HERE / args.out).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
