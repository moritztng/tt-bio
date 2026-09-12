#!/usr/bin/env python3
"""Paired interleaved A/B of a 16-bit DST against the shipped 32-bit one, on a real
Boltz-2 512 aa PairformerLayer.

Arm A (shipped): fp32_dest_acc_en=True, so the dest register file holds 4 tiles and every
program config caps out_subblock_h * out_subblock_w at 4.
Arm B:           fp32_dest_acc_en=False, dest holds 8, subblock cap 8.

Both arms run in ONE process, alternating rep by rep, because a whglx chip answers exactly one
device open per reset (state/b2z/FINDINGS.md) and because the box is shared, so only a ratio
against its own A/A floor is worth anything. `_L1_OUT_RUNG` and the four lru-cached program
config factories are reset at every arm switch; leave either and the arms measure each other's
leftovers.

The block is grabbed the way ws:b2z-kernel-cycle-census grabs it: run the real fold, let the
stack settle, steal the second settled PairformerLayer call with its arguments cloned, abort
the fold. Nothing about the block is synthesised.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

FENCE_N, FENCE_DIM = 4, 2048
OUT: dict = {}
OUT_PATH: Path | None = None


class Grabbed(Exception):
    pass


def dump():
    if OUT_PATH:
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def loadavg():
    return [round(x, 2) for x in os.getloadavg()]


def make_fence(ttnn, dev):
    """A fixed op big enough that the queue is provably drained between timed reps."""
    import torch
    t = ttnn.from_torch(torch.zeros(1, 1, FENCE_DIM, FENCE_DIM), device=dev,
                        layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    def fence():
        for _ in range(FENCE_N):
            ttnn.exp(t)
        ttnn.synchronize_device(dev)
    return fence


def collect_ckcs(obj, seen=None, out=None, depth=0):
    """Every compute kernel config object reachable from a module, deduped by id."""
    if seen is None:
        seen, out = set(), []
    if depth > 6 or id(obj) in seen:
        return out
    seen.add(id(obj))
    if hasattr(obj, "fp32_dest_acc_en") and hasattr(obj, "math_fidelity"):
        out.append(obj)
        return out
    vals = []
    if hasattr(obj, "__dict__"):
        vals = list(vars(obj).values())
    elif isinstance(obj, (list, tuple)):
        vals = list(obj)
    elif isinstance(obj, dict):
        vals = list(obj.values())
    for v in vals:
        if v is None or isinstance(v, (int, float, str, bytes, bool)):
            continue
        collect_ckcs(v, seen, out, depth + 1)
    return out


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=7, help="timed reps per arm, per leg")
    ap.add_argument("--aa", type=int, default=5, help="reps for the A/A floor leg")
    a = ap.parse_args()
    OUT_PATH = a.out
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T

    OUT["env"] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "card": os.environ.get("TT_VISIBLE_DEVICES"),
                  "size": a.size, "reps": a.reps,
                  "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                  "tt_bio": T.__file__, "loadavg_start": loadavg()}
    dump()

    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    T.get_device(trace_region_size=1 << 29)          # whglx caps at 512 MiB, see CONTEXT
    fix = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, state = B.build_fold("boltz2", HERE / f".msa_{a.size}",
                                         fix / f"cdk2x2_{a.size}.yaml",
                                         fix / f"cdk2x2_{a.size}.a3m")
    OUT["env"].update({k: meta[k] for k in ("hardware", "grid", "card_type") if k in meta})
    dev = T.get_device()
    dump()

    grabs, counts = {}, {"n": 0}
    cls = T.PairformerLayer
    orig = cls.__dict__["__call__"]

    def clone(x):
        return ttnn.clone(x) if isinstance(x, ttnn.Tensor) else x

    def wrapper(self_obj, *args, **kw):
        counts["n"] += 1
        out = orig(self_obj, *args, **kw)
        if not grabs and counts["n"] >= 2 and getattr(self_obj, "transform_s", False):
            grabs["obj"] = self_obj
            grabs["args"] = tuple(clone(x) for x in args)
            grabs["kwargs"] = {k: clone(v) for k, v in kw.items()}
            print(f"  grabbed PairformerLayer on call {counts['n']}", flush=True)
            raise Grabbed
        return out
    cls.__call__ = wrapper

    print("=== precursor fold (aborted at the grab) ===", flush=True)
    t0 = time.perf_counter()
    try:
        one_fold()
    except Grabbed:
        pass
    finally:
        cls.__call__ = orig
    OUT["precursor_s"] = round(time.perf_counter() - t0, 3)
    dump()
    if not grabs:
        OUT["error"] = "PairformerLayer was never grabbed"
        dump()
        print("FAILED " + OUT["error"], flush=True)
        return 1

    obj, args, kwargs = grabs["obj"], grabs["args"], grabs["kwargs"]
    ckcs = collect_ckcs(obj)
    OUT["ckc_objects"] = len(ckcs)
    OUT["arg_shapes"] = [list(x.shape) if hasattr(x, "shape") else type(x).__name__ for x in args]
    print(f"  {len(ckcs)} compute-kernel-config objects reachable from the block", flush=True)
    dump()

    caches = [f for f in (getattr(T, n, None) for n in
                          ("_batched_matmul_search", "_pair_proj_program_config",
                           "_attn_value_program_config", "_qkv_l1_config"))
              if f is not None and hasattr(f, "cache_clear")]
    OUT["caches_cleared"] = len(caches)

    def set_arm(fp32_acc: bool):
        T._DST_SUBBLOCK_TILES = 4 if fp32_acc else 8
        for c in ckcs:
            c.fp32_dest_acc_en = fp32_acc
        for f in caches:
            f.cache_clear()
        T._L1_OUT_RUNG.clear()

    fence = make_fence(ttnn, dev)

    def one_rep():
        fence()
        t = time.perf_counter()
        out = obj(*args, **kwargs)
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t) * 1e3
        return ms, out

    def digest(out):
        """max|.| and the first few entries of the block's own output, for parity."""
        ts = [x for x in (out if isinstance(out, (list, tuple)) else [out])
              if isinstance(x, ttnn.Tensor)]
        return [torch.Tensor(ttnn.to_torch(x)).to(torch.float32) for x in ts]

    # --- warm both arms so neither pays a program compile inside a timed rep -----------
    ref = {}
    for name, acc in (("A", True), ("B", False)):
        set_arm(acc)
        for _ in range(2):
            _, out = one_rep()
        ref[name] = digest(out)
        print(f"  warmed arm {name} (fp32_dest_acc_en={acc})", flush=True)
    dump()

    # --- parity, block level ----------------------------------------------------------
    par = []
    for x, y in zip(ref["A"], ref["B"]):
        d = (x - y).abs()
        par.append({"shape": list(x.shape),
                    "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                    "ref_max_abs": float(x.abs().max()),
                    "rel_rms": float((d.pow(2).mean().sqrt() /
                                      x.pow(2).mean().sqrt().clamp(min=1e-30))),
                    "bit_exact": bool(torch.equal(x, y))})
    OUT["block_parity"] = par
    print("  block parity: " + json.dumps(par), flush=True)
    dump()

    def leg(tag, arms):
        """Interleave the given arms rep by rep and return per-arm sample lists."""
        samples = {k: [] for k, _ in arms}
        for i in range(a.reps if tag == "AB" else a.aa):
            for k, acc in arms:
                set_arm(acc)
                ms, _ = one_rep()
                samples[k].append(round(ms, 4))
            print(f"  {tag} rep {i + 1}: " +
                  " ".join(f"{k}={samples[k][-1]:.4f}" for k, _ in arms), flush=True)
            OUT[f"leg_{tag}"] = samples
            dump()
        return samples

    print("=== A/A floor leg ===", flush=True)
    aa = leg("AA", [("A0", True), ("A1", True)])
    print("=== A/B leg ===", flush=True)
    ab = leg("AB", [("A", True), ("B", False)])

    def med(v):
        return statistics.median(v)

    OUT["result"] = {
        "AA_ratio": round(med(aa["A0"]) / med(aa["A1"]), 5),
        "AA_median_ms": [round(med(aa["A0"]), 4), round(med(aa["A1"]), 4)],
        "A_median_ms": round(med(ab["A"]), 4),
        "B_median_ms": round(med(ab["B"]), 4),
        "AB_ratio": round(med(ab["A"]) / med(ab["B"]), 5),
        "A_spread_ms": [min(ab["A"]), max(ab["A"])],
        "B_spread_ms": [min(ab["B"]), max(ab["B"])],
        "loadavg_end": loadavg(),
    }
    print("RESULT " + json.dumps(OUT["result"]), flush=True)
    dump()
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
