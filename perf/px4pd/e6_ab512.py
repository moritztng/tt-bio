#!/usr/bin/env python3
"""E6 (fused trimul chunk+gate forward move) on protenix-v2: the arm the shared harness cannot run.

`REBLOCK_PERMUTE_GATED` is True on main, but the fused move is a PER-INSTANCE opt-in
(`TriangleMultiplication.gated_move`). opendde.py and esmfold2.py pass `gated_move=True`;
protenix.py never does, at any of its five construction sites. `perf/size512/fold_ab512.py`'s
`e6` arm only flips the module-level master switch `reblock_permute._ENABLED_GATED`, so on
protenix-v2 that arm has always been an A/A -- which is exactly why every prior pass recorded
E6 as "dark" on this model.

This harness flips BOTH levels: the master switch and every live TriangleMultiplication
instance's `gated_move`. Arms are interleaved one fold at a time in one process, one device
context, so an A/A pair bounds the noise floor in the same session as the A/B.

Recorded per fold: wall, plDDT, CIF sha256, the fused kernel's served/declined census and its
reject reasons, and a synced module wall for TriangleMultiplication.
"""
import argparse, hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
STATE = {"dev": None, "model": None}


def timed(name, fn, self, *a, **k):
    import ttnn
    ttnn.synchronize_device(STATE["dev"])
    t0 = time.perf_counter()
    out = fn(self, *a, **k)
    ttnn.synchronize_device(STATE["dev"])
    w = WALL[name]
    w["n"] += 1
    w["s"] += time.perf_counter() - t0
    return out


def sha_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(Path(d).glob("*.cif"))}


def walk_trimuls(root, cls):
    """Every live TriangleMultiplication reachable from the model object."""
    seen, out, stack = set(), [], [root]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, cls):
            out.append(o)
        if isinstance(o, (list, tuple)):
            stack.extend(o)
        elif isinstance(o, dict):
            stack.extend(o.values())
        elif hasattr(o, "__dict__"):
            stack.extend(o.__dict__.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="off,on,off,on,off,on")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_bio.reblock_permute as RB
    import tt_baseline as B

    saved = []
    for cls, nm in ((T.TriangleMultiplication, "TriangleMultiplication"),
                    (T.TriangleAttention, "TriangleAttention")):
        f = cls.__call__
        saved.append((cls, f))
        cls.__call__ = (lambda g, n: lambda self, *x, **k: timed(f"body:{n}", g, self, *x, **k))(f, nm)

    fixdir = ROOT / "perf" / "size512" / "fixtures"
    tgt, a3m = fixdir / f"cdk2x2_{a.size}.yaml", fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold("protenix-v2", ROOT / f".msa_px4pd_{a.size}", tgt, a3m)
    STATE["dev"] = T.get_device()
    STATE["model"] = state.model
    struct_dir = Path(meta["struct_dir"])

    tms = walk_trimuls(state.model, T.TriangleMultiplication)
    import importlib.metadata as im, os, socket
    res = {"ttnn": im.version("ttnn"), "host": socket.gethostname(),
           "chip": os.environ.get("TT_VISIBLE_DEVICES", "?"), "size": a.size,
           "grid": list(T.COMPUTE_GRID_MAIN), "n_trimul_instances": len(tms),
           "master_switch": RB._ENABLED_GATED, "runs": []}
    print(f"trimul instances reachable: {len(tms)}, master gate {RB._ENABLED_GATED}", flush=True)

    def set_arm(on):
        RB.set_enabled_gated(True)      # master switch on in both arms; the instance flag is the arm
        for m in tms:
            m.gated_move = bool(on)
        RB.STATS_GATED[0] = RB.STATS_GATED[1] = 0
        RB.REJECTS.clear()

    set_arm(False)
    print("=== cold fold (discarded) ===", flush=True)
    cold_s, cold_m = one_fold()
    print(f"  cold {cold_s:.2f}s plddt={cold_m.get('plddt')}", flush=True)

    for arm in a.arms.split(","):
        set_arm(arm == "on")
        WALL.clear()
        fold_s, m = one_fold()
        rec = {"arm": arm, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "n_tokens": m.get("n_tokens"), "cif_sha256": sha_dir(struct_dir),
               "gated": {"enabled": RB._ENABLED_GATED, "served": RB.STATS_GATED[0],
                         "declined": RB.STATS_GATED[1],
                         "rejects": {f"{k}": v for k, v in RB.REJECTS.items()}},
               "wall_ms": {k: {"calls": v["n"], "ms": round(v["s"] * 1e3, 2)}
                           for k, v in sorted(WALL.items())}}
        res["runs"].append(rec)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  {arm}: {fold_s:.3f}s plddt={rec['plddt']} sha={list(rec['cif_sha256'].values())} "
              f"gated served={rec['gated']['served']} declined={rec['gated']['declined']} "
              f"trimul={rec['wall_ms'].get('body:TriangleMultiplication')}", flush=True)


if __name__ == "__main__":
    main()
