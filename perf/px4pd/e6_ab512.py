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
TR = {"l1": 0, "dram": 0}
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
    ap.add_argument("--instrument", type=int, default=0,
                    help="1 = split every fold into featurize / model.fold / CIF write. The "
                         "page's GPU leg times runner.predict + cuda.synchronize() only, so "
                         "this is what says whether the two legs share a boundary.")
    ap.add_argument("--wall", type=int, default=1,
                    help="0 = no per-module sync walls. The walls cost ~4.8 s/fold in BOTH arms "
                         "(tt-bio-isolated-op-timing-oversync-inflates-cost), so only --wall 0 "
                         "measures the shipped path a page cell may quote.")
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_bio.reblock_permute as RB
    import tt_baseline as B
    import ttnn as _tn

    _pt = T._pair_transpose

    def _counted(t, mc):
        TR["l1" if mc.buffer_type == _tn.BufferType.L1 else "dram"] += 1
        return _pt(t, mc)

    T._pair_transpose = _counted

    saved = []
    if a.wall:
        for cls, nm in ((T.TriangleMultiplication, "TriangleMultiplication"),
                        (T.TriangleAttention, "TriangleAttention")):
            f = cls.__call__
            saved.append((cls, f))
            cls.__call__ = (lambda g, n: lambda self, *x, **k: timed(f"body:{n}", g, self, *x, **k))(f, nm)

    fixdir = ROOT / "perf" / "size512" / "fixtures"
    tgt, a3m = fixdir / f"cdk2x2_{a.size}.yaml", fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold("protenix-v2", ROOT / f".msa_px4pd_{a.size}", tgt, a3m,
                                         instrument=bool(a.instrument))
    STATE["dev"] = T.get_device()
    STATE["model"] = state.model
    struct_dir = Path(meta["struct_dir"])

    tms = walk_trimuls(state.model, T.TriangleMultiplication)
    # As CONSTRUCTED, before this harness touches anything: this is the shipped default.
    shipped_default = sorted({bool(getattr(m, "gated_move", False)) for m in tms})
    import importlib.metadata as im, os, socket
    res = {"ttnn": im.version("ttnn"), "host": socket.gethostname(),
           "chip": os.environ.get("TT_VISIBLE_DEVICES", "?"), "size": a.size,
           "grid": list(T.COMPUTE_GRID_MAIN), "n_trimul_instances": len(tms), "wall": a.wall,
           "master_switch": RB._ENABLED_GATED,
           "shipped_default_gated_move": shipped_default,
           "timed_region": meta["timed_region"], "runs": []}
    print(f"trimul instances reachable: {len(tms)}, master gate {RB._ENABLED_GATED}, "
          f"shipped default gated_move {shipped_default}", flush=True)

    def set_arm(arm):
        """off = E6 off. on = E6 on. ontr = E6 on + the tight L1 transpose headroom.

        The transpose lever is the second unlanded lever from wk/integrated-ab-h200-gap: at
        the shipped 2.5 the 512 aa pair tensor (134.22 MB of 168.57 MB of grid L1) can never
        take the L1 route, so every pair transpose pays DRAM. 1.25 lets it in and
        `_pair_transpose`'s refusal fallback is what makes it safe.
        """
        RB.set_enabled_gated(True)      # master switch on in every arm; the instance flag is the arm
        for m in tms:
            m.gated_move = arm in ("on", "ontr")
        T._TRANSPOSE_L1_HEADROOM = 1.25 if arm == "ontr" else 2.5
        T._TRANSPOSE_L1_REFUSED.clear()
        TR["l1"] = TR["dram"] = 0
        RB.STATS_GATED[0] = RB.STATS_GATED[1] = 0
        RB.REJECTS.clear()

    set_arm("off")
    print("=== cold fold (discarded) ===", flush=True)
    cold_s, cold_m = one_fold()
    print(f"  cold {cold_s:.2f}s plddt={cold_m.get('plddt')}", flush=True)

    for arm in a.arms.split(","):
        set_arm(arm)
        WALL.clear()
        fold_s, m = one_fold()
        rec = {"arm": arm, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "n_tokens": m.get("n_tokens"), "cif_sha256": sha_dir(struct_dir),
               "transpose": {"headroom": T._TRANSPOSE_L1_HEADROOM, "l1": TR["l1"],
                             "dram": TR["dram"], "refused": len(T._TRANSPOSE_L1_REFUSED)},
               "gated": {"enabled": RB._ENABLED_GATED, "served": RB.STATS_GATED[0],
                         "declined": RB.STATS_GATED[1],
                         "rejects": {f"{k}": v for k, v in RB.REJECTS.items()}},
               "wall_ms": {k: {"calls": v["n"], "ms": round(v["s"] * 1e3, 2)}
                           for k, v in sorted(WALL.items())},
               "phases": (meta["phase_times"] or [None])[-1] if a.instrument else None}
        res["runs"].append(rec)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  {arm}: {fold_s:.3f}s plddt={rec['plddt']} sha={list(rec['cif_sha256'].values())} "
              f"gated served={rec['gated']['served']} declined={rec['gated']['declined']} "
              f"tr={rec['transpose']} "
              f"trimul={rec['wall_ms'].get('body:TriangleMultiplication')}", flush=True)


if __name__ == "__main__":
    main()
