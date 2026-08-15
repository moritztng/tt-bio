#!/usr/bin/env python3
"""OpenDDE stage census: timers that cover the WHOLE fold, not just the Pairformer bodies.

`perf/other512/fold_ab_multi.py` patches `Pairformer`, `PairformerLayer` and the four bodies, which
is why 45 % of the OpenDDE fold has never been decomposed on either machine (state/wh-perf-opendde.md
section 3). This adds the missing stages and, crucially, a LEVEL-1 PARTITION that has to reconcile:

    fold  =  trunk_cond + expand_refine + diff_pair_cond + denoise + confidence + remainder

Every stage timer puts a `synchronize_device` on both sides of the call, so the wall this script
prints is INFLATED and is for attribution only -- the fold wall is `xmodel_ab.py`'s
(state/wh-perf-opendde.md section 7 A2, standing rule).

Level 2/3 keys are nested inside the level-1 ones and MUST NOT be added to them. Pairformer-family
keys carry their sequence length, so OpenDDE's residue trunk (S=512) and its structural-token
refiner (Ns=995 at 512 aa) separate without a second run.

The gate counters at the end are read off the live modules, so a gate that went dark at a size is a
measured fact rather than an inference.
"""
import argparse, hashlib, json, os, sys, time
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
GROUPS = Counter()
STATE = {"dev": None}


def timed(key, fn, *a, **kw):
    import ttnn
    ttnn.synchronize_device(STATE["dev"])
    t0 = time.perf_counter()
    try:
        return fn(*a, **kw)
    finally:
        ttnn.synchronize_device(STATE["dev"])
        d = WALL[key]
        d["n"] += 1
        d["s"] += time.perf_counter() - t0


def _seq_of(args):
    """The pair/single sequence length of a call, read off the first tensor argument.

    Used to split the shared `Pairformer` class between OpenDDE's residue trunk and its
    structural-token refiner without patching them separately -- they are the same object.
    """
    for a in args:
        sh = getattr(a, "shape", None)
        if sh is None or len(sh) < 2:
            continue
        d = [int(x) for x in sh]
        return max(d[:-1]) if len(d) > 2 else d[-2]
    return None


def patch_method(obj, name, key, *, by_seq=False):
    """Wrap one bound method. Returns True if it was there, so a missing name is a measured
    fact and not a silently absent timer."""
    cls = obj if isinstance(obj, type) else obj
    f = getattr(cls, name, None)
    if f is None or not callable(f):
        return False

    def wrapped(self, *x, _f=f, _k=key, **kw):
        k = _k
        if by_seq:
            s = _seq_of(x)
            if s is not None:
                k = f"{_k}|S={s}"
        return timed(k, _f, self, *x, **kw)

    setattr(cls, name, wrapped)
    return True


def sha_dir(d):
    h = hashlib.sha256()
    for p in sorted(Path(d).glob("*")):
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="opendde")
    ap.add_argument("--sizes", default="512")
    ap.add_argument("--repeat", type=int, default=1, help="warm folds per size (cold is discarded)")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--target", type=Path, default=None, help="explicit yaml (abag variants)")
    ap.add_argument("--a3m", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timers", default="on", choices=("on", "off"),
                    help="off installs no timers, so the wall is a CLEAN wall. The stage census "
                         "needs the timers and pays 2.6 %% for them; a size sweep or an abag wall "
                         "must not.")
    a = ap.parse_args()

    import tt_baseline as B
    import tt_bio.tenstorrent as T
    import tt_bio.protenix as P
    import tt_bio.opendde as OD

    # --- decision counters that only exist while the fold runs -------------------------------
    ORIG_GROUP = T._trimul_inproj_group

    def group_census(seq_len, chunk, batch, n_pairs):
        g = ORIG_GROUP(seq_len, chunk, batch, n_pairs)
        GROUPS[f"S={seq_len},n_pairs={n_pairs}->g={g}"] += 1
        return g

    T._trimul_inproj_group = group_census

    # --- stage timers -------------------------------------------------------------------------
    installed, missing = [], []

    def P_(obj, name, key, **kw):
        (installed if patch_method(obj, name, key, **kw) else missing).append(key)

    if a.timers == "off":
        installed, missing = [], ["ALL (--timers off)"]
        print("timers OFF: this is a clean wall", flush=True)
    # level 1: the partition the closure check reconciles against
    P_ = P_ if a.timers == "on" else (lambda *x, **k: None)
    P_(P.Protenix, "_trunk_cond", "L1:trunk_cond")
    P_(OD.OpenDDE, "expand_and_refine", "L1:expand_refine")
    P_(P.Protenix, "_diffusion_pair_cond", "L1:diff_pair_cond")
    P_(P.DiffusionModule, "denoise", "L1:denoise")
    P_(P.DiffusionModule, "_denoise_multiplicity", "L1:denoise_mult")
    P_(P.DiffusionModule, "denoise_traced", "L1:denoise_traced")
    P_(P.ConfidenceHead, "confidence", "L1:confidence")
    P_(P.ConfidenceHead, "confidence_device", "L1:confidence_dev")
    P_(P.ConfidenceHead, "plddt", "L1:plddt")

    # level 2: inside the trunk, and the expander seam
    P_(P.Trunk, "__call__", "L2:Trunk")
    P_(P.Trunk, "_msa", "L2:trunk_msa")
    P_(P.Trunk, "_template", "L2:trunk_template")
    P_(P.TrunkInput, "__call__", "L2:TrunkInput")
    P_(OD.StructuralTokenExpander, "__call__", "L2:expander")
    P_(P.DiffusionModule, "_atom_cond", "L2:atom_cond")

    # level 3: bodies. Pairformer-family keys carry S so trunk and refiner separate.
    P_(T.Pairformer, "__call__", "L3:Pairformer", by_seq=True)
    # Kept deliberately short. Every body timer costs two `synchronize_device` per call, and
    # `AttentionPairBias` alone is 5288 calls -- `OuterProductMean`, `MSALayer` and the atom/DiT
    # transformers are already covered by their L2/L1 parents, so patching them buys resolution
    # nobody has asked a question about and inflates the instrumented wall for everyone.
    for nm in ("TriangleMultiplication", "TriangleAttention", "AttentionPairBias",
               "PairWeightedAveraging", "Transition"):
        if a.timers == "off":
            break
        cls = getattr(T, nm, None) or getattr(P, nm, None)
        if cls is None:
            missing.append(f"L3:{nm}")
            continue
        # `Transition` is keyed by W as well: its 1538 calls are spread over the trunk pair
        # shape, the MSA stack and the S=995 refiner, and the row-chunk lever's landing is a
        # bound rather than a number until they are split.
        (installed if patch_method(cls, "__call__", f"L3:{nm}",
                                   by_seq=(nm in ("Pairformer", "Transition")))
         else missing).append(f"L3:{nm}")

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "model": a.model,
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "timers_installed": installed, "timers_missing": missing, "runs": []}
    print(f"timers installed {len(installed)}  missing {missing}", flush=True)

    sizes = [int(s) for s in a.sizes.split(",")] if a.target is None else [0]
    for size in sizes:
        tgt = a.target if a.target is not None else a.fixdir / f"cdk2x2_{size}.yaml"
        a3m = a.a3m if a.a3m is not None else a.fixdir / f"cdk2x2_{size}.a3m"
        tag = a.target.stem if a.target is not None else str(size)
        try:
            one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_cen_{tag}", tgt, a3m)
        except Exception as e:                                                  # noqa: BLE001
            import traceback; traceback.print_exc()
            res["runs"].append({"size": size, "tag": tag, "phase": "build",
                                "error": f"{type(e).__name__}: {e}"[:600]})
            a.out.write_text(json.dumps(res, indent=1)); continue
        STATE["dev"] = T.get_device()
        g = STATE["dev"].compute_with_storage_grid_size()
        res["grid"] = [g.x, g.y]
        struct_dir = Path(meta["struct_dir"])
        print(f"=== {a.model} {tag} rec={B.RECYCLING_STEPS} steps={B.SAMPLING_STEPS} "
              f"grid {g.x}x{g.y}: cold ===", flush=True)
        try:
            cold_s, cold_m = one_fold()
        except Exception as e:                                                  # noqa: BLE001
            import traceback; traceback.print_exc()
            res["runs"].append({"size": size, "tag": tag, "phase": "cold",
                                "error": f"{type(e).__name__}: {e}"[:600]})
            a.out.write_text(json.dumps(res, indent=1)); continue
        print(f"  cold {cold_s:.2f}s n_tokens={cold_m.get('n_tokens')} "
              f"plddt={cold_m.get('plddt')}", flush=True)

        for it in range(a.repeat):
            WALL.clear(); GROUPS.clear()
            try:
                fold_s, m = one_fold()
            except Exception as e:                                              # noqa: BLE001
                import traceback; traceback.print_exc()
                res["runs"].append({"size": size, "tag": tag, "iter": it,
                                    "error": f"{type(e).__name__}: {e}"[:600]})
                a.out.write_text(json.dumps(res, indent=1)); continue
            RB = __import__("tt_bio.reblock_permute", fromlist=["x"])
            HM = __import__("tt_bio.triatt_qkv", fromlist=["x"])
            PM = __import__("tt_bio.triatt_sdpa", fromlist=["x"])
            wall = {k: {"calls": v["n"], "ms": round(v["s"] * 1e3, 2)}
                    for k, v in sorted(WALL.items(), key=lambda kv: -kv[1]["s"])}
            l1 = {k: v for k, v in wall.items() if k.startswith("L1:")}
            l1_sum = sum(v["ms"] for v in l1.values())
            rec = {"size": size, "tag": tag, "iter": it,
                   "instrumented_fold_s": round(fold_s, 3),
                   "n_tokens": m.get("n_tokens"), "plddt": m.get("plddt"),
                   "cif_sha256": sha_dir(struct_dir),
                   "l1_partition_ms": l1, "l1_sum_ms": round(l1_sum, 2),
                   "l1_remainder_ms": round(fold_s * 1e3 - l1_sum, 2),
                   "l1_closure_pct": round(100.0 * l1_sum / (fold_s * 1e3), 2),
                   "trimul_inproj_groups": dict(GROUPS),
                   "reblock_fwd": [RB._ENABLED, list(RB.STATS)],
                   "gated_kernel": [RB._ENABLED_GATED, list(RB.STATS_GATED)],
                   "head_major_qkv": {"enabled": HM._ENABLED, "served": HM.STATS[0],
                                      "declined": HM.STATS[1],
                                      "rejects": {f"{r}:{sh}": n for (r, sh), n in HM.REJECTS.items()},
                                      "tail_served": HM.TAIL_STATS[0],
                                      "tail_declined": HM.TAIL_STATS[1]},
                   "persistent_mask": {"enabled": PM._ENABLED, "served": PM.STATS[0],
                                       "declined": PM.STATS[1],
                                       "rejects": {f"{r}:{sh}": n for (r, sh), n in PM.REJECTS.items()}},
                   "fp32_softmax_chain": dict(T.FP32_SOFTMAX_STATS),
                   "loadavg": open("/proc/loadavg").read().split()[:3],
                   "wall_ms": wall}
            res["runs"].append(rec)
            a.out.write_text(json.dumps(res, indent=1))
            print(f"  [{tag} it{it}] instrumented {fold_s:.2f}s  n_tokens={m.get('n_tokens')} "
                  f"plddt={m.get('plddt')} cif={rec['cif_sha256']}", flush=True)
            print(f"      L1 closure {rec['l1_closure_pct']:.1f}%  remainder "
                  f"{rec['l1_remainder_ms']/1000:.2f}s", flush=True)
            for k, v in sorted(l1.items(), key=lambda kv: -kv[1]["ms"]):
                print(f"      {k:24s} {v['ms']/1000:8.2f}s over {v['calls']:5d} calls", flush=True)
            for k, v in list(wall.items())[:22]:
                if k.startswith("L1:"):
                    continue
                print(f"        {k:34s} {v['ms']/1000:8.2f}s over {v['calls']:5d} calls", flush=True)
            print(f"      groups {dict(GROUPS)}", flush=True)
            print(f"      K1 {rec['head_major_qkv']['served']}/{rec['head_major_qkv']['declined']} "
                  f"K2 {rec['persistent_mask']['served']}/{rec['persistent_mask']['declined']} "
                  f"E6 {rec['gated_kernel']}", flush=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
