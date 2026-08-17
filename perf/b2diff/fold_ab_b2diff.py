#!/usr/bin/env python3
"""Fold-level A/B for the three boltz-2 diffusion levers, one process, arms alternated.

Pre-registered in `state/boltz2-diffusion-perf.md` §10 before any of the three was built:

  l7   the per-layer bias slices cut once per fold instead of once per denoise step (bit-exact)
  l6   AdaLN's atom-path conditioning half memoised on the rollout-invariant `s`  (bit-exact)
  s6   the token DiT's attention routed through the fused ttnn SDPA               (NOT bit-exact)

`on` writes all three OFF, so it is main verbatim whatever the shipped defaults become. Integrated
arms (`l7l6`, `l7l6s6`) are folded and MEASURED; a sum or a product of the single-lever numbers is
never quoted. Every arm sets every flag, so no arm can inherit the previous one's state, and every
arm records its own served/declined counters -- a lever reporting 0 served is UNTESTED, not inert.

The region walls below carry the pre-committed kill rules, so they are timed on every arm with a
device sync at each end. That costs ~0.4 s of fold wall uniformly across arms; the delta is what
the kill rules read, and the instrumented `on` wall is recorded so nothing is compared against the
production 24.822 s by accident.
"""
import argparse, hashlib, json, os, sys, time
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "other512"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
STATS = Counter()
STATE = {"dev": None}
ARMS = ("on", "l7", "l6", "l7l6", "s6", "l6s6", "l7s6", "l7l6s6")


def timed(key, fn, *a, **kw):
    import ttnn
    ttnn.synchronize_device(STATE["dev"])
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    ttnn.synchronize_device(STATE["dev"])
    w = WALL[key]
    w["n"] += 1
    w["s"] += time.perf_counter() - t0
    return out


def sha_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(Path(d).glob("*")) if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512")
    ap.add_argument("--arms", default="on,on")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import (_resolve_recycling_steps, _resolve_sampling_steps,
                             _detect_p300_devices, _find_ttnn_mesh_graph_descriptor)
    from fold_ab_multi import patch_boltz2_cfg

    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, "boltz2")
    patch_boltz2_cfg()

    # ---- lever counters, read off the live fold rather than inferred from the flag -----------
    ORIG_HOIST = T.DiffusionModule._hoist_layer_bias

    def hoist(self, bias, transformer):
        out = ORIG_HOIST(self, bias, transformer)
        STATS["l7_served" if isinstance(out, (list, tuple)) else "l7_declined"] += 1
        return out

    T.DiffusionModule._hoist_layer_bias = hoist

    ORIG_STERMS = T.AdaLN.s_terms

    def s_terms(self, s, large_seq_len=False):
        out = ORIG_STERMS(self, s, large_seq_len)
        if self.atom_level:
            STATS["l6_hit" if out is self._s_memo else "l6_miss"] += 1
        return out

    T.AdaLN.s_terms = s_terms

    ORIG_APB = T.AttentionPairBias.__call__

    def apb(self, s, z, *x, **k):
        site = "atom" if self.atom_level else ("token" if self.token_dit else "trunk")
        if self.token_dit:
            STATS["s6_fused" if (T._B2_TOKEN_DIT_SDPA and z is not None) else "s6_unfused"] += 1
        return timed(f"body:AttentionPairBias|{site}", ORIG_APB, self, s, z, *x, **k)

    T.AttentionPairBias.__call__ = apb

    # ---- the five regions the pre-committed kill rules read ----------------------------------
    installed = ["body:AttentionPairBias|<site>"]

    def patch(cls, key_fn):
        f = cls.__call__
        cls.__call__ = lambda self, *x, **k: timed(key_fn(self), f, self, *x, **k)
        installed.append(key_fn.__doc__ or cls.__name__)

    def dt_key(self):
        """stage:DiffusionTransformer"""
        return "stage:DiffusionTransformer"

    def dtl_key(self):
        """block:DiffusionTransformerLayer|<atom|token>"""
        return "block:DiffusionTransformerLayer|" + ("atom" if self.atom_level else "token")

    def adaln_key(self):
        """body:AdaLN|<atom|token>"""
        return "body:AdaLN|" + ("atom" if self.atom_level else "token")

    def pfl_key(self):
        """block:PairformerLayer"""
        return "block:PairformerLayer"

    patch(T.DiffusionTransformer, dt_key)
    patch(T.DiffusionTransformerLayer, dtl_key)
    patch(T.AdaLN, adaln_key)
    patch(T.PairformerLayer, pfl_key)

    def set_arm(name):
        assert name in ARMS, f"unknown arm {name}"
        T._B2_BIAS_SLICE_HOIST = "l7" in name
        T._B2_ADALN_S_MEMO = "l6" in name
        T._B2_TOKEN_DIT_SDPA = "s6" in name
        STATS.clear()

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "model": "boltz2",
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "samples": a.samples, "timers_installed": installed, "runs": []}

    for size in [int(s) for s in a.sizes.split(",")]:
        tgt = a.fixdir / f"cdk2x2_{size}.yaml"
        a3m = a.fixdir / f"cdk2x2_{size}.a3m"
        set_arm("on")
        one_fold, meta, state = B.build_fold("boltz2", ROOT / f".msa_b2diff_{size}", tgt, a3m,
                                             samples=a.samples)
        STATE["dev"] = T.get_device()
        g = STATE["dev"].compute_with_storage_grid_size()
        res["grid"] = [g.x, g.y]
        struct_dir = Path(meta["struct_dir"])
        print(f"=== boltz2 {size} aa rec={B.RECYCLING_STEPS} steps={B.SAMPLING_STEPS} "
              f"samples={a.samples}: cold ===", flush=True)
        cold_s, cold_m = one_fold()
        print(f"  cold {cold_s:.2f}s n_tokens={cold_m.get('n_tokens')} "
              f"plddt={cold_m.get('plddt')}", flush=True)

        keep = Path(__file__).resolve().parent / f"ab{size}.arms"
        run_ix = Counter()
        for arm in a.arms.split(","):
            set_arm(arm)
            WALL.clear()
            try:
                fold_s, m = one_fold()
            except Exception as e:                                          # noqa: BLE001
                import traceback; traceback.print_exc()
                res["runs"].append({"size": size, "arm": arm,
                                    "error": f"{type(e).__name__}: {e}"[:600]})
                a.out.write_text(json.dumps(res, indent=1)); continue
            ix = run_ix[arm]; run_ix[arm] += 1
            dst = keep / f"{size}_{arm}_{ix}"
            dst.mkdir(parents=True, exist_ok=True)
            for f in Path(struct_dir).glob("*"):
                if f.is_file():
                    dst.joinpath(f.name).write_bytes(f.read_bytes())
            res["runs"].append({
                "size": size, "arm": arm, "ix": ix, "fold_s": round(fold_s, 3),
                "n_tokens": m.get("n_tokens"), "plddt": m.get("plddt"),
                "cif_sha256": sha_dir(struct_dir),
                "lever_stats": dict(STATS),
                "walls_ms": {k: round(v["s"] * 1e3, 1) for k, v in sorted(WALL.items())},
                "wall_calls": {k: v["n"] for k, v in sorted(WALL.items())},
            })
            a.out.write_text(json.dumps(res, indent=1))
            w = res["runs"][-1]["walls_ms"]
            print(f"  {arm:8s} #{ix} {fold_s:8.3f}s  plddt={m.get('plddt')}  "
                  f"DT={w.get('stage:DiffusionTransformer')}  "
                  f"DTLtok={w.get('block:DiffusionTransformerLayer|token')}  "
                  f"DTLatom={w.get('block:DiffusionTransformerLayer|atom')}  "
                  f"AdaLNatom={w.get('body:AdaLN|atom')}  "
                  f"APBtok={w.get('body:AttentionPairBias|token')}  "
                  f"PFL={w.get('block:PairformerLayer')}  {dict(STATS)}", flush=True)

    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}", flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
