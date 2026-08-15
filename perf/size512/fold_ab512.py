#!/usr/bin/env python3
"""Deliverable 1 — the ON/OFF fold A/B for the org's L1-capacity-gated wins, swept over size.

The question is whether C2FIX + X7 + X10 (1664.5 of the org's 1704.4 ms/fold, all behind a silent
`if it fits` branch) still pay at 512 aa. The analytic pass says all three take the DRAM branch at
512; this measures the fold, reads the branch each call actually took, and hashes the CIF so a null
result is provably a no-op rather than a broken run.

One process, one device context, several targets and several arms. The gate flags are module
globals read at call time, so an arm is a flag flip between folds -- no reload, no second device
open, and the weights and the MSA cache are shared, which is what makes a 4-arm sweep fit in a turn.

Arms:
  on   production defaults; every gate decides for itself.
  hifi4/hifi3/hifi2
       the trunk's matmul fidelity, capacity gates held at production defaults. The arm is a
       single in-place write to the trunk's own compute kernel config object, which the trunk
       threads to every submodule and which the diffusion module does not share -- so one
       variable moves and the fp32 diffusion boundary is provably untouched. Both fidelities
       are recorded per arm so the scoping is auditable from the JSON.
  off  the five capacity-gated wins forced off:
         _transpose_memory_config -> DRAM   (C2FIX)
         _PAIR_PROJ_L1_OUT = False          (X7 L1 output)
         _PAIR_BIAS_L1_NORM = False         (X7 L1 layer_norm source)
         _PWA_L1_NORM = False               (X10)
         _TEMPLATE_L1_NORM = False          (X10)
       `_PAIR_PROJ_BW` and `_NARROW_PROJ_BW` are identical in both arms: they are program config,
       not capacity, and moving them would change parity as well as the thing under test.

The headline instrument is the block wall, `PairformerLayer.__call__` synchronised on both sides and
summed over its executions, not the fold wall -- this harness's fold-wall A/A floor is 758.3 ms and
cannot resolve a null. Run the `on` arm twice to get this session's own A/A floor.

Results are written after every fold, so a turn that runs out of time still lands what it measured.
"""
import argparse, hashlib, json, statistics as st, sys, time
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
DEC = defaultdict(Counter)
STATE = {"dev": None, "gates": "on", "model": None}
FIDELITY = {"lofi": "LoFi", "hifi2": "HiFi2", "hifi3": "HiFi3", "hifi4": "HiFi4"}
# trimul in-projection group width (_TRIMUL_INPROJ_GROUP). g1 is the pre-pass-9 default.
GROUP = {"g1": 1, "g2": 2, "g4": 4, "g8": 8}
# trimul channel-move-back kernel (reblock_permute_back). Both arms hold the in-projection at
# the branch default of 8 so the only thing that moves is the back move.
BACK = {"g8nobk": False, "g8bk": True}
# E6: the four-way chunk and both input gates folded into the forward channel move. Both arms
# hold the in-projection at the branch default of 8, so the only thing that moves is the gate.
E6 = {"e6": True, "noe6": False}
# Tri-attention SDPA arms: (_SDPA_WIDE_Q, _TRIATT_BIAS_B8). narrowq is the pre-C1 shipped q_chunk,
# wideq is C1. `_tri_att_q_chunks` reads _SDPA_WIDE_Q from inside an lru_cache, so an arm flip that
# forgets the cache_clear silently runs an A/A pair and labels it an A/B -- see set_arm.
SDPA = {"narrowq": (False, False), "wideq": (True, False),
        "narrowq_b8": (False, True), "wideq_b8": (True, True)}
# Head-major qkv projection (tt_bio/triatt_qkv.py): the qkv matmul writes q, k and v itself instead
# of nlp_create_qkv_heads reordering them afterwards. `nohmqkv` is today's shipped path. Every other
# arm sets it explicitly to the production default so no arm inherits the previous one's setting.
HMQKV = {"hmqkv": True, "nohmqkv": False}
HMQKV_DEFAULT = True
# The tail half (K1b): the gate projection writes head-major and `out` reads head-major, so
# nlp_concat_heads never runs. It needs the qkv half on, so `hmtail` turns both on and `hmqkv`
# turns only the qkv half on -- the pair `hmqkv` vs `hmtail` isolates the tail exactly.
HMTAIL = {"hmtail": True, "hmtail_l1": True, "hmqkv": False, "nohmqkv": False}
HMTAIL_DEFAULT = True
# `hmtail` keeps the conservative gate, which declines any call whose `out` would have used the
# L1-output leg. `hmtail_l1` lets the head-major tail take those calls too, which is the only way
# to find out whether deleting nlp_concat_heads beats keeping `out`'s result in L1.
HMTAIL_OVER_L1 = {"hmtail_l1": True, "hmtail": False}
HMTAIL_OVER_L1_DEFAULT = True
# K2: the SDPA bias held in a permanently fronted CB (tt_bio/triatt_sdpa.py). `k2` turns it on on
# top of everything else; `nok2` is K1-complete, which is what it must be measured against.
PMASK = {"k2": True, "nok2": False, "hmtail_l1": False}
PMASK_DEFAULT = True
# F1: the trimul output tail's two projections and its gate in one generic_op. Bit-exact against
# the three ops it replaces, so `f1` must produce the same CIF sha as `nof1`; the instrument that
# resolves it is the TriangleMultiplication body wall, not the fold wall.
TAILF1 = {"f1": True, "nof1": False}


def timed_call(key, fn, *a, **kw):
    import ttnn
    ttnn.synchronize_device(STATE["dev"])
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    ttnn.synchronize_device(STATE["dev"])
    w = WALL[key]
    w["n"] += 1
    w["s"] += time.perf_counter() - t0
    return out


def sha_dir(d: Path):
    """sha256 of every structure file the fold just wrote, by name."""
    out = {}
    for p in sorted(Path(d).glob("*")):
        if p.is_file():
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protenix-v2", help="any model tt_baseline.build_fold loads")
    ap.add_argument("--sizes", default="512", help="comma list of fixture token counts")
    ap.add_argument("--arms", default="on,on,off", help="comma list, run in this order per size")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.protenix as P
    import tt_baseline as B

    import os
    from tt_bio.main import (_detect_p300_devices, _find_ttnn_mesh_graph_descriptor,
                             _resolve_recycling_steps, _resolve_sampling_steps)
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    if a.model == "boltz2":
        sys.path.insert(0, str(ROOT / "perf" / "other512"))
        import fold_ab_multi as _FAM
        _FAM.patch_boltz2_cfg()

    # ---- decision counters: read the branch actually taken, do not infer it from the shape ----
    ORIG_TMC = T._transpose_memory_config
    ORIG_LN = T._l1_layer_norm
    ORIG_PPC = T._pair_proj_config

    def is_l1(mc):
        return mc.buffer_type == ttnn.BufferType.L1

    def tmc(t):
        mc = ttnn.DRAM_MEMORY_CONFIG if STATE["gates"] == "off" else ORIG_TMC(t)
        DEC[f"transpose|{'x'.join(str(int(d)) for d in t.shape)}"]["L1" if is_l1(mc) else "DRAM"] += 1
        return mc

    def ln(x, headroom, **kw):
        out, in_l1 = ORIG_LN(x, headroom, **kw)
        DEC[f"l1_layer_norm|h={headroom}|{'x'.join(str(int(d)) for d in x.shape)}"][
            "L1" if in_l1 else "DRAM"] += 1
        return out, in_l1

    def ppc(x, w, bw_cap=-1, out_l1=False, **kw):
        cfg = ORIG_PPC(x, w, bw_cap=bw_cap, out_l1=out_l1, **kw)
        if out_l1:
            DEC[f"pair_proj_out_l1|{'x'.join(str(int(d)) for d in x.shape)}"
                f"@{int(list(w.shape)[-1])}"]["L1" if cfg is not None else "DRAM"] += 1
        return cfg

    T._transpose_memory_config = tmc
    T._l1_layer_norm = ln
    P._l1_layer_norm = ln          # protenix.py imports it by name, so patch both namespaces
    T._pair_proj_config = ppc

    SDPA_DEFAULT = (T._SDPA_WIDE_Q, T._TRIATT_BIAS_B8)

    # Record the q_chunk the tri-att SDPA actually ran at, per call. The candidate list is not the
    # answer: a candidate that overflows L1 is dropped into _SDPA_Q_CHUNK_OVER_L1 and the next one
    # runs, so the branch has to be read rather than inferred from the flag.
    ORIG_TAS = T._tri_att_sdpa

    def tas(qq, kk, vv, bias, scale):
        ql, kl = int(qq.shape[2]), int(kk.shape[2])
        fits = [c for c in T._tri_att_q_chunks(ql, kl)
                if (ql, kl, c) not in T._SDPA_Q_CHUNK_OVER_L1]
        DEC[f"tri_att_sdpa|q{ql}k{kl}"][
            f"q_chunk={fits[0]} k_chunk={T._sdpa_chunks_shipped(ql, kl)[1]} "
            f"bias={'bfp8' if T._TRIATT_BIAS_B8 else 'bf16'}"] += 1
        return ORIG_TAS(qq, kk, vv, bias, scale)

    T._tri_att_sdpa = tas

    saved = []
    for cls in (T.Pairformer,):
        f = cls.__call__
        saved.append((cls, f))
        cls.__call__ = (lambda g: lambda self, *x, **k: timed_call("stage:Pairformer", g, self, *x, **k))(f)
    for cls in (T.PairformerLayer,):
        f = cls.__call__
        saved.append((cls, f))
        cls.__call__ = (lambda g: lambda self, *x, **k: timed_call("block:PairformerLayer", g, self, *x, **k))(f)
    for cls in (T.TriangleMultiplication, T.TriangleAttention, T.AttentionPairBias,
                T.PairWeightedAveraging):
        f = cls.__call__
        saved.append((cls, f))
        cls.__call__ = (lambda g, nm: lambda self, *x, **k: timed_call(f"body:{nm}", g, self, *x, **k))(f, cls.__name__)

    def set_arm(name):
        # A fidelity arm holds the capacity gates at production defaults; the only thing that moves
        # is the trunk's own compute kernel config, written in place. So does an SDPA arm.
        fid = FIDELITY.get(name)
        grp = GROUP.get(name)
        bk = BACK.get(name)
        sdpa = SDPA.get(name)
        e6 = E6.get(name)
        f1 = TAILF1.get(name)
        hm = HMQKV.get(name)
        if name in ("hmtail", "hmtail_l1", "k2", "nok2"):
            hm = True
        if name in ("k2", "nok2"):
            hmt = True
        hmt = HMTAIL.get(name)
        # `prev` reverts the two extracted engine fixes (the `_MM_BLOCK[8]` block config and the
        # pair-projection `minimal_matmul` leg) and holds every capacity gate at the production
        # default, so the arm difference is exactly those two and nothing else.
        prev = name == "prev"
        T._PAIR_PROJ_MM = not prev
        T._MM_BLOCK[(8, 8)] = (2, 8, 1, 2, 1) if prev else (4, 8, 1, 4, 1)
        STATE["gates"] = ("on" if (fid or grp or bk is not None or sdpa or prev
                                   or e6 is not None or f1 is not None or hm is not None
                                   or hmt is not None)
                          else name)
        # Every arm sets the tail kernel, so a non-F1 arm provably runs the three shipped ops
        # rather than inheriting the previous arm's flag.
        T._TRIMUL_TAIL_F1 = bool(f1)
        import tt_bio.trimul_tail as F1MOD
        F1MOD.STATS[0] = F1MOD.STATS[1] = 0
        F1MOD.REJECTS.clear()
        # Every arm sets the SDPA flags, so an arm that is not an SDPA arm provably runs the
        # production pick rather than inheriting the previous arm's.
        T._SDPA_WIDE_Q, T._TRIATT_BIAS_B8 = sdpa if sdpa else SDPA_DEFAULT
        T._tri_att_q_chunks.cache_clear()
        import tt_bio.triatt_qkv as HM
        HM._ENABLED = HMQKV_DEFAULT if hm is None else hm
        HM._TAIL_ENABLED = HMTAIL_DEFAULT if hmt is None else hmt
        HM._TAIL_OVER_L1 = (True if name in ("k2", "nok2")
                            else HMTAIL_OVER_L1.get(name, HMTAIL_OVER_L1_DEFAULT))
        import tt_bio.triatt_sdpa as PM
        PM._ENABLED = PMASK.get(name, PMASK_DEFAULT)
        PM.STATS[0] = PM.STATS[1] = 0
        PM.REJECTS.clear()
        HM.STATS[0] = HM.STATS[1] = 0
        HM.TAIL_STATS[0] = HM.TAIL_STATS[1] = 0
        HM.REJECTS.clear()
        HM.TAIL_REJECTS.clear()
        on = STATE["gates"] == "on"
        T._PAIR_PROJ_L1_OUT = on
        T._PAIR_BIAS_L1_NORM = on
        T._PWA_L1_NORM = on
        T._TEMPLATE_L1_NORM = on
        T._pair_proj_program_config.cache_clear()
        T._L1_OUT_REFUSED.clear()
        if grp:
            # The weight cache is keyed on (chunk_size, group), so an arm flip builds the widened
            # weights once and never reloads the model.
            T._TRIMUL_INPROJ_GROUP = grp
        if bk is not None:
            import tt_bio.reblock_permute as RB
            T._TRIMUL_INPROJ_GROUP = 8
            RB.set_enabled_back(bk)
            RB.STATS_BACK[0] = RB.STATS_BACK[1] = 0
        # Every arm sets the fused gate, so a non-E6 arm provably runs without it rather than
        # inheriting the previous arm's state.
        import tt_bio.reblock_permute as _RB
        if e6 is not None:
            T._TRIMUL_INPROJ_GROUP = 8
        _RB.set_enabled_gated(bool(e6))
        _RB.STATS_GATED[0] = _RB.STATS_GATED[1] = 0
        if fid:
            ckc = STATE["model"].trunk.compute_kernel_config
            ckc.math_fidelity = getattr(ttnn.MathFidelity, fid)
            assert str(ckc.math_fidelity).endswith(fid), ckc.math_fidelity

    def fidelities():
        m = STATE["model"]
        if m is None:
            return {}
        trunk = getattr(m, "trunk", None)
        out = {"trunk": str(trunk.compute_kernel_config.math_fidelity)} if trunk is not None else {}
        dit = getattr(getattr(m, "diffusion", None), "_dit_ckc", None)
        if dit is not None:
            out["diffusion"] = str(dit.math_fidelity)
        conf = getattr(getattr(m, "confidence_head", None), "compute_kernel_config", None)
        if conf is not None:
            out["confidence"] = str(conf.math_fidelity)
        return out

    import importlib.metadata as im
    import os, socket
    res = {"ttnn": im.version("ttnn"), "host": socket.gethostname(),
           "chip": os.environ.get("TT_VISIBLE_DEVICES", "?"),
           "note": "qb1 (tt-quietbox) is ttnn 0.67.4 and qb2 is 0.68.0 -- never put an absolute "
                   "from one in a table with the other; the arms in THIS file share a process",
           "model": a.model, "recycling_steps": B.RECYCLING_STEPS,
           "sampling_steps": B.SAMPLING_STEPS, "runs": []}

    for size in [int(s) for s in a.sizes.split(",")]:
        tgt = a.fixdir / f"cdk2x2_{size}.yaml"
        a3m = a.fixdir / f"cdk2x2_{size}.a3m"
        set_arm("on")
        one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_s512_{a.model}_{size}", tgt, a3m)
        STATE["dev"] = T.get_device()
        STATE["model"] = state.model
        struct_dir = Path(meta["struct_dir"])
        print(f"=== size {size}: cold fold ===", flush=True)
        try:
            cold_s, cold_m = one_fold()
        except Exception as e:                                                  # noqa: BLE001
            res["runs"].append({"size": size, "arm": "cold", "error": f"{type(e).__name__}: {e}"[:400]})
            a.out.write_text(json.dumps(res, indent=1))
            print(f"  COLD FOLD FAILED at {size}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        assert (a.model.startswith("esmfold2") or cold_m.get("msa")
            or (a.model == "boltz2" and meta.get("n_msa"))), "fold ran without an MSA"
        print(f"  cold {cold_s:.2f}s n_tokens={cold_m.get('n_tokens')} plddt={cold_m.get('plddt')}",
              flush=True)
        WALL.clear()
        DEC.clear()

        for arm in a.arms.split(","):
            set_arm(arm)
            WALL.clear()
            DEC.clear()
            t0 = time.perf_counter()
            try:
                fold_s, m = one_fold()
            except Exception as e:                                              # noqa: BLE001
                rec = {"size": size, "arm": arm, "error": f"{type(e).__name__}: {e}"[:400]}
                res["runs"].append(rec)
                a.out.write_text(json.dumps(res, indent=1))
                print(f"  {arm} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
                continue
            cifs = sha_dir(struct_dir)
            # Keep every arm's structure files: a sha mismatch is only diagnosable with the
            # files in hand, and identical shas make the copies free to delete.
            keep = a.out.parent / f"{a.out.stem}_cifs" / f"{size}_{arm}_{len(res['runs'])}"
            keep.mkdir(parents=True, exist_ok=True)
            for p in struct_dir.glob("*"):
                if p.is_file():
                    (keep / p.name).write_bytes(p.read_bytes())
            rec = {"size": size, "arm": arm, "fold_s": round(fold_s, 3),
                   "n_tokens": m.get("n_tokens"), "plddt": m.get("plddt"),
                   "cif_sha256": cifs,
                   "grid": list(T.COMPUTE_GRID_MAIN),
                   "pair_proj_mm": T._PAIR_PROJ_MM,
                   "mm_block_8": list(T._MM_BLOCK[(8, 8)]),
                   "trimul_inproj_group": T._TRIMUL_INPROJ_GROUP,
                   "back_kernel": (lambda RB: [RB._ENABLED_BACK, list(RB.STATS_BACK)])(
                       __import__("tt_bio.reblock_permute", fromlist=["x"])),
                   "gated_kernel": (lambda RB: [RB._ENABLED_GATED, list(RB.STATS_GATED)])(
                       __import__("tt_bio.reblock_permute", fromlist=["x"])),
                   "head_major_qkv": (lambda HM: {
                       "enabled": HM._ENABLED, "served": HM.STATS[0], "declined": HM.STATS[1],
                       "rejects": {f"{r}:{sh}": n for (r, sh), n in HM.REJECTS.items()},
                       "tail_enabled": HM._TAIL_ENABLED, "tail_over_l1": HM._TAIL_OVER_L1,
                       "tail_served": HM.TAIL_STATS[0],
                       "tail_declined": HM.TAIL_STATS[1],
                       "tail_rejects": {f"{r}:{sh}": n for (r, sh), n in HM.TAIL_REJECTS.items()}})(
                       __import__("tt_bio.triatt_qkv", fromlist=["x"])),
                   "persistent_mask": (lambda PM: {
                       "enabled": PM._ENABLED, "served": PM.STATS[0], "declined": PM.STATS[1],
                       "rejects": {f"{r}:{sh}": n for (r, sh), n in PM.REJECTS.items()}})(
                       __import__("tt_bio.triatt_sdpa", fromlist=["x"])),
                   "trimul_tail_f1": (lambda F1M: {
                       "enabled": T._TRIMUL_TAIL_F1, "served": F1M.STATS[0],
                       "declined": F1M.STATS[1],
                       "rejects": {f"{r}:{sh}": n for (r, sh), n in F1M.REJECTS.items()}})(
                       __import__("tt_bio.trimul_tail", fromlist=["x"])),
                   "sdpa_wide_q": T._SDPA_WIDE_Q,
                   "triatt_bias_b8": T._TRIATT_BIAS_B8,
                   "sdpa_q_chunk_over_l1": sorted(str(k) for k in T._SDPA_Q_CHUNK_OVER_L1),
                   "fidelity": fidelities(),
                   "loadavg": open("/proc/loadavg").read().split()[:3],
                   "wall_ms": {k: {"calls": v["n"], "ms": round(v["s"] * 1e3, 2)}
                               for k, v in sorted(WALL.items(), key=lambda kv: -kv[1]["s"])},
                   "decisions": {k: dict(v) for k, v in sorted(DEC.items())},
                   "l1_out_refused": [str(k) for k in T._L1_OUT_REFUSED]}
            blk = rec["wall_ms"].get("block:PairformerLayer", {})
            rec["block_wall_ms"] = blk.get("ms")
            rec["block_calls"] = blk.get("calls")
            ta = rec["wall_ms"].get("body:TriangleAttention", {})
            rec["triatt_wall_ms"] = ta.get("ms")
            rec["triatt_calls"] = ta.get("calls")
            res["runs"].append(rec)
            a.out.write_text(json.dumps(res, indent=1))
            print(f"  {arm}: fold {fold_s:.2f}s  block {blk.get('ms')} ms over {blk.get('calls')} "
                  f"calls  plddt {m.get('plddt')}  ({time.perf_counter()-t0:.0f}s)", flush=True)
            for k, v in sorted(DEC.items()):
                print(f"      DEC {k:52s} {dict(v)}", flush=True)

    # ---- per-size ON/OFF ratios ----------------------------------------------------------
    ratios = {}
    for size in {r["size"] for r in res["runs"] if "block_wall_ms" in r}:
        on = [r["block_wall_ms"] for r in res["runs"] if r["size"] == size and r["arm"] == "on"]
        off = [r["block_wall_ms"] for r in res["runs"] if r["size"] == size and r["arm"] == "off"]
        onf = [r["fold_s"] for r in res["runs"] if r["size"] == size and r["arm"] == "on"]
        offf = [r["fold_s"] for r in res["runs"] if r["size"] == size and r["arm"] == "off"]
        e = {"on_block_ms": on, "off_block_ms": off, "on_fold_s": onf, "off_fold_s": offf}
        if len(on) > 1:
            e["aa_floor_block_ms"] = round(abs(on[0] - on[1]), 2)
            e["aa_floor_fold_s"] = round(abs(onf[0] - onf[1]), 3)
        if on and off:
            e["off_over_on_block"] = round(st.median(off) / st.median(on), 4)
            e["off_minus_on_block_ms"] = round(st.median(off) - st.median(on), 2)
            e["off_over_on_fold"] = round(st.median(offf) / st.median(onf), 4)
        ratios[size] = e
    res["ratios"] = ratios
    # Per-arm medians, for arms that are not the on/off pair (fidelity sweeps). Report the median
    # over an arm's repeats and the arm-to-arm spread of the reference arm: a difference smaller
    # than that spread is not resolved (perfwar-single-shot-ab-resolution-floor).
    arms = {}
    for size in sorted({r["size"] for r in res["runs"] if "block_wall_ms" in r}):
        per = {}
        for r in res["runs"]:
            if r["size"] != size or "block_wall_ms" not in r:
                continue
            per.setdefault(r["arm"], {"block_ms": [], "fold_s": [], "plddt": []})
            per[r["arm"]]["block_ms"].append(r["block_wall_ms"])
            per[r["arm"]]["fold_s"].append(r["fold_s"])
            per[r["arm"]]["plddt"].append(r["plddt"])
        for arm, v in per.items():
            v["n"] = len(v["block_ms"])
            v["block_ms_median"] = round(st.median(v["block_ms"]), 2)
            v["fold_s_median"] = round(st.median(v["fold_s"]), 3)
            # The first fold of an arm pays the one-off JIT compile of that arm's SDPA program
            # config -- ~1.6 s, larger than the effect under test and always in the same direction.
            # min over repeats is the only summary immune to it; the raw list stays for audit.
            v["block_ms_min"] = round(min(v["block_ms"]), 2)
            v["fold_s_min"] = round(min(v["fold_s"]), 3)
            v["block_ms_spread"] = round(max(v["block_ms"]) - min(v["block_ms"]), 2)
            v["fold_s_spread"] = round(max(v["fold_s"]) - min(v["fold_s"]), 3)
        arms[size] = per
    res["arm_medians"] = arms
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"ratios": ratios, "arm_medians": arms}, indent=1), flush=True)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
