#!/usr/bin/env python3
"""Does the integrated campaign's win transfer to boltz2 --fast on examples/686.yaml?

Every number the integration campaign produced is protenix-v2 at 512 aa, bf16 trunk. boltz2
`--fast` runs a bfloat8_b trunk at 686 tokens, and the levers are tuned around bf16 L1 budgets and
tile widths, so a lever can fire, do nothing, or invert. This harness measures the fold and, more
importantly, records what every gate actually returned per call, so a null result is provably
"fired and flat" rather than "silently declined at this size and dtype".

One process, one device context, one model load, arms alternating. The gate flags are module
globals read at call time, so an arm is a flag flip between folds. The model is loaded once with
fast=True: `--fast` is a load-time property (the trunk weights are stored bfloat8_b), so it is not
an arm and every fold here is a fast fold.

Arms (the same four levers as the 512 aa integration A/B):
  main       shipped defaults: E6 gate off, K1/K2 on (merged at c9bfcaef8), transpose headroom 2.5
  int        all four on: E6 on, K1 on, K2 on, transpose headroom 1.25
  int_noe6 / int_nok1k2 / int_notr
             leave-one-out knockouts inside the integrated stack, which is the only way to
             attribute a share without multiplying ratios taken against different baselines
"""
import argparse, hashlib, json, statistics as st, sys, tempfile, time
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
DEC = defaultdict(Counter)
STATE = {"dev": None, "model": None}

INTEG = {
    "main":       {"e6": False, "k1": True,  "k2": True,  "tr": False},
    "int":        {"e6": True,  "k1": True,  "k2": True,  "tr": True},
    "int_noe6":   {"e6": False, "k1": True,  "k2": True,  "tr": True},
    "int_nok1k2": {"e6": True,  "k1": False, "k2": False, "tr": True},
    "int_notr":   {"e6": True,  "k1": True,  "k2": True,  "tr": False},
}
# `main` is what `tt-bio predict --model boltz2 --fast` runs today on origin/main c9bfcaef8: K1 and
# K2 ship default-ON since that commit, E6's gate ships False, and the transpose headroom ships 2.5.
# So the code delta between the two arms is the transpose lever alone; the other three are runtime
# gate flips, which is what makes a per-lever attribution possible in one process.


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
    out = {}
    for p in sorted(Path(d).glob("*")):
        if p.is_file():
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def build_fold(target: Path, msa_dir: Path, struct_dir: Path, recycles: int, steps: int,
               fast: bool = True):
    """Open the card, load boltz2 with fast=True, return (one_fold, meta, state).

    Modelled on scripts/gpu_vs_tt/tt_baseline.build_fold, which hardwires fast=False and
    protenix-v2's 10 recycles. Everything else is boltz2's own shipped default: 3 recycling
    steps (main._resolve_recycling_steps), 200 sampling steps, 1 diffusion sample, seed 0.
    """
    import torch
    torch.set_grad_enabled(False)
    from tt_bio.tenstorrent import get_device, arch_name
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E

    _noop = lambda *a, **k: None
    _E.set_progress(_noop)
    get_device()
    hw = arch_name()
    struct_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)

    # The two kwargs dicts boltz2 needs at load time, copied verbatim from tt_bio.main.predict's
    # CLI defaults so this harness folds through the same configuration `tt-bio predict --model
    # boltz2 --fast` builds: 64 Pairformer blocks / 16 heads / v2, no steering potentials,
    # kernels on, no trace, no MSA subsampling, max_parallel_samples 5.
    _diffusion = {"step_scale": 1.5, "gamma_0": 0.8, "gamma_min": 1.0, "noise_scale": 1.003,
                  "rho": 7, "sigma_min": 0.0001, "sigma_max": 160.0, "sigma_data": 16.0,
                  "P_mean": -1.2, "P_std": 1.5, "coordinate_augmentation": True,
                  "alignment_reverse_diff": True, "synchronize_sigmas": True}
    _pairformer = {"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True}
    _msa = {"subsample_msa": False, "num_subsampled_msa": 1024, "use_paired_feature": True,
            "msa_s": 64, "msa_blocks": 4, "msa_dropout": 0.15, "z_dropout": 0.25,
            "pairwise_head_width": 32, "pairwise_num_heads": 4,
            "activation_checkpointing": True}
    _steering = {"fk_steering": False, "physical_guidance_update": False,
                 "contact_guidance_update": True, "num_particles": 3, "fk_lambda": 4.0,
                 "fk_resampling_interval": 3, "num_gd_steps": 20}
    conf_kwargs = dict(
        predict_args={"recycling_steps": recycles, "sampling_steps": steps,
                      "diffusion_samples": 1, "max_parallel_samples": 5},
        diffusion_process_args=_diffusion, pairformer_args=_pairformer, msa_args=_msa,
        steering_args=_steering, use_kernels=True, use_tenstorrent=True, trace=False,
        diffusion_trace=False)
    aff_kwargs = dict(
        predict_args={"recycling_steps": 5, "sampling_steps": 200,
                      "diffusion_samples": 5, "max_parallel_samples": 1},
        diffusion_process_args=_diffusion, pairformer_args=_pairformer, msa_args=_msa,
        steering_args=dict(_steering, contact_guidance_update=False),
        affinity_mw_correction=False, use_tenstorrent=True, trace=False, diffusion_trace=False)

    cfg = dict(
        model="boltz2", fast=fast, output_format="cif",
        recycling_steps=recycles, sampling_steps=steps, diffusion_samples=1, seed=0, trace=False,
        conf_kwargs=conf_kwargs, aff_kwargs=aff_kwargs,
        msa_dir=str(msa_dir), struct_dir=str(struct_dir),
        use_msa_server=True, msa_db_path=None, use_envdb=False, msa_endpoint=None,
        single_sequence=False, msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy", msa_server_username=None, msa_server_password=None,
        api_key_value=None, max_msa_seqs=8192,
        write_pae=False, write_pde=False, write_embeddings=False, method=None,
    )
    _ensure_local_artifacts(cfg)
    state = _WorkerState("tenstorrent")
    t_load = time.perf_counter()
    state.load_model(cfg)
    load_s = time.perf_counter() - t_load
    state.bind_run("b2fast686", cfg)
    state.pfn = _noop
    job_cfg = dict(cfg)

    def one_fold():
        job_cfg["struct_dir"] = str(struct_dir)
        for p in struct_dir.glob("*"):
            p.unlink()
        t0 = time.perf_counter()
        metrics, _best, _feats = state.predict_one(target, job_cfg)
        return time.perf_counter() - t0, metrics

    return one_fold, dict(hardware=hw, load_s=round(load_s, 2), cfg=cfg), state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, default=ROOT / "examples" / "686.yaml")
    ap.add_argument("--msa-dir", type=Path, default=ROOT / ".msa_686")
    ap.add_argument("--arms", default="main,int,main,int,int")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--no-fast", dest="fast", action="store_false",
                    help="bf16 control arm: the same four levers with the trunk in bf16, "
                         "which is what separates 'boltz2 is different' from 'bf8 is different'")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.reblock_permute as RB
    import tt_bio.triatt_qkv as HM
    import tt_bio.triatt_sdpa as PM

    # ---- decision counters: read the branch actually taken, never infer it from the shape -----
    ORIG_TMC = T._transpose_memory_config
    ORIG_GRP = T._trimul_inproj_group
    ORIG_PPMM = T._pair_proj_minimal_matmul
    ORIG_QKV = T._qkv_mm_config
    ORIG_TAS = T._tri_att_sdpa
    ORIG_LN = T._l1_layer_norm

    def shp(t):
        return "x".join(str(int(d)) for d in t.shape)

    def tmc(t):
        mc = ORIG_TMC(t)
        # dtype in the key: the whole question is whether a bf16-tuned budget still fits under bf8.
        DEC[f"transpose|{shp(t)}|{t.dtype}"][
            "L1" if mc.buffer_type == ttnn.BufferType.L1 else "DRAM"] += 1
        return mc

    def grp_of(seq_len, chunk, batch, n_pairs):
        g = ORIG_GRP(seq_len, chunk, batch, n_pairs)
        DEC[f"trimul_inproj_group|n{seq_len}c{chunk}"][f"G={g}"] += 1
        return g

    def ppmm(x, w, ckc, dtype):
        out = ORIG_PPMM(x, w, ckc, dtype)
        DEC[f"pair_proj_minimal_matmul|{shp(x)}@{int(list(w.shape)[-1])}"][
            "minimal_matmul" if out is not None else "declined"] += 1
        return out

    def qkv(x, w):
        cfg = ORIG_QKV(x, w)
        nt = -(-int(list(w.shape)[-1]) // 32)
        DEC[f"qkv_mm_config|nt{nt}|{shp(x)}"][
            "None" if cfg is None else f"blk={T._MM_BLOCK.get(nt)}"] += 1
        return cfg

    def tas(qq, kk, vv, bias, scale):
        ql, kl = int(qq.shape[2]), int(kk.shape[2])
        fits = [c for c in T._tri_att_q_chunks(ql, kl)
                if (ql, kl, c) not in T._SDPA_Q_CHUNK_OVER_L1]
        DEC[f"tri_att_sdpa|q{ql}k{kl}|{qq.dtype}"][
            f"q_chunk={fits[0] if fits else '?'} "
            f"k_chunk={T._sdpa_chunks_shipped(ql, kl)[1]}"] += 1
        return ORIG_TAS(qq, kk, vv, bias, scale)

    def ln(x, headroom, **kw):
        out, in_l1 = ORIG_LN(x, headroom, **kw)
        DEC[f"l1_layer_norm|h={headroom}|{shp(x)}"]["L1" if in_l1 else "DRAM"] += 1
        return out, in_l1

    T._transpose_memory_config = tmc
    T._trimul_inproj_group = grp_of
    T._pair_proj_minimal_matmul = ppmm
    T._qkv_mm_config = qkv
    T._tri_att_sdpa = tas
    T._l1_layer_norm = ln

    for cls, key in ((T.Pairformer, "stage:Pairformer"),
                     (T.PairformerLayer, "block:PairformerLayer"),
                     (T.TriangleMultiplication, "body:TriangleMultiplication"),
                     (T.TriangleAttention, "body:TriangleAttention"),
                     (T.AttentionPairBias, "body:AttentionPairBias"),
                     (T.PairWeightedAveraging, "body:PairWeightedAveraging"),
                     (T.MSA, "stage:MSA"),
                     (T.Diffusion, "stage:Diffusion")):
        f = cls.__call__
        cls.__call__ = (lambda g, k: lambda self, *x, **kw: timed_call(k, g, self, *x, **kw))(f, key)

    def set_arm(name):
        """Every arm sets all four levers explicitly, so no arm can inherit the previous one's."""
        g = INTEG[name]
        RB.set_enabled_gated(g["e6"])
        RB.STATS_GATED[0] = RB.STATS_GATED[1] = 0
        RB.STATS_BACK[0] = RB.STATS_BACK[1] = 0
        HM._ENABLED = HM._TAIL_ENABLED = g["k1"]
        HM._TAIL_OVER_L1 = True
        HM.STATS[0] = HM.STATS[1] = 0
        HM.TAIL_STATS[0] = HM.TAIL_STATS[1] = 0
        HM.REJECTS.clear()
        HM.TAIL_REJECTS.clear()
        PM._ENABLED = g["k2"]
        PM.STATS[0] = PM.STATS[1] = 0
        PM.REJECTS.clear()
        T._TRANSPOSE_L1_HEADROOM = T.TRANSPOSE_L1_HEADROOM if g["tr"] else 2.5
        T._TRANSPOSE_L1_REFUSED.clear()
        T._L1_OUT_REFUSED.clear()
        # _tri_att_q_chunks reads _SDPA_WIDE_Q from inside an lru_cache; clearing it keeps an arm
        # flip from silently running an A/A pair and labelling it an A/B.
        T._tri_att_q_chunks.cache_clear()

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": "qb2", "model": "boltz2", "fast": a.fast,
           "target": str(a.target), "recycles": a.recycles, "steps": a.steps, "runs": []}

    struct_dir = Path(tempfile.mkdtemp(prefix="b2fast686-")) / "out"
    set_arm("main")
    one_fold, meta, state = build_fold(a.target, a.msa_dir, struct_dir, a.recycles, a.steps,
                                      fast=a.fast)
    STATE["dev"] = T.get_device()
    STATE["model"] = state.model
    res["meta"] = {k: v for k, v in meta.items() if k != "cfg"}
    res["cfg"] = {k: v for k, v in meta["cfg"].items() if k not in ("msa_dir", "struct_dir")}
    res["fast_mode_at_fold"] = T._FAST_MODE
    res["grid"] = list(T.COMPUTE_GRID_MAIN)
    a.out.write_text(json.dumps(res, indent=1))

    print("=== cold fold (discarded: first-kernel compile, and it seeds the MSA cache) ===",
          flush=True)
    t0 = time.perf_counter()
    cold_s, cold_m = one_fold()
    print(f"  cold {cold_s:.2f}s ({time.perf_counter()-t0:.0f}s wall)  n_tokens={cold_m.get('n_tokens')} "
          f"plddt={cold_m.get('plddt')} msa={cold_m.get('msa')}", flush=True)
    res["cold_fold_s"] = round(cold_s, 3)
    res["cold_plddt"] = cold_m.get("complex_plddt", cold_m.get("plddt"))
    res["n_tokens"] = cold_m.get("n_tokens")
    res["msa_rows"] = len(list(a.msa_dir.glob("*.a3m")))
    a.out.write_text(json.dumps(res, indent=1))

    for arm in a.arms.split(","):
        set_arm(arm)
        WALL.clear()
        DEC.clear()
        t0 = time.perf_counter()
        try:
            fold_s, m = one_fold()
        except Exception as e:                                                  # noqa: BLE001
            res["runs"].append({"arm": arm, "error": f"{type(e).__name__}: {e}"[:500]})
            a.out.write_text(json.dumps(res, indent=1))
            print(f"  {arm} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        cifs = sha_dir(struct_dir)
        keep = a.out.parent / f"{a.out.stem}_cifs" / f"{arm}_{len(res['runs'])}"
        keep.mkdir(parents=True, exist_ok=True)
        for p in struct_dir.glob("*"):
            if p.is_file():
                (keep / p.name).write_bytes(p.read_bytes())
        rec = {"arm": arm, "fold_s": round(fold_s, 3), "n_tokens": m.get("n_tokens"),
               "plddt": m.get("complex_plddt", m.get("plddt")), "metrics": {
                   k: v for k, v in m.items() if isinstance(v, (int, float, str, bool))},
               "cif_sha256": cifs,
               "fast_mode": T._FAST_MODE,
               "gated_kernel": [RB._ENABLED_GATED, list(RB.STATS_GATED)],
               "back_kernel": [RB._ENABLED_BACK, list(RB.STATS_BACK)],
               "head_major_qkv": {"enabled": HM._ENABLED, "served": HM.STATS[0],
                                  "declined": HM.STATS[1],
                                  "rejects": {f"{r}:{sh}": n for (r, sh), n in HM.REJECTS.items()},
                                  "tail_enabled": HM._TAIL_ENABLED,
                                  "tail_over_l1": HM._TAIL_OVER_L1,
                                  "tail_served": HM.TAIL_STATS[0],
                                  "tail_declined": HM.TAIL_STATS[1],
                                  "tail_rejects": {f"{r}:{sh}": n
                                                   for (r, sh), n in HM.TAIL_REJECTS.items()}},
               "persistent_mask": {"enabled": PM._ENABLED, "served": PM.STATS[0],
                                   "declined": PM.STATS[1],
                                   "rejects": {f"{r}:{sh}": n for (r, sh), n in PM.REJECTS.items()}},
               "transpose_l1_headroom": T._TRANSPOSE_L1_HEADROOM,
               "transpose_l1_refused": [str(k) for k in T._TRANSPOSE_L1_REFUSED],
               "trimul_inproj_group": T._TRIMUL_INPROJ_GROUP,
               "trimul_inproj_fused_bytes": T._TRIMUL_INPROJ_FUSED_BYTES,
               "pair_proj_mm": T._PAIR_PROJ_MM,
               "mm_block_8": list(T._MM_BLOCK[8]),
               "sdpa_wide_q": T._SDPA_WIDE_Q,
               "sdpa_q_chunk_over_l1": sorted(str(k) for k in T._SDPA_Q_CHUNK_OVER_L1),
               "l1_out_refused": [str(k) for k in T._L1_OUT_REFUSED],
               "loadavg": open("/proc/loadavg").read().split()[:3],
               "wall_ms": {k: {"calls": v["n"], "ms": round(v["s"] * 1e3, 2)}
                           for k, v in sorted(WALL.items(), key=lambda kv: -kv[1]["s"])},
               "decisions": {k: dict(v) for k, v in sorted(DEC.items())}}
        blk = rec["wall_ms"].get("block:PairformerLayer", {})
        rec["block_wall_ms"] = blk.get("ms")
        rec["block_calls"] = blk.get("calls")
        res["runs"].append(rec)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  {arm}: fold {fold_s:.2f}s  block {blk.get('ms')} ms over {blk.get('calls')} calls"
              f"  plddt {m.get('plddt')}  ({time.perf_counter()-t0:.0f}s)", flush=True)
        for k, v in sorted(DEC.items()):
            print(f"      DEC {k:56s} {dict(v)}", flush=True)

    per = {}
    for r in res["runs"]:
        if "fold_s" not in r:
            continue
        per.setdefault(r["arm"], {"fold_s": [], "block_ms": [], "plddt": []})
        per[r["arm"]]["fold_s"].append(r["fold_s"])
        per[r["arm"]]["block_ms"].append(r["block_wall_ms"])
        per[r["arm"]]["plddt"].append(r["plddt"])
    summ = {}
    for arm, d in per.items():
        e = {"n": len(d["fold_s"]), "fold_median_s": round(st.median(d["fold_s"]), 3),
             "fold_s": d["fold_s"], "plddt": d["plddt"]}
        if all(b is not None for b in d["block_ms"]) and d["block_ms"]:
            e["block_median_ms"] = round(st.median(d["block_ms"]), 2)
        if len(d["fold_s"]) > 1:
            e["spread_s"] = round(max(d["fold_s"]) - min(d["fold_s"]), 3)
        summ[arm] = e
    if "main" in summ and "int" in summ:
        summ["int_over_main"] = round(summ["main"]["fold_median_s"] / summ["int"]["fold_median_s"], 4)
        if "block_median_ms" in summ["main"] and "block_median_ms" in summ["int"]:
            summ["int_over_main_block"] = round(
                summ["main"]["block_median_ms"] / summ["int"]["block_median_ms"], 4)
    res["summary"] = summ
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
