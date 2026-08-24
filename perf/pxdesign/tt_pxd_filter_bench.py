"""The Protenix filter of the PXDesign pipeline, on Blackhole, at the PXDesign-pinned width.

`pxdesign-perf` pass 1 floored the generator (52.4% of the H200's device seconds) and left the
filter as the only unmeasured stage on the ported half: 10.9% of device seconds at the
saturating cell, 43.36 s, split tgt_template 9.55 s (one fold of the bare target) and the
8-design ptx pass 33.80 s.

tt-bio has run Protenix at c_z=256 and 384 and never at 128, and `pxdesign-port` pass 2's
`trunk_width_probe.py` stopped at N=384 while the saturating cell is 848 tokens -- past the
640-token boundary that `tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa` is about. So both
halves of the question are open at this shape: does it fit, and what does it cost.

Cells (shapes from the GPU reference's own fixtures and its runtime model choice):
  filter : 848 tokens = 768-residue target crop + 80-residue binder, mini_tmpl, one design
  probe  : 768 tokens = bare target,                                  base,     one fold
Settings are the reference's eval defaults: N_cycle=4, N_sample=1, N_step=2.

MSA depth is 1 by default (the filter fires with `use_msa=False` when it picks mini_tmpl);
`--msa-depth D` attaches a synthetic D-row a3m per chain to price the probe leg's real MSA.

Template conditioning IS exercised, contrary to pass 2's note. `build_complex_features` calls
`dummy_template_features(N)` unconditionally, so `template_aatype` is always present and nt is
4, not 0 -- the template embedder runs 4 slots x 4 cycles on every leg. What the pinned
checkpoints lack is a template PAIRFORMER STACK (n_template=0 blocks), which is a different
thing from the embedder being skipped; the projections still run. `shape_facts` records nt off
the tensors so this cannot be assumed again.

Rep 0 of every cell is a discarded warm-up. It pays the ttnn kernel JIT and the weight upload,
and `pxdesign-port` pass 2 already recorded a cold single-shot leg on this exact comparison
reporting the wrong SIGN.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")

CKPT_DIR = os.path.expanduser("~/pxdesign_release_data/checkpoint")
VARIANTS = {
    "base": "protenix_base_default_v0.5.0.pt",
    "mini_tmpl": "protenix_mini_tmpl_v0.5.0.pt",
}
# H200 device seconds at the saturating cell, from state/pxdesign-gpu-reference.md
H200 = {"probe": 9.55, "filter_per_design": 33.80 / 8.0}

_AA = "ARNDCQEGHILKMFPSTWYV"


def seq(n, phase=0):
    return "".join(_AA[(i * 7 + 13 + phase) % 20] for i in range(n))


def load(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ck = ck.get("model", ck)
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}


def gates(T, trunk, N):
    """Every width- or size-conditioned choice the trunk's pair path makes at this shape."""
    c_z = trunk.C_Z
    chunk = T._trimul_chunk_size(N, c_z, 1)
    return {
        "c_z": c_z,
        "trimul_chunk": chunk,
        "trimul_l1_max_seq": T._trimul_l1_max_seq(),
        "trimul_l1_resident": N <= T._trimul_l1_max_seq(),
        "trimul_inproj_group": T._trimul_inproj_group(N, chunk, 1, c_z // chunk),
        "n_tri_heads": c_z // trunk.TRI_HEAD_DIM,
        "grid": list(T.COMPUTE_GRID_MAIN),
        "n_pairformer": len(trunk.PF.layers) if hasattr(trunk.PF, "layers") else None,
        "n_msa": len(trunk.MSA),
        "n_template": len(trunk.TPL),
    }


def shape_facts(feats):
    """Template slots and MSA depth actually present in the features.

    `build_complex_features` calls `dummy_template_features(N)` unconditionally, so
    `template_aatype` is ALWAYS there and nt is 4, not 0: the template embedder runs on every
    leg. Read off the tensors rather than assumed, so a feature change shows up here."""
    return {
        "nt_template_slots": (int(feats["template_aatype"].shape[0])
                              if "template_aatype" in feats else 0),
        "msa_depth": int(feats["msa"].shape[0]) if "msa" in feats else 0,
    }


def a3m(query, depth, mutate=6):
    """An a3m of `depth` rows aligned to `query`. Row 0 is the query; each further row
    substitutes every `mutate`-th column with a rotated residue. Match columns only (no
    lowercase insertions), so every row is len(query) wide and the MSA is (depth, len(query)).
    The values are synthetic; the depth, width and op sequence are what the MSA module costs."""
    rows = [">q\n" + query]
    for r in range(1, depth):
        q = list(query)
        for i in range(r % mutate, len(q), mutate):
            q[i] = _AA[(_AA.index(q[i]) + r) % 20]
        rows.append(">h%d\n" % r + "".join(q))
    return "\n".join(rows) + "\n"


def fold_split(model, feats, *, n_cycles, n_step, seed):
    """fold()'s body with a sync-bracketed timer on each of its three device stages."""
    import ttnn
    from tt_bio.protenix import edm_sample

    dev = model.dev
    ttnn.synchronize_device(dev); t0 = time.time()
    cond, aux = model._trunk_cond(feats, n_cycles=n_cycles)
    ttnn.synchronize_device(dev); t1 = time.time()
    coords = edm_sample(model.diffusion, cond, aux["N"], n_step=n_step, seed=seed)
    ttnn.synchronize_device(dev); t2 = time.time()
    # Through fold()'s OWN dispatch, not the host head directly. Calling
    # `confidence_head.confidence(...)` here pins the host path whatever
    # TT_PROTENIX_CONF_DEVICE says, so a device-path A/B would come back a silent A/A
    # (`two-level-optin-ab-arm-and-page-provenance-drop`).
    conf = model._confidence_for(aux, feats, coords[0])
    ttnn.synchronize_device(dev); t3 = time.time()
    if isinstance(cond.get("dit_z"), ttnn.Tensor):
        ttnn.deallocate(cond["dit_z"])
    return dict(trunk=round(t1 - t0, 3), diffusion=round(t2 - t1, 3),
                confidence=round(t3 - t2, 3), total=round(t3 - t0, 3)), coords, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=["filter", "probe"])
    ap.add_argument("--target", type=int, default=768)
    ap.add_argument("--binder", type=int, default=80)
    ap.add_argument("--n-cycle", type=int, default=4)
    ap.add_argument("--n-step", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--msa-depth", type=int, default=1,
                    help="rows in the synthetic a3m per chain (1 = single sequence)")
    ap.add_argument("--conf-arms", default=None,
                    help="comma list of confidence arms to interleave in ONE process, e.g. "
                         "host,device. `device_confidence_enabled()` reads os.environ on every "
                         "call, so both arms run on the same weights, the same program cache and "
                         "the same box state, which removes the ~1%% cross-process drift pass 2 "
                         "measured on this cell. Omit for a single arm off the env.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.protenix import Protenix
    from tt_bio.protenix_data import build_complex_features

    variant = "mini_tmpl" if a.cell == "filter" else "base"
    path = os.path.join(CKPT_DIR, VARIANTS[variant])
    tseq, bseq = seq(a.target), seq(a.binder, phase=3)
    ta = a3m(tseq, a.msa_depth) if a.msa_depth > 1 else None
    ba = a3m(bseq, a.msa_depth) if a.msa_depth > 1 else None
    chains = ([(tseq, ta), (bseq, ba)] if a.cell == "filter" else [(tseq, ta)])

    t0 = time.time()
    feats = build_complex_features(chains)
    t_feat = round(time.time() - t0, 3)
    NT = int(feats["restype"].shape[0])
    N_atom = int(feats["ref_pos"].shape[0])

    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    sd = load(path)
    t0 = time.time()
    model = Protenix(sd, ckc, dev, gated_move=True)
    t_build = round(time.time() - t0, 3)
    del sd

    # The arm has to be asserted, not assumed. TT_PROTENIX_CONF_DEVICE=1 is only half the
    # opt-in: `device_confidence_enabled()` also feature-detects ttnn, and the call site adds
    # NT>=128. If the flag is set and the path is not actually live, this is not a null result,
    # it is a broken arm, so it stops here.
    def conf_arm_state():
        asked = os.environ.get("TT_PROTENIX_CONF_DEVICE", "0") in ("1", "true", "True")
        enabled = bool(model.confidence_head.device_confidence_enabled())
        active = enabled and NT >= 128
        if asked and not active:
            raise SystemExit(f"TT_PROTENIX_CONF_DEVICE=1 but the device path is not live "
                             f"(enabled={enabled}, NT={NT}). An A/A is not a null result.")
        return asked, active

    conf_asked, conf_active = conf_arm_state()

    rec = dict(cell=a.cell, variant=variant, ckpt=os.path.basename(path),
               conf_device_asked=conf_asked, conf_device_active=conf_active,
               target_aa=a.target, binder_aa=(a.binder if a.cell == "filter" else 0),
               n_tokens=NT, n_atoms=N_atom, n_cycle=a.n_cycle, n_step=a.n_step,
               feat_host_s=t_feat, build_s=t_build,
               gates=gates(T, model.trunk, NT), shape_facts=shape_facts(feats),
               msa_depth_arg=a.msa_depth, reps=[], error=None,
               force_grid=os.environ.get("TT_BIO_FORCE_GRID"))
    print(json.dumps({k: rec[k] for k in ("cell", "variant", "n_tokens", "n_atoms", "gates",
                                          "shape_facts", "conf_device_active")}), flush=True)

    arms = [x.strip() for x in a.conf_arms.split(",")] if a.conf_arms else [None]
    # Interleaved: warm-up, then arm0 arm1 arm0 arm1 ... so neither ordering nor program-cache
    # state can be mistaken for the effect (`perf-gate-single-shot-legs-recurring-false-alarm`).
    schedule = [arms[0]] + [arms[i % len(arms)] for i in range(a.reps * len(arms))]
    for r, arm in enumerate(schedule):
        try:
            if arm is not None:
                os.environ["TT_PROTENIX_CONF_DEVICE"] = "1" if arm == "device" else "0"
                _, act = conf_arm_state()
                assert act == (arm == "device"), f"arm {arm} did not take (active={act})"
            split, coords, conf = fold_split(model, feats, n_cycles=a.n_cycle,
                                             n_step=a.n_step, seed=r)
            split["cold"] = (r == 0)
            split["arm"] = arm
            split["conf_device_active"] = (arm == "device") if arm is not None else conf_active
            rg = float((coords[0] - coords[0].mean(0)).pow(2).sum(-1).mean().sqrt())
            split["rg"] = round(rg, 2)
            split["finite"] = bool(torch.isfinite(coords).all())
            split["plddt"] = round(float(conf["plddt"]), 4)
            rec["reps"].append(split)
            print(json.dumps(split), flush=True)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:600]
            print("ERROR", rec["error"], flush=True)
            break

    warm = [x for x in rec["reps"] if not x["cold"]]
    if a.conf_arms:
        rec["arms"] = {}
        for arm in arms:
            w = [x for x in warm if x["arm"] == arm]
            if not w:
                continue
            med = lambda k, w=w: sorted(x[k] for x in w)[len(w) // 2]
            rec["arms"][arm] = {
                "n": len(w),
                "median": {k: med(k) for k in ("trunk", "diffusion", "confidence", "total")},
                "conf_legs": sorted(x["confidence"] for x in w),
                "total_legs": sorted(x["total"] for x in w),
                "plddt": sorted({x["plddt"] for x in w}),
            }
        if "host" in rec["arms"] and "device" in rec["arms"]:
            h, d = rec["arms"]["host"], rec["arms"]["device"]
            rec["arms"]["delta"] = {
                "confidence_x": round(h["median"]["confidence"] / d["median"]["confidence"], 4),
                "total_x": round(h["median"]["total"] / d["median"]["total"], 4),
                "total_pct": round(100 * (h["median"]["total"] - d["median"]["total"])
                                   / h["median"]["total"], 3),
            }
        print(json.dumps(rec["arms"]), flush=True)
    if warm:
        med = lambda k: sorted(x[k] for x in warm)[len(warm) // 2]
        rec["warm_median"] = {k: med(k) for k in ("trunk", "diffusion", "confidence", "total")}
        spread = (max(x["total"] for x in warm) - min(x["total"] for x in warm)) / med("total")
        rec["warm_spread_pct"] = round(100 * spread, 2)
        h200 = H200["probe"] if a.cell == "probe" else H200["filter_per_design"]
        rec["h200_device_s"] = h200
        rec["ratio_vs_h200_device"] = round(rec["warm_median"]["total"] / h200, 3)
        rec["bar_4x_s"] = round(4 * h200, 2)
        print(json.dumps({k: rec[k] for k in ("warm_median", "warm_spread_pct",
                                              "h200_device_s", "ratio_vs_h200_device",
                                              "bar_4x_s")}), flush=True)
    # Lever census on the filter path. Pass 1 read these on the GENERATOR path and got
    # served=0 AND declined=0 everywhere, because that path has no trunk. The filter has one,
    # so this is the first time these counters can say anything, and `l1_refused` is where the
    # circular-buffer overflow at 848 tokens lands.
    import tt_bio.protenix as _P
    import tt_bio.triatt_qkv as _TQ
    import tt_bio.triatt_sdpa as _TS
    import tt_bio.trimul_tail as _TT
    import tt_bio.reblock_permute as _RB
    import tt_bio.mm_dualnoc as _DN
    # Every [served, declined] counter on the trunk's pair path, so a lever that is never
    # invoked is distinguishable from one that is invoked and refuses. The three pass 2 read
    # are not the whole set: TriMul E6 lives in reblock_permute.STATS_GATED, the fused output
    # tail in trimul_tail.STATS, and TriAtt K1/K2 in triatt_qkv.{STATS,TAIL_STATS}.
    rec["census"] = {
        "FP32_SOFTMAX_STATS": dict(T.FP32_SOFTMAX_STATS),
        "RELP_STATS": list(_P.RELP_STATS),
        "trimul_E6_gated_move": list(_RB.STATS_GATED),
        "reblock_fwd": list(_RB.STATS),
        "reblock_back": list(_RB.STATS_BACK),
        "trimul_tail_F1": list(_TT.STATS),
        "triatt_qkv_K2": list(_TQ.STATS),
        "triatt_tail_K1": list(_TQ.TAIL_STATS),
        "triatt_sdpa": list(_TS.STATS),
        "mm_dualnoc": list(_DN.STATS),
        # Renamed by the 349d7614 merge: the counter is keyed on the k chunk, not the q one.
        "sdpa_k_chunk": list(T.SDPA_K_CHUNK_STATS),
    }
    rec["rejects"] = {
        "trimul_tail_F1": {str(k): v for k, v in _TT.REJECTS.items()},
        "triatt_qkv_K2": {str(k): v for k, v in _TQ.REJECTS.items()},
        "triatt_tail_K1": {str(k): v for k, v in _TQ.TAIL_REJECTS.items()},
        "reblock": {str(k): v for k, v in _RB.REJECTS.items()},
        "triatt_sdpa": {str(k): v for k, v in _TS.REJECTS.items()},
    }
    # Renamed by the 349d7614 merge, and it is now a dict: the value is the reason the L1
    # geometry was retired, which is the whole point of reading it here.
    rec["fp32_softmax_l1_refused"] = {str(k): v for k, v in T._FP32_SOFTMAX_L1_REFUSALS.items()}
    # The other silent L1-output refusal on the pair path. It catches bare `Exception` and
    # records only a shape key -- no served/declined counter -- so a refusal here is invisible
    # to the census proper. Pass 2 attributed the 2005504 B circular-buffer TT_THROW at 848
    # tokens to the fp32-softmax shard; that path reports calls=0 on this stage, so it cannot
    # have been. This is where to look instead.
    rec["l1_out_refused_keys"] = sorted(str(k) for k in T._L1_OUT_REFUSED)
    rec["bmm_cfg_refused"] = sorted(str(k) for k in T._BMM_CFG_REFUSED)
    rec["sdpa_q_chunk_over_l1"] = sorted(str(k) for k in T._SDPA_Q_CHUNK_OVER_L1)
    print(json.dumps({"census": rec["census"],
                      "fp32_softmax_l1_refused": rec["fp32_softmax_l1_refused"],
                      "l1_out_refused_keys": rec["l1_out_refused_keys"],
                      "bmm_cfg_refused": rec["bmm_cfg_refused"],
                      "sdpa_q_chunk_over_l1": rec["sdpa_q_chunk_over_l1"]}), flush=True)
    print(json.dumps(rec["rejects"])[:1200], flush=True)
    rec["clashes"] = {str(k): v for k, v in T._TRIMUL_CHUNK_CLASH.items()}
    rec["dram_shapes"] = sorted(str(x) for x in T._TRIMUL_DRAM_SHAPES)
    print(json.dumps({"clashes": rec["clashes"], "dram_shapes": rec["dram_shapes"]}), flush=True)
    out = a.out or os.path.join(os.path.dirname(__file__), f"tt_pxd_filter_{a.cell}.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
