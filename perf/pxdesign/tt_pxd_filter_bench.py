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

Single sequence per chain: the filter fires with `use_msa=False` when it picks mini_tmpl, and
the reference measured the MSA axis at under 1% anyway. Template conditioning is NOT exercised
-- every PXDesign-pinned checkpoint ships a template embedder that is projections only, no
pairformer stack, so tt-bio derives 0 template blocks from it.

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
    conf = model.confidence_head.confidence(aux["s_inputs"], aux["s_trunk"], aux["z_trunk"],
                                            coords[0], feats)
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
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.protenix import Protenix
    from tt_bio.protenix_data import build_complex_features

    variant = "mini_tmpl" if a.cell == "filter" else "base"
    path = os.path.join(CKPT_DIR, VARIANTS[variant])
    chains = ([(seq(a.target), None), (seq(a.binder, phase=3), None)] if a.cell == "filter"
              else [(seq(a.target), None)])

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

    rec = dict(cell=a.cell, variant=variant, ckpt=os.path.basename(path),
               target_aa=a.target, binder_aa=(a.binder if a.cell == "filter" else 0),
               n_tokens=NT, n_atoms=N_atom, n_cycle=a.n_cycle, n_step=a.n_step,
               feat_host_s=t_feat, build_s=t_build,
               gates=gates(T, model.trunk, NT), reps=[], error=None,
               force_grid=os.environ.get("TT_BIO_FORCE_GRID"))
    print(json.dumps({k: rec[k] for k in ("cell", "variant", "n_tokens", "n_atoms", "gates")}),
          flush=True)

    for r in range(a.reps + 1):
        try:
            split, coords, conf = fold_split(model, feats, n_cycles=a.n_cycle,
                                             n_step=a.n_step, seed=r)
            split["cold"] = (r == 0)
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
    rec["clashes"] = {str(k): v for k, v in T._TRIMUL_CHUNK_CLASH.items()}
    rec["dram_shapes"] = sorted(str(x) for x in T._TRIMUL_DRAM_SHAPES)
    print(json.dumps({"clashes": rec["clashes"], "dram_shapes": rec["dram_shapes"]}), flush=True)
    out = a.out or os.path.join(os.path.dirname(__file__), f"tt_pxd_filter_{a.cell}.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
