#!/usr/bin/env python3
"""One warm PXDesign generator cell for the 512 aa perf page: seconds per design.

The page's design rows (BoltzGen, RFdiffusion3) are a warm in-process per-design median with
the cold design dropped. Neither existing PXDesign harness measures that: `perf_regression.py
--model pxdesign` times whole CLI subprocesses at 196 tokens, so every reading carries a
checkpoint load and an interpreter start, and `perf/pxdesign/tt_pxd_generator_bench.py` fits
the sampler on synthetic conditioning with no featurisation and no CIF write. So this file is
`scripts/release_gate.py:run_pxdesign` in a loop: the same five imports, the same float64
narrowing, the same three calls, with the checkpoint load hoisted out of the timed region.

The region is featurise + generate + write, which is exactly `gen_feat_s + gen_device_s +
gen_write_s = gen_total_s` on the H200 reference (`perf/pxdesign/gpu_reference.json`), and
excludes `model_init_s` on both sides. Batch is one design per call: `n_sample` is the number
of backbones out of ONE diffusion trajectory, so `--num_designs 4` is batch 4, not four
designs at batch 1.

Round 0 is cold and is excluded from every statistic (first-use kernel compile). Seeds are
[0, 1, 2, 3, 0] so the last warm round repeats the cold round's seed: its coordinate digest
must match, which is the run-to-run determinism check.
"""
import argparse, hashlib, json, os, statistics, subprocess, sys, time
from pathlib import Path


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _loadavg():
    return open("/proc/loadavg").read().split()[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True, help="source tree to import tt_bio from")
    ap.add_argument("--yaml", type=Path, required=True, help="PXDesign target YAML")
    ap.add_argument("--n_step", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--label", default="px")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    tree = a.tree.resolve()
    sys.path.insert(0, str(tree))

    import torch
    import tt_bio.tenstorrent as T
    from tt_bio.pxdesign.inputs import design_inputs_from_yaml
    from tt_bio.pxdesign.model import ProtenixDesign
    from tt_bio.pxdesign.write import write_design_cifs
    from tt_bio.main import ensure_pxdesign_weights, ensure_p300_mesh_descriptor
    assert Path(T.__file__).resolve().is_relative_to(tree), \
        f"tt_bio came from {T.__file__}, not {tree}"

    torch.set_grad_enabled(False)
    import importlib.metadata as im
    head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()

    spec = a.yaml.resolve()
    cif = None
    import yaml as _y
    tf = _y.safe_load(spec.read_text())["target"]["file"]
    cand = (spec.parent / tf).resolve()
    cif = cand if cand.exists() else Path(tf)

    res = {"label": a.label, "model": "pxdesign", "tree": str(tree), "git_head": head,
           "yaml": str(spec), "yaml_sha256": _sha(spec),
           "target_cif": str(cif), "target_cif_sha256": _sha(cif) if cif.exists() else None,
           "n_step": a.n_step, "n_sample_per_call": 1, "rounds": a.rounds,
           "ttnn": im.version("ttnn"), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "loadavg_start": _loadavg(), "designs": []}

    def flush():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))

    # Checkpoint load, once, outside every timed region. `ensure_p300_mesh_descriptor` before it,
    # because it sets the env the device open reads.
    ckpt = ensure_pxdesign_weights(Path(os.path.expanduser("~/.boltz")))
    ensure_p300_mesh_descriptor()
    t0 = time.monotonic()
    model = ProtenixDesign.load_from_checkpoint(str(ckpt))
    res["model_init_s"] = round(time.monotonic() - t0, 3)
    res["checkpoint"] = str(ckpt)
    res["checkpoint_bytes"] = Path(ckpt).stat().st_size
    # Read after the checkpoint load: the grid thresholds are applied at device open, not import.
    res["arch"] = T.arch_name()
    res["grid"] = [int(T.COMPUTE_GRID_MAIN[0]), int(T.COMPUTE_GRID_MAIN[1])]
    # The shipped default is fp32 diffusion (model.py:82, env_flag default True); the H200
    # reference ran bf16. Record what this process resolved rather than restating the default.
    from tt_bio.pxdesign.model import env_flag as _ef
    res["diffusion_fp32"] = bool(_ef("PROTENIX_DIFFUSION_FP32_DEVICE", True))
    res["env_PROTENIX_DIFFUSION_FP32_DEVICE"] = os.environ.get(
        "PROTENIX_DIFFUSION_FP32_DEVICE")
    flush()

    outdir = a.out.parent / f"designs_{a.label}"
    seeds = [0, 1, 2, 3, 0][:a.rounds] if a.rounds <= 5 else \
        [0] + list(range(1, a.rounds - 1)) + [0]

    for i, seed in enumerate(seeds):
        la = _loadavg()
        t = time.monotonic()
        feats = design_inputs_from_yaml(spec)
        feats = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float64 else v)
                 for k, v in feats.items()}
        t_feat = time.monotonic() - t

        n_token = int(feats["restype"].shape[0])
        t = time.monotonic()
        coords = model.design(feats, n_step=a.n_step, n_sample=1, seed=seed)
        t_design = time.monotonic() - t

        t = time.monotonic()
        rows = write_design_cifs(coords, feats, outdir / f"r{i}", stem="laczc_512")
        t_write = time.monotonic() - t

        from tt_bio.pxdesign.write import BINDER_RESTYPE
        _atoms = [l for l in Path(rows[0]["cif"]).read_text().splitlines()
                  if l.startswith("ATOM ")]
        binder_tok = int((feats["restype"].argmax(-1) == BINDER_RESTYPE).sum())
        rec = {"round": i, "seed": seed, "cold": i == 0,
               "t_feat_s": round(t_feat, 3), "t_design_s": round(t_design, 3),
               "t_write_s": round(t_write, 3),
               "round_s": round(t_feat + t_design + t_write, 3),
               "n_token": n_token, "binder_tokens": binder_tok,
               "target_tokens": n_token - binder_tok,
               "n_sample": int(coords.shape[0]), "n_atom": int(coords.shape[1]),
               "coords_finite": bool(torch.isfinite(coords).all()),
               "coord_sha16": hashlib.sha256(
                   coords.contiguous().numpy().tobytes()).hexdigest()[:16],
               "binder_residues": rows[0]["binder_residues"],
               "binder_atoms": rows[0]["binder_atoms"],
               "conditioned_tokens": rows[0]["conditioned_tokens"],
               "fit_rmsd": round(rows[0]["fit_rmsd"], 4),
               "cif_sha16": _sha(rows[0]["cif"])[:16],
               # write.py writes 18 _atom_site columns; auth_seq_id is field 15 and
               # auth_asym_id is 16. Only the binder is in the CIF by design (write.py:41),
               # so this counts binder residues, not the 512-residue target.
               "cif_residues": len({l.split()[15] for l in _atoms}),
               "cif_chains": sorted({l.split()[16] for l in _atoms}),
               "cif_atoms": len(_atoms),
               "loadavg": la}
        res["designs"].append(rec)
        print(f"[{a.label}] r{i} seed={seed} {rec['round_s']:.3f}s "
              f"(feat {rec['t_feat_s']:.3f} design {rec['t_design_s']:.3f} "
              f"write {rec['t_write_s']:.3f}) rmsd={rec['fit_rmsd']} "
              f"digest={rec['coord_sha16']}{' COLD' if i == 0 else ''}", flush=True)
        flush()

    warm = [d for d in res["designs"] if not d["cold"]]
    if warm:
        w = sorted(d["round_s"] for d in warm)
        res["warm_n"] = len(w)
        res["warm_median_s"] = round(statistics.median(w), 3)
        res["warm_min_s"], res["warm_max_s"] = w[0], w[-1]
        res["warm_spread_pct"] = round((w[-1] - w[0]) / statistics.median(w) * 100, 3)
        for k, f in (("t_feat_s", "feat"), ("t_design_s", "design"), ("t_write_s", "write")):
            res[f"warm_median_{f}_s"] = round(statistics.median([d[k] for d in warm]), 3)
        res["warm_median_host_s"] = round(
            res["warm_median_feat_s"] + res["warm_median_write_s"], 3)
        # The residual the split leaves: the three leaves partition round_s by construction,
        # so this is a check on the arithmetic, not an unattributed term.
        res["split_residual_s"] = round(
            res["warm_median_s"] - res["warm_median_feat_s"]
            - res["warm_median_design_s"] - res["warm_median_write_s"], 4)
        print(f"[{a.label}] warm median {res['warm_median_s']:.3f}s over n={len(w)} "
              f"spread {res['warm_spread_pct']:.2f}% "
              f"host {res['warm_median_host_s']:.3f}s", flush=True)
    res["loadavg_end"] = _loadavg()
    flush()
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
