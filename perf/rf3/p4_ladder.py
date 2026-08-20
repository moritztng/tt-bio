#!/usr/bin/env python3
"""The two rungs pass 3 left unmeasured: 768 and 1024 aa, both new levers, one session.

Pass 3 shipped the confidence-head global-layer-norm row fold and the t-independent
rollout hoist default OFF for one reason: they were scored at 128/256/512 only, and the
hoist keeps the atom-pair track and both windowed bias stacks resident for the whole
rollout. This port's residency has broken at 1024 aa before. So the gate is not just the
ratio, it is whether the lever arm still fits, and both arms run off one checkpoint load
so the base arm is a live measurement rather than a quoted cell.

Peak DRAM is reported per arm where the runtime exposes it, because "it completed" and
"it has headroom" are different answers to the residency question.
"""
from __future__ import annotations

import argparse
import enum
import json
import statistics
import sys
import time
from pathlib import Path

if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self):
            return str(self.value)
    enum.StrEnum = StrEnum

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

ARMS = {
    "base": {"hoist": False, "gln": False},
    "hoist": {"hoist": True, "gln": False},
    "gln": {"hoist": False, "gln": True},
    "levers": {"hoist": True, "gln": True},
    "levers_opm": {"hoist": True, "gln": True, "opm": True},
    # Pass 5. `p4` is everything pass 4 recommends, which is the only honest base for a pass-5
    # lever: scoring against `base` would re-bank four levers that are already banked. `p4_aa` is
    # the same arm again at the end, so the A/A floor is measured on the fold and not assumed --
    # qb2's co-tenant load ran 15-28 through this pass and the effect here is 3-4 %.
    "p4": {"hoist": True, "gln": True, "opm": True, "hifi": True, "qkv": False},
    "p4_qkv": {"hoist": True, "gln": True, "opm": True, "hifi": True, "qkv": True},
    "p4_aa": {"hoist": True, "gln": True, "opm": True, "hifi": True, "qkv": False},
}


def dram_peak(device):
    """Peak DRAM bytes, if this runtime exposes the allocator. None if it does not."""
    for name in ("allocator_statistics", "get_memory_statistics"):
        fn = getattr(device, name, None)
        if fn is None:
            continue
        try:
            import ttnn
            st = fn(ttnn.BufferType.DRAM)
            for attr in ("total_allocated_bytes", "peak_allocated_bytes"):
                v = getattr(st, attr, None)
                if v is not None:
                    return {"stat": f"{name}.{attr}", "bytes": int(v)}
        except Exception:
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, default=768)
    ap.add_argument("--arms", default="base,levers")
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--n_recycles", type=int, default=10)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--diffusion_batch_size", type=int, default=1)
    ap.add_argument("--reps", type=int, default=2,
                    help="rep 0 discarded as cold, as every harness in this campaign does")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feat_cache", default="/home/ttuser/rf3_perf_work/featcache",
                    help="empty string disables; featurisation is host work and must not "
                         "run inside the box lock")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import ttnn
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.rf3 import confidence_head as rf3_conf
    from tt_bio import tenstorrent as tts
    from tt_bio import triatt_qkv
    from tt_bio.tenstorrent import get_device
    from perf.rf3.featcache import featurized
    from perf.rf3.tt_rf3_bench import PHASES, Timer, net_config, one_fold

    inp = str(REPO / f"perf/rf3/inputs/rf3_{args.aa}.json")
    t0 = time.perf_counter()
    fo = featurized(inp, n_recycles=args.n_recycles,
                    diffusion_batch_size=args.diffusion_batch_size, seed=args.seed,
                    cache_dir=args.feat_cache or None)
    featurize_s = time.perf_counter() - t0
    f = fo["feats"]
    rep_atom_idxs = fo.get("ground_truth", {}).get("rep_atom_idxs")
    print(f"[featurize] {featurize_s:.1f} s", flush=True)

    cfg = net_config(args.ckpt)
    device = get_device()
    kcfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    t0 = time.perf_counter()
    tt = rf3_model.load(
        args.ckpt, kcfg,
        n_pairformer_blocks=cfg["recycler"]["n_pairformer_blocks"],
        n_msa_blocks=cfg["recycler"]["msa_module"]["n_block"],
        n_dit_blocks=cfg["diffusion_module"]["diffusion_transformer"]["n_block"],
        num_timesteps=args.num_steps,
        with_confidence="confidence_head" in cfg)
    load_s = time.perf_counter() - t0
    print(f"[load] {load_s:.1f} s", flush=True)

    rungs = json.loads((REPO / "perf/rf3/gpu_reference.json").read_text())["rungs"]
    match = [r for r in rungs if r["rung_aa"] == args.aa
             and r["batch"] == args.diffusion_batch_size]
    target = match[0]["tt_target_device_s"] if match else None

    out: dict = {"aa": args.aa, "featurize_s": featurize_s, "ckpt_load_s": load_s,
                 "tt_target_device_s": target, "n_recycles": args.n_recycles,
                 "num_steps": args.num_steps,
                 "diffusion_batch_size": args.diffusion_batch_size, "arms": {}}

    for arm in args.arms.split(","):
        cfg_arm = ARMS[arm]
        rf3_model._HOIST_ROLLOUT = cfg_arm["hoist"]
        rf3_conf._GLN_ROW_FOLD = cfg_arm["gln"]
        tts._OPM_SMALL_DEPTH = cfg_arm.get("opm", False)
        # Absent keys leave the module default (and so the env flag) alone, so every pre-pass-5
        # arm above measures exactly what it measured before.
        if "hifi" in cfg_arm:
            tts._TRIATT_FUSED_HIFI = cfg_arm["hifi"]
        if "qkv" in cfg_arm:
            triatt_qkv._ENABLED = cfg_arm["qkv"]
        reps = []
        failed = None
        for rep in range(args.reps):
            tm = Timer(device)
            t0 = time.perf_counter()
            try:
                r = one_fold(tt, f, rep_atom_idxs, device, tm,
                             n_recycles=args.n_recycles,
                             diffusion_batch_size=args.diffusion_batch_size,
                             want_confidence=True, breakdown=False)
            except Exception as exc:   # OOM is the answer this rung is asking for
                failed = f"{type(exc).__name__}: {exc}"
                print(f"[{arm} rep {rep}] FAILED {failed}", flush=True)
                break
            rec = dict(tm.t)
            rec["infer_s"] = time.perf_counter() - t0
            rec["cold"] = rep == 0
            rec["finite"] = bool(torch.isfinite(r["coords"]).all())
            rec["coord_rms"] = float(r["coords"].pow(2).mean().sqrt())
            rec["plddt_logit_mean"] = r.get("plddt_logit_mean")
            reps.append(rec)
            print(f"[{arm} rep {rep}{' cold' if rep == 0 else ''}] "
                  + "  ".join(f"{k}={rec[k]:.3f}" for k in PHASES if k in rec)
                  + f"  infer={rec['infer_s']:.3f}  finite={rec['finite']}", flush=True)
        entry = {"flags": cfg_arm, "reps": reps, "failed": failed,
                 "dram_peak": dram_peak(device)}
        if reps:
            warm = [r for r in reps if not r["cold"]] or reps
            keys = [k for k in warm[0] if isinstance(warm[0][k], float)]
            entry["median_warm"] = {k: statistics.median([r[k] for r in warm])
                                   for k in keys}
            if target:
                entry["ratio_vs_target"] = entry["median_warm"]["infer_s"] / target
                print(f"[{arm}] {entry['median_warm']['infer_s']:.3f} s, "
                      f"{entry['ratio_vs_target']:.4f}x of target", flush=True)
        out["arms"][arm] = entry

    dst = Path(args.out or f"/home/ttuser/rf3_perf_work/res/p4_ladder_{args.aa}.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(f"wrote {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
