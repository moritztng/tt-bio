#!/usr/bin/env python3
"""Per-step device cost of a diffusion designer, timed inside the process.

Timing a design CLI from outside measures kernel compilation and weight load as much as it
measures steps: run 1 pays a cold ttnn cache and run 3 does not, so a two-point
"wall(2N) - wall(N)" difference can come out negative (it did, on this Galaxy, 2026-09-11).
This probe instead wraps the diffusion module's own denoise call and timestamps every
invocation, so the answer is a distribution over steps in ONE trajectory: step 0 carries the
compile, the rest are warm, and the trunk shows up as the gap before the first step.

Nothing in tt_bio is modified or needed for this -- the wrap is around the bound method on the
loaded model, so the same probe fits any model whose sampler calls a denoise module per step.

Usage:
  TT_VISIBLE_DEVICES=<n> python3 perf/whde/design_step_probe.py --model pxdesign \
      --inputs perf/pxdesign/targets/laczc_512.yaml --n-step 40 --num-designs 1 --out r.json
  TT_VISIBLE_DEVICES=<n> python3 perf/whde/design_step_probe.py --model boltzgen \
      --inputs examples/binder.yaml --n-step 50 --num-designs 1 --out r.json
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path


def summarize(gaps):
    """ms stats over a list of per-call seconds."""
    if not gaps:
        return None
    ms = sorted(g * 1e3 for g in gaps)
    return {"n": len(ms), "median_ms": round(statistics.median(ms), 2),
            "min_ms": round(ms[0], 2), "max_ms": round(ms[-1], 2),
            "p10_ms": round(ms[max(0, len(ms) // 10)], 2),
            "spread_pct": round((ms[-1] - ms[0]) / ms[0] * 100, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pxdesign", choices=["pxdesign", "rfd3", "boltzgen"])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--n-step", type=int, default=40)
    ap.add_argument("--num-designs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default=os.environ.get("BOLTZ_CACHE",
                                                      str(Path("~/.boltz").expanduser())))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch

    torch.set_grad_enabled(False)
    stamps = []

    def stamp(call):
        """Wrap a per-step callable so every invocation lands in ``stamps``."""
        def timed(*a, **kw):
            s = time.perf_counter()
            out = call(*a, **kw)
            stamps.append((s, time.perf_counter()))
            return out
        return timed

    n_token = None
    if args.model == "pxdesign":
        from tt_bio.main import ensure_p300_mesh_descriptor, ensure_pxdesign_weights
        from tt_bio.pxdesign.inputs import design_inputs_from_yaml
        from tt_bio.pxdesign.model import ProtenixDesign

        feats = design_inputs_from_yaml(Path(args.inputs))
        feats = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float64 else v)
                 for k, v in feats.items()}
        n_token = int(feats["restype"].shape[0])
        ckpt = ensure_pxdesign_weights(Path(args.cache).expanduser())
        ensure_p300_mesh_descriptor()

        t0 = time.perf_counter()
        model = ProtenixDesign.load_from_checkpoint(str(ckpt))
        load_s = time.perf_counter() - t0

        model.diffusion.denoise = stamp(model.diffusion.denoise)
        t_start = time.perf_counter()
        coords = model.design(feats, n_step=args.n_step, n_sample=args.num_designs,
                              seed=args.seed)
        total_s = time.perf_counter() - t_start
        extra = {"coords_shape": list(coords.shape),
                 "coords_finite": bool(torch.isfinite(coords).all())}
    elif args.model == "boltzgen":
        # BoltzGen runs its pipeline through a CLI, and one --device_ids chip runs it
        # in-process, so the hook goes on the class rather than on an instance: every
        # AtomDiffusion step calls TTScoreModelAdapter.forward exactly once.
        import sys

        from tt_bio.boltzgen import adapter as bg_adapter
        from tt_bio.boltzgen.cli.boltzgen import main as bg_main
        from tt_bio.main import ensure_p300_mesh_descriptor

        bg_adapter.TTScoreModelAdapter.forward = stamp(
            bg_adapter.TTScoreModelAdapter.forward)
        ensure_p300_mesh_descriptor()

        out_dir = Path(args.out).with_suffix(".bgout")
        chip = os.environ.get("TT_VISIBLE_DEVICES", "0")
        sys.argv = ["tt-bio design", "run", args.inputs,
                    "--output", str(out_dir),
                    "--num_designs", str(args.num_designs),
                    "--device_ids", chip,
                    "--steps", "design",
                    "--config", "design", f"sampling_steps={args.n_step}"]
        load_s = None
        t_start = time.perf_counter()
        bg_main()
        total_s = time.perf_counter() - t_start
        extra = {"out_dir": str(out_dir),
                 "n_cif_out": len(list(out_dir.rglob("*.cif"))) + len(list(out_dir.rglob("*.pdb")))}
    else:
        # rfd3's sampler calls the diffusion module directly, so the hook goes on the
        # builder's product rather than on a method of a loaded model. Same probe, same
        # stamps list, no tt_bio change either way.
        import json as _json

        from tt_bio.main import ensure_rfd3_weights
        from tt_bio.rfd3 import design as rfd3_design

        specs = _json.loads(Path(args.inputs).read_text())
        ckpt_dir = ensure_rfd3_weights(Path(args.cache).expanduser())
        build = rfd3_design.build_diffusion_module

        def build_timed(*a, **kw):
            mod = build(*a, **kw)
            inner = mod.__call__
            wrapped = stamp(inner)
            mod.__call__ = wrapped
            return lambda *aa, **kk: wrapped(*aa, **kk)

        rfd3_design.build_diffusion_module = build_timed
        t0 = time.perf_counter()
        t_start = t0
        rows = rfd3_design.run_design(
            specs, args.out + ".designs", checkpoint_dir=str(ckpt_dir), from_pdb=True,
            num_timesteps=args.n_step, seed=args.seed, num_designs=args.num_designs,
            device_visible=os.environ.get("TT_VISIBLE_DEVICES", "0"), verbose=False)
        total_s = time.perf_counter() - t_start
        load_s = None
        # rfd3's size axis is atoms, not tokens; keeping them in one field would let a
        # later reader compare an atom count against pxdesign's token count.
        n_token = None
        extra = {"n_designs_out": len(rows),
                 "n_atoms": int(getattr(rows[0], "n_atoms", 0) or 0),
                 "out_paths": [str(r.out_path) for r in rows][:3],
                 "cif_exists": [Path(r.out_path).is_file() for r in rows][:3]}

    per_call = [b - a for a, b in stamps]
    trunk_s = (stamps[0][0] - t_start) if stamps else None
    # Step 0 pays the ttnn compile for every kernel the denoise stream uses; it is reported
    # separately rather than folded into the step cost it would otherwise dominate.
    warm = per_call[1:]

    out = {
        "model": args.model, "inputs": args.inputs, "n_token": n_token,
        "n_step_requested": args.n_step, "num_designs": args.num_designs,
        "denoise_calls": len(per_call),
        "host": os.uname().nodename, "device": os.environ.get("TT_VISIBLE_DEVICES"),
        "loadavg_1m": round(os.getloadavg()[0], 2),
        "load_s": None if load_s is None else round(load_s, 1), "trunk_s": None if trunk_s is None else round(trunk_s, 2),
        "total_design_s": round(total_s, 2),
        "cold_step_ms": round(per_call[0] * 1e3, 2) if per_call else None,
        "warm_step": summarize(warm),
        **extra,
        "warm_step_ms_all": [round(g * 1e3, 2) for g in warm],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "warm_step_ms_all"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
