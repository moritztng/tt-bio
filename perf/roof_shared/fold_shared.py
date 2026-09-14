#!/usr/bin/env python3
"""The TT half of the arithmetic-only TT-vs-upstream comparison `k10-p1-accuracy-anchor` owes.

The anchor measured main against upstream `boltz==2.2.1` at fp32 and got 0.90077 A (298 aa) and
1.42726 A (512 aa, worst per-pseudo-domain). Both stacks drew their own diffusion noise, so those
numbers carry the sampler's full seed floor: the reference compared to ITSELF with only the noise
realisation changed reads 0.92565 A and 1.82527 A. Shared draws remove exactly that term.

`TT_BIO_SHARED_DRAW_SEED` reseeds the CPU RNG at the sampler's first draw (`tt_bio/boltz2.py`),
and `perf/k10_anchor/gpu_ref_fold.py --mode shared` does the same thing on the reference stack:
every `torch.randn` on the CPU generator, reseeded at `AtomDiffusion.sample` entry. Two arms per
size, in one process on one card:

  shared    TT_BIO_SHARED_DRAW_SEED=<seed>, paired against `gpurefshared-s<seed>`
  plain     the variable unset, paired against `gpuref-s<seed>` -- the anchor's own comparison,
            re-run on this card and this commit so the paired and unpaired numbers differ by the
            draws and nothing else

The draw trace is recorded, not assumed. Both stacks reseed to the same integer and then draw
`(1, n_atoms, 3)`; if the atom axis were padded differently on one side the streams would diverge
at the first draw and the whole comparison would be void, so the shape and digest of the first
three sampler draws go into the run record next to the CIF digest.

    fold_shared.py --out out/folds.json --cifdir cif --sizes 298,512 --seed 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))

import ab_flag_levers as AB  # noqa: E402  -- the fixtures, cfg and MSA seeding, unmodified

FLAG = "TT_BIO_SHARED_DRAW_SEED"
TRACE_DRAWS = 3


def install_draw_trace(torch, sink: list) -> None:
    """Record shape + digest of the draws that follow the LAST reseed of a fold.

    The shared arm reseeds twice: once when the worker sets the fold's seed, once inside
    `AtomDiffusion.sample`. The draws that matter are the ones after the second, so the buffer is
    reset on every `manual_seed` and what survives to the end of the fold is the sampler's own
    stream. Values are untouched: the wrapper calls the original and hashes the result, so a
    traced fold consumes byte for byte the noise an untraced one would.
    """
    orig_randn, orig_seed = torch.randn, torch.manual_seed

    def randn(*a, **kw):
        t = orig_randn(*a, **kw)
        if sink:
            rec = sink[-1]
            rec["n_randn"] += 1
            if len(rec["draws"]) < TRACE_DRAWS:
                c = t.detach().to("cpu").contiguous()
                rec["draws"].append({
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "sha256": hashlib.sha256(c.numpy().tobytes()).hexdigest()[:16],
                })
        return t

    def manual_seed(seed):
        if sink:
            sink[-1]["draws"] = []
            sink[-1]["seeds"].append(int(seed))
        return orig_seed(seed)

    torch.randn, torch.manual_seed = randn, manual_seed


def n_atoms(cif: Path) -> int:
    return sum(1 for ln in cif.read_text().splitlines()
               if ln.startswith("ATOM ") or ln.startswith("HETATM"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--sizes", default="298,512")
    ap.add_argument("--arms", default="shared,plain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    ap.add_argument("--extra-seeds", default="",
                    help="extra seeds for the TT arms only (the reference has s0 shared), for an "
                         "in-session seed floor. Comma separated.")
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this checkout")
    import tt_bio.boltz2 as B2
    assert FLAG not in os.environ, f"{FLAG} may not be pinned; the arm is set per fold"
    src = Path(B2.__file__).read_text()
    assert 'os.environ.get("TT_BIO_SHARED_DRAW_SEED")' in src, (
        "this checkout's sampler does not read TT_BIO_SHARED_DRAW_SEED; both arms would be equal")

    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = args.steps, args.recycles
    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    traces: list = []
    install_draw_trace(torch, traces)

    out = {"doc": __doc__, "env": {
        "host": socket.gethostname(), "grid": [g.x, g.y], "arch": str(dev.arch()),
        "torch": torch.__version__,
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {"recycling_steps": args.recycles, "sampling_steps": args.steps,
                     "diffusion_samples": AB.DIFFUSION_SAMPLES, "seed": args.seed,
                     "arms": args.arms},
    }, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))

    dump()

    work = Path(tempfile.mkdtemp(prefix="roof-shared-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for name in ("cdk2x2_512", "cdk2x2_298"):
        AB._seed_msa(AB.FIX / f"{name}.yaml", (AB.FIX / f"{name}.a3m").read_text(), msa_dir)

    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("roof-shared", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()

    def fold(arm, seed, target, keep):
        os.environ.pop(FLAG, None)
        if arm == "shared":
            os.environ[FLAG] = str(seed)
        cfg["seed"] = seed
        try:
            state.model.structure_module.score_model.reset_static_cache()
        except Exception:
            pass
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        traces.append({"draws": [], "n_randn": 0, "seeds": []})
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cifs[0], keep / cifs[0].name)
        body = (keep / cifs[0].name).read_bytes()
        return {"arm": f"tt{arm}", "seed": seed, "target": target.stem,
                "fold_s": round(wall, 3), "mode": arm,
                "sha256": hashlib.sha256(body).hexdigest()[:16],
                "shared_draw_seed": os.environ.get(FLAG),
                "n_atoms_cif": n_atoms(keep / cifs[0].name),
                "sampler_draws": traces[-1]["draws"], "n_randn": traces[-1]["n_randn"],
                "manual_seeds": traces[-1]["seeds"],
                "plddt": round(float(metrics.get("plddt", metrics.get("confidence_score", 0))), 6)}

    seeds = [args.seed] + [int(s) for s in args.extra_seeds.split(",") if s.strip()]
    for size in args.sizes.split(","):
        target = AB.FIX / f"cdk2x2_{size}.yaml"
        for seed in seeds:
            for arm in args.arms.split(","):
                tag = f"tt{arm}-s{seed}"
                r = fold(arm, seed, target, args.cifdir / f"{size}_{tag}")
                r["tag"] = tag
                out["runs"].append(r)
                d0 = r["sampler_draws"][0] if r["sampler_draws"] else {}
                print(f"  {size} {tag:14s} {r['fold_s']:8.3f}s sha={r['sha256']} "
                      f"plddt={r['plddt']} draw0={d0.get('shape')} {d0.get('sha256')} "
                      f"atoms={r['n_atoms_cif']}", flush=True)
                dump()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
