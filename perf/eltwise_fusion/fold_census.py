#!/usr/bin/env python3
"""Count the fusion helpers' calls, and their shapes, in a real fold.

This exists because the cdk2x2_298 parity run came back byte-identical across arms, and a
digest that does not move is indistinguishable from a site that never fired (memories
`pcc-gate-can-pass-without-the-op-it-names`, `negative-control-must-break-what-check-reads`).
The census is the positive control: it proves the arm switch changes what executes, and it
prices the aggregate win from COUNTED calls rather than an assumed block x step product.

Wraps the three helpers in `tt_bio.eltwise_fusion` -- the one place all eleven sites route
through -- keyed by the caller's file:line, so a site that never fires is visible as an
absent row rather than as a silence.

    fold_census.py --model openfold3 --fixture cdk2x2_298 --out <json>
"""
from __future__ import annotations

import argparse, collections, json, os, shutil, socket, sys, tempfile, time, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "eltwise_fusion"))

from fold_parity import build_cfg, _seed_msa, FIX  # noqa: E402  -- same protocol, one source


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", default="cdk2x2_298")
    ap.add_argument("--recycles", type=int, default=1)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), \
        f"imported tt_bio from {_TB.__file__}, not this worktree"
    import tt_bio.eltwise_fusion as EF

    counts: dict = collections.defaultdict(lambda: collections.Counter())

    def wrap(name):
        orig = getattr(EF, name)

        def w(*args, **kw):
            f = sys._getframe(1)
            site = f"{Path(f.f_code.co_filename).name}:{f.f_lineno}"
            shapes = " ".join(str(tuple(t.shape)) for t in args
                              if hasattr(t, "shape"))
            counts[name][f"{site} {shapes}"] += 1
            return orig(*args, **kw)
        setattr(EF, name, w)
        return orig

    for n in ("scale_add", "mask_add", "norm_residual"):
        wrap(n)
    # The model modules bound the helpers at import, so rebind those names too.
    import importlib
    for mod in ("protenix", "openfold3", "openfold3_atom_transformer",
                "openfold3_diffusion_transformer", "openfold3_diffusion_module",
                "esmfold2", "tenstorrent"):
        m = importlib.import_module("tt_bio." + mod)
        for n in ("scale_add", "mask_add", "norm_residual"):
            if hasattr(m, n):
                setattr(m, n, getattr(EF, n))

    dev = get_device()
    work = Path(tempfile.mkdtemp(prefix="eltfuse-census-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    _seed_msa(FIX / f"{a.fixture}.yaml", (FIX / f"{a.fixture}.a3m").read_text(), msa_dir)
    cfg = build_cfg(a.model, msa_dir, struct_dir, a.recycles, a.steps, a.samples, a.seed)
    _ensure_local_artifacts(cfg)
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("ttx-eltwise-fusion-census", cfg)

    t0 = time.perf_counter()
    err = None
    try:
        state.predict_one(FIX / f"{a.fixture}.yaml", cfg)
    except Exception:
        err = traceback.format_exc()[-2000:]
    ttnn.synchronize_device(dev)
    wall = time.perf_counter() - t0

    out = dict(
        model=a.model, fixture=a.fixture, fold_s=round(wall, 3), error=err,
        host=socket.gethostname(), card=os.environ.get("TT_VISIBLE_DEVICES"),
        protocol=dict(recycles=a.recycles, steps=a.steps, samples=a.samples, seed=a.seed),
        totals={k: sum(v.values()) for k, v in counts.items()},
        per_site={k: dict(v.most_common()) for k, v in counts.items()})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"CENSUS {a.model} {a.fixture} fold {wall:.2f}s totals={out['totals']}", flush=True)
    for k, v in out["per_site"].items():
        for site, n in v.items():
            print(f"  {k:13s} {n:7d}  {site}", flush=True)
    if err:
        print("FOLD ERROR:\n" + err, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
