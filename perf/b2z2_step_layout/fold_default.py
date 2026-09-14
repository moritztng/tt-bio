#!/usr/bin/env python3
"""One fold per size with the checkout untouched, sha256 of the CIF.

The control the step-layout branch's claim rests on: with neither elision set, the branch folds
the same structure as a tree without it. Run it on the merged branch, swap `tt_bio/` to
`origin/main`, run it again, compare the digests. It sets no flag and imports no branch-only
symbol, so the same file runs on both trees.

    fold_default.py --out <json> --cifdir <dir> [--sizes 298,512] [--seed 0]
"""
import argparse, hashlib, json, os, shutil, socket, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))
import ab_flag_levers as AB  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--sizes", default="298,512")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for f in ("TT_BIO_DIT_FUSED_QKV", "TT_BIO_APB_CONCAT_HEADS", "TT_BIO_UNFUSED_SILU"):
        assert f not in os.environ, f"{f} is pinned; this run would not be the default path"

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree")

    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    out = {"doc": __doc__, "env": {
        "host": socket.gethostname(), "grid": [g.x, g.y], "arch": str(dev.arch()),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "tt_bio_diff_vs_main": os.popen(
            f"git -C {REPO} diff --stat origin/main -- tt_bio/").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
    }, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    work = Path(tempfile.mkdtemp(prefix="b2z2-default-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for name in ("cdk2x2_512", "cdk2x2_298"):
        AB._seed_msa(AB.FIX / f"{name}.yaml", (AB.FIX / f"{name}.a3m").read_text(), msa_dir)

    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-default", cfg)

    for size in args.sizes.split(","):
        target = AB.FIX / f"cdk2x2_{size}.yaml"
        cfg["seed"] = args.seed
        try:
            state.model.structure_module.score_model.reset_static_cache()
        except Exception:
            pass
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        keep = args.cifdir / f"{size}_default-s{args.seed}"
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cifs[0], keep / cifs[0].name)
        body = (keep / cifs[0].name).read_bytes()
        r = {"size": size, "seed": args.seed, "fold_s": round(wall, 3),
             "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
             "plddt": round(float(metrics.get("plddt",
                                              metrics.get("confidence_score", 0))), 6)}
        out["runs"].append(r)
        print(f"  {size} {r['fold_s']:7.3f}s sha={r['sha256'][:16]} plddt={r['plddt']}",
              flush=True)
        args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
