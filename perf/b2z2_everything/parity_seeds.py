#!/usr/bin/env python3
"""The union's parity: base vs all nine levers, both fixture sizes, several seeds.

Six of the nine levers are bit-exact and the fold A/B proved it end to end -- they write the base
CIF byte for byte. The other three move the diffusion conditioning, the trunk's z_init and the
confidence pair assembly onto the device in bf16, so the union's structure is NOT byte-identical
and owes a reading.

`cdk2x2_512` cannot be read whole: it is CDK2 fused to its own first 214 residues with no
inter-domain interface, so the hinge between the two pseudo-domains saturates whole-molecule RMSD
for any reassociation (memory `cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`). What a
per-domain reading needs is what this produces: the SAME arm at several seeds, so the sampler's own
spread is measured rather than assumed, and both arms at each seed, so the lever's move is priced
against that spread. 298 aa is the monomeric control and has one domain, not two.

**The parity evidence here is a whole fold against a whole fold with real weights.** It is not a
block output against `tt_bio.reference`, which zero-initialises 23 of a PairformerLayer's weights
and would pass for an arm that computes nothing (`b2z2-trunk-byte-round2`, 2026-09-13). The
negative control is built in and it demonstrably breaks the comparison: the `on` arm writes a
different sha256 from `off` at both sizes, and the A/A repeat writes the same one.

Arm names are `off` and `on` so `perf/b2z2_fusebias/score.py` reads this run unchanged.

    parity_seeds.py --out <json> --cifdir <dir> [--seeds 0,1,2] [--sizes 512,298]
"""
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

ARMS = ("off", "on")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--sizes", default="512,298")
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    sizes = args.sizes.split(",")

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
    import tt_bio.tenstorrent as TT
    import tt_bio.triatt_qkv as QK
    import tt_bio.boltz2 as B2
    for k in ("TT_BIO_ATOM_KEY_WINDOW", "TT_BIO_ATOM_KV_PREPROJ", "TT_BIO_ATOM_L1",
              "TT_BIO_ATOM_SHIFT_GATHER", "TT_BIO_SDPA_GRID_Q_CHUNK", "TT_BIO_TRIATT_FUSED_QKVG",
              "TT_BIO_PWA_RESIDENCY", "TT_BIO_PWA_BATCH_HEAD_WEIGHTS", "TT_BIO_PWA_L1_NORM_M",
              "TT_BIO_PWA_L1_ROWS", "TT_BIO_DEVICE_CONDITIONING", "TT_BIO_DEVICE_ZINIT",
              "TT_BIO_DEVICE_CONFIDENCE"):
        assert k not in os.environ, f"{k} is pinned in the env; the arms are set per fold"
    assert "TT_METAL_DEVICE_PROFILER" not in os.environ, (
        "TT_METAL_DEVICE_PROFILER must be ABSENT, not 0")

    def set_arm(arm: str) -> dict:
        on = arm == "on"
        TT._ATOM_KEY_WINDOW = False               # never taken: it gathers the wrong atoms
        TT._ATOM_SHIFT_GATHER = on
        TT._ATOM_KV_PREPROJ = on
        TT._ATOM_L1 = on
        TT._SDPA_GRID_Q_CHUNK = on
        TT._sdpa_program_config_for_lengths.cache_clear()
        TT._grid_q_chunk.cache_clear()
        QK._QKVG_ENABLED = on
        TT._PWA_BATCH_HEAD_WEIGHTS = on
        TT._PWA_L1_NORM_M = on
        TT._PWA_L1_ROWS = 0 if on else -1
        for env in ("TT_BIO_DEVICE_CONDITIONING", "TT_BIO_DEVICE_ZINIT",
                    "TT_BIO_DEVICE_CONFIDENCE"):
            os.environ[env] = "1" if on else "0"
        return {"shift_gather": TT._ATOM_SHIFT_GATHER, "kv_preproj": TT._ATOM_KV_PREPROJ,
                "atom_l1": TT._ATOM_L1, "sdpa_grid_q": TT._SDPA_GRID_Q_CHUNK,
                "qkvg": QK._QKVG_ENABLED, "pwa_heads": TT._PWA_BATCH_HEAD_WEIGHTS,
                "pwa_l1_rows": TT._PWA_L1_ROWS, "conditioning": B2._device_conditioning(),
                "zinit": B2._device_zinit(), "confidence": B2._device_confidence()}

    # rule 0 is not negotiable, and the driver asserts it rather than the author
    assert args.steps == 200 and args.recycles == 3, "200 sampling steps, 3 recycles"
    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = args.steps, args.recycles
    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    out = {"doc": __doc__, "env": {
        "host": socket.gethostname(), "grid": [g.x, g.y], "arch": str(dev.arch()),
        "torch": torch.__version__,
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(),
        "protocol": {"recycling_steps": args.recycles, "sampling_steps": args.steps,
                     "seeds": seeds, "diffusion_trace": False},
    }, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))

    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z2-everything-parity-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for name in ("cdk2x2_512", "cdk2x2_298"):
        AB._seed_msa(AB.FIX / f"{name}.yaml", (AB.FIX / f"{name}.a3m").read_text(), msa_dir)

    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-everything-union-wh", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()

    def fold(arm, seed, target, keep):
        flags = set_arm(arm)
        QK.QKVG_STATS[0] = QK.QKVG_STATS[1] = 0
        TT.ATOM_SHIFT_GATHER_STATS[0] = TT.ATOM_SHIFT_GATHER_STATS[1] = 0
        cfg["seed"] = seed
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
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cifs[0], keep / cifs[0].name)
        body = (keep / cifs[0].name).read_bytes()
        return {"arm": arm, "seed": seed, "target": target.stem, "fold_s": round(wall, 3),
                "sha256": hashlib.sha256(body).hexdigest()[:16],
                "plddt": round(float(metrics.get("plddt", metrics.get("confidence_score", 0))), 6),
                "flags": flags, "gather_stats": list(TT.ATOM_SHIFT_GATHER_STATS),
                "qkvg_stats": list(QK.QKVG_STATS)}

    # 512 aa first: it is the question. The 298 aa control follows, on the same device open, so
    # the two sizes are comparable on this box without a second model load.
    for size in sizes:
        target = AB.FIX / f"cdk2x2_{size}.yaml"
        plan = [(arm, s) for s in seeds for arm in ARMS]
        plan.append(("off", seeds[0]))                     # the A/A repeat, last
        seen = set()
        for arm, seed in plan:
            tag = f"{arm}-s{seed}" + ("_r1" if (arm, seed) in seen else "")
            seen.add((arm, seed))
            r = fold(arm, seed, target, args.cifdir / f"{size}_{tag}")
            r["tag"] = tag
            out["runs"].append(r)
            print(f"  {size} {tag:10s} {r['fold_s']:7.3f}s sha={r['sha256']} "
                  f"plddt={r['plddt']}", flush=True)
            dump()

    aa = [r for r in out["runs"] if r["tag"].startswith(f"off-s{seeds[0]}")]
    out["aa_floor_identical"] = {
        size: len({r["sha256"] for r in aa if r["target"].endswith(size)}) == 1 for size in sizes}
    dump()
    print(json.dumps(out["aa_floor_identical"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
