#!/usr/bin/env python3
"""Is `TT_BIO_HEAD_PAD_TAIL` safe at 512 aa? The folds that question needs.

`b2z2-fusebias-512-parity`'s `fold_seeds.py`, taken as-is except that the arm is a module global
rather than an env var, because this lever's flag is read at import. The flag it was written for is
named in `--flag`; everything else -- the fixtures, the per-seed plan, the trailing A/A repeat -- is
that row's and is deliberately unchanged so the two levers are scored by one instrument.

The lever under test here: the diffusion tail carries the head split's zero pad lanes into the gate
multiply and the output projection instead of spending four programs stripping them. Exact in real
arithmetic; NOT bit-exact, because padding K from 768 to 1024 regroups the output projection's
partial sums at tile granularity.


The flag is on by default on main and is NOT bit-exact: `_fuse_bias_stack` collapses a stack of
LayerNorm(C)+Linear(C,H) into one affine-free LayerNorm plus one [C, H*len] Linear, which is exact
in real arithmetic and 2e-7 mean / 2.1e-5 max relative in floating point. Its only accuracy
evidence is 0.218 A on the 298-residue monomeric control. At 512 aa it has never been scored, and
three levers in this wave passed everything upstream and then failed at 512 aa.

The fixture cannot be read whole. `cdk2x2_512` is CDK2 fused to its own first 214 residues with no
inter-domain interface, so the hinge between the two pseudo-domains saturates whole-molecule RMSD
for any reassociation (memory `cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`; the
shipped default itself moves 8.60 A on it). So this run produces what a per-domain reading needs:
the SAME arm at several seeds, so the sampler's own spread is measured rather than assumed, and
both arms at each of those seeds, so the lever's move is priced against that spread.

Arms, in ONE process so they share a device, a program cache and a model load:

    off   TT_BIO_FUSE_BIAS_STACKS=0, the per-layer stack
    on    TT_BIO_FUSE_BIAS_STACKS=1, main's shipped default

Nothing else is touched: the two bit-exact levers that shipped with it stay at their defaults in
both arms, so the only difference between `off` and `on` is this flag. The flag is read per call
(`boltz2._fuse_bias_stacks`) and changes no device program, so flipping it in-process is legal and
neither arm can be served the other's compiled program.

The last fold repeats the first (`off`, first seed): that repeat is the A/A floor, and it must come
back byte-identical or no other number here is interpretable.

    fold_seeds.py --out <json> --cifdir <dir> [--seeds 0,1,2,3] [--sizes 512,298]
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
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--sizes", default="512,298")
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    ap.add_argument("--flag", default="TT_BIO_HEAD_PAD_TAIL")
    ap.add_argument("--attr", default="_HEAD_PAD_TAIL")
    # `module:attribute` pairs set together as ONE arm, for a lever that lives in two modules.
    ap.add_argument("--attrs", default="")
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
    import importlib
    import tt_bio.tenstorrent as T
    assert args.flag not in os.environ, \
        "the arm under test may not be pinned in the environment; it is set per fold"
    if args.attrs:
        targets = []
        for spec in args.attrs.split(","):
            mod, _, attr = spec.partition(":")
            m = importlib.import_module(mod)
            assert hasattr(m, attr), f"{spec} does not exist"
            targets.append((m, attr))
    else:
        assert hasattr(T, args.attr), f"{args.attr} is not a tt_bio.tenstorrent global"
        targets = [(T, args.attr)]
    out["env"]["arm_targets"] = [f"{m.__name__}:{a}" for m, a in targets]

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
                     "seeds": seeds, "diffusion_trace": False, "flag": args.flag},
    }, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))

    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z2-layout-tail-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for name in ("cdk2x2_512", "cdk2x2_298"):
        AB._seed_msa(AB.FIX / f"{name}.yaml", (AB.FIX / f"{name}.a3m").read_text(), msa_dir)

    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-layout-tail", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()

    def fold(arm, seed, target, keep):
        for _m, _a in targets:
            setattr(_m, _a, arm == "on")
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
                "plddt": round(float(metrics.get("plddt", metrics.get("confidence_score", 0))), 6)}

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
