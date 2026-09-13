#!/usr/bin/env python3
"""Does unfusing the transition SiLU change a fold on the models that are NOT Boltz-2?

`TT_BIO_UNFUSED_SILU` is read at one site, in `tt_bio.tenstorrent.Transition`, and that class is
built by `PairformerLayer`, `MiniformerLayer`, `MSALayer` and `Diffusion`. Protenix-v1/v2 import it
directly (`protenix.py:907`, `:2038`) and OpenFold3 reaches it through
`openfold3_msa_embedder.py:81` and its template pair stack. So "shared code, so it is fine" is an
argument, not a measurement, and the flag is not bit-exact anywhere: an identical digest is not
available as a bar on any of these models.

The bar that IS available is the one Boltz-2 was scored against: the lever's structural move
against the same model's own seed spread. This folds each model at two seeds in both arms, in one
process on one device, and writes the layout `perf/b2z2_fusebias/score.py` already reads, so every
model is scored by the same instrument on the same axis rather than by a new one per model.

The last fold repeats the first. That repeat is the A/A floor and must come back byte-identical or
nothing else here is interpretable.

    xmodel_silu_ab.py --model protenix-v2 --out <json> --cifdir <dir> [--seeds 0,1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--arms", default="base,usilu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    a = ap.parse_args()

    assert "TT_BIO_UNFUSED_SILU" not in os.environ, (
        "the arms are set in-process; a pinned env value would make both arms the same arm")

    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(ROOT), (
        f"imported tt_bio from {_TB.__file__}, not this worktree")
    import tt_bio.tenstorrent as T
    import tt_baseline as B

    seeds = [int(s) for s in a.seeds.split(",")]
    arms = a.arms.split(",")
    size = a.size
    tgt = a.fixdir / f"cdk2x2_{size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{size}.a3m"
    one_fold, meta, _state = B.build_fold(
        a.model, Path(__file__).resolve().parent / "msa" / f"{a.model}_{size}", tgt, a3m)
    struct_dir = Path(meta["struct_dir"])
    job_cfg = meta["job_cfg"]

    import importlib.metadata as im
    out = {"doc": __doc__, "env": {
        "host": socket.gethostname(), "model": a.model, "size": size,
        "ttnn": im.version("ttnn"), "hardware": meta["hardware"], "grid": meta["grid"],
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "recycling_steps": meta["recycling_steps"], "sampling_steps": B.SAMPLING_STEPS,
        "diffusion_samples": meta["diffusion_samples"], "n_msa": meta["n_msa"],
        "commit": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(),
    }, "runs": []}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        a.out.write_text(json.dumps(out, indent=1))

    dump()

    def fold(arm, seed, tag):
        T._UNFUSED_SILU = (arm == "usilu")
        job_cfg["seed"] = seed
        t0 = time.perf_counter()
        _s, m = one_fold()
        wall = time.perf_counter() - t0
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        keep = a.cifdir / f"{size}_{tag}"
        keep.mkdir(parents=True, exist_ok=True)
        body = cifs[0].read_bytes()
        (keep / cifs[0].name).write_bytes(body)
        r = {"arm": arm, "seed": seed, "tag": tag, "target": tgt.stem,
             "fold_s": round(wall, 3), "sha256": hashlib.sha256(body).hexdigest()[:16],
             "unfused_silu": bool(T._UNFUSED_SILU),
             "plddt": round(float(m.get("plddt") or 0), 6)}
        out["runs"].append(r)
        dump()
        print(f"  {a.model} {size} {tag:14s} {r['fold_s']:7.2f}s sha={r['sha256']} "
              f"plddt={r['plddt']}", flush=True)
        return r

    # Cold fold, discarded: it compiles the kernels. Comparing a cold structure to a warm one
    # compares two different things, and the arms must both be warm to be comparable at all.
    T._UNFUSED_SILU = False
    t0 = time.perf_counter()
    one_fold()
    out["cold_fold_s"] = round(time.perf_counter() - t0, 3)
    print(f"  {a.model} {size} cold (discarded) {out['cold_fold_s']:.2f}s", flush=True)
    dump()

    for seed in seeds:
        for arm in arms:
            fold(arm, seed, f"{arm}-s{seed}")
    first = arms[0]
    rep = fold(first, seeds[0], f"{first}-s{seeds[0]}_r1")          # the A/A floor, last

    base0 = next(r for r in out["runs"] if r["tag"] == f"{first}-s{seeds[0]}")
    out["aa_floor_bitexact"] = rep["sha256"] == base0["sha256"]
    out["arms_differ"] = len({r["sha256"] for r in out["runs"]
                              if r["seed"] == seeds[0] and not r["tag"].endswith("_r1")}) > 1
    dump()
    print(json.dumps({k: out[k] for k in ("aa_floor_bitexact", "arms_differ")}, indent=1))
    return 0 if out["aa_floor_bitexact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
