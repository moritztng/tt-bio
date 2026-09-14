#!/usr/bin/env python3
"""Split the standing TT-vs-upstream CA-lDDT deficit into port code and device arithmetic.

`perf/roof_shared` closed the owed arithmetic-only comparison: with `TT_BIO_SHARED_DRAW_SEED=0`
on both stacks the port lands 0.43277 A (298 aa) / 0.68922 A (512 aa) from upstream
`boltz==2.2.1` fp32, and the CA-lDDT deficit against the experimental structure 1HCL SURVIVES the
pairing -- -0.00586 at 298 aa, -0.01487 / -0.01462 on the two 512 aa pseudo-domains. That number
is a single scalar covering everything between the two stacks: our port"s own torch code, the
bf16 dtype boundary, and ttnn"s arithmetic. This run splits it.

tt-bio carries its own torch CPU path for Boltz-2 (the one `scripts/full_parity_gate.py` uses as
`reference_fp32`), so the same three arms the release gate already defines answer the question,
all at seed 0 with shared draws, all scored against the committed `gpurefshared-s0` CIFs:

  ttcpufp32   tt-bio"s torch code, CPU, fp32, no kernels.  d(ttcpufp32, gpurefshared) is the
              PORT CODE delta: two fp32 torch implementations of the same checkpoint, so
              anything left is our code differing from upstream"s, not a dtype.
  ttcpubf16   the same, under `TT_BIO_REF_BF16=1` (bf16 autocast), i.e. tt_bio/worker.py"s own
              reference-bf16 hook.  d(ttcpubf16, ttcpufp32) is the intrinsic bf16 cost of the
              full sampler trajectory in torch.
  ttshared    the device arm, already committed under perf/roof_shared/cif/.  Its distance to
              ttcpufp32 is what ttnn adds on top of the two above.

Whichever arm carries the -0.015 lDDT is where the deficit lives. No device is opened here.

    fold_cpu_ref.py --out out/cpu_folds.json --cifdir cif --sizes 298 --arms fp32
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
ARMS = {"fp32": "ttcpufp32", "bf16": "ttcpubf16"}


def install_draw_trace(torch, sink: list) -> None:
    """Shape + digest of the draws after the LAST reseed, exactly as perf/roof_shared does, so
    the CPU arms" sampler stream can be checked against the device arm"s digests rather than
    assumed equal."""
    orig_randn, orig_seed = torch.randn, torch.manual_seed

    def randn(*a, **kw):
        t = orig_randn(*a, **kw)
        if sink:
            rec = sink[-1]
            rec["n_randn"] += 1
            if len(rec["draws"]) < TRACE_DRAWS:
                c = t.detach().to("cpu").contiguous()
                rec["draws"].append({"shape": list(t.shape), "dtype": str(t.dtype),
                                     "sha256": hashlib.sha256(c.numpy().tobytes()).hexdigest()[:16]})
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
    ap.add_argument("--sizes", default="298")
    ap.add_argument("--arms", default="fp32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this checkout")
    import tt_bio.boltz2 as B2
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    assert FLAG not in os.environ, f"{FLAG} may not be pinned; it is set per fold"
    assert "os.environ.get(\"TT_BIO_SHARED_DRAW_SEED\")" in Path(B2.__file__).read_text(), (
        "this checkout does not read the shared-draw flag in its sampler")
    assert not os.environ.get("TT_BIO_REF_BF16"), "TT_BIO_REF_BF16 may not be pinned"

    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = args.steps, args.recycles
    traces: list = []
    install_draw_trace(torch, traces)

    out = {"doc": __doc__, "env": {
        "host": socket.gethostname(), "accelerator": "cpu", "torch": torch.__version__,
        "threads": torch.get_num_threads(), "tt_bio_file": _TB.__file__,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "float32_matmul_precision": "highest", "use_kernels": False,
        "protocol": {"recycling_steps": args.recycles, "sampling_steps": args.steps,
                     "diffusion_samples": AB.DIFFUSION_SAMPLES, "seed": args.seed,
                     "arms": args.arms},
    }, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))

    dump()

    work = Path(tempfile.mkdtemp(prefix="k10p2-cpu-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for name in ("cdk2x2_512", "cdk2x2_298"):
        AB._seed_msa(AB.FIX / f"{name}.yaml", (AB.FIX / f"{name}.a3m").read_text(), msa_dir)

    cfg = AB.build_cfg(msa_dir, struct_dir)
    # The reference is the slowest honest path: tt-bio"s torch code, no fused kernels, no device.
    cfg["conf_kwargs"] = {**cfg["conf_kwargs"], "use_tenstorrent": False, "use_kernels": False}
    _ensure_local_artifacts(cfg)

    t0 = time.perf_counter()
    state = _WorkerState("cpu")
    state.load_model(cfg)
    state.bind_run("k10-p2-cpu", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()

    def fold(arm, seed, target, keep):
        os.environ[FLAG] = str(seed)
        os.environ.pop("TT_BIO_REF_BF16", None)
        if arm == "bf16":
            os.environ["TT_BIO_REF_BF16"] = "1"
        cfg["seed"] = seed
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        traces.append({"draws": [], "n_randn": 0, "seeds": []})
        t = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        wall = time.perf_counter() - t
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cifs[0], keep / cifs[0].name)
        body = (keep / cifs[0].name).read_bytes()
        return {"arm": ARMS[arm], "seed": seed, "target": target.stem,
                "fold_s": round(wall, 3), "mode": f"cpu-{arm}",
                "sha256": hashlib.sha256(body).hexdigest()[:16],
                "shared_draw_seed": os.environ.get(FLAG),
                "ref_bf16": os.environ.get("TT_BIO_REF_BF16"),
                "n_atoms_cif": n_atoms(keep / cifs[0].name),
                "sampler_draws": traces[-1]["draws"], "n_randn": traces[-1]["n_randn"],
                "manual_seeds": traces[-1]["seeds"],
                "plddt": round(float(metrics.get("plddt", metrics.get("confidence_score", 0))), 6)}

    for size in args.sizes.split(","):
        target = AB.FIX / f"cdk2x2_{size}.yaml"
        for arm in args.arms.split(","):
            tag = f"{ARMS[arm]}-s{args.seed}"
            r = fold(arm, args.seed, target, args.cifdir / f"{size}_{tag}")
            r["tag"] = tag
            out["runs"].append(r)
            d0 = r["sampler_draws"][0] if r["sampler_draws"] else {}
            print(f"  {size} {tag:16s} {r['fold_s']:9.1f}s sha={r['sha256']} "
                  f"plddt={r['plddt']} draw0={d0.get('shape')} {d0.get('sha256')} "
                  f"nrandn={r['n_randn']} atoms={r['n_atoms_cif']}", flush=True)
            dump()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
