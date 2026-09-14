#!/usr/bin/env python3
"""Upstream Boltz-2 2.2.1 reference folds of the cdk2x2 fixtures, on a rented GPU.

The accuracy budget the K10 campaign rations is measured against a Tenstorrent fold, so every
Angstrom in it is drift-from-last-shipped rather than error-against-truth. This produces the
non-TT side of that comparison: the same fixture, the same protocol, the pip package a
researcher installs.

Reference means the slowest honest path, not the fast one:
  * plain ``boltz==2.2.1``, not ``boltz[cuda]`` -- cuEquivariance kernels are a different
    arithmetic, and ``--no_kernels`` is passed as well so the choice cannot leak back in.
  * TF32 off on matmul and cuDNN, ``float32_matmul_precision("highest")``. TF32 is a 10-bit
    mantissa; a reference that runs it is an approximation of the thing it references.
  * one dataloader worker process, so the run does not depend on worker RNG seeding.

Two draw modes, because Boltz-2 is a diffusion sampler and two runs that do not share their
noise differ by the full sampling spread rather than by arithmetic (memory
``diffusion-port-parity-shared-draws``):

  cuda      the stock path. ``--seed`` seeds everything and the sampler draws on the GPU. This
            is the reference a researcher gets, and the arm the seed floor is measured on.
  shared    every ``torch.randn``/``randn_like`` is drawn on the CPU and moved, and the RNG is
            reseeded immediately before the sampler's first draw. That is byte-for-byte what
            ``TT_BIO_SHARED_DRAW_SEED`` does on the TT side (tt_bio/boltz2.py), so a TT fold run
            with that variable and a reference fold run in this mode consume the same noise and
            the only thing left between them is arithmetic.

    gpu_ref_fold.py --yaml F.yaml --a3m F.a3m --size 512 --seeds 0,1,2,3 \
        --mode cuda --outdir /root/results/ref --runs /root/results/runs_512.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RECYCLING_STEPS = 3
SAMPLING_STEPS = 200
DIFFUSION_SAMPLES = 1


def install_cpu_draws(seed: int) -> None:
    """All noise on the CPU RNG, reseeded at the sampler's first draw. Mirrors
    tt_bio/boltz2.py's TT_BIO_SHARED_DRAW_SEED hook."""
    import torch

    orig_randn, orig_randn_like = torch.randn, torch.randn_like
    state = {"reseeded": False}

    def randn(*a, **kw):
        dev = kw.pop("device", None)
        t = orig_randn(*a, **kw)
        return t if dev is None else t.to(dev)

    def randn_like(t, *a, **kw):
        return orig_randn(tuple(t.shape), dtype=kw.get("dtype", t.dtype)).to(t.device)

    torch.randn, torch.randn_like = randn, randn_like

    # Reseed at the sampler entry, not at CLI start: featurization and the trunk consume the
    # global stream by different amounts on the two stacks, which is exactly the divergence the
    # TT-side hook was written to kill.
    cls = None
    for name in ("boltz.model.modules.diffusionv2", "boltz.model.modules.diffusion"):
        try:
            m = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(m, "AtomDiffusion"):
            cls = m.AtomDiffusion
            break
    assert cls is not None and hasattr(cls, "sample"), \
        "cannot find boltz AtomDiffusion.sample: the shared-draw hook has nothing to attach to"
    orig_sample = cls.sample

    def sample(self, *a, **kw):
        if not state["reseeded"]:
            state["reseeded"] = True
        torch.manual_seed(seed)
        return orig_sample(self, *a, **kw)

    cls.sample = sample


def mean_ca_bfactor(cif: Path) -> float:
    vals, lines, i = [], cif.read_text().splitlines(), 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        j, cols = i + 1, []
        while j < len(lines) and lines[j].lstrip().startswith("_"):
            cols.append(lines[j].strip())
            j += 1
        if not cols or not cols[0].startswith("_atom_site."):
            i = j
            continue
        idx = {c.split(".", 1)[1]: k for k, c in enumerate(cols)}
        while j < len(lines):
            s = lines[j].strip()
            if not s or s.startswith("#") or s == "loop_" or s.startswith("_"):
                break
            f = s.split()
            if len(f) >= len(cols) and f[idx["label_atom_id"]].strip('"') == "CA":
                vals.append(float(f[idx["B_iso_or_equiv"]]))
            j += 1
        break
    return round(sum(vals) / len(vals), 4) if vals else float("nan")


def fold_one(args, seed: int) -> dict:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    work = Path(args.work) / f"{args.size}_s{seed}_{args.mode}"
    shutil.rmtree(work, ignore_errors=True)
    inp = work / "input"
    inp.mkdir(parents=True)

    base = args.yaml.read_text()
    assert "msa:" not in base, "fixture YAML already carries an msa key"
    seq = [ln.split("sequence:", 1)[1].strip() for ln in base.splitlines()
           if ln.strip().startswith("sequence:")][0]
    a3m_rows = args.a3m.read_text().split("\n")
    assert a3m_rows[1] == seq, "a3m query row does not match the fixture sequence"
    y = base.replace(f"sequence: {seq}",
                     f"sequence: {seq}\n      msa: {args.a3m.resolve()}")
    assert "msa:" in y
    (inp / f"cdk2x2_{args.size}.yaml").write_text(y)

    if args.mode == "shared":
        install_cpu_draws(seed)

    # boltz 2.2.1 hard-codes `precision="bf16-mixed"` for every boltz2 run (main.py:1262).
    # bf16 has an 8-bit mantissa: a reference measured on it is an approximation of the thing it
    # references, which is the whole defect this task exists to fix. Force fp32.
    import boltz.main as bmain
    _Trainer = bmain.Trainer

    class _Fp32Trainer(_Trainer):
        def __init__(self, *a, **kw):
            kw["precision"] = args.precision
            super().__init__(*a, **kw)

    bmain.Trainer = _Fp32Trainer

    from boltz.main import cli
    argv = ["predict", str(inp), "--out_dir", str(work / "out"),
            "--cache", "/root/.boltz", "--devices", "1", "--accelerator", "gpu",
            "--recycling_steps", str(RECYCLING_STEPS),
            "--sampling_steps", str(SAMPLING_STEPS),
            "--diffusion_samples", str(DIFFUSION_SAMPLES),
            "--seed", str(seed), "--no_kernels",
            "--output_format", "mmcif", "--num_workers", "1"]
    print("boltz argv:", " ".join(argv), flush=True)
    t0 = time.perf_counter()
    try:
        cli.main(args=argv, standalone_mode=False)
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    fold_s = round(time.perf_counter() - t0, 3)

    cifs = sorted((work / "out").rglob("*model_0.cif")) or sorted((work / "out").rglob("*.cif"))
    assert cifs, f"no CIF under {work/'out'}"
    cif = cifs[0]
    tag = f"{args.arm}-s{seed}"
    dst = Path(args.outdir) / f"{args.size}_{tag}"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(cif, dst / f"cdk2x2_{args.size}.cif")
    conf = sorted((work / "out").rglob("confidence_*model_0.json"))
    cj = json.loads(conf[0].read_text()) if conf else {}
    return {"target": f"cdk2x2_{args.size}", "tag": tag, "seed": seed, "arm": args.arm,
            "mode": args.mode, "fold_s": fold_s,
            "sha256": hashlib.sha256((dst / f"cdk2x2_{args.size}.cif").read_bytes()).hexdigest(),
            "plddt": mean_ca_bfactor(dst / f"cdk2x2_{args.size}.cif"),
            "confidence_plddt": cj.get("complex_plddt"),
            "cif": str(dst / f"cdk2x2_{args.size}.cif")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", type=Path, required=True)
    ap.add_argument("--a3m", type=Path, required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--mode", default="cuda", choices=["cuda", "shared"])
    ap.add_argument("--arm", default=None, help="default: gpuref / gpurefshared")
    ap.add_argument("--outdir", default="/root/results/cif")
    ap.add_argument("--work", default="/root/work")
    ap.add_argument("--precision", default="32",
                    help='Lightning precision; boltz2 hard-codes "bf16-mixed", the reference is 32')
    ap.add_argument("--runs", type=Path, required=True)
    args = ap.parse_args()
    if args.arm is None:
        args.arm = "gpuref" if args.mode == "cuda" else "gpurefshared"

    import torch
    # Set before the env block is captured, not only inside fold_one: otherwise the record shows
    # cuDNN's default True while the folds themselves ran with it off.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    env = {"host": os.uname().nodename, "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                     "--format=csv,noheader"], capture_output=True,
                                    text=True).stdout.strip(),
           "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
           "tf32_cudnn": torch.backends.cudnn.allow_tf32,
           "boltz": importlib.import_module("boltz").__version__
           if hasattr(importlib.import_module("boltz"), "__version__") else "2.2.1",
           "no_kernels": True, "precision": args.precision, "recycling_steps": RECYCLING_STEPS,
           "sampling_steps": SAMPLING_STEPS, "diffusion_samples": DIFFUSION_SAMPLES,
           "mode": args.mode}

    out = {"doc": "upstream boltz 2.2.1 reference folds", "env": env, "runs": []}
    if args.runs.exists():
        prev = json.loads(args.runs.read_text())
        out["runs"] = [r for r in prev.get("runs", [])]
    for s in [int(x) for x in args.seeds.split(",")]:
        r = fold_one(args, s)
        print(json.dumps(r), flush=True)
        out["runs"] = [x for x in out["runs"] if x["tag"] != r["tag"]
                       or x["target"] != r["target"]] + [r]
        args.runs.write_text(json.dumps(out, indent=1))
    print("wrote", args.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
