"""p41 -- what the 32 per-step host->device uploads carry, and how much of it repeats.

p39's census attributes 12.6 ms/step to `ttnn.from_torch` in 36 calls: 8.5 to the device-tilize
branch of `_tt` (model.py:384, 26 calls at 328.8 us) and 3.0 to the host-tilize branch
(model.py:387, 6 calls at 505.1 us -- SLOWER per call than the branch the threshold sends the
BIG tensors down, which is the first thing this checks).

This wraps `_tt` itself, so the recorded callsite is the model line that wanted the upload rather
than the helper. Per call it records the caller, shape, dtype, ms and a blake2b of the tensor
bytes. A callsite whose per-step digest tuple is the same every step is uploading step-invariant
data and is cacheable outright; one that repeats a digest WITHIN a step is re-uploading the same
bytes across the two recycles. The sum over the repeating callsites is the cap on an upload
cache, measured before anything is built.

Run: TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_HOLDER=worker:rfd3-host-half PYTHONPATH=$PWD \
     python3 scripts/rfd3_port/p41_upload_census.py --num_timesteps 8 --out perf/p41/uploads.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402

import tt_bio.rfd3.model as R  # noqa: E402
from tt_bio.rfd3.design import build_diffusion_module, build_token_initializer  # noqa: E402
from tt_bio.rfd3.featurize import featurize  # noqa: E402
from tt_bio.rfd3.input import InputSpecification  # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler  # noqa: E402

CALLS: list = []
STEP = [0]


def _digest(x: torch.Tensor) -> str:
    t = x.detach().cpu().contiguous()
    if t.dtype == torch.bool:
        t = t.to(torch.uint8)
    return hashlib.blake2b(t.numpy().tobytes(), digest_size=8).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb")
    ap.add_argument("--contig", default="A1-10,230,A31-40")
    ap.add_argument("--ckpt", default="/home/ttuser/.boltz/rfd3/weights")
    ap.add_argument("--num_timesteps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    orig_tt = R._tt

    def wrapped(x, dev, dtype=ttnn.bfloat16):
        fr = sys._getframe(1)
        t0 = time.perf_counter()
        try:
            return orig_tt(x, dev, dtype)
        finally:
            CALLS.append({
                "step": STEP[0],
                "fn": fr.f_code.co_name,
                "line": fr.f_lineno,
                "shape": list(x.shape),
                "numel": int(x.numel()),
                "in_dtype": str(x.dtype),
                "out_dtype": str(dtype),
                "branch": 384 if (R._TORCH_DTYPE.get(dtype) is not None
                                  and x.numel() >= R._DEVICE_TILIZE_MIN_ELEMENTS) else 387,
                "ms": (time.perf_counter() - t0) * 1e3,
                "sha": _digest(x),
            })

    R._tt = wrapped

    spec = InputSpecification.from_dict({"input": a.pdb, "contig": a.contig})
    spec.validate()
    f = featurize(a.pdb, spec)
    cap = Path(a.ckpt)
    ti_w = torch.load(cap / "token_initializer.real_weights.pt", map_location="cpu", weights_only=True)
    dm_w = torch.load(cap / "diffusion_module.real_weights.pt", map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti_w)
    dev_dm = build_diffusion_module(dm_w)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    L = init["Q_L_init"].shape[0]
    coord0 = f["motif_pos"].float().unsqueeze(0) if "motif_pos" in f else torch.zeros(1, L, 3)

    CALLS.clear()
    walls = []
    cls = type(dev_dm)
    dm_call = cls.__call__

    def stepped(self, *ar, **kw):
        t0 = time.perf_counter()
        try:
            return dm_call(self, *ar, **kw)
        finally:
            walls.append((time.perf_counter() - t0) * 1e3)
            STEP[0] += 1

    cls.__call__ = stepped

    sampler = RFD3Sampler(num_timesteps=a.num_timesteps)
    with torch.no_grad():
        sampler.sample(dev_dm, 1, L, coord0, f, init,
                       f["is_motif_atom_with_fixed_coord"],
                       generator=torch.Generator().manual_seed(a.seed))
    cls.__call__ = dm_call
    R._tt = orig_tt

    # --- analysis: warm steps only, step 0 registers ~3400 programs ------------------------
    warm = [c for c in CALLS if c["step"] >= 2]
    n_steps = len({c["step"] for c in warm})
    by_site = defaultdict(list)
    for c in warm:
        by_site[(c["fn"], c["line"], tuple(c["shape"]), c["branch"], c["in_dtype"])].append(c)

    rows = []
    for key, cs in by_site.items():
        fn, line, shape, branch, in_dtype = key
        shas_by_step = defaultdict(list)
        for c in cs:
            shas_by_step[c["step"]].append(c["sha"])
        per_step = len(cs) / n_steps
        tuples = {tuple(v) for v in shas_by_step.values()}
        within = (per_step > 1
                  and all(len(set(v)) < len(v) for v in shas_by_step.values()))
        rows.append({
            "fn": fn, "line": line, "shape": list(shape), "branch": branch,
            "in_dtype": in_dtype,
            "calls_per_step": per_step,
            "ms_per_step": sum(c["ms"] for c in cs) / n_steps,
            "us_per_call": 1e3 * sum(c["ms"] for c in cs) / len(cs),
            "invariant_across_steps": len(tuples) == 1,
            "repeats_within_step": within,
            "distinct_shas": len({c["sha"] for c in cs}),
            "n_calls": len(cs),
        })
    rows.sort(key=lambda r: -r["ms_per_step"])

    tot = sum(r["ms_per_step"] for r in rows)
    across = sum(r["ms_per_step"] for r in rows if r["invariant_across_steps"])
    # a within-step repeat saves only the duplicate calls, not the first one
    within_ms = sum(r["ms_per_step"] * (1 - r["distinct_shas"] / r["n_calls"])
                    for r in rows if not r["invariant_across_steps"] and r["repeats_within_step"])
    med_step = statistics.median(walls[2:]) if len(walls) > 3 else float("nan")

    print(f"\nL={L} atoms  median warm step={med_step:.1f} ms  warm steps={n_steps}")
    print(f"_tt total       {tot:7.3f} ms/step over {sum(r['calls_per_step'] for r in rows):.0f} calls")
    print(f"  invariant across steps  {across:7.3f} ms/step   <- cache once per design")
    print(f"  duplicated within step  {within_ms:7.3f} ms/step   <- cache across recycles\n")
    print(f"{'caller':24} {'line':>5} {'shape':26} {'in':>16} {'br':>4} {'n/st':>5} "
          f"{'ms/st':>7} {'us/call':>8} {'inv':>4} {'rep':>4} {'shas':>5}")
    for r in rows:
        print(f"{r['fn'][:24]:24} {r['line']:5d} {str(r['shape'])[:26]:26} "
              f"{r['in_dtype'].replace('torch.',''):>16} {r['branch']:4d} "
              f"{r['calls_per_step']:5.1f} {r['ms_per_step']:7.3f} {r['us_per_call']:8.1f} "
              f"{str(r['invariant_across_steps'])[:4]:>4} {str(r['repeats_within_step'])[:4]:>4} "
              f"{r['distinct_shas']:5d}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            "atoms": L, "median_warm_step_ms": med_step, "warm_steps": n_steps,
            "step_walls_ms": walls,
            "tt_ms_per_step": tot,
            "cacheable_across_steps_ms_per_step": across,
            "cacheable_within_step_ms_per_step": within_ms,
            "rows": rows,
        }, indent=1))
        print(f"\n[done] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
