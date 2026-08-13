"""Where the whole-invocation wall goes, phase by phase, inside one process.

An unlocked both-on design walled 413 s against an unlocked both-off 391 s -- the wrong sign against
the -16.43 % the locked ms/step A/B measured. Two candidate explanations: co-tenancy, or a fixed
per-process cost the median warm step cannot see by construction.

The second half is answerable without a quiet box, because it is a WITHIN-process split rather than
a comparison of two walls. Each phase is timestamped in the same process:

  build     weights -> device modules
  init      the token initializer
  first     the first diffusion step: JIT compile, program-config calibration, kernel build
  warm      the median of the remaining steps

Co-tenancy inflates all four, so the phase SHARES and the arm-to-arm difference in `first` are what
this is read for -- not the absolute seconds.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402

from tt_bio.rfd3.design import build_diffusion_module, build_token_initializer  # noqa: E402
from tt_bio.rfd3.featurize import featurize  # noqa: E402
from tt_bio.rfd3.input import InputSpecification  # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler  # noqa: E402

STEPS: list[float] = []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="perf/dsfix/targets/R4_9q6y_A.pdb")
    ap.add_argument("--contig", default="A1-585,100")
    ap.add_argument("--ckpt", default="/home/ttuser/.boltz/rfd3/weights")
    ap.add_argument("--num_timesteps", type=int, default=6)
    ap.add_argument("--designs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    t_start = time.perf_counter()
    spec = InputSpecification.from_dict({"input": a.pdb, "contig": a.contig})
    spec.validate()
    f = featurize(a.pdb, spec)
    cap = Path(a.ckpt)
    dev_ti = build_token_initializer(
        torch.load(cap / "token_initializer.real_weights.pt", map_location="cpu", weights_only=True))
    dev_dm = build_diffusion_module(
        torch.load(cap / "diffusion_module.real_weights.pt", map_location="cpu", weights_only=True))
    t_build = time.perf_counter()
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    L = init["Q_L_init"].shape[0]
    coord0 = f["motif_pos"].float().unsqueeze(0) if "motif_pos" in f else torch.zeros(1, L, 3)
    t_init = time.perf_counter()

    cls = type(dev_dm)
    dm_call = cls.__call__

    def stepped(self, *ar, **kw):
        t0 = time.perf_counter()
        try:
            return dm_call(self, *ar, **kw)
        finally:
            STEPS.append(time.perf_counter() - t0)

    cls.__call__ = stepped

    sampler = RFD3Sampler(num_timesteps=a.num_timesteps)
    with torch.no_grad():
        sampler.sample(dev_dm, a.designs, L, coord0, f, init,
                       f["is_motif_atom_with_fixed_coord"],
                       generator=[torch.Generator().manual_seed(a.seed + i)
                                  for i in range(a.designs)])
    t_end = time.perf_counter()

    build = t_build - t_start
    initt = t_init - t_build
    first = STEPS[0]
    warm = statistics.median(STEPS[2:]) if len(STEPS) > 3 else STEPS[-1]
    total = t_end - t_start
    # what a 200-step invocation would cost if the warm step held: fixed + 200 * warm
    fixed = build + initt + (first - warm)
    proj200 = fixed + 200 * warm

    row = {"tag": a.tag, "build_s": build, "init_s": initt, "first_step_s": first,
           "warm_step_s": warm, "total_s": total, "fixed_s": fixed,
           "projected_200step_s": proj200, "n_steps": len(STEPS)}
    print(f"[{a.tag or 'phase'}] build {build:6.2f}s  init {initt:6.2f}s  "
          f"first-step {first:6.2f}s  warm {warm * 1e3:7.1f} ms  total {total:6.2f}s", flush=True)
    print(f"[{a.tag or 'phase'}] FIXED (build+init+first-step excess) = {fixed:6.2f}s   "
          f"projected 200-step invocation = {proj200:6.1f}s", flush=True)

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(row, indent=2))
        print(f"[done] {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
