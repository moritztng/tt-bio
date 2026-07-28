"""p30: interleaved A/B for the decoder's dense-bias reuse across recycles.

The decoder runs its three atom blocks twice per diffusion step (n_recycle=2) against the
same gathered pair features and the same neighbour index, so each block's dense attention
bias is bit-identical between the two calls. `GatedCrossAttention._sparse_bias_f32` builds
it on the first call only when the caller passes a cache dict; `CompactStreamingDecoder`
does. That removes three of a step's nine `ttnn.scatter` calls (4.66 ms each at 3359 atoms)
and three fp32 typecasts (0.82 ms each).

Interleaved, because p14 measured a non-interleaved first read at +13.3% against an honest
+5.2%: the two legs alternate on one hot card in one process, `--alternations` times, and
the report is the median leg. Setting `decoder._bias_cache = None` is the OFF leg, so both
legs run the identical code path.

Every leg re-seeds and compares its sampled coordinates against the first OFF leg's, so
bit-exactness is proven at TRAJECTORY level in the same run that measures the speedup --
not from an op-level PCC (p25: an op-level win can wash out in the loop).

Run: TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=... PYTHONPATH=$PWD \
       python3 scripts/rfd3_port/p30_bias_cache_ab.py --contig "A1-10,230,A31-40" --batch 1
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contig", default="A1-10,230,A31-40")
    ap.add_argument("--pdb", type=Path, default=PDB)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--timesteps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--alternations", type=int, default=3)
    ap.add_argument("--json_out", type=Path)
    args = ap.parse_args()

    import tt_bio.rfd3 as R
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification
    from tt_bio.rfd3_sampler import RFD3Sampler

    data = {"input": str(args.pdb), "contig": args.contig}
    spec = InputSpecification.from_dict(data)
    spec.validate()
    f = featurize(data["input"], spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
         for k, v in f.items()}
    L = int(f["ref_pos"].shape[0])

    ti = R.build_token_initializer(torch.load(
        GOLDEN_DIR / "token_initializer.real_weights.pt", map_location="cpu", weights_only=True))
    dm = R.build_diffusion_module(torch.load(
        GOLDEN_DIR / "diffusion_module.real_weights.pt", map_location="cpu", weights_only=True))
    steps = args.timesteps - 1          # the sampler walks consecutive schedule pairs

    coord0 = f["motif_pos"].float().unsqueeze(0)
    with torch.no_grad():
        init = ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})

        def leg(on):
            """One timed trajectory with the cache on or off. Returns (ms/step, coords)."""
            dm.decoder._bias_cache = {} if on else None
            g = torch.Generator().manual_seed(42)
            t0 = time.perf_counter()
            out = RFD3Sampler(num_timesteps=args.timesteps).sample(
                dm, args.batch, L, coord0, f, init,
                f["is_motif_atom_with_fixed_coord"], generator=g)
            ms = (time.perf_counter() - t0) / steps * 1e3
            x = out[0] if isinstance(out, tuple) else out    # sample() -> (X_L, traj)
            return ms, x.detach().clone().float()

        # p17: a cold read is 13x. Warm both paths before the first timed leg.
        for on in (False, True):
            dm.decoder._bias_cache = {} if on else None
            RFD3Sampler(num_timesteps=args.warmup).sample(
                dm, args.batch, L, coord0, f, init,
                f["is_motif_atom_with_fixed_coord"],
                generator=torch.Generator().manual_seed(7))

        off, on, ref, maxabs = [], [], None, 0.0
        for i in range(args.alternations):
            ms_off, x_off = leg(False)
            ms_on, x_on = leg(True)
            if ref is None:
                ref = x_off
            for x in (x_off, x_on):
                maxabs = max(maxabs, (x - ref).abs().max().item())
            off.append(ms_off)
            on.append(ms_on)
            print(f"  alternation {i + 1}: OFF {ms_off:7.2f} ms/step   "
                  f"ON {ms_on:7.2f} ms/step   ({(ms_off / ms_on - 1) * 100:+5.1f}%)",
                  flush=True)

    m_off, m_on = statistics.median(off), statistics.median(on)
    print(f"\n=== p30 bias-reuse A/B  L={L} atoms  batch={args.batch}  "
          f"steps/leg={steps}  alternations={args.alternations} ===")
    print(f"OFF (shipped)      median {m_off:7.2f} ms/step   legs {[round(v, 1) for v in off]}")
    print(f"ON  (bias reused)  median {m_on:7.2f} ms/step   legs {[round(v, 1) for v in on]}")
    print(f"speedup            {m_off / m_on:.4f}x  ({(m_off / m_on - 1) * 100:+.1f}%)")
    print(f"trajectory maxabs vs first OFF leg: {maxabs:.3e}  "
          f"{'BIT-EXACT' if maxabs == 0.0 else 'NOT BIT-EXACT -- do not ship'}")
    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"L": L, "batch": args.batch, "steps_per_leg": steps, "off_ms": off,
             "on_ms": on, "off_median": m_off, "on_median": m_on,
             "speedup": m_off / m_on, "trajectory_maxabs": maxabs}, indent=1))


if __name__ == "__main__":
    main()
