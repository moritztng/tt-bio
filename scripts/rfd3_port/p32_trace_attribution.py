"""p32: where the traced decoder's 12.3% loss goes at 3359 atoms.

The traced path is not "the eager path plus a trace". `_run_device_sparse_traced` has to
stage the gathered pair features into a persistent buffer, and `ttnn.embedding` cannot write
into one -- so it keeps the HOST advanced-index gather that p26's resident pair table
replaced with a device gather, and it uploads the head-replicated scatter index in full
instead of uploading one head and concatenating on device. This times those pieces against
the trace replay itself, per leg, so the sign of the trade is measured rather than argued.

Run: TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=... PYTHONPATH=$PWD \
       python3 scripts/rfd3_port/p32_trace_attribution.py --contig "A1-10,230,A31-40"
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contig", default="A1-10,230,A31-40")
    ap.add_argument("--pdb", type=Path, default=PDB)
    ap.add_argument("--timesteps", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    import ttnn
    import tt_bio.rfd3.model as R
    import tt_bio.tenstorrent as TTd
    from tt_bio.rfd3.featurize import featurize
    from tt_bio.rfd3.input import InputSpecification
    from tt_bio.rfd3.sampler import RFD3Sampler

    TTd.get_device(trace_region_size=1 << 30)
    acc = defaultdict(float)

    def timed(name, fn):
        def wrapper(*a, **kw):
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            acc[name] += time.perf_counter() - t0
            return out
        return wrapper

    R._sparse_qk_host = timed("host pair gather (traced only)", R._sparse_qk_host)
    R._sparse_qk_inputs = timed("device pair gather + index (eager only)", R._sparse_qk_inputs)
    R._tt_refresh = timed("persistent-buffer refresh (traced only)", R._tt_refresh)
    ttnn.execute_trace = timed("trace replay (blocking)", ttnn.execute_trace)
    # Host dispatch time of EXACTLY the region the trace replaces: run_device is the
    # captured graph, and eagerly it only enqueues (no sync), so this is the ceiling on
    # what tracing the decoder can ever save.
    R.CompactStreamingDecoder.run_device = timed(
        "run_device host dispatch (eager only)", R.CompactStreamingDecoder.run_device)
    R.ttnn = ttnn

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
    steps = args.timesteps - 1
    coord0 = f["motif_pos"].float().unsqueeze(0)

    with torch.no_grad():
        init = ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})

        def leg(name, traced, timesteps):
            dm.decoder.trace = traced
            acc.clear()
            t0 = time.perf_counter()
            RFD3Sampler(num_timesteps=timesteps).sample(
                dm, 1, L, coord0, f, init, f["is_motif_atom_with_fixed_coord"],
                generator=torch.Generator().manual_seed(42))
            total = (time.perf_counter() - t0) / (timesteps - 1) * 1e3
            print(f"\n{name}: {total:.2f} ms/step total")
            for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
                per = v / (timesteps - 1) * 1e3
                print(f"    {k:<42} {per:7.2f} ms/step  ({per / total * 100:4.1f}%)")

        for traced in (False, True):
            dm.decoder.trace = traced
            RFD3Sampler(num_timesteps=args.warmup).sample(
                dm, 1, L, coord0, f, init, f["is_motif_atom_with_fixed_coord"],
                generator=torch.Generator().manual_seed(7))
        print(f"=== p32 decoder-trace attribution  L={L} atoms  steps/leg={steps} ===")
        leg("eager decoder", False, args.timesteps)
        leg("traced decoder", True, args.timesteps)


if __name__ == "__main__":
    main()
