"""p18: segment ONE real RFD3 diffusion step into host-torch / host<->device transfer /
device-dispatch, at a chosen design size and batch.

p13 segmented the DEVICE side with hardware counters and found only ~52% of wall clock is
device-busy at 3359 atoms. It never segmented the other ~48%. This does that: it counts every
``ttnn.from_torch`` / ``ttnn.to_torch`` crossing (time AND bytes) and times the five host-only
torch kernels the step calls, so the non-device half stops being one undifferentiated number.

The interesting quantity is how each bucket SCALES with design size, because the TT-vs-GPU gap
is size-dependent (3.5x at 419 atoms, 15.6x at 3359) and a bucket that is flat in size cannot
explain it.

Usage (shipped config -- do NOT export RFD3_TRACE_DECODER):
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:... \
  python3 scripts/rfd3_port/profile_host_boundary.py --contig "A1-10,230,A31-40" --batch 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()

_t = defaultdict(float)
_n = defaultdict(int)
_b = defaultdict(int)
_on = False


def _acc(key, dt, nbytes=0):
    if _on:
        _t[key] += dt
        _n[key] += 1
        _b[key] += nbytes


def _wrap(mod, name, key, bytes_of):
    orig = getattr(mod, name)

    def wrapped(*a, **kw):
        t0 = time.perf_counter()
        out = orig(*a, **kw)
        _acc(key, time.perf_counter() - t0, bytes_of(a, out))
        return out

    setattr(mod, name, wrapped)
    return orig


def _tensor_bytes(t):
    try:
        return int(t.numel()) * int(t.element_size())
    except Exception:
        try:  # ttnn tensor
            n = 1
            for d in t.shape:
                n *= int(d)
            return n * 2
        except Exception:
            return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contig", default="A1-10,20,A31-40")
    ap.add_argument("--spec", type=Path)
    ap.add_argument("--pdb", type=Path, default=PDB)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--timesteps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=2, help="steps excluded from the accounting")
    args = ap.parse_args()

    import ttnn
    import tt_bio.rfd3 as R
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification
    from tt_bio.rfd3_sampler import RFD3Sampler

    if args.spec:
        data = json.loads(args.spec.read_text())
        p = Path(data["input"])
        data["input"] = str((p if p.is_absolute() else args.spec.parent / p).resolve())
    else:
        data = {"input": str(args.pdb), "contig": args.contig}
    spec = InputSpecification.from_dict(data)
    spec.validate()
    f = featurize(data["input"], spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in f.items()}
    L = int(f["ref_pos"].shape[0])
    I = int(f["atom_to_token_map"].max().item()) + 1

    ti_w = torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt", map_location="cpu", weights_only=True)
    dm_w = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt", map_location="cpu", weights_only=True)
    ti = R.build_token_initializer(ti_w)
    dm = R.build_diffusion_module(dm_w)

    # --- transfers: every host<->device crossing, by direction ---
    _wrap(ttnn, "to_torch", "xfer.to_torch(D->H)", lambda a, o: _tensor_bytes(o))
    _wrap(ttnn, "from_torch", "xfer.from_torch(H->D)", lambda a, o: _tensor_bytes(a[0]) if a else 0)
    _wrap(ttnn, "copy_host_to_device_tensor", "xfer.copy_h2d", lambda a, o: _tensor_bytes(a[0]) if a else 0)

    # --- host-only torch kernels called from the per-step path ---
    for name, key in (
        ("_create_attention_indices", "host.attn_indices"),
        ("_sparse_qk_host", "host.sparse_qk_gather"),
        ("_scatter_mean", "host.scatter_mean"),
        ("_scaled_distogram_bins", "host.distogram_bins"),
        ("_dense_attention_mask", "host.dense_mask"),
    ):
        _wrap(R, name, key, lambda a, o: 0)

    # --- components (wall clock, inclusive; they nest, reported separately) ---
    # The token encoder is wrapped on run_device, NOT __call__: since p19 kept z on the card
    # the per-step path calls run_device directly, and wrapping __call__ silently dropped the
    # component from the table.
    for obj, attr, key in (
        (type(dm.encoder), "__call__", "comp.encoder"),
        (type(dm.decoder), "__call__", "comp.decoder"),
        (type(dm.diffusion_transformer), "__call__", "comp.dit"),
        (type(dm.diffusion_token_encoder), "run_device", "comp.token_encoder"),
    ):
        _wrap(obj, attr, key, lambda a, o: 0)

    # --- inside the decoder, now the largest single component (p19 s7) ---
    _wrap(R, "_sparse_qk_inputs", "dec.sparse_qk_inputs", lambda a, o: 0)
    for attr, key in (
        ("run_device", "dec.core_loop"),
        ("_pack_atoms_device", "dec.pack"),
        ("_unpack_atoms_device", "dec.unpack"),
    ):
        _wrap(type(dm.decoder), attr, key, lambda a, o: 0)
    _wrap(type(dm.decoder.downcast), "run_device", "dec.downcast", lambda a, o: 0)
    _wrap(dm, "_downcast_c", "comp.downcast_c", lambda a, o: 0)
    _wrap(dm, "_downcast_q", "comp.downcast_q", lambda a, o: 0)

    coord0 = f["motif_pos"].float().unsqueeze(0)
    global _on
    with torch.no_grad():
        init = ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
        g = torch.Generator().manual_seed(42)
        # warmup run: JIT, kernel cache, program cache all land here (p17: a cold read is 13x)
        RFD3Sampler(num_timesteps=args.warmup).sample(
            dm, args.batch, L, coord0, f, init, f["is_motif_atom_with_fixed_coord"], generator=g)
        _t.clear(), _n.clear(), _b.clear()
        _on = True
        g = torch.Generator().manual_seed(42)
        t0 = time.perf_counter()
        RFD3Sampler(num_timesteps=args.timesteps).sample(
            dm, args.batch, L, coord0, f, init, f["is_motif_atom_with_fixed_coord"], generator=g)
        wall = time.perf_counter() - t0
        _on = False

    steps = args.timesteps
    print(f"\n=== host-boundary breakdown  L={L} atoms  I={I} tokens  batch={args.batch} "
          f"steps={steps}  trace_decoder={os.environ.get('RFD3_TRACE_DECODER', '0')} ===")
    print(f"wall {wall * 1e3:.1f} ms total, {wall / steps * 1e3:.1f} ms/step\n")
    print(f"{'bucket':<28}{'ms/step':>10}{'% step':>8}{'calls/step':>12}{'MB/step':>10}")
    xfer = host = 0.0
    for k in sorted(_t):
        ms = _t[k] / steps * 1e3
        if k.startswith("xfer."):
            xfer += ms
        elif k.startswith("host."):
            host += ms
        mb = _b[k] / steps / 1e6
        pct = ms / (wall / steps * 1e3) * 100
        mbs = f"{mb:>10.1f}" if mb else f"{'-':>10}"
        print(f"{k:<28}{ms:>10.2f}{pct:>7.1f}%{_n[k] / steps:>12.1f}{mbs}")
    step_ms = wall / steps * 1e3
    print(f"\n{'TRANSFER total':<28}{xfer:>10.2f}{xfer / step_ms * 100:>7.1f}%")
    print(f"{'HOST-TORCH total':<28}{host:>10.2f}{host / step_ms * 100:>7.1f}%")
    print(f"{'REST (device+dispatch)':<28}{step_ms - xfer - host:>10.2f}"
          f"{(step_ms - xfer - host) / step_ms * 100:>7.1f}%")


if __name__ == "__main__":
    main()
