"""Attribute the RFD3 in-forward batching shortfall to individual stages.

p10 left one question open: above ~2000 atoms a D=8 sampler step costs MORE than
eight D=1 steps, so batching stops amortizing and mildly regresses. Answering it
needs a different instrument from `profile_batch_forward.py`:

* a single warm forward is the wrong unit. p10 measured a host-side change as
  exactly 0 on one forward and 1.12x over the sampler loop, because in a
  one-shot forward host work overlaps device work. This script therefore runs
  the real ``RFD3Sampler`` loop and reports per-stage ms **per sampler step**.
* the interesting quantity is not a stage's cost but its *amortization*: the
  ratio ``D=8 cost / (8 x D=1 cost)``. 1.0 means the stage is purely
  per-design (batching neither helps nor hurts), <1.0 means batching shares
  work, >1.0 means batching actively costs more than running the designs one
  at a time.

Both batch sizes run in one process against one device context, back to back,
so neither compilation nor thermal drift lands on only one of them. Fast
runtime mode stays on; the sync-bracketed stage timers add a measurable but
uniform overhead, reported as ``inflation``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import functools
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()

# Stages that nest inside another stage in this list. Their time is already
# counted by the parent, so they are reported in a separate block.
NESTED = {
    "atom_block", "gca", "gca.mask_upload", "sparse_qk_inputs", "sparse_qk_host",
    "dec.pack", "dec.unpack", "dec.traced_core", "pairformer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--timesteps", type=int, default=6)
    parser.add_argument("--pdb", type=Path, default=PDB)
    parser.add_argument("--contig", default="A1-10,20,A31-40")
    parser.add_argument("--spec", type=Path, help="JSON InputSpecification; overrides --pdb/--contig")
    parser.add_argument("--trace-decoder", action="store_true")
    parser.add_argument("--out", type=Path, help="write the raw per-stage rows as JSON")
    return parser.parse_args()


def _stage_wrapper(label, original, ttnn, device, rows):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        ttnn.synchronize_device(device)
        start_ns = time.perf_counter_ns()
        output = original(*args, **kwargs)
        ttnn.synchronize_device(device)
        rows.append((label, time.perf_counter_ns() - start_ns))
        return output

    return wrapped


def _stage_methods(rfd3):
    """(label, owner, attribute) triples, top-level stages first."""
    m = (
        # --- host glue in RFD3DiffusionModule.__call__ / _process_ ---
        ("attn_indices", rfd3, "_create_attention_indices"),
        ("scatter_mean", rfd3, "_scatter_mean"),
        ("bucketize", rfd3, "_scaled_distogram_bins"),
        ("scale_in", rfd3.RFD3DiffusionModule, "scale_positions_in"),
        ("scale_out", rfd3.RFD3DiffusionModule, "scale_positions_out"),
        ("process_time", rfd3.RFD3DiffusionModule, "_process_time"),
        ("downcast_c", rfd3.RFD3DiffusionModule, "_downcast_c"),
        ("downcast_q", rfd3.RFD3DiffusionModule, "_downcast_q"),
        ("grouping_buffers", rfd3.RFD3DiffusionModule, "_grouping_buffers"),
        ("encoder_downcast_traced", rfd3.RFD3DiffusionModule, "_encoder_downcast_traced"),
        # --- the model stages ---
        ("atom_encoder", rfd3.LocalAtomTransformer, "__call__"),
        ("token_encoder", rfd3.DiffusionTokenEncoder, "__call__"),
        ("token_dit", rfd3.LocalTokenTransformer, "__call__"),
        ("decoder", rfd3.CompactStreamingDecoder, "__call__"),
        ("sequence_head", rfd3.LinearSequenceHead, "__call__"),
        # --- nested, for attribution inside the stages above ---
        ("atom_block", rfd3.RFD3AtomBlock, "__call__"),
        ("pairformer", rfd3.PairformerBlock, "__call__"),
        ("gca", rfd3.GatedCrossAttention, "run_device"),
        ("gca.mask_upload", rfd3.GatedCrossAttention, "_prepare_additive_mask"),
        ("sparse_qk_inputs", rfd3, "_sparse_qk_inputs"),
        ("sparse_qk_host", rfd3, "_sparse_qk_host"),
        ("dec.pack", rfd3.CompactStreamingDecoder, "_pack_atoms_device"),
        ("dec.unpack", rfd3.CompactStreamingDecoder, "_unpack_atoms_device"),
        ("dec.traced_core", rfd3.CompactStreamingDecoder, "_run_device_sparse_traced"),
    )
    return m


def main() -> None:
    args = parse_args()
    if args.trace_decoder:
        os.environ["RFD3_TRACE_DECODER"] = "1"
        os.environ.setdefault("TT_BIO_TRACE_REGION_SIZE", str(1 << 28))

    import ttnn
    import tt_bio.rfd3 as rfd3
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification
    from tt_bio.rfd3_sampler import RFD3Sampler

    if args.spec:
        spec_data = json.loads(args.spec.read_text())
        input_path = Path(spec_data["input"])
        if not input_path.is_absolute():
            input_path = args.spec.parent / input_path
        spec_data["input"] = str(input_path.resolve())
        fixture = f"spec={args.spec}"
    else:
        spec_data = {"input": str(args.pdb), "contig": args.contig}
        fixture = f"pdb={args.pdb.name} contig={args.contig!r}"
    spec = InputSpecification.from_dict(spec_data)
    spec.validate()
    features = featurize(spec_data["input"], spec)
    features = {
        key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
        for key, value in features.items()
    }
    token_weights = torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt",
                               map_location="cpu", weights_only=True)
    diffusion_weights = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt",
                                   map_location="cpu", weights_only=True)
    token_initializer = rfd3.build_token_initializer(token_weights)
    diffusion_module = rfd3.build_diffusion_module(diffusion_weights)
    with torch.no_grad():
        initial = token_initializer({k: (v.clone() if torch.is_tensor(v) else v)
                                     for k, v in features.items()})

    length = features["ref_pos"].shape[0]
    fixed = features["is_motif_atom_with_fixed_coord"]
    coord = features["motif_pos"].float().unsqueeze(0)
    steps = args.timesteps - 1
    print(f"fixture: {fixture} I={features['restype'].shape[0]} L={length} "
          f"trace_decoder={os.environ.get('RFD3_TRACE_DECODER') == '1'} "
          f"timesteps={args.timesteps} steps={steps}", flush=True)

    def run(batch, instrument):
        rows = []
        sampler = RFD3Sampler(num_timesteps=args.timesteps)
        with torch.no_grad(), ExitStack() as stack:
            if instrument:
                for label, owner, attr in _stage_methods(rfd3):
                    stack.enter_context(patch.object(
                        owner, attr,
                        _stage_wrapper(label, getattr(owner, attr), ttnn,
                                       diffusion_module.device, rows)))
            start = time.perf_counter()
            output, _ = sampler.sample(
                diffusion_module, batch, length, coord, features, initial, fixed,
                generator=torch.Generator().manual_seed(2000 + batch))
            elapsed = time.perf_counter() - start
        finite = bool(torch.isfinite(output).all().item())
        return elapsed / steps * 1000.0, rows, finite

    per_batch = {}
    for batch in args.batches:
        with torch.no_grad():  # warmup: compile every program at this shape
            RFD3Sampler(num_timesteps=4).sample(
                diffusion_module, batch, length, coord, features, initial, fixed,
                generator=torch.Generator().manual_seed(1000 + batch))
        plain_ms, _, _ = run(batch, instrument=False)
        inst_ms, rows, finite = run(batch, instrument=True)
        totals, counts = defaultdict(float), defaultdict(int)
        for label, ns in rows:
            totals[label] += ns / 1e6 / steps
            counts[label] += 1
        per_batch[batch] = {
            "plain_ms_per_step": plain_ms,
            "instrumented_ms_per_step": inst_ms,
            "inflation": inst_ms / plain_ms,
            "finite": finite,
            "stages": {k: {"ms_per_step": totals[k], "calls_per_step": counts[k] / steps}
                       for k in sorted(totals)},
        }
        print(f"D={batch:<3d} plain {plain_ms:9.2f} ms/step   instrumented {inst_ms:9.2f} "
              f"(inflation {inst_ms / plain_ms:.2f}x)  finite={finite}", flush=True)

    base = args.batches[0]
    for batch in args.batches[1:]:
        b0, bn = per_batch[base], per_batch[batch]
        scale = batch / base
        print(f"\n=== amortization D={batch} vs D={base} "
              f"(ratio = D{batch} / ({scale:g} x D{base}); 1.00 = purely per-design, "
              f"<1 = shared, >1 = batching costs extra) ===")
        header = (f"{'stage':<26}{'D' + str(base) + ' ms':>10}{'D' + str(batch) + ' ms':>10}"
                  f"{'ratio':>8}{'excess ms':>11}{'calls':>7}")
        for block, title in ((False, "top-level"), (True, "nested (already counted above)")):
            print(f"-- {title}")
            print(header)
            keys = [k for k in bn["stages"] if (k in NESTED) == block]
            keys.sort(key=lambda k: -(bn["stages"][k]["ms_per_step"]
                                      - scale * b0["stages"].get(k, {"ms_per_step": 0.0})["ms_per_step"]))
            for k in keys:
                d0 = b0["stages"].get(k, {"ms_per_step": 0.0})["ms_per_step"]
                dn = bn["stages"][k]["ms_per_step"]
                ratio = dn / (scale * d0) if d0 else float("nan")
                print(f"{k:<26}{d0:10.2f}{dn:10.2f}{ratio:8.2f}{dn - scale * d0:11.2f}"
                      f"{bn['stages'][k]['calls_per_step']:7.1f}")
        top0 = sum(v["ms_per_step"] for k, v in b0["stages"].items() if k not in NESTED)
        topn = sum(v["ms_per_step"] for k, v in bn["stages"].items() if k not in NESTED)
        print(f"{'SUM top-level':<26}{top0:10.2f}{topn:10.2f}"
              f"{topn / (scale * top0):8.2f}{topn - scale * top0:11.2f}")
        print(f"{'step (uninstrumented)':<26}{b0['plain_ms_per_step']:10.2f}"
              f"{bn['plain_ms_per_step']:10.2f}"
              f"{bn['plain_ms_per_step'] / (scale * b0['plain_ms_per_step']):8.2f}"
              f"{bn['plain_ms_per_step'] - scale * b0['plain_ms_per_step']:11.2f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"fixture": fixture, "atoms": int(length),
             "tokens": int(features["restype"].shape[0]),
             "timesteps": args.timesteps, "batches": per_batch}, indent=2))


if __name__ == "__main__":
    main()
