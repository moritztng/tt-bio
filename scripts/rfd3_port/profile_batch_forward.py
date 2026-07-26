"""Profile one warm RFD3 denoiser forward at a requested in-tensor batch size.

The measured call is bracketed by Tracy signposts so its ttnn operations can be
isolated from model construction, compilation, and warmup in the generated
``ops_perf_results_*.csv`` report.

Example:
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:rfd3-batch \
    python3 -m tracy -p -r -v --profiler-capture-perf-counters=fpu \
    -o /tmp/rfd3-profile-b1 -m scripts.rfd3_port.profile_batch_forward \
    --batch 1 --trace-decoder
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--pdb", type=Path, default=PDB)
    parser.add_argument("--contig", default="A1-10,20,A31-40")
    parser.add_argument("--spec", type=Path, help="JSON InputSpecification; overrides --pdb/--contig")
    parser.add_argument("--trace-decoder", action="store_true")
    parser.add_argument("--compare-sparse-qk", action="store_true")
    parser.add_argument(
        "--sync-profile",
        type=Path,
        help="Write synchronized per-ttnn-op timings as JSON (does not need a Tracy build).",
    )
    parser.add_argument(
        "--stage-profile",
        type=Path,
        help="Write synchronized model-stage timings as JSON.",
    )
    parser.add_argument(
        "--substage-profile",
        action="store_true",
        help="With --stage-profile: also time the methods INSIDE the atom encoder and "
             "decoder (pack/unpack, cross-attention, atom block, host sparse gather). "
             "Fast runtime mode stays on, so unlike --sync-profile the totals stay "
             "comparable to an uninstrumented run; check the reported inflation.",
    )
    return parser.parse_args()


def _shape(value):
    try:
        return list(value.shape)
    except (AttributeError, RuntimeError, TypeError):
        return None


class _SynchronizedOperationProfiler:
    """Measure registered ttnn operations with a device sync at each boundary."""

    def __init__(self, ttnn, device):
        self.ttnn = ttnn
        self.device = device
        self.started_ns = 0
        self.pending = None
        self.rows = []

    def pre(self, operation, args, kwargs):
        self.ttnn.synchronize_device(self.device)
        self.pending = {
            "operation": getattr(operation, "python_fully_qualified_name", str(operation)),
            "input_shapes": [shape for value in args if (shape := _shape(value)) is not None],
            "core_grid": str(kwargs.get("core_grid")) if kwargs.get("core_grid") is not None else None,
        }
        self.started_ns = time.perf_counter_ns()

    def post(self, operation, args, kwargs, output):
        self.ttnn.synchronize_device(self.device)
        row = self.pending or {
            "operation": getattr(operation, "python_fully_qualified_name", str(operation)),
            "input_shapes": [],
            "core_grid": None,
        }
        row["elapsed_ns"] = time.perf_counter_ns() - self.started_ns
        row["output_shape"] = _shape(output)
        self.rows.append(row)


def _stage_wrapper(label, original, ttnn, device, rows):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        ttnn.synchronize_device(device)
        start_ns = time.perf_counter_ns()
        output = original(*args, **kwargs)
        ttnn.synchronize_device(device)
        rows.append(
            {
                "stage": label,
                "input_shapes": [
                    shape for value in args[1:] if (shape := _shape(value)) is not None
                ],
                "elapsed_ns": time.perf_counter_ns() - start_ns,
            }
        )
        return output

    return wrapped


def main() -> None:
    args = parse_args()
    if args.trace_decoder:
        os.environ["RFD3_TRACE_DECODER"] = "1"
        os.environ.setdefault("TT_BIO_TRACE_REGION_SIZE", str(1 << 28))
    if args.sync_profile:
        overrides = json.loads(os.environ.get("TTNN_CONFIG_OVERRIDES", "{}"))
        overrides["enable_fast_runtime_mode"] = False
        os.environ["TTNN_CONFIG_OVERRIDES"] = json.dumps(overrides)

    import ttnn
    from tracy import signpost
    from tt_bio.rfd3 import (
        CompactStreamingDecoder,
        DiffusionTokenEncoder,
        GatedCrossAttention,
        LinearSequenceHead,
        LocalAtomTransformer,
        LocalTokenTransformer,
        RFD3AtomBlock,
        RFD3DiffusionModule,
        build_diffusion_module,
        build_token_initializer,
    )
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification

    if args.spec:
        spec_data = json.loads(args.spec.read_text())
        input_path = Path(spec_data["input"])
        if not input_path.is_absolute():
            input_path = args.spec.parent / input_path
        spec_data["input"] = str(input_path.resolve())
        fixture = f"spec={args.spec}"
    else:
        spec_data = {"input": str(args.pdb), "contig": args.contig}
        fixture = f"pdb={args.pdb} contig={args.contig!r}"
    spec = InputSpecification.from_dict(spec_data)
    spec.validate()
    features = featurize(spec_data["input"], spec)
    features = {
        key: value.float()
        if torch.is_tensor(value) and value.is_floating_point()
        else value
        for key, value in features.items()
    }
    token_weights = torch.load(
        GOLDEN_DIR / "token_initializer.real_weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    diffusion_weights = torch.load(
        GOLDEN_DIR / "diffusion_module.real_weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    token_initializer = build_token_initializer(token_weights)
    diffusion_module = build_diffusion_module(diffusion_weights)
    with torch.no_grad():
        initial = token_initializer(
            {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in features.items()
            }
        )

    length = features["ref_pos"].shape[0]
    generator = torch.Generator().manual_seed(42)
    noisy = torch.randn(args.batch, length, 3, generator=generator) * 16.0
    times = torch.full((args.batch,), 8.0)

    with torch.no_grad():
        if args.compare_sparse_qk:
            os.environ["RFD3_SPARSE_QK"] = "0"
        baseline = diffusion_module(X_noisy_L=noisy, t=times, f=features, **initial)
        ttnn.synchronize_device(diffusion_module.device)
        if args.compare_sparse_qk:
            os.environ["RFD3_SPARSE_QK"] = "1"
        op_profiler = _SynchronizedOperationProfiler(ttnn, diffusion_module.device)
        stage_rows = []
        with ExitStack() as stack:
            if args.sync_profile:
                stack.enter_context(ttnn.register_pre_operation_hook(op_profiler.pre))
                stack.enter_context(ttnn.register_post_operation_hook(op_profiler.post))
            if args.stage_profile:
                stage_methods = (
                    ("process_time", RFD3DiffusionModule, "_process_time"),
                    ("downcast_c", RFD3DiffusionModule, "_downcast_c"),
                    ("atom_encoder", LocalAtomTransformer, "__call__"),
                    ("downcast_q", RFD3DiffusionModule, "_downcast_q"),
                    ("token_encoder", DiffusionTokenEncoder, "__call__"),
                    ("token_dit", LocalTokenTransformer, "__call__"),
                    ("decoder", CompactStreamingDecoder, "__call__"),
                    ("sequence_head", LinearSequenceHead, "__call__"),
                )
                if args.substage_profile:
                    import tt_bio.rfd3 as _rfd3

                    stage_methods += (
                        # decoder: the traced core, then the eager tail around it
                        ("dec.traced_core", CompactStreamingDecoder, "_run_device_sparse_traced"),
                        ("dec.eager_core", CompactStreamingDecoder, "run_device"),
                        ("dec.pack", CompactStreamingDecoder, "_pack_atoms_device"),
                        ("dec.unpack", CompactStreamingDecoder, "_unpack_atoms_device"),
                        ("dec.design_buffers", CompactStreamingDecoder, "_design_buffers"),
                        # shared by the decoder's 3 upcasts + its downcast
                        ("gca", GatedCrossAttention, "run_device"),
                        ("gca.mask_upload", GatedCrossAttention, "_prepare_additive_mask"),
                        # shared by encoder + decoder atom stacks
                        ("atom_block", RFD3AtomBlock, "__call__"),
                        # host-side sparse pair work
                        ("sparse_qk_host", _rfd3, "_sparse_qk_host"),
                        ("sparse_qk_inputs", _rfd3, "_sparse_qk_inputs"),
                    )
                for label, cls, method_name in stage_methods:
                    original = getattr(cls, method_name)
                    stack.enter_context(
                        patch.object(
                            cls,
                            method_name,
                            _stage_wrapper(
                                label,
                                original,
                                ttnn,
                                diffusion_module.device,
                                stage_rows,
                            ),
                        )
                    )
            signpost(f"RFD3_BATCH_{args.batch}_START")
            start = time.perf_counter()
            output = diffusion_module(
                X_noisy_L=noisy, t=times, f=features, **initial
            )
            ttnn.synchronize_device(diffusion_module.device)
            elapsed = time.perf_counter() - start
            signpost(f"RFD3_BATCH_{args.batch}_STOP")

    if args.sync_profile:
        args.sync_profile.parent.mkdir(parents=True, exist_ok=True)
        args.sync_profile.write_text(
            json.dumps(
                {
                    "batch": args.batch,
                    "fixture": fixture,
                    "tokens": int(features["restype"].shape[0]),
                    "atoms": int(length),
                    "trace_decoder": args.trace_decoder,
                    "wall_elapsed_ns": round(elapsed * 1e9),
                    "operations": op_profiler.rows,
                },
                indent=2,
            )
        )
    if args.stage_profile:
        args.stage_profile.parent.mkdir(parents=True, exist_ok=True)
        args.stage_profile.write_text(
            json.dumps(
                {
                    "batch": args.batch,
                    "fixture": fixture,
                    "tokens": int(features["restype"].shape[0]),
                    "atoms": int(length),
                    "trace_decoder": args.trace_decoder,
                    "wall_elapsed_ns": round(elapsed * 1e9),
                    "stages": stage_rows,
                },
                indent=2,
            )
        )

    print(
        f"PROFILE_RESULT batch={args.batch} trace_decoder={args.trace_decoder} "
        f"elapsed_ms={elapsed * 1000:.3f} finite={torch.isfinite(output['X_L']).all().item()}",
        flush=True,
    )

    if args.compare_sparse_qk:
        for key in ("X_L", "sequence_logits_I"):
            ref = baseline[key].float().flatten()
            got = output[key].float().flatten()
            ref_c, got_c = ref - ref.mean(), got - got.mean()
            pcc = torch.dot(ref_c, got_c) / (ref_c.norm() * got_c.norm())
            print(
                f"SPARSE_QK_PARITY {key} pcc={pcc.item():.9f} "
                f"maxabs={(ref - got).abs().max().item():.6f}", flush=True,
            )


if __name__ == "__main__":
    main()
