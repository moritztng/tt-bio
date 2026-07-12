"""Profile Pairformer component costs on the real 48-block ESMFold2 trunk."""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from types import MethodType

import torch
import ttnn


def _compare(a: torch.Tensor, b: torch.Tensor, chunk: int = 1 << 22) -> dict[str, float | bool]:
    x, y = a.reshape(-1), b.reshape(-1)
    n = x.numel()
    sx = sy = 0.0
    max_abs = 0.0
    finite = True
    for start in range(0, n, chunk):
        xd = x[start : start + chunk].double()
        yd = y[start : start + chunk].double()
        finite = finite and bool(torch.isfinite(xd).all() and torch.isfinite(yd).all())
        max_abs = max(max_abs, float((xd - yd).abs().max()))
        sx += float(xd.sum())
        sy += float(yd.sum())
    mx, my = sx / n, sy / n
    vx = vy = cov = 0.0
    for start in range(0, n, chunk):
        xd = x[start : start + chunk].double() - mx
        yd = y[start : start + chunk].double() - my
        vx += float((xd * xd).sum())
        vy += float((yd * yd).sum())
        cov += float((xd * yd).sum())
    return {
        "pcc": cov / max((vx * vy) ** 0.5, 1e-30),
        "max_abs": max_abs,
        "finite": finite,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--checkpoint", default="biohub/ESMFold2")
    parser.add_argument("--swiglu-ab", action="store_true")
    parser.add_argument("--precision-diag", action="store_true")
    parser.add_argument("--fc2-profile", action="store_true")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(20260711)

    from tt_bio import esmfold2 as E
    from tt_bio._vendor.esmfold2_hf.modeling_esmfold2 import ESMFold2Model

    print("loading checkpoint and 48-block folding trunk", flush=True)
    reference = ESMFold2Model.from_pretrained(args.checkpoint, load_esmc=False).eval()
    prefix = "folding_trunk."
    trunk_state = {
        key[len(prefix):]: value.float()
        for key, value in reference.state_dict().items()
        if key.startswith(prefix)
    }
    n_layers = reference.config.folding_trunk.n_layers
    del reference
    gc.collect()
    trunk = E.FoldingTrunk(n_layers=n_layers)
    trunk.load_state_dict(trunk_state, strict=False)
    del trunk_state
    gc.collect()
    assert n_layers == 48

    if args.precision_diag:
        if len(args.sizes) != 1:
            raise ValueError("--precision-diag accepts exactly one size")
        size = args.sizes[0]
        transition = trunk.module.blocks[0].transition
        generator = torch.Generator().manual_seed(20260711 + size)
        z = torch.randn((1, size, size, E.C_Z), generator=generator)
        z_tt = trunk._from_torch(z)
        x_norm = ttnn.layer_norm(
            z_tt,
            weight=transition.norm_weight,
            bias=transition.norm_bias,
            epsilon=1e-5,
            compute_kernel_config=transition.compute_kernel_config,
        )

        h_linear = transition._lin(x_norm, transition.fc1_weight)
        h_minimal = ttnn.experimental.minimal_matmul(
            input_tensor=x_norm,
            weight_tensor=transition.fc1_weight,
            compute_kernel_config=transition.compute_kernel_config,
            dtype=transition.fc1_weight.dtype,
        )

        weight = transition.weights["1.weight"]
        packed = weight.t()
        rows, two_n = packed.shape
        packed = packed.reshape(rows, 2, -1, 32).permute(0, 2, 1, 3).reshape(rows, two_n)
        packed_weight = ttnn.from_torch(
            packed,
            layout=ttnn.TILE_LAYOUT,
            device=transition.device,
            dtype=transition.fc1_weight.dtype,
        )
        gated_fused = ttnn.experimental.minimal_matmul(
            input_tensor=x_norm,
            weight_tensor=packed_weight,
            compute_kernel_config=transition.compute_kernel_config,
            dtype=transition.fc1_weight.dtype,
            fuse_swiglu=True,
        )
        ttnn.deallocate(x_norm)
        ttnn.deallocate(z_tt)

        linear_1, linear_2 = ttnn.chunk(h_linear, 2, dim=-1)
        minimal_1, minimal_2 = ttnn.chunk(h_minimal, 2, dim=-1)
        gated_linear = ttnn.multiply(ttnn.silu(linear_1), linear_2)
        gated_minimal = ttnn.multiply(ttnn.silu(minimal_1), minimal_2)
        out_linear = transition._lin(gated_linear, transition.fc2_weight)
        out_minimal = transition._lin(gated_minimal, transition.fc2_weight)
        out_fused = transition._lin(gated_fused, transition.fc2_weight)
        ttnn.synchronize_device(trunk.tt_device)

        host = {
            "h_linear": trunk._to_torch(h_linear),
            "h_minimal": trunk._to_torch(h_minimal),
            "gated_linear": trunk._to_torch(gated_linear),
            "gated_minimal": trunk._to_torch(gated_minimal),
            "gated_fused": trunk._to_torch(gated_fused),
            "out_linear": trunk._to_torch(out_linear),
            "out_minimal": trunk._to_torch(out_minimal),
            "out_fused": trunk._to_torch(out_fused),
        }
        record = {
            "N": size,
            "matmul_schedule": _compare(host["h_linear"], host["h_minimal"]),
            "schedule_after_swiglu": _compare(host["gated_linear"], host["gated_minimal"]),
            "fused_epilogue": _compare(host["gated_minimal"], host["gated_fused"]),
            "schedule_after_fc2": _compare(host["out_linear"], host["out_minimal"]),
            "fused_after_fc2": _compare(host["out_minimal"], host["out_fused"]),
            "total_after_fc2": _compare(host["out_linear"], host["out_fused"]),
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        return

    def enable_fused_swiglu() -> None:
        """Use tt-metal main's FC1 matmul+SwiGLU epilogue fusion."""
        for block in trunk.module.blocks:
            transition = block.transition
            weight = transition.weights["1.weight"]
            packed = weight.t()
            rows, two_n = packed.shape
            packed = packed.reshape(rows, 2, -1, 32).permute(0, 2, 1, 3).reshape(rows, two_n)
            fused_weight = ttnn.from_torch(
                packed,
                layout=ttnn.TILE_LAYOUT,
                device=transition.device,
                dtype=transition.fc1_weight.dtype,
            )
            ttnn.deallocate(transition.fc1_weight)
            transition.fc1_weight = fused_weight

            def fused_ffn(self, x):
                x_norm = ttnn.layer_norm(
                    x,
                    weight=self.norm_weight,
                    bias=self.norm_bias,
                    epsilon=1e-5,
                    compute_kernel_config=self.compute_kernel_config,
                )
                gated = ttnn.experimental.minimal_matmul(
                    input_tensor=x_norm,
                    weight_tensor=self.fc1_weight,
                    compute_kernel_config=self.compute_kernel_config,
                    dtype=self.fc1_weight.dtype,
                    fuse_swiglu=True,
                )
                ttnn.deallocate(x_norm)
                out = self._lin(gated, self.fc2_weight)
                ttnn.deallocate(gated)
                return out

            transition._ffn = MethodType(fused_ffn, transition)

    component_attrs = {
        "trimul_out": "tri_out",
        "trimul_in": "tri_in",
        "transition": "transition",
    }

    def execute(z: torch.Tensor, timing: str | None = None):
        totals = defaultdict(float)
        calls = defaultdict(int)
        originals = []
        if timing is not None:
            synchronize = timing == "sync"
            for block in trunk.module.blocks:
                for label, attr in component_attrs.items():
                    original = getattr(block, attr)
                    originals.append((block, attr, original))

                    def wrapped(*call_args, _original=original, _label=label, **call_kwargs):
                        if synchronize:
                            ttnn.synchronize_device(trunk.tt_device)
                        started = time.perf_counter()
                        result = _original(*call_args, **call_kwargs)
                        if synchronize:
                            ttnn.synchronize_device(trunk.tt_device)
                        totals[_label] += time.perf_counter() - started
                        calls[_label] += 1
                        return result

                    setattr(block, attr, wrapped)
        try:
            z_tt = trunk._from_torch(z)
            ttnn.synchronize_device(trunk.tt_device)
            started = time.perf_counter()
            out_tt = trunk.module(z_tt, None)
            ttnn.synchronize_device(trunk.tt_device)
            elapsed = time.perf_counter() - started
            out = trunk._to_torch(out_tt)
            ttnn.deallocate(out_tt)
            return elapsed, out, dict(totals), dict(calls)
        finally:
            for block, attr, original in originals:
                setattr(block, attr, original)

    def execute_fc2_profile(z: torch.Tensor):
        """Synchronously time FC2 and its following residual add in every block."""
        totals = defaultdict(float)
        calls = defaultdict(int)
        fc2_weights = {id(block.transition.fc2_weight) for block in trunk.module.blocks}
        original_linear = ttnn.linear
        residuals = []

        def timed(label, operation, *op_args, **op_kwargs):
            ttnn.synchronize_device(trunk.tt_device)
            started = time.perf_counter()
            result = operation(*op_args, **op_kwargs)
            ttnn.synchronize_device(trunk.tt_device)
            totals[label] += time.perf_counter() - started
            calls[label] += 1
            return result

        def wrapped_linear(*linear_args, **linear_kwargs):
            weight = linear_args[1] if len(linear_args) > 1 else linear_kwargs.get("input_tensor_b")
            if id(weight) in fc2_weights:
                return timed("fc2", original_linear, *linear_args, **linear_kwargs)
            return original_linear(*linear_args, **linear_kwargs)

        ttnn.linear = wrapped_linear
        for block in trunk.module.blocks:
            original_residual = block._residual
            residuals.append((block, original_residual))
            call_index = {"value": 0}

            def wrapped_residual(z_in, update, _original=original_residual, _index=call_index):
                index = _index["value"]
                _index["value"] += 1
                if index == 2:
                    return timed("fc2_residual", _original, z_in, update)
                return _original(z_in, update)

            block._residual = wrapped_residual
        try:
            z_tt = trunk._from_torch(z)
            ttnn.synchronize_device(trunk.tt_device)
            started = time.perf_counter()
            out_tt = trunk.module(z_tt, None)
            ttnn.synchronize_device(trunk.tt_device)
            elapsed = time.perf_counter() - started
            out = trunk._to_torch(out_tt)
            ttnn.deallocate(out_tt)
            return elapsed, out, dict(totals), dict(calls)
        finally:
            ttnn.linear = original_linear
            for block, original_residual in residuals:
                block._residual = original_residual

    for size in args.sizes:
        if args.swiglu_ab and len(args.sizes) != 1:
            raise ValueError("--swiglu-ab accepts exactly one size per process")
        generator = torch.Generator().manual_seed(20260711 + size)
        z = torch.randn((1, size, size, E.C_Z), generator=generator)
        print(f"warming N={size}", flush=True)
        _, warm, _, _ = execute(z)
        del warm
        gc.collect()
        base_s, base, _, _ = execute(z)
        host_s, host, host_parts, calls = execute(z, "host")
        sync_s, sync, sync_parts, _ = execute(z, "sync")
        fc2_record = {}
        fc2_profile = None
        if args.fc2_profile:
            fc2_profile_s, fc2_profile, fc2_parts, fc2_calls = execute_fc2_profile(z)
            residual_s = fc2_parts["fc2_residual"]
            fc2_record = {
                "fc2_profile_trunk_s": fc2_profile_s,
                "fc2_sync_component_s": fc2_parts,
                "fc2_calls": fc2_calls,
                "fc2_profile_parity": _compare(base, fc2_profile),
                "residual_free_trunk_ceiling": base_s / (base_s - residual_s),
            }
        fused_record = {}
        fused = None
        if args.swiglu_ab:
            enable_fused_swiglu()
            _, fused_warm, _, _ = execute(z)
            del fused_warm
            gc.collect()
            fused_s, fused, _, _ = execute(z)
            fused_record = {
                "fused_swiglu_s": fused_s,
                "fused_swiglu_speedup": base_s / fused_s,
                "fused_swiglu_parity": _compare(base, fused),
            }
        record = {
            "N": size,
            "blocks": n_layers,
            "baseline_s": base_s,
            "host_profile_trunk_s": host_s,
            "sync_profile_trunk_s": sync_s,
            "host_enqueue_s": host_parts,
            "sync_component_s": sync_parts,
            "calls": calls,
            "host_parity": _compare(base, host),
            "sync_parity": _compare(base, sync),
            **fc2_record,
            **fused_record,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        del z, base, host, sync
        if fused is not None:
            del fused
        if fc2_profile is not None:
            del fc2_profile
        gc.collect()


if __name__ == "__main__":
    main()
