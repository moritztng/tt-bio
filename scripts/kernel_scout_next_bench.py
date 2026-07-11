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
            **fused_record,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        del z, base, host, sync
        if fused is not None:
            del fused
        gc.collect()


if __name__ == "__main__":
    main()
