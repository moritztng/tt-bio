"""Profile TriangleAttention and OuterProductMean with real Protenix-v2 weights."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager

import torch
import ttnn


def _compare(a: torch.Tensor, b: torch.Tensor, chunk: int = 1 << 22) -> dict[str, float | bool]:
    x, y = a.reshape(-1), b.reshape(-1)
    n = x.numel()
    sx = sy = sxx = syy = sxy = 0.0
    max_abs = 0.0
    finite = True
    for start in range(0, n, chunk):
        xd = x[start : start + chunk].double()
        yd = y[start : start + chunk].double()
        finite = finite and bool(torch.isfinite(xd).all() and torch.isfinite(yd).all())
        max_abs = max(max_abs, float((xd - yd).abs().max()))
        sx += float(xd.sum())
        sy += float(yd.sum())
        sxx += float((xd * xd).sum())
        syy += float((yd * yd).sum())
        sxy += float((xd * yd).sum())
    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    return {
        "pcc": cov / max((vx * vy) ** 0.5, 1e-30),
        "max_abs": max_abs,
        "finite": finite,
    }


def _load_checkpoint(path: str) -> dict[str, torch.Tensor]:
    print(f"loading real checkpoint: {path}", flush=True)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    checkpoint = checkpoint.get("model", checkpoint)
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }


@contextmanager
def _replace_permute(variant: str | None):
    original = ttnn.permute

    def replacement(x, dims, *args, **kwargs):
        dims = tuple(dims)
        if variant == "triatt-ending" and len(x.shape) == 3 and dims == (1, 0, 2):
            return ttnn.transpose(x, 0, 1, *args, **kwargs)
        if variant == "triatt-bias" and len(x.shape) == 4 and dims == (0, 3, 1, 2):
            first = ttnn.transpose(x, 2, 3)
            out = ttnn.transpose(first, 1, 2, *args, **kwargs)
            ttnn.deallocate(first)
            return out
        if variant == "opm" and len(x.shape) == 3:
            if dims == (1, 2, 0):
                first = ttnn.transpose(x, 0, 1)
                out = ttnn.transpose(first, 1, 2, *args, **kwargs)
                ttnn.deallocate(first)
                return out
            if dims == (2, 1, 0):
                return ttnn.transpose(x, 0, 2, *args, **kwargs)
            if dims == (0, 2, 1):
                return ttnn.transpose(x, 1, 2, *args, **kwargs)
        return original(x, dims, *args, **kwargs)

    if variant is not None:
        ttnn.permute = replacement
    try:
        yield
    finally:
        ttnn.permute = original


@contextmanager
def _profile_operations(device, paths: list[tuple[object, str]], synchronize: bool):
    totals = defaultdict(float)
    calls = defaultdict(int)
    originals = []
    for owner, name in paths:
        original = getattr(owner, name)
        originals.append((owner, name, original))

        def wrapped(*args, _name=name, _original=original, **kwargs):
            if synchronize:
                ttnn.synchronize_device(device)
            started = time.perf_counter()
            result = _original(*args, **kwargs)
            if synchronize:
                ttnn.synchronize_device(device)
            totals[_name] += time.perf_counter() - started
            calls[_name] += 1
            return result

        setattr(owner, name, wrapped)
    try:
        yield totals, calls
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


def _operation_paths() -> list[tuple[object, str]]:
    return [
        (ttnn, "reshape"),
        (ttnn, "layer_norm"),
        (ttnn, "linear"),
        (ttnn, "permute"),
        (ttnn, "transpose"),
        (ttnn, "to_layout"),
        (ttnn, "matmul"),
        (ttnn, "multiply_"),
        (ttnn, "add"),
        (ttnn, "unsqueeze"),
        (ttnn, "squeeze"),
        (ttnn.experimental, "minimal_matmul"),
        (ttnn.experimental, "nlp_create_qkv_heads"),
        (ttnn.experimental, "nlp_concat_heads"),
        (ttnn.transformer, "scaled_dot_product_attention"),
    ]


def _device_setup():
    from tt_bio import tenstorrent as T

    T.set_fast_mode(False)
    device = T.get_device()
    config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    return T, device, config


def _build_pairformer(state, config):
    from tt_bio.protenix_weights import remap_pairformer_block
    from tt_bio.tenstorrent import Pairformer

    blocks = 1 + max(
        int(key.split("pairformer_stack.blocks.", 1)[1].split(".", 1)[0])
        for key in state
        if key.startswith("pairformer_stack.blocks.")
    )
    combined = {}
    for index in range(blocks):
        prefix = f"pairformer_stack.blocks.{index}."
        block = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        for key, value in remap_pairformer_block(block).items():
            combined[f"layers.{index}.{key}"] = value
    return Pairformer(blocks, 32, 8, 384 // 16, 16, True, combined, config)


def _pairformer(args, state, T, device, config):
    pairformer = _build_pairformer(state, config)
    component_attrs = {
        "trimul_out": "triangle_multiplication_start",
        "trimul_in": "triangle_multiplication_end",
        "triatt_start": "triangle_attention_start",
        "triatt_end": "triangle_attention_end",
        "transition_z": "transition_z",
        "attention_pair_bias": "attention_pair_bias",
        "transition_s": "transition_s",
    }

    def execute(s_host, z_host, timing=None, variant=None):
        totals = defaultdict(float)
        calls = defaultdict(int)
        originals = []
        if timing is not None:
            synchronize = timing == "sync"
            for block in pairformer.blocks:
                for label, attr in component_attrs.items():
                    original = getattr(block, attr)
                    originals.append((block, attr, original))

                    def wrapped(*op_args, _label=label, _original=original, **op_kwargs):
                        if synchronize:
                            ttnn.synchronize_device(device)
                        started = time.perf_counter()
                        result = _original(*op_args, **op_kwargs)
                        if synchronize:
                            ttnn.synchronize_device(device)
                        totals[_label] += time.perf_counter() - started
                        calls[_label] += 1
                        return result

                    setattr(block, attr, wrapped)
        try:
            s = ttnn.from_torch(s_host, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
            z = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
            with _replace_permute(variant):
                ttnn.synchronize_device(device)
                started = time.perf_counter()
                s_out, z_out = pairformer(s, z)
                ttnn.synchronize_device(device)
                elapsed = time.perf_counter() - started
            s_cpu = torch.Tensor(ttnn.to_torch(s_out)).float().reshape(s_host.shape)
            z_cpu = torch.Tensor(ttnn.to_torch(z_out)).float().reshape(z_host.shape)
            ttnn.deallocate(s_out)
            ttnn.deallocate(z_out)
            return elapsed, s_cpu, z_cpu, dict(totals), dict(calls)
        finally:
            for block, attr, original in originals:
                setattr(block, attr, original)

    for size in args.sizes:
        generator = torch.Generator().manual_seed(20260712 + size)
        s_host = torch.randn((1, size, 384), generator=generator, dtype=torch.bfloat16)
        z_host = torch.randn((1, size, size, 256), generator=generator, dtype=torch.bfloat16)
        print(f"warming real 48-block Pairformer N={size}", flush=True)
        _, warm_s, warm_z, _, _ = execute(s_host, z_host)
        del warm_s, warm_z
        baseline_s, base_s, base_z, _, _ = execute(s_host, z_host)
        host_s, host_out_s, host_out_z, host_parts, calls = execute(s_host, z_host, "host")
        sync_s, sync_out_s, sync_out_z, sync_parts, _ = execute(s_host, z_host, "sync")

        variants = {}
        for variant in ("triatt-ending", "triatt-bias"):
            _, warm_s, warm_z, _, _ = execute(s_host, z_host, variant=variant)
            del warm_s, warm_z
            elapsed, out_s, out_z, _, _ = execute(s_host, z_host, variant=variant)
            variants[variant] = {
                "trunk_s": elapsed,
                "speedup": baseline_s / elapsed,
                "s_parity": _compare(base_s, out_s),
                "z_parity": _compare(base_z, out_z),
            }
            del out_s, out_z

        z_device = ttnn.from_torch(z_host, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        with _profile_operations(device, _operation_paths(), synchronize=True) as (op_times, op_calls):
            ttnn.synchronize_device(device)
            started = time.perf_counter()
            for block in pairformer.blocks:
                for attention in (block.triangle_attention_start, block.triangle_attention_end):
                    out = attention(z_device)
                    ttnn.deallocate(out)
            ttnn.synchronize_device(device)
            triatt_isolated_s = time.perf_counter() - started
        ttnn.deallocate(z_device)

        record = {
            "component": "pairformer",
            "N": size,
            "blocks": len(pairformer.blocks),
            "baseline_trunk_s": baseline_s,
            "host_profile_trunk_s": host_s,
            "sync_profile_trunk_s": sync_s,
            "host_enqueue_s": host_parts,
            "sync_component_s": sync_parts,
            "component_calls": calls,
            "host_parity": {"s": _compare(base_s, host_out_s), "z": _compare(base_z, host_out_z)},
            "sync_parity": {"s": _compare(base_s, sync_out_s), "z": _compare(base_z, sync_out_z)},
            "triatt_isolated_s": triatt_isolated_s,
            "triatt_op_sync_s": dict(op_times),
            "triatt_op_calls": dict(op_calls),
            "variants": variants,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        del s_host, z_host, base_s, base_z, host_out_s, host_out_z, sync_out_s, sync_out_z
        gc.collect()


def _build_opm(state, config):
    from tt_bio.protenix_weights import remap_outer_product_mean
    from tt_bio.tenstorrent import OuterProductMean

    modules = []
    for index in range(4):
        prefix = f"msa_module.blocks.{index}.outer_product_mean_msa."
        weights = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        modules.append(OuterProductMean(remap_outer_product_mean(weights), config))
    return modules


def _real_trunk(args, state, T, device, config):
    import yaml

    from tt_bio.protenix import AtomAttentionEncoder, Protenix, Trunk
    from tt_bio.protenix_data import build_protein_features

    def under(prefix):
        return {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}

    input_aae = AtomAttentionEncoder(
        under("input_embedder.atom_attention_encoder."), config
    )
    trunk = Trunk(state, config)
    dummy = object.__new__(Protenix)
    component_attrs = {
        "trimul_out": "triangle_multiplication_start",
        "trimul_in": "triangle_multiplication_end",
        "triatt_start": "triangle_attention_start",
        "triatt_end": "triangle_attention_end",
        "transition_z": "transition_z",
        "attention_pair_bias": "attention_pair_bias",
        "transition_s": "transition_s",
    }

    def to_device(x):
        return ttnn.from_torch(
            x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
        )

    def execute(feats, s_inputs, relp, timing=False):
        totals = defaultdict(float)
        calls = defaultdict(int)
        originals = []
        msa_originals = []

        def wrap(label, operation):
            def timed(*op_args, **op_kwargs):
                if timing:
                    ttnn.synchronize_device(device)
                started = time.perf_counter()
                result = operation(*op_args, **op_kwargs)
                if timing:
                    ttnn.synchronize_device(device)
                totals[label] += time.perf_counter() - started
                calls[label] += 1
                return result

            return timed

        if timing:
            for block in trunk.PF.blocks:
                for label, attr in component_attrs.items():
                    original = getattr(block, attr)
                    originals.append((block, attr, original))
                    setattr(block, attr, wrap(label, original))
            for index, (opm, pwa, transition, pair_layer) in enumerate(trunk.MSA):
                msa_originals.append((index, trunk.MSA[index]))
                trunk.MSA[index] = (wrap("opm", opm), pwa, transition, pair_layer)
        try:
            ttnn.synchronize_device(device)
            started = time.perf_counter()
            s_out, z_out = trunk(
                feats,
                s_inputs,
                relp,
                feats["token_bonds"],
                n_cycles=1,
            )
            ttnn.synchronize_device(device)
            elapsed = time.perf_counter() - started
            s_host = torch.Tensor(ttnn.to_torch(s_out)).float().reshape(
                s_inputs.shape[0], 384
            )
            z_host = torch.Tensor(ttnn.to_torch(z_out)).float().reshape(
                1, s_inputs.shape[0], s_inputs.shape[0], trunk.C_Z
            )
            ttnn.deallocate(s_out)
            ttnn.deallocate(z_out)
            return elapsed, s_host, z_host, dict(totals), dict(calls)
        finally:
            for block, attr, original in originals:
                setattr(block, attr, original)
            for index, original in msa_originals:
                trunk.MSA[index] = original

    examples = Path(args.examples)
    for size in args.sizes:
        source = examples / ("615.yaml" if size <= 615 else "1303.yaml")
        sequence = yaml.safe_load(source.read_text())["sequences"][0]["protein"]["sequence"][:size]
        if len(sequence) != size:
            raise ValueError(f"{source} has only {len(sequence)} residues, need {size}")
        print(f"building actual protein features from {source.name}, N={size}", flush=True)
        feats = build_protein_features(sequence)
        fi = dummy._atom_feat_inputs(feats)
        n_atoms = fi["N"]
        n_tokens = fi["NT"]
        atom_to_token_mean = fi["S"].t()
        atom_to_token_mean = atom_to_token_mean / (
            atom_to_token_mean.sum(-1, keepdim=True) + 1e-6
        )
        deletion_mean = feats["deletion_mean"]
        if deletion_mean.dim() == 1:
            deletion_mean = deletion_mean.reshape(-1, 1)
        s_inputs_tt = input_aae(
            to_device(feats["ref_pos"]),
            to_device(fi["ref_charge_asinh"]),
            to_device(feats["ref_mask"].reshape(n_atoms, 1)),
            to_device(fi["f_in"]),
            to_device(fi["d"]),
            to_device(fi["v"]),
            to_device(fi["invd"]),
            fi["mt"],
            to_device(atom_to_token_mean),
            to_device(feats["restype"]),
            to_device(feats["profile"]),
            to_device(deletion_mean),
        )
        s_inputs = torch.Tensor(ttnn.to_torch(s_inputs_tt)).float()[:n_tokens]
        ttnn.deallocate(s_inputs_tt)
        relp = dummy._generate_relp(feats)

        print(f"warming one complete actual-input Protenix trunk cycle N={size}", flush=True)
        _, warm_s, warm_z, _, _ = execute(feats, s_inputs, relp)
        elapsed, out_s, out_z, parts, calls = execute(feats, s_inputs, relp, timing=True)
        record = {
            "component": "actual_input_trunk",
            "N": size,
            "msa_depth": int(feats["msa"].shape[0]),
            "atoms": n_atoms,
            "cycle_s": elapsed,
            "sync_component_s": parts,
            "component_calls": calls,
            "parity": {
                "s": _compare(warm_s, out_s),
                "z": _compare(warm_z, out_z),
            },
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        del feats, fi, s_inputs, relp, warm_s, warm_z, out_s, out_z
        gc.collect()


def _triatt_projection(args, state, T, device, config):
    pairformer = _build_pairformer(state, config)
    attention = pairformer.blocks[0].triangle_attention_start
    packed_host = torch.cat(
        [
            attention.weights["linear_q.weight"],
            attention.weights["linear_k.weight"],
            attention.weights["linear_v.weight"],
            attention.weights["linear_g.weight"],
        ],
        dim=0,
    ).t()
    packed_weight = ttnn.from_torch(
        packed_host, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
    )
    qkv_width = attention.n_heads * attention.head_dim * 3

    def projections(x_norm, packed):
        if packed:
            both = ttnn.experimental.minimal_matmul(
                input_tensor=x_norm,
                weight_tensor=packed_weight,
                compute_kernel_config=config,
                dtype=ttnn.bfloat16,
            )
            qkv = both[..., :qkv_width]
            gate = both[..., qkv_width:]
            return qkv, gate, both
        qkv = ttnn.experimental.minimal_matmul(
            input_tensor=x_norm,
            weight_tensor=attention.qkv_weight,
            compute_kernel_config=config,
            dtype=ttnn.bfloat16,
        )
        gate = ttnn.experimental.minimal_matmul(
            input_tensor=x_norm,
            weight_tensor=attention.g_weight,
            compute_kernel_config=config,
            dtype=ttnn.bfloat16,
        )
        return qkv, gate, None

    def timed(x_norm, packed):
        ttnn.synchronize_device(device)
        started = time.perf_counter()
        qkv, gate, both = projections(x_norm, packed)
        ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - started
        ttnn.deallocate(qkv)
        ttnn.deallocate(gate)
        if both is not None:
            ttnn.deallocate(both)
        return elapsed

    def capture(x_norm, packed):
        qkv, gate, both = projections(x_norm, packed)
        ttnn.synchronize_device(device)
        qkv_host = torch.Tensor(ttnn.to_torch(qkv)).float()
        gate_host = torch.Tensor(ttnn.to_torch(gate)).float()
        ttnn.deallocate(qkv)
        ttnn.deallocate(gate)
        if both is not None:
            ttnn.deallocate(both)
        return qkv_host, gate_host

    for size in args.sizes:
        generator = torch.Generator().manual_seed(20260712 + size)
        x_host = torch.randn((size, size, 256), generator=generator, dtype=torch.bfloat16)
        x = ttnn.from_torch(x_host, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        x_norm = ttnn.layer_norm(
            x,
            weight=attention.layer_norm_weight,
            bias=attention.layer_norm_bias,
            epsilon=1e-5,
            compute_kernel_config=config,
        )
        timed(x_norm, False)
        timed(x_norm, True)
        baseline_samples = [timed(x_norm, False) for _ in range(args.repeats)]
        packed_samples = [timed(x_norm, True) for _ in range(args.repeats)]
        baseline = capture(x_norm, False)
        packed = capture(x_norm, True)
        baseline_s = statistics.median(baseline_samples)
        packed_s = statistics.median(packed_samples)
        record = {
            "component": "triangle_attention_qkvg_projection",
            "N": size,
            "baseline_s": baseline_s,
            "baseline_samples_s": baseline_samples,
            "packed_s": packed_s,
            "packed_samples_s": packed_samples,
            "speedup": baseline_s / packed_s,
            "qkv_parity": _compare(baseline[0], packed[0]),
            "gate_parity": _compare(baseline[1], packed[1]),
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        ttnn.deallocate(x_norm)
        ttnn.deallocate(x)
        del x_host, baseline, packed
        gc.collect()


def _opm(args, state, T, device, config):
    modules = _build_opm(state, config)

    def execute(m, variant=None, profile=False):
        paths = _operation_paths() if profile else []
        with _replace_permute(variant), _profile_operations(device, paths, synchronize=profile) as (times, calls):
            ttnn.synchronize_device(device)
            started = time.perf_counter()
            for module in modules:
                out = module(m, None, args.msa_depth)
                ttnn.deallocate(out)
            ttnn.synchronize_device(device)
            elapsed = time.perf_counter() - started
        return elapsed, dict(times), dict(calls)

    def capture(m, variant=None):
        saved = {}
        with _replace_permute(variant):
            for index in (0, len(modules) - 1):
                out = modules[index](m, None, args.msa_depth)
                ttnn.synchronize_device(device)
                saved[index] = torch.Tensor(ttnn.to_torch(out)).float()
                ttnn.deallocate(out)
        return saved

    for size in args.sizes:
        generator = torch.Generator().manual_seed(20260712 + size)
        m_host = torch.randn(
            (1, args.msa_depth, size, 128), generator=generator, dtype=torch.bfloat16
        )
        m = ttnn.from_torch(m_host, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        print(f"warming four real OPM blocks N={size}, MSA={args.msa_depth}", flush=True)
        execute(m)
        baseline_samples = [execute(m)[0] for _ in range(args.repeats)]
        execute(m, variant="opm")
        variant_samples = [execute(m, variant="opm")[0] for _ in range(args.repeats)]
        profile_s, op_times, op_calls = execute(m, profile=True)
        baseline = capture(m)
        variant = capture(m, variant="opm")
        profiled = capture(m)
        baseline_s = statistics.median(baseline_samples)
        variant_s = statistics.median(variant_samples)

        record = {
            "component": "outer_product_mean",
            "N": size,
            "msa_depth": args.msa_depth,
            "blocks": len(modules),
            "baseline_s": baseline_s,
            "baseline_samples_s": baseline_samples,
            "transpose_decomposition_s": variant_s,
            "transpose_decomposition_samples_s": variant_samples,
            "transpose_decomposition_speedup": baseline_s / variant_s,
            "transpose_decomposition_parity": {
                str(index): _compare(baseline[index], variant[index]) for index in baseline
            },
            "sync_profile_s": profile_s,
            "sync_profile_parity": {
                str(index): _compare(baseline[index], profiled[index]) for index in baseline
            },
            "op_sync_s": op_times,
            "op_calls": op_calls,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        ttnn.deallocate(m)
        del m_host, baseline, variant, profiled
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "component", choices=("pairformer", "real-trunk", "triatt-projection", "opm")
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--msa-depth", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--checkpoint", default="/home/moritz/.boltz/protenix-v2.pt")
    parser.add_argument("--examples", default="examples")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(20260712)

    state = _load_checkpoint(args.checkpoint)
    T, device, config = _device_setup()
    if args.component == "pairformer":
        _pairformer(args, state, T, device, config)
    elif args.component == "real-trunk":
        _real_trunk(args, state, T, device, config)
    elif args.component == "triatt-projection":
        _triatt_projection(args, state, T, device, config)
    else:
        _opm(args, state, T, device, config)


if __name__ == "__main__":
    main()
