#!/usr/bin/env python3
"""Reproduce synthetic source contracts and archived-call refusals on CPU."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from .reduce import (ASSUMPTION, HERE, MANIFEST, ROLES, digest, inspect_observer,
                     reduce_contract, round32, sha)


def operand(role, dims):
    return dict(role=role, shape=dims,
                padded_shape=dims[:-2] + [round32(dims[-2]), round32(dims[-1])],
                dtype="bfloat16", layout="TILE", storage=role)


def contract(family, shapes, compile, name=None):
    known = MANIFEST["families"][family]
    d = dict(family=family, function=known["function"], sources=copy.deepcopy(known["sources"]),
        operands=[operand(r, s) for r, s in zip(ROLES[family], shapes)],
        compile=compile, extra_defines={}, assumptions=ASSUMPTION,
        ownership=dict(id=name or family, scope="one_generic_call", cover="full_disjoint_logical_domain"))
    return dict(schema=1, basis="normalized_source_contract", descriptor=d, descriptor_sha256=digest(d))


def examples():
    result = {}
    mm = dict(transpose_a=False, transpose_b=False, bias=False, activation=False,
              ternary=False, head_major=False, split_outputs=False,
              grid=[2, 2], block=[2, 2, 2], subblock=[1, 1])
    for name, dims in (("matmul_dense", [[32, 32], [32, 32], [32, 32]]),
                       ("matmul_rectangular", [[64, 32], [32, 96], [64, 96]]),
                       ("matmul_flattened", [[2, 64, 32], [32, 96], [2, 64, 96]])):
        result[name] = contract("matmul", dims, copy.deepcopy(mm), name)
    for name, family, dims in (
        ("reblock_dense", "reblock", [[1, 32, 32, 32], [1, 32, 32, 32]]),
        ("reblock_ragged", "reblock", [[1, 33, 33, 64], [1, 64, 33, 33]]),
        ("reblock_reverse", "reblock_back", [[1, 64, 32, 32], [1, 32, 32, 64]])):
        result[name] = contract(family, dims, dict(walk="block"), name)
    for name, dims, off in (
        ("reblock_gated", [[1, 32, 32, 128], [1, 32, 32, 32]], 0),
        ("reblock_gated_row", [[1, 32, 64, 128], [1, 32, 64, 64]], 32),
        ("reblock_gated_tail", [[1, 1, 33, 128], [1, 32, 33, 33]], 32)):
        result[name] = contract("reblock_gated", dims, dict(walk="block", skip_sigmoid=False,
            granularity=4, p_slice=64, g_slice=0, row_off=off), name)
    sd = dict(causal=False, provided_mask=True, padded_mask=False, chunked=False,
        sliding_window=0, attention_sink=False, streaming=False, phases=1,
        q_chunk=32, k_chunk=32, persistent_mask=False, scale=0.125,
        exp_approx_mode=False, qkv_shape=None, fuse_qkv=False, gate_epilogue=False,
        heads_per_worker=1, q_chunks_per_worker=1, kv_buffer_factor=2)
    for name, q, k, mask, family in (
        ("sdpa_dense", [1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32], "sdpa"),
        ("sdpa_rectangular", [1, 2, 32, 32], [1, 2, 64, 32], [1, 2, 32, 64], "sdpa"),
        ("sdpa_broadcast_mask", [2, 2, 32, 32], [2, 2, 64, 32], [1, 1, 32, 64], "sdpa"),
        ("sdpa_gated", [1, 2, 32, 32], [1, 2, 64, 32], [1, 2, 32, 64], "sdpa_gated")):
        dims = [q, k, k, mask, q] + ([q] if family == "sdpa_gated" else [])
        cfg = dict(sd, gate_epilogue=family == "sdpa_gated")
        result[name] = contract(family, dims, cfg, name)
    result["sdpa_persistent_mask"] = copy.deepcopy(result["sdpa_broadcast_mask"])
    result["sdpa_persistent_mask"]["descriptor"]["compile"]["persistent_mask"] = True
    result["sdpa_persistent_mask"]["descriptor"]["ownership"]["id"] = "sdpa_persistent_mask"
    result["sdpa_persistent_mask"]["descriptor_sha256"] = digest(result["sdpa_persistent_mask"]["descriptor"])
    result["sdpa_fused_qkv"] = contract("sdpa_fused_qkv",
        [[2, 32, 64], [64, 192], [1, 2, 32, 32], [2, 2, 32, 32]],
        dict(sd, fuse_qkv=True, qkv_shape=[2, 2, 32, 32]), "sdpa_fused_qkv")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = {}
    for name, value in examples().items():
        (args.out / (name + ".json")).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        rows[name] = reduce_contract(value)
    archived = {name: inspect_observer(HERE / "evidence" / name, 1)
                for name in ("matmul_on.jsonl", "reblock_on.jsonl")}
    report = dict(schema=1, basis="CPU arithmetic only; synthetic cases are not observations",
        source_contract_sha256=sha((HERE / "contracts.json").read_bytes()),
        controls=rows, archived=archived, exact_total=False,
        predicted_model_cycles_saved=0, measured_model_cycles_saved=None, aggregate=None,
        verdict="GO for conditional components; STOP for observed work and exact totals")
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if all(r["status"] == "SOURCE_CONDITIONAL" for r in rows.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
