#!/usr/bin/env python3
"""Source-conditional work components. Standard library only; never imports a TT runtime."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from perf.c10_flop_contract.audit import matrix_work, shape

MANIFEST = json.loads((HERE / "contracts.json").read_text())
ROLE_REGISTRY = json.loads((ROOT / "perf/c10_generic_identity/roles.json").read_text())
ASSUMPTION = "listed source; no injected defines; API lowering and cached binary unverified"


class Refusal(ValueError):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode())


def require(ok, why):
    if not ok:
        raise Refusal(why)


def keys(value, expected, where):
    require(isinstance(value, dict) and set(value) == set(expected.split()),
            f"{where}: missing or unsupported fields")


def positive(value):
    return type(value) is int and value > 0


def round32(n):
    return (n + 31) // 32 * 32


def unknown(reason, **identity):
    return dict(status="REFUSED", reason=reason, logical_matrix_flops=None,
                conventional=None, source_schedule=None, compulsory_logical_bytes=None,
                issued_arithmetic=None, sfpu_flops=None, physical_bytes=None,
                exact_total=False, **identity)


def tensor(t):
    keys(t, "role shape padded_shape dtype layout storage", "operand")
    s, p = shape(t["shape"]), shape(t["padded_shape"])
    require(len(s) == len(p) and all(x <= y for x, y in zip(s, p)), "invalid padding")
    require(t["dtype"] == "bfloat16" and t["layout"] == "TILE", "requires bf16 TILE")
    require(isinstance(t["storage"], str) and bool(t["storage"]), "missing storage identity")
    return s


def aligned(t):
    s = tensor(t)
    require(s == tuple(t["padded_shape"]) and s[-1] % 32 == s[-2] % 32 == 0,
            "this contract requires tile-aligned unpadded operands")
    return s


def flags(c, expected):
    keys(c, " ".join(expected), "compile flags")
    for k, v in expected.items():
        if v is not None:
            require(type(c[k]) is type(v) and c[k] == v, f"unsupported compile flag {k}")


def count_matmul(d, row):
    c = d["compile"]
    flags(c, dict(transpose_a=False, transpose_b=False, bias=False, activation=False,
                  ternary=False, head_major=False, split_outputs=False, grid=None,
                  block=None, subblock=[1, 1]))
    a, b, o = [aligned(t) for t in d["operands"]]
    require(all(x == 1 for x in b[:-2]), "batched right operand is not supported by this wrapper")
    m, k, n = math.prod(a[:-1]), a[-1], b[-1]
    require(b[-2] == k and o == a[:-1] + (n,), "matmul role/shape mismatch")
    work = matrix_work((m, k), (k, n), (m, n))
    row["logical_matrix_flops"] = work["shape_flops"]
    row["matrix_components"] = {"AB": work["shape_flops"]}
    gx, gy = c["grid"]
    mb, kb, nb = c["block"]
    require(all(positive(x) for x in (gx, gy, mb, kb, nb)), "invalid grid/block")
    # The wrapper's transpose flag changes the core axes, NOT the tensor contraction.
    ma, na = (gx, gy) if m > n else (gy, gx)
    mt, nt, kt = m // 32, n // 32, k // 32
    mp, np, kp = ((mt + ma - 1) // ma * ma,
                  (nt + na - 1) // na * na, (kt + kb - 1) // kb * kb)
    # compute.cpp clips the last M/N block to the core's padded range. Subblock=1
    # makes every clipped block divisible; no extra subblock lanes are assumed.
    tile_products = mp * np * kp
    row["source_schedule"] = dict(
        basis="conditional source loops, not machine instructions",
        matrix_padded_extents=[mp * 32, kp * 32, np * 32],
        matmul_block_calls=tile_products,
        padded_domain_matrix_flops=matrix_work((mp * 32, kp * 32), (kp * 32, np * 32))["shape_flops"],
        conditions="full rectangular grid; wrapper ownership; 1x1 subblocks; no extra defines")
    row["compulsory_logical_bytes"] = byte_terms(
        {"left": math.prod(a), "right": math.prod(b)}, math.prod(o))


def byte_terms(reads, written):
    return dict(read_by_role={k: 2 * n for k, n in reads.items()}, write=2 * written,
                total=2 * (sum(reads.values()) + written),
                basis="unique required logical elements per call, bf16, distinct storage; not DRAM traffic")


def count_reblock(d, row):
    family, c = d["family"], d["compile"]
    gated = family == "reblock_gated"
    flags(c, dict(walk="block", **(dict(skip_sigmoid=False, granularity=None,
          p_slice=None, g_slice=None, row_off=None) if gated else {})))
    x, out = d["operands"]
    s, o = tensor(x), tensor(out)
    require(len(s) == len(o) == 4 and s[0] == o[0] == 1, "requires rank-four batch one")
    if family == "reblock_back":
        _, channels, n, n2 = s
        require(n == n2 and n % 32 == 0 and o == (1, n, n, channels), "reverse shape/ragged refusal")
        r = n
        groups = (n // 32)**2 * (channels // 32)
        tiles = groups * 32
    else:
        _, r, n, width = s
        _, channels, n1, n2 = o
        require(n == n1 == n2 and (r <= n if gated else r == n), "forward shape mismatch")
        if gated:
            ps, gs, off, gran = (c[k] for k in ("p_slice", "g_slice", "row_off", "granularity"))
            require(all(type(v) is int and v >= 0 and v % 32 == 0 for v in (ps, gs, off)),
                    "slice/row offsets must be nonnegative tile boundaries")
            require(positive(gran) and gran in (1, 2, 4), "unsupported gate granularity")
            require(ps + channels <= width and gs + channels <= width, "slice outside projection")
            require(ps + channels <= gs or gs + channels <= ps, "overlapping gate slices unsupported")
            require(off + r <= n and (r % 32 == 0 or off + r == n),
                    "row block must end on tile boundary or logical tail")
            groups = ((r + 31) // 32) * ((n + 31) // 32) * (channels // 32)
        else:
            require(width == channels, "channel mismatch")
            groups = ((n + 31) // 32)**2
        require(width % 32 == 0, "input channels not tile aligned")
        tiles = groups * 32 * (1 if gated else channels // 32)
    require(channels > 0 and channels % 32 == 0, "channels not tile aligned")
    for t in (x, out):
        z = t["shape"]
        require(t["padded_shape"] == z[:-2] + [round32(z[-2]), round32(z[-1])],
                "unexpected reblock padded shape")
    elements = r * n * channels
    row["logical_matrix_flops"] = 0  # Proven permutation/gate, not opaque generic zero.
    row["matrix_components"] = {}
    row["conventional"] = dict(multiply=elements, sigmoid_evaluations=elements) if gated else {}
    row["source_schedule"] = dict(
        basis="conditional source loops; full disjoint group cover assumed",
        groups=groups, transpose_wh_tile_calls=tiles,
        sigmoid_tile_calls=tiles if gated else 0,
        mul_binary_tile_calls=tiles if gated else 0,
        padded_lane_slots=tiles * 1024,
        note="tile API calls/lane slots are not scalar issued FLOPs or SFPU instructions")
    row["compulsory_logical_bytes"] = byte_terms(
        {"projection_slice": elements, "gate_slice": elements} if gated else {"input": elements}, elements)
    if gated:
        row["output_region"] = {"rows": [c["row_off"], c["row_off"] + r], "channels": channels}


def count_sdpa(d, row):
    family, c = d["family"], d["compile"]
    fused, gated = family == "sdpa_fused_qkv", family == "sdpa_gated"
    flags(c, dict(causal=False, provided_mask=True, padded_mask=False, chunked=False,
                  sliding_window=0, attention_sink=False, streaming=False, phases=1,
                  q_chunk=None, k_chunk=None, persistent_mask=None,
                  scale=None, exp_approx_mode=None, qkv_shape=None,
                  fuse_qkv=fused, gate_epilogue=gated, heads_per_worker=None,
                  q_chunks_per_worker=None, kv_buffer_factor=None))
    require(type(c["scale"]) in (int, float) and math.isfinite(c["scale"]) and c["scale"] > 0,
            "requires finite positive scale")
    require(type(c["persistent_mask"]) is bool and type(c["exp_approx_mode"]) is bool,
            "missing mask/exp flags")
    dims = [aligned(t) for t in d["operands"]]
    if fused:
        x, w, mask, out = dims
        q = k = v = shape(c["qkv_shape"])
        require(len(q) == 4, "missing explicit internal QKV shape")
    else:
        q, k, v, mask, out = dims[:5]
        require(c["qkv_shape"] is None, "internal QKV shape only belongs to fused projection")
    require(len(q) == len(k) == len(v) == len(mask) == len(out) == 4, "SDPA requires rank four")
    b, h, sq, dh = q
    sk = k[-2]
    require(k == v == (b, h, sk, dh) and out == q, "SDPA head/batch/value dimensions unsupported")
    require(all(z % 32 == 0 for z in (sq, sk, dh)), "ragged SDPA unsupported")
    require(mask[0] in (1, b) and mask[1] in (1, h) and mask[2:] == (sq, sk), "mask broadcast mismatch")
    qc, kc = c["q_chunk"], c["k_chunk"]
    require(all(positive(z) and z % 32 == 0 for z in (qc, kc)), "invalid SDPA chunk")
    require(sq % qc == sk % kc == 0, "padded chunk/mask schedule unsupported")
    require(all(positive(c[k]) for k in ("heads_per_worker", "q_chunks_per_worker", "kv_buffer_factor")),
            "missing worker/buffer limits")
    if c["persistent_mask"]:
        require(mask[0] == 1 and qc == sq and c["heads_per_worker"] == c["q_chunks_per_worker"] == 1,
                "persistent mask requires batch broadcast and one head/Q chunk per worker")
    if gated or fused:
        require(dh == 32 and qc == sq, "fusion requires DH=32 and one Q chunk")
    if gated:
        require(dims[-1] == out, "gate shape mismatch")
    qk = matrix_work(q, k, transpose_b=True)
    pv = matrix_work(qk["output_shape"], v, out)
    terms = {"QK_transpose": qk["shape_flops"], "PV": pv["shape_flops"]}
    if fused:
        require(c["heads_per_worker"] == c["q_chunks_per_worker"] == 1
                and c["kv_buffer_factor"] >= sk // kc, "fused QKV worker/buffer requirements")
        require(sq == sk and x == (b, sq, x[-1]) and len(w) == 2
                and w == (x[-1], 3 * h * dh), "fused projection shape mismatch")
        terms["QKV_projection"] = matrix_work((b * sq, x[-1]), w)["shape_flops"]
    row["matrix_components"] = terms
    row["logical_matrix_flops"] = sum(terms.values())
    scores, rows = b * h * sq * sk, b * h * sq
    row["conventional"] = dict(score_scale_multiply=scores, mask_add=scores,
        stable_softmax_subtract=scores, max_comparisons=rows * (sk - 1),
        sum_add=rows * (sk - 1), exp_evaluations=scores,
        reciprocal_evaluations=rows, probability_normalize_multiply=scores)
    if gated:
        row["conventional"].update(gate_multiply=math.prod(out), sigmoid_evaluations=math.prod(out))
    row["conventional_basis"] = (
        "dense real-arithmetic reference decomposition only; online softmax rescales accumulators, "
        "fuses scale into exp and uses matmul_reduce; these are NOT source operation totals")
    row["source_schedule"] = dict(basis="source-conditional primary contractions only",
        qk_chunk_pairs=b * h * (sq // qc) * (sk // kc),
        pv_chunk_pairs=b * h * (sq // qc) * (sk // kc),
        primary_matrix_flops=sum(terms.values()),
        reduction_matrix_flops=None,
        conditions="single phase; full disjoint B/head/Q cover; no causal/padded mask; fusion one head per worker",
        note="matmul_reduce, online recurrence, SFPU and packing remain uncounted")
    reads = ({"projection_input": math.prod(x), "projection_weight": math.prod(w)} if fused else
             {"query": math.prod(q), "key": math.prod(k), "value": math.prod(v)})
    reads["mask"] = math.prod(mask)
    if gated:
        reads["gate"] = math.prod(out)
    row["compulsory_logical_bytes"] = byte_terms(reads, math.prod(out))


ROLES = {
    "matmul": ["left", "right", "output"],
    "reblock": ["input", "output"], "reblock_back": ["input", "output"],
    "reblock_gated": ["projection_and_gate", "output"],
    "sdpa": ["query", "key", "value", "mask", "output"],
    "sdpa_gated": ["query", "key", "value", "mask", "output", "gate"],
    "sdpa_fused_qkv": ["projection_input", "projection_weight", "mask", "output"],
}


def reduce_contract(item):
    """One normalized, explicitly hypothetical source contract; never a model aggregate."""
    try:
        keys(item, "schema basis descriptor descriptor_sha256", "contract")
        require(type(item["schema"]) is int and item["schema"] == 1
                and item["basis"] == "normalized_source_contract", "unsupported input basis")
        d = item["descriptor"]
        require(digest(d) == item["descriptor_sha256"], "normalized descriptor hash mismatch")
        keys(d, "family function sources operands compile extra_defines ownership assumptions", "descriptor")
        family = d["family"]
        require(family in MANIFEST["families"], "unsupported family")
        known = MANIFEST["families"][family]
        require(d["function"] == known["function"] and d["sources"] == known["sources"],
                "source hash/function mismatch")
        for path, expected in known["sources"].items():
            require(sha((ROOT / path).read_bytes()) == expected, f"source bytes changed: {path}")
        wrapper = ("tt_bio/mm_generic.py" if family == "matmul" else
                   "tt_bio/reblock_permute.py" if family.startswith("reblock") else "tt_bio/sdpa_generic.py")
        require(d["function"] in ROLE_REGISTRY.get(known["sources"][wrapper], {}),
                "wrapper missing from shared observer role registry")
        require(d["extra_defines"] == {} and d["assumptions"] == ASSUMPTION,
                "unreviewed defines or source assumptions")
        own = d["ownership"]
        keys(own, "id scope cover", "ownership")
        require(isinstance(own["id"], str) and bool(own["id"]) and own["scope"] == "one_generic_call"
                and own["cover"] == "full_disjoint_logical_domain", "missing disjoint single-call ownership")
        require([t["role"] for t in d["operands"]] == ROLES[family], "wrong operand roles/order")
        for t in d["operands"]:
            tensor(t)
        storage = [t["storage"] for t in d["operands"]]
        require(len(set(storage)) == len(storage), "storage aliases unsupported; distinct storage required")
        row = dict(status="SOURCE_CONDITIONAL", family=family, id=own["id"],
            descriptor_sha256=item["descriptor_sha256"], source_manifest_sha256=digest(known),
            logical_matrix_flops=None, conventional={}, matrix_components={},
            issued_arithmetic=None, sfpu_flops=None, physical_bytes=None, exact_total=False,
            uncounted=["API/header lowering and cached machine-code linkage",
                       "issued arithmetic, conversion/packing and SFPU instructions",
                       "physical rereads, spills, transactions and memory-level placement"],
            ownership="one call only; do not sum overlapping captures or inclusive parents")
        if family == "matmul":
            count_matmul(d, row)
        elif family.startswith("reblock"):
            count_reblock(d, row)
        else:
            count_sdpa(d, row)
        return row
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return unknown(str(exc))


def inspect_observer(path, sequence):
    """Audit an archived call without inventing a normalized descriptor from missing data."""
    raw = Path(path).read_bytes()
    ident = dict(archive_sha256=sha(raw), sequence=sequence)
    try:
        records = [json.loads(line) for line in raw.splitlines()]
        require(records[0]["kind"] == "header" and records[-1]["kind"] == "footer", "capture incomplete")
        require(records[-1]["observation_ok"] and not records[-1]["body_raised"], "observer/body failure")
        calls = [r for r in records if r["kind"] == "call" and r["sequence"] == sequence]
        require(len(calls) == 1, "missing/duplicate selected sequence")
        r = calls[0]
        ident["execution_sha256"] = r["execution_sha256"]
        ident["descriptor_sha256"] = digest(r["descriptor"])
        require(digest(dict(descriptor=r["descriptor"], operands=r["operands"],
                            environment=r["environment_at_call"])) == r["execution_sha256"],
                "observer execution hash mismatch")
        sources = {}
        for s in records:
            if s["kind"] == "source":
                data = base64.b64decode(s["content"], validate=True)
                require(sha(data) == s["sha256"], "embedded source hash mismatch")
                sources[s["sha256"]] = data
        require(r["caller"]["file"]["sha256"] in sources and r["caller"].get("role_basis")
                and r["caller"].get("loaded_code_sha256"), "missing wrapper/loaded-code role evidence")
        caller = r["caller"]
        declared = ROLE_REGISTRY.get(caller["file"]["sha256"], {}).get(caller["function"])
        require(declared is not None, "wrapper source/function not in observer role registry")
        roles = [t["role"] for t in r["operands"]]
        if isinstance(declared, list):
            require(roles == [pair[0] for pair in declared], "archived declared role mismatch")
        elif declared == "matmul":
            require(roles[:2] == ["left", "right"] and len(roles) >= 3
                    and all(role == "output" for role in roles[2:]), "archived matmul role mismatch")
        elif declared == "sdpa":
            require(roles in [ROLES[f] for f in ("sdpa", "sdpa_gated", "sdpa_fused_qkv")],
                    "unsupported archived SDPA role combination")
        outcomes = [o for o in records if o["kind"] == "outcome" and o["sequence"] == sequence]
        require(len(outcomes) == 1 and outcomes[0]["outcome"] == "returned", "call did not return")
        gaps = []
        for i, kernel in enumerate(r["descriptor"]["fields"]["kernels"]):
            f = kernel["fields"]
            require(f["source_identity"]["sha256"] in sources, "missing embedded kernel source")
            if "unavailable" in f["runtime_args"]:
                gaps.append(f"kernel {i}: per-core runtime arguments/ownership unavailable")
        if any("unavailable" in t.get("storage_alias", {"unavailable": "missing"}) for t in r["operands"]):
            gaps.append("storage alias identity absent")
        gaps.append("observer-to-normalized translation requires a reviewed source/flag/ownership mapping")
        return unknown("; ".join(gaps), **ident)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return unknown(str(exc), **ident)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--contract", type=Path)
    g.add_argument("--observer", type=Path)
    p.add_argument("--sequence", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--require-exact-total", action="store_true")
    a = p.parse_args()
    if a.contract:
        result = reduce_contract(json.loads(a.contract.read_text()))
    else:
        result = inspect_observer(a.observer, a.sequence)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 2 if a.require_exact_total or result["status"] == "REFUSED" else 0


if __name__ == "__main__":
    sys.exit(main())
