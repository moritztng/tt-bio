#!/usr/bin/env python3
"""Package definitions for the 0.68.0 -> current tt-metal diff, plus their measured volume.

Each package is a set of path prefixes over the tt-metal tree. Volume is computed from
`git diff --numstat v0.68.0 v0.79.0-dev20260913` (artifacts/ttx/numstat_068_079.tsv) and the
commit count from `git rev-list --count` over the same paths.
"""
import collections, json, subprocess, sys
from pathlib import Path

# The mirror is a bare, object-shared clone of tt-metal, rebuilt in seconds if it is gone:
#   git clone --bare --shared /home/moritz/tt-metal/.git .ttm.git
#   git -C .ttm.git config remote.origin.url https://github.com/tenstorrent/tt-metal.git
#   git -C .ttm.git fetch --no-tags origin tag v0.76.0 tag v0.77.0 tag v0.78.0 \
#       tag v0.79.0-dev20260913

ROOT = Path(__file__).resolve().parent
MIRROR = ROOT.parent.parent / ".ttm.git"
OLD, NEW = "v0.68.0", "v0.79.0-dev20260913"

PACKAGES = {
 "P1-matmul": [
   "ttnn/cpp/ttnn/operations/matmul/",
   "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/",
   "ttnn/cpp/ttnn/operations/experimental/matmul/",
   "ttnn/cpp/ttnn/operations/experimental/matmul_decode/",
 ],
 "P2-sdpa-attention": [
   "ttnn/cpp/ttnn/operations/transformer/",
   "ttnn/cpp/ttnn/operations/experimental/transformer/",
 ],
 "P3-cb-dataflow": [
   "tt_metal/hw/inc/",
   "tt_metal/api/tt-metalium/circular_buffer",
   "tt_metal/api/tt-metalium/global_circular_buffer",
   "tt_metal/impl/buffers/",
   "tt_metal/impl/dataflow_buffer/",
   "ttnn/cpp/ttnn/kernel/",
   "ttnn/cpp/ttnn/kernel_lib/",
   "tt_metal/api/tt-metalium/experimental/metal2_host_api/",
 ],
 "P4-sharding-alloc": [
   "ttnn/cpp/ttnn/operations/data_movement/sharded/",
   "ttnn/cpp/ttnn/operations/data_movement/sharded_partial/",
   "tt_metal/impl/allocator/",
   "tt_metal/impl/per_core_allocation/",
   "tt_metal/impl/range_lockstep_allocation/",
   "tt_metal/impl/tensor/",
   "ttnn/core/tensor/",
   "ttnn/api/ttnn/tensor",
   "tt_metal/api/tt-metalium/tensor",
   "tt_metal/api/tt-metalium/shard_data_transfer.hpp",
 ],
 "P5-layout-reblock": [
   "ttnn/cpp/ttnn/operations/data_movement/tilize",
   "ttnn/cpp/ttnn/operations/data_movement/untilize",
   "ttnn/cpp/ttnn/operations/data_movement/transpose/",
   "ttnn/cpp/ttnn/operations/data_movement/permute/",
   "ttnn/cpp/ttnn/operations/data_movement/reshape_view/",
   "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/",
   "ttnn/cpp/ttnn/operations/data_movement/concat/",
   "ttnn/cpp/ttnn/operations/data_movement/slice/",
   "ttnn/cpp/ttnn/operations/data_movement/pad/",
   "ttnn/cpp/ttnn/operations/data_movement/fill_pad/",
   "ttnn/cpp/ttnn/operations/data_movement/chunk/",
   "ttnn/cpp/ttnn/operations/data_movement/split/",
   "ttnn/cpp/ttnn/operations/data_movement/repeat/",
   "ttnn/cpp/ttnn/operations/data_movement/clone/",
   "ttnn/cpp/ttnn/operations/data_movement/copy/",
   "ttnn/cpp/ttnn/operations/data_movement/view/",
   "ttnn/cpp/ttnn/operations/data_movement/squeeze/",
   "ttnn/cpp/ttnn/operations/data_movement/unsqueeze/",
   "ttnn/cpp/ttnn/operations/data_movement/common/",
   "ttnn/cpp/ttnn/operations/data_movement/gather/",
   "ttnn/cpp/ttnn/operations/data_movement/scatter/",
   "ttnn/cpp/ttnn/operations/experimental/padded_slice/",
   "ttnn/cpp/ttnn/operations/experimental/slice_write/",
   "ttnn/cpp/ttnn/operations/copy/",
   "ttnn/cpp/ttnn/operations/embedding/",
   "ttnn/cpp/ttnn/operations/creation/",
 ],
 "P6-dispatch-trace-cache": [
   "tt_metal/impl/dispatch/",
   "tt_metal/impl/program/",
   "tt_metal/impl/kernels/",
   "tt_metal/impl/jit_server/",
   "tt_metal/jit_build/",
   "tt_metal/tools/jit_compile_server/",
   "tt_metal/distributed/",
   "ttnn/cpp/ttnn/operations/trace",
   "ttnn/cpp/ttnn/graph/",
   "tt_metal/impl/graph/",
   "ttnn/cpp/ttnn/operations/generic/",
   "tt_metal/api/tt-metalium/experimental/program_descriptor_patching.hpp",
 ],
 "P7-llk-compute-api": [
   "tt_metal/tt-llk/tt_llk_blackhole/",
   "tt_metal/tt-llk/tt_llk_wormhole_b0/",
   "tt_metal/tt-llk/common/",
   "tt_metal/hw/ckernels/",
 ],
 "P8-blackhole": [
   "tt_metal/hw/ckernels/blackhole/",
   "tt_metal/tt-llk/tt_llk_blackhole/",
   "tt_metal/llrt/hal/",
   "tt_metal/hw/firmware/",
   "tt_metal/hw/inc/blackhole/",
   "tt_metal/soc_descriptors/",
 ],
 "P9-normalization": [
   "ttnn/cpp/ttnn/operations/normalization/layernorm/",
   "ttnn/cpp/ttnn/operations/normalization/rmsnorm/",
   "ttnn/cpp/ttnn/operations/normalization/softmax/",
   "ttnn/cpp/ttnn/operations/normalization/kernel_util/",
   "ttnn/cpp/ttnn/operations/normalization/shard_spec_validation",
   "ttnn/cpp/ttnn/operations/normalization/normalization_nanobind",
 ],
 "P10-eltwise": [
   "ttnn/cpp/ttnn/operations/eltwise/binary/",
   "ttnn/cpp/ttnn/operations/eltwise/binary_ng/",
   "ttnn/cpp/ttnn/operations/eltwise/unary/",
   "ttnn/cpp/ttnn/operations/eltwise/unary_ng/",
   "ttnn/cpp/ttnn/operations/eltwise/ternary/",
   "ttnn/cpp/ttnn/operations/reduction/generic/",
   "ttnn/cpp/ttnn/operations/data_movement/bcast/",
 ],
 "P11-dtype-precision": [
   "tt_metal/api/tt-metalium/mxfp",
   "tt_metal/api/tt-metalium/mxint.hpp",
   "tt_metal/api/tt-metalium/int8.hpp",
   "tt_metal/api/tt-metalium/uint8.hpp",
   "tt_metal/api/tt-metalium/bfloat",
   "tt_metal/api/tt-metalium/tile.hpp",
   "tt_metal/api/tt-metalium/face_geometry.hpp",
   "tt_metal/impl/data_format/",
   "tt_metal/common/",
 ],
 "P12-new-ops": [
   "ttnn/cpp/ttnn/operations/experimental/fusion/",
   "ttnn/cpp/ttnn/operations/experimental/core_subset_write/",
   "ttnn/cpp/ttnn/operations/experimental/tensor_prefetcher/",
   "ttnn/cpp/ttnn/operations/experimental/indexer_score/",
   "ttnn/cpp/ttnn/operations/experimental/reduction/",
   "ttnn/cpp/ttnn/operations/experimental/plusone/",
   "ttnn/cpp/ttnn/operations/prefetcher/",
 ],
 "P13-python-api": [
   "ttnn/ttnn/",
   "ttnn/cpp/ttnn-nanobind/",
   "ttnn/api/ttnn/",
   "ttnn/core/",
   "ttnn/cpp/ttnn/decorators.hpp",
   "ttnn/cpp/ttnn/device_operation.hpp",
   "ttnn/cpp/ttnn/run_operation",
   "ttnn/cpp/ttnn/operations/core/",
   "ttnn/cpp/ttnn/types.hpp",
   "ttnn/cpp/ttnn/tensor_",
 ],
 "P14-host-api-device": [
   "tt_metal/api/tt-metalium/host_api.hpp",
   "tt_metal/impl/host_api/",
   "tt_metal/impl/metal2_host_api/",
   "tt_metal/impl/device/",
   "tt_metal/impl/context/",
   "tt_metal/impl/memory_tracking/",
   "tt_metal/impl/threading/",
   "tt_metal/tt_metal.cpp",
   "tt_metal/llrt/",
   "tt_metal/api/tt-metalium/",
 ],
}

EXCLUDED = {
 "models/": "reference model zoo; we do not use it",
 "tests/": "upstream test suite",
 "tt-train/": "training framework; inference only here",
 ".github/": "CI",
 "docs/": "upstream docs (read only as an index)",
 "tt_metal/tt-llk/tests/": "LLK test suite",
 "tt_metal/tt-llk/tt_llk_quasar/": "Quasar is a part we do not own",
 "ttnn/cpp/ttnn/operations/experimental/quasar/": "Quasar",
 "ttnn/cpp/ttnn/operations/experimental/ccl/": "collectives; the fold is one chip, no TP",
 "ttnn/cpp/ttnn/operations/ccl/": "collectives",
 "ttnn/cpp/ttnn/operations/experimental/deepseek": "LLM-specific",
 "ttnn/cpp/ttnn/operations/moreh/": "training ops",
 "ttnn/cpp/ttnn/operations/conv/": "no convolutions in the fold",
 "ttnn/cpp/ttnn/operations/pool/": "no pooling in the fold",
 "ttnn/cpp/ttnn/operations/sliding_window/": "conv support",
 "tt_metal/fabric/": "multi-host fabric; single-chip fold",
 "tt_metal/impl/emulation/": "emulator",
 "ttnn/cpp/ttnn/operations/experimental/fft/": "no FFT in the fold",
 "ttnn/cpp/ttnn/operations/experimental/kda/": "LLM kernel-descriptor attention",
 "tt-train": "training",
}


def ensure_numstat():
    """The per-file numstat is 1.5 MB and regenerates in ~1.3 s, so it is not committed."""
    f = ROOT / "numstat_068_079.tsv"
    if not f.is_file():
        out = subprocess.run(["git", "-C", str(MIRROR), "diff", "--numstat", "-M", OLD, NEW],
                             capture_output=True, text=True).stdout
        f.write_text(out)
    return f


def tree(rev):
    out = subprocess.run(["git", "-C", str(MIRROR), "ls-tree", "-r", "--name-only", rev],
                         capture_output=True, text=True).stdout
    return set(out.split("\n"))


def volume():
    old_files, new_files = tree(OLD), tree(NEW)
    rows = []
    for ln in open(ensure_numstat()):
        a, b, p = ln.rstrip("\n").split("\t", 2)
        if a == "-":
            a = b = "0"
        rows.append((int(a) + int(b), p))
    out = {}
    claimed = set()
    for name, prefixes in PACKAGES.items():
        lines = files = 0
        for n, p in rows:
            if any(p.startswith(x) for x in prefixes):
                lines += n
                files += 1
                claimed.add(p)
        added = sum(1 for f in new_files
                    if f not in old_files and any(f.startswith(x) for x in prefixes))
        commits = int(subprocess.run(
            ["git", "-C", str(MIRROR), "rev-list", "--count", f"{OLD}..{NEW}", "--", *prefixes],
            capture_output=True, text=True).stdout.strip() or 0)
        out[name] = {"prefixes": prefixes, "lines": lines, "files": files,
                     "new_files": added, "commits": commits}
    excl_lines = sum(n for n, p in rows
                     if p not in claimed and any(p.startswith(x) for x in EXCLUDED))
    unclaimed = sum(n for n, p in rows if p not in claimed
                    and not any(p.startswith(x) for x in EXCLUDED))
    out["_totals"] = {
        "all_lines": sum(n for n, _ in rows), "all_files": len(rows),
        "packaged_lines": sum(v["lines"] for k, v in out.items() if k != "_totals"),
        "excluded_lines": excl_lines, "unclaimed_lines": unclaimed,
    }
    return out


if __name__ == "__main__":
    v = volume()
    json.dump(v, open(ROOT / "packages.json", "w"), indent=1)
    t = v.pop("_totals")
    print(f"{'package':26} {'lines':>8} {'files':>6} {'new':>5} {'commits':>8}")
    for k, d in sorted(v.items(), key=lambda x: -x[1]["lines"]):
        print(f"{k:26} {d['lines']:8d} {d['files']:6d} {d['new_files']:5d} {d['commits']:8d}")
    print(f"\npackaged {t['packaged_lines']} of {t['all_lines']} lines "
          f"({100*t['packaged_lines']/t['all_lines']:.1f} %); "
          f"explicitly excluded {t['excluded_lines']} "
          f"({100*t['excluded_lines']/t['all_lines']:.1f} %); "
          f"unclaimed {t['unclaimed_lines']} ({100*t['unclaimed_lines']/t['all_lines']:.1f} %)")
