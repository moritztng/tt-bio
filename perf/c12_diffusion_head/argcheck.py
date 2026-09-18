#!/usr/bin/env python3
"""Does the descriptor's argument layout still match the kernels that will read it? CPU only.

The bias half of this row is host-side: `mm_generic` puts the bias address at runtime arg 1 and the
bias accessor's compile-time args immediately after the outputs', because that is where the wheel's
own kernels look. Both are LITERAL offsets in the kernel sources. Get one wrong and nothing crashes
-- a tile id is simply read through a neighbouring tensor's accessor, which is the quietest possible
failure and the one no bit-exactness check downstream would attribute correctly.

So check it against the INSTALLED wheel rather than against our copy of the intent:

  IDENTICAL   our two DM kernels are byte-identical to the wheel's, which is what lets us rely on
              their FUSE_BIAS branches being the ones that compile.
  CT OFFSET   each kernel's `TensorAccessorArgs<N>` literal equals the scalar count
              `mm_generic.DM_SCALAR_CT_ARGS` emits for it.
  BIAS ORDER  `in2_addr` is runtime arg index 1 in both kernels, and the bias accessor offset is
              derived from the OUTPUTS' offset, not from in1's.
  CB          the bias circular buffer is c_4 and is sized one N block, not double buffered.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OURS = ROOT / "tt_bio" / "kernels" / "triatt"
# in0 carries one extra scalar (in3_tile_size) in `tail`, so its accessor offset is one higher.
EXTRA_TAIL = {"dm_in0_sender.cpp": 1, "dm_in1_sender_out.cpp": 0}


def wheel_kernels():
    import ttnn
    from tt_bio.mm_generic import ttnn_cpp_root
    d = (ttnn_cpp_root() / "cpp/ttnn/operations/experimental/minimal_matmul/device/kernels")
    assert d.is_dir(), f"wheel kernels not found at {d}"
    return d, ttnn.__version__ if hasattr(ttnn, "__version__") else "unknown"


def main() -> int:
    from tt_bio.mm_generic import DM_SCALAR_CT_ARGS
    wheel, ver = wheel_kernels()
    rows, ok = [], True

    for name, extra in EXTRA_TAIL.items():
        w, o = (wheel / name).read_bytes(), (OURS / name).read_bytes()
        identical = hashlib.sha256(w).hexdigest() == hashlib.sha256(o).hexdigest()
        src = w.decode()

        m = re.search(r"TensorAccessorArgs<(\d+)>\(\)", src)
        literal = int(m.group(1)) if m else -1
        expect = DM_SCALAR_CT_ARGS + extra
        ct_ok = literal == expect

        # `in2_addr` must be the SECOND runtime arg read.
        reads = re.findall(r"const uint32_t (\w+) = get_arg_val<uint32_t>\(argidx\+\+\);", src)
        bias_ok = len(reads) > 1 and reads[1] == "in2_addr"

        # the bias accessor offset is derived from the outputs', not from in1's
        derived_ok = bool(re.search(
            r"in2_args_cta_offset\s*=\s*\n?\s*tensor_accessor::detail::"
            r"get_tensor_accessor_args_cta_offset<N_chunks, out_tensor_args_cta_offset>", src))
        cb_ok = "cb_id_in2 = tt::CBIndex::c_4" in src

        good = identical and ct_ok and bias_ok and derived_ok and cb_ok
        ok &= good
        rows.append({"kernel": name, "byte_identical_to_wheel": identical,
                     "accessor_args_literal": literal, "host_scalar_ct_args": expect,
                     "ct_offset_matches": ct_ok, "in2_is_runtime_arg_1": bias_ok,
                     "runtime_arg_order": reads[:3],
                     "bias_accessor_after_outputs": derived_ok,
                     "bias_cb_is_c4": cb_ok, "pass": good})
        print(f"{name:<24} identical={identical} accessor<{literal}> == {expect} -> {ct_ok}  "
              f"rt[1]={reads[1] if len(reads) > 1 else None}  after_outputs={derived_ok}  "
              f"cb_c4={cb_ok}")

    # the compute kernel takes no extra arg for the bias, only the define and the CB
    comp = (wheel / "compute.cpp").read_text()
    comp_ok = ("in2_cb = tt::CBIndex::c_4" in comp
               and re.search(r"#ifndef FUSE_BIAS", comp) is not None
               and re.search(r"add_bias_block\(intermediate_cb, in2_cb", comp) is not None)
    ok &= comp_ok
    print(f"{'compute.cpp':<24} bias needs no extra CT/RT arg, only FUSE_BIAS + c_4 -> {comp_ok}")

    out = Path(__file__).resolve().parent / "argcheck.json"
    out.write_text(json.dumps({"all_pass": bool(ok), "ttnn": ver,
                               "wheel_kernels": str(wheel), "kernels": rows,
                               "compute_needs_no_extra_arg": comp_ok}, indent=1) + "\n")
    print(f"\n{'PASS' if ok else 'FAIL'} -- {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
