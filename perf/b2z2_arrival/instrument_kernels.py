#!/usr/bin/env python3
"""Add device-profiler zones to the trunk's matmul dataflow kernels.

The Pairformer block's dominant programs are tt-bio's own ``generic_op`` matmuls
(``mm_split``, ``triatt``, ``trimul_tail``). Their operands are not multicast: one
``is_injector_core`` per grid axis reads the block from DRAM and every other core on that
axis receives it by a serial unicast hop from its predecessor. So the math thread's
``cb_wait_front`` can be waiting for three different things, and the campaign has never
separated them:

  B2Z2-INx-SRC        the injector's own DRAM read (issue + barrier)
  B2Z2-INx-RDBAR      the barrier inside that read -- bytes actually in flight
  B2Z2-INx-CHAINWAIT  a receiver core blocked on the predecessor that has not forwarded yet
  B2Z2-INx-DOWNWAIT   a forwarding core blocked on its successor not yet asking
  B2Z2-INx-FWD        the unicast forward itself
  B2Z2-INx-CBRES      the reader blocked on circular-buffer room

``tools/profiler/kernel_profiler.hpp`` defines every macro as empty when the build has no
Tracy, so the instrumented kernels still compile and run against the shipped wheel.

Idempotent: re-running on an already-patched tree is a no-op.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("mm_split", "triatt", "trimul_tail")
INCLUDE = '#include "tools/profiler/kernel_profiler.hpp"'
MARK = "B2Z2-IN"


def add_include(src: str) -> str:
    if INCLUDE in src:
        return src
    # after the last #include of the preamble
    idx = src.rfind("#include")
    end = src.index("\n", idx) + 1
    return src[:end] + INCLUDE + "\n" + src[end:]


def zone(body: str, name: str, indent: str) -> str:
    inner = "\n".join(indent + "    " + l.strip() if l.strip() else "" for l in body.split("\n"))
    return (f'{indent}{{\n{indent}    DeviceZoneScopedN("{name}");\n{inner}\n{indent}}}')


def patch_sender(path: Path) -> int:
    src = path.read_text()
    if MARK in src:
        return 0
    n = 0

    # 1. circular-buffer room
    def cbres(m):
        return (f'{m.group(1)}{{ DeviceZoneScopedN("B2Z2-{m.group(2).upper()}-CBRES"); '
                f'{m.group(0).strip()} }}')
    src, k = re.subn(r'(\n\s*)cb_reserve_back\(cb_id_(in\d), in\d_block_num_tiles\);', cbres, src)
    n += k

    # 2. the receiver's wait on its predecessor: three statements, one zone
    def chain(m):
        ind, x = m.group(1), m.group(2)
        return zone(m.group(0).strip(), f"B2Z2-{x.upper()}-CHAINWAIT", ind.strip("\n"))

    src, k = re.subn(
        r'(\n[ ]*)noc_semaphore_set\((in\d)_receiver_semaphore_addr_ptr, INVALID\);'
        r'\n[ ]*noc_semaphore_inc\(\2_sender_semaphore_noc_addr, 1\);'
        r'\n[ ]*noc_semaphore_wait\(\2_receiver_semaphore_addr_ptr, VALID\);',
        lambda m: "\n" + chain(m), src)
    n += k

    # 3. the forwarder's wait on its successor
    def down(m):
        return (f'{m.group(1)}{{ DeviceZoneScopedN("B2Z2-{m.group(2).upper()}-DOWNWAIT"); '
                f'noc_semaphore_wait({m.group(2)}_sender_semaphore_addr_ptr, 1); }}')
    src, k = re.subn(r'(\n[ ]*)noc_semaphore_wait\((in\d)_sender_semaphore_addr_ptr, 1\);',
                     down, src)
    n += k

    # 4. the forward itself: from the semaphore reset to the remote valid set
    def fwd(m):
        ind = m.group(1).strip("\n")
        x = m.group(2)
        return "\n" + zone(m.group(0).strip(), f"B2Z2-{x.upper()}-FWD", ind)
    src, k = re.subn(
        r'(\n[ ]*)noc_semaphore_set\((in\d)_sender_semaphore_addr_ptr, 0\);'
        r'(?:.|\n)*?noc_semaphore_set_remote\(\2_valid_semaphore_addr, \2_receiver_semaphore_noc_addr\);',
        fwd, src)
    n += k

    path.write_text(add_include(src))
    return n


def patch_common(path: Path) -> int:
    """Zone the DRAM read helpers: whole call, and the barrier inside it."""
    src = path.read_text()
    if MARK in src:
        return 0
    n = 0
    for x in ("in0", "in1"):
        # whole helper body
        pat = re.compile(r'(void read_%s_block_sync\((?:.|\n)*?\n\s*ASSERT\(d1_end > d1_start\);\n)'
                         % x)
        src, k = pat.subn(
            lambda m: m.group(1) + f'    DeviceZoneScopedN("B2Z2-{x.upper()}-SRC");\n', src)
        n += k
    # the barrier that closes each helper
    src, k = re.subn(
        r'(\n[ ]*)noc_async_read_barrier\(\);(\n\})',
        lambda m: f'{m.group(1)}{{ DeviceZoneScopedN("B2Z2-RDBAR"); noc_async_read_barrier(); }}'
                  + m.group(2), src)
    n += k
    path.write_text(add_include(src))
    return n


def main() -> int:
    total = 0
    for fam in FAMILIES:
        d = ROOT / "tt_bio" / "kernels" / fam
        for f in ("dm_in0_sender.cpp", "dm_in1_sender_out.cpp"):
            k = patch_sender(d / f)
            print(f"  {fam}/{f}: {k} zones")
            total += k
        k = patch_common(d / "matmul_dataflow_common.hpp")
        print(f"  {fam}/matmul_dataflow_common.hpp: {k} zones")
        total += k
    print(f"total {total}")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
