#!/usr/bin/env python3
"""Two ways to stop daisy-chaining the matmul operand, shared by all three kernel families.

`mm_split`, `triatt` and `trimul_tail` each keep their own copy of the wheel's
``dm_in0_sender.cpp`` / ``dm_in1_sender_out.cpp``. All three carry the same operand broadcast: one
injector core per grid axis reads the block from DRAM and every other core takes a semaphore-gated
unicast hop from its predecessor, 8-9 hops on an 8x9 Wormhole grid and 10-11 on an 11x10 Blackhole
one. `b2z2-tile-arrival-latency` measured that chain: the handshakes are 67.6 % of the reader's own
time and the forward another 15.2 %.

Two arms replace it, both selected from `tt_bio/mm_generic.py` and both no-ops unless their macro is
defined, so the generated files stay the wheel's kernels byte for byte on the default path:

``MM_BCAST_FANOUT``
    The injector writes the block to every receiver itself, one `noc_async_write` each, followed by
    that receiver's valid flag so per-destination NOC ordering is the same guarantee the chain
    already relies on. This deletes the whole dependency chain -- the 67.6 % term -- and keeps the
    forwarded bytes, which were measured at 0.63 % of the reader's time in flight. Same primitives
    the shipped kernel already uses, so there is no new addressing hazard.

``MM_MCAST_OPERAND``
    One `noc_async_write_multicast` to the whole axis, which is what tt-metal's own
    `matmul_multi_core_reuse_mcast` does. It deletes the chain AND the repeated bytes.
    **It is not yet correct.** On whglx card 16 it left the chip unable to run any program, with
    `Read unexpected run_mailbox value from core (x=25,y=17)` -- an ETHERNET core, i.e. the
    multicast reached outside the worker rectangle. Recovered with `tt-smi -r 16`. Do not run this
    arm again without `TT_METAL_WATCHER` armed from process start so the NOC sanitiser names the
    illegal transaction.

Bit-exact by construction in both arms: the same bytes reach the same cores at the same L1
addresses, and nothing about the accumulation changes.

The edits live here rather than in each family's patch script so there is one copy of them.
"""

# Injector-side arguments, both arms. They go after `defer_write_k_block` and before the output
# addresses, which is what `mm_generic.rebind` indexes from.
#
# For the multicast, `in*_dest_noc_x/y` is already the first core the chain reaches, which is also
# the first corner of the rectangle in this NOC's own traversal order, so only the far corner and
# the count are new. For the fan-out, the injector needs every receiver's coordinates.
_ARGS = """#ifdef MM_MCAST_OPERAND
    const uint32_t {p}_mcast_end_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t {p}_mcast_end_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t {p}_mcast_num_dests = get_arg_val<uint32_t>(argidx++);
#endif
#ifdef MM_BCAST_FANOUT
    const uint32_t {p}_bcast_num_dests = get_arg_val<uint32_t>(argidx++);
    tt_l1_ptr uint32_t* {p}_bcast_noc_xy = (tt_l1_ptr uint32_t*)get_arg_addr(argidx);
    argidx += 2 * {p}_bcast_num_dests;
#endif
"""

_ADDRS = """#ifdef MM_MCAST_OPERAND
    const uint64_t {p}_mcast_data_base_addr = get_noc_multicast_addr(
        {p}_dest_noc_x, {p}_dest_noc_y, {p}_mcast_end_noc_x, {p}_mcast_end_noc_y, 0);
    const uint64_t {p}_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        {p}_dest_noc_x, {p}_dest_noc_y, {p}_mcast_end_noc_x, {p}_mcast_end_noc_y,
        {p}_receiver_semaphore_addr);
#endif
"""

_IN0_FORWARD = """                if (!is_sink_core) {
                    noc_semaphore_wait(in0_sender_semaphore_addr_ptr, 1);
                    noc_semaphore_set(in0_sender_semaphore_addr_ptr, 0);

                    uint64_t in0_unicast_data_addr = get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_start_address);

                    /**
                     * in0 is M_block_tiles x K_block_tiles. When M block is partial, we don't need to write the
                     * padded tiles. Use `current_block_bytes`.
                     */
                    noc_async_write(in0_start_address, in0_unicast_data_addr, current_block_bytes);

#ifdef ARCH_BLACKHOLE
                    noc_async_writes_flushed();
#endif

                    noc_semaphore_set_remote(in0_valid_semaphore_addr, in0_receiver_semaphore_noc_addr);
                }
"""

# `linked` stays false on the multicast. A linked transaction holds the NOC path until the next
# unlinked one on the SAME command buffer, and `noc_semaphore_set_multicast` issues on the register
# command buffer, not the write one, so a linked block multicast would never be released.
_IN0_MCAST = """                        noc_async_write_multicast(
                            in0_start_address,
                            in0_mcast_data_base_addr | in0_start_address,
                            current_block_bytes,
                            in0_mcast_num_dests);
#ifdef ARCH_BLACKHOLE
                        noc_async_writes_flushed();
#endif
                        noc_semaphore_set_multicast(
                            in0_valid_semaphore_addr,
                            in0_mcast_receiver_semaphore_noc_addr,
                            in0_mcast_num_dests);
"""

# Block then flag, per destination, so each receiver keeps exactly the ordering guarantee the chain
# gave it and the first one can start while the last is still being written.
_IN0_FANOUT = """                        for (uint32_t d = 0; d < in0_bcast_num_dests; d++) {
                            const uint32_t dx = in0_bcast_noc_xy[2 * d];
                            const uint32_t dy = in0_bcast_noc_xy[2 * d + 1];
                            noc_async_write(in0_start_address,
                                            get_noc_addr(dx, dy, in0_start_address),
                                            current_block_bytes);
#ifdef ARCH_BLACKHOLE
                            noc_async_writes_flushed();
#endif
                            noc_semaphore_set_remote(
                                in0_valid_semaphore_addr,
                                get_noc_addr(dx, dy, in0_receiver_semaphore_addr));
                        }
"""

_IN1_FORWARD = """                if (!is_sink_core) {
                    noc_semaphore_wait(in1_sender_semaphore_addr_ptr, 1);
                    noc_semaphore_set(in1_sender_semaphore_addr_ptr, 0);

                    /**
                     * in1 is K_block_tiles x N_block_tiles. When N block is partial, we don't need to write the
                     * padded tiles. For each tile in the K block, write only the non-padded N tiles. Use
                     * `current_N_tiles_bytes`.
                     */
                    for (uint32_t i = 0; i < K_block_tiles; i++) {
                        uint64_t in1_unicast_data_addr = in1_unicast_data_base_addr | in1_start_address;
                        noc_async_write(in1_start_address, in1_unicast_data_addr, current_N_tiles_bytes);
                        in1_start_address += full_N_tiles_bytes;
                    }

#ifdef ARCH_BLACKHOLE
                    noc_async_writes_flushed();
#endif

                    noc_semaphore_set_remote(in1_valid_semaphore_addr, in1_receiver_semaphore_noc_addr);
                }
"""

# in1's block is K_block_tiles rows of `current_N_tiles_bytes` strided by `full_N_tiles_bytes`, so
# both arms walk it row by row and must not clobber the loop's own cursor.
_IN1_MCAST = """                        uint32_t in1_mcast_src_address = in1_start_address;
                        for (uint32_t i = 0; i < K_block_tiles; i++) {
                            noc_async_write_multicast(
                                in1_mcast_src_address,
                                in1_mcast_data_base_addr | in1_mcast_src_address,
                                current_N_tiles_bytes,
                                in1_mcast_num_dests);
                            in1_mcast_src_address += full_N_tiles_bytes;
                        }
#ifdef ARCH_BLACKHOLE
                        noc_async_writes_flushed();
#endif
                        noc_semaphore_set_multicast(
                            in1_valid_semaphore_addr,
                            in1_mcast_receiver_semaphore_noc_addr,
                            in1_mcast_num_dests);
"""

_IN1_FANOUT = """                        for (uint32_t d = 0; d < in1_bcast_num_dests; d++) {
                            const uint32_t dx = in1_bcast_noc_xy[2 * d];
                            const uint32_t dy = in1_bcast_noc_xy[2 * d + 1];
                            const uint64_t dbase = get_noc_addr(dx, dy, 0);
                            uint32_t in1_fanout_src_address = in1_start_address;
                            for (uint32_t i = 0; i < K_block_tiles; i++) {
                                noc_async_write(in1_fanout_src_address,
                                                dbase | in1_fanout_src_address,
                                                current_N_tiles_bytes);
                                in1_fanout_src_address += full_N_tiles_bytes;
                            }
#ifdef ARCH_BLACKHOLE
                            noc_async_writes_flushed();
#endif
                            noc_semaphore_set_remote(
                                in1_valid_semaphore_addr,
                                get_noc_addr(dx, dy, in1_receiver_semaphore_addr));
                        }
"""

# The injector is the only core that sends in either arm, so the `is_sink_core` test that ends the
# chain has no counterpart; a destination count of zero (a one-core axis) sends nothing, which is
# also what `noc_async_write_multicast` requires.
_GUARD = """#if defined(MM_MCAST_OPERAND) || defined(MM_BCAST_FANOUT)
                if constexpr (is_injector_core) {
#ifdef MM_MCAST_OPERAND
                    if (%(p)s_mcast_num_dests) {
                        noc_semaphore_wait(%(p)s_sender_semaphore_addr_ptr, %(p)s_mcast_num_dests);
                        noc_semaphore_set(%(p)s_sender_semaphore_addr_ptr, 0);
%(mcast)s                    }
#else
                    if (%(p)s_bcast_num_dests) {
                        noc_semaphore_wait(%(p)s_sender_semaphore_addr_ptr, %(p)s_bcast_num_dests);
                        noc_semaphore_set(%(p)s_sender_semaphore_addr_ptr, 0);
%(fanout)s                    }
#endif
                }
#else
"""

_ARG_ANCHOR = "    const uint32_t defer_write_k_block = get_arg_val<uint32_t>(argidx++);\n"

_SPEC = {
    "dm_in0_sender.cpp": (
        "in0",
        "    const uint64_t in0_receiver_semaphore_noc_addr =\n"
        "        get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_receiver_semaphore_addr);\n",
        _IN0_FORWARD, _IN0_MCAST, _IN0_FANOUT),
    "dm_in1_sender_out.cpp": (
        "in1",
        "    const uint64_t in1_unicast_data_base_addr = get_noc_addr(in1_dest_noc_x, in1_dest_noc_y, 0);\n",
        _IN1_FORWARD, _IN1_MCAST, _IN1_FANOUT),
}


def edits(name):
    """The (old, new) pairs for one kernel file, exact-match so a wheel bump fails loudly."""
    p, addr_anchor, forward, mcast, fanout = _SPEC[name]
    guard = _GUARD % {"p": p, "mcast": mcast, "fanout": fanout}
    return [
        (_ARG_ANCHOR, _ARG_ANCHOR + _ARGS.format(p=p)),
        (addr_anchor, addr_anchor + _ADDRS.format(p=p)),
        (forward, guard + forward + "#endif  // MM_MCAST_OPERAND || MM_BCAST_FANOUT\n"),
    ]


def apply(text, name):
    """Apply both arms to one kernel source. Fails loudly rather than patching the wrong place."""
    for old, new in edits(name):
        if text.count(old) != 1:
            raise SystemExit("mcast patch site not unique in %s: %r" % (name, old[:70]))
        text = text.replace(old, new)
    return text
