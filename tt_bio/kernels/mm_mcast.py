#!/usr/bin/env python3
"""The multicast arm of the two matmul operand readers, shared by all three kernel families.

`mm_split`, `triatt` and `trimul_tail` each keep their own copy of the wheel's
``dm_in0_sender.cpp`` / ``dm_in1_sender_out.cpp``. All three carry the same operand broadcast: one
injector core per grid axis reads the block from DRAM and every other core takes a semaphore-gated
unicast hop from its predecessor, 8-9 hops on an 8x9 Wormhole grid and 10-11 on an 11x10 Blackhole
one. `b2z2-tile-arrival-latency` measured that chain: the handshakes are 67.6 % of the reader's own
time and the forward another 15.2 %.

``MM_MCAST_OPERAND`` replaces the chain with `noc_async_write_multicast`, which is what tt-metal's
own `matmul_multi_core_reuse_mcast` does. The injector waits for one credit from each receiver,
writes the block to the whole axis in one transaction and multicasts the valid flag after it. Every
receiver then waits on the injector instead of on its predecessor, which is a runtime-arg change in
`tt_bio/mm_generic.py`, not a kernel one.

The arm is a no-op unless the macro is defined, so the generated files stay the wheel's kernels byte
for byte on the default path. Bit-exact by construction: the same bytes reach the same cores at the
same L1 addresses, and nothing about the accumulation changes.

The edits live here rather than in each family's patch script so there is one copy of them.
"""

# Injector-side arguments. `in*_dest_noc_x/y` is already the first core the chain reaches, which is
# also the first corner of the multicast rectangle in this NOC's own traversal order, so only the
# far corner and the destination count are new. They go after `defer_write_k_block` and before the
# output addresses, which is what `mm_generic.rebind` indexes from.
_ARGS = """#ifdef MM_MCAST_OPERAND
    const uint32_t {p}_mcast_end_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t {p}_mcast_end_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t {p}_mcast_num_dests = get_arg_val<uint32_t>(argidx++);
#endif
"""

# The rectangle, resolved once. `get_noc_multicast_addr` with a zero address gives a base the block
# address ORs into, exactly as the in1 unicast path already does.
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

# The two multicasts ride the same NOC, VC and command buffer, so they are ordered against each
# other and need no barrier between them. Blackhole still needs the flush: its NOC latency is above
# the L1-to-RISCV latency, so the source block can be overwritten before the write has issued.
_IN0_SEND = """                        noc_async_write_multicast(
                            in0_start_address,
                            in0_mcast_data_base_addr | in0_start_address,
                            current_block_bytes,
                            in0_mcast_num_dests,
                            true);
#ifdef ARCH_BLACKHOLE
                        noc_async_writes_flushed();
#endif
                        noc_semaphore_set_multicast(
                            in0_valid_semaphore_addr,
                            in0_mcast_receiver_semaphore_noc_addr,
                            in0_mcast_num_dests);
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
# it multicasts row by row and must not clobber the loop's own cursor.
_IN1_SEND = """                        uint32_t in1_mcast_src_address = in1_start_address;
                        for (uint32_t i = 0; i < K_block_tiles; i++) {
                            noc_async_write_multicast(
                                in1_mcast_src_address,
                                in1_mcast_data_base_addr | in1_mcast_src_address,
                                current_N_tiles_bytes,
                                in1_mcast_num_dests,
                                true);
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

_GUARD = """#ifdef MM_MCAST_OPERAND
                if constexpr (is_injector_core) {
                    if (%s_mcast_num_dests) {
                        noc_semaphore_wait(%s_sender_semaphore_addr_ptr, %s_mcast_num_dests);
                        noc_semaphore_set(%s_sender_semaphore_addr_ptr, 0);
%s                    }
                }
#else
"""

_ARG_ANCHOR = "    const uint32_t defer_write_k_block = get_arg_val<uint32_t>(argidx++);\n"

_SPEC = {
    "dm_in0_sender.cpp": (
        "in0",
        "    const uint64_t in0_receiver_semaphore_noc_addr =\n"
        "        get_noc_addr(in0_dest_noc_x, in0_dest_noc_y, in0_receiver_semaphore_addr);\n",
        _IN0_FORWARD, _IN0_SEND),
    "dm_in1_sender_out.cpp": (
        "in1",
        "    const uint64_t in1_unicast_data_base_addr = get_noc_addr(in1_dest_noc_x, in1_dest_noc_y, 0);\n",
        _IN1_FORWARD, _IN1_SEND),
}


def edits(name):
    """The (old, new) pairs for one kernel file, exact-match so a wheel bump fails loudly.

    The injector is the only core that sends, so the `is_sink_core` test that ends the chain has no
    counterpart in the multicast arm; a `num_dests` of zero (a one-core axis) sends nothing, which
    is what `noc_async_write_multicast` requires.
    """
    p, addr_anchor, forward, send = _SPEC[name]
    return [
        (_ARG_ANCHOR, _ARG_ANCHOR + _ARGS.format(p=p)),
        (addr_anchor, addr_anchor + _ADDRS.format(p=p)),
        (forward, _GUARD % (p, p, p, p, send) + forward + "#endif  // MM_MCAST_OPERAND\n"),
    ]


def apply(text, name):
    """Apply the arm to one kernel source. Fails loudly rather than patching the wrong place."""
    for old, new in edits(name):
        if text.count(old) != 1:
            raise SystemExit("mcast patch site not unique in %s: %r" % (name, old[:70]))
        text = text.replace(old, new)
    return text
