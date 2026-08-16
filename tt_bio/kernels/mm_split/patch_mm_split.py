#!/usr/bin/env python3
"""Generate `tt_bio/kernels/mm_split/` from the wheel's own `minimal_matmul` kernels.

Two guarded arms, both no-ops unless their macro is defined, so the generated files are the
wheel's kernels byte for byte on the default path:

  MM_NOWRITE   the output CB is drained and popped exactly as it is today, but the DRAM write
               is not issued. Prices the write half of the op, which is the whole question the
               DRAM-write-serialisation diagnosis turns on. Output is garbage by construction,
               so this arm is a stopwatch, never a result.
  MM_DUAL_NOC  every second output tile goes out on the OTHER NOC. The writer RISC carries
               512 MiB on NOC_1 while the reader RISC carries 128 MiB on NOC_0, and
               `noc_async_write_tile` takes an explicit noc index, so half the drain can move
               across without a second RISC, a second CB or a semaphore. Bit-exact by
               construction: same tiles, same addresses, same bytes, different wire.

Run from the repo root; overwrites the generated files.
"""

import sys
from pathlib import Path

OUT = Path("tt_bio/kernels/mm_split")

HELPER = r"""
// --- tt-bio: the two screen arms, see tt_bio/kernels/mm_split/patch_mm_split.py -----------------
#if defined(MM_DUAL_NOC)
// Alternate the wire, not the transaction. `noc_index` is this kernel's own NOC; 1 - noc_index is
// the one the other DM RISC uses for its DRAM reads and which carries a quarter of the traffic.
#define MM_WRITE_TILE(tid, acc, ptr, i) \
    noc_async_write_tile((tid), (acc), (ptr), (uint8_t)(((i) & 1) ? (1 - noc_index) : noc_index))
#define MM_WRITES_FLUSHED()                            \
    do {                                               \
        noc_async_writes_flushed(noc_index);           \
        noc_async_writes_flushed(1 - noc_index);       \
    } while (0)
#define MM_WRITE_BARRIER()                             \
    do {                                               \
        noc_async_write_barrier(noc_index);            \
        noc_async_write_barrier(1 - noc_index);        \
    } while (0)
#elif defined(MM_NOWRITE)
#define MM_WRITE_TILE(tid, acc, ptr, i) do { (void)(tid); (void)(ptr); } while (0)
#define MM_WRITES_FLUSHED() noc_async_writes_flushed()
#define MM_WRITE_BARRIER() noc_async_write_barrier()
#else
#define MM_WRITE_TILE(tid, acc, ptr, i) noc_async_write_tile((tid), (acc), (ptr))
#define MM_WRITES_FLUSHED() noc_async_writes_flushed()
#define MM_WRITE_BARRIER() noc_async_write_barrier()
#endif
// -----------------------------------------------------------------------------------------------
"""

# (file, old, new). Every edit is exact-match and asserted, so a wheel bump that moves any of
# these lines fails loudly instead of silently generating the stock kernel.
HPP_EDITS = [
    # write_block_sync: block-at-a-time drain, used on the deferred-write path.
    ("""            uint32_t tile_id = i * shape.logical_d1 + j;
            noc_async_write_tile(tile_id, tensor_accessor, read_ptr);
            read_ptr += tile_size_bytes;
        }
        // finish up incrementing read_ptr if (d1_end - d1_start) < N_block_tiles
        read_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc_async_writes_flushed();""",
     """            uint32_t tile_id = i * shape.logical_d1 + j;
            MM_WRITE_TILE(tile_id, tensor_accessor, read_ptr, mm_write_seq);
            mm_write_seq++;
            read_ptr += tile_size_bytes;
        }
        // finish up incrementing read_ptr if (d1_end - d1_start) < N_block_tiles
        read_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    MM_WRITES_FLUSHED();"""),
    # write_block_sync_granular: row-at-a-time drain, the path N_chunks == 1 actually takes.
    ("""                uint32_t tile_id = m_tile * shape.logical_d1 + n_tile_id;
                noc_async_write_tile(tile_id, tensor_accessor, out_read_ptr);
                out_read_ptr += tile_size_bytes;
            }
        }
        cb_pop_front(cb_id_out, N_block_tiles);
    }
    noc_async_writes_flushed();""",
     """                uint32_t tile_id = m_tile * shape.logical_d1 + n_tile_id;
                MM_WRITE_TILE(tile_id, tensor_accessor, out_read_ptr, mm_write_seq);
                mm_write_seq++;
                out_read_ptr += tile_size_bytes;
            }
        }
        cb_pop_front(cb_id_out, N_block_tiles);
    }
    MM_WRITES_FLUSHED();"""),
]

# A single counter for the whole kernel, so the alternation keeps striping across block
# boundaries instead of restarting in phase on every block.
HPP_COUNTER = ("""template <uint32_t M_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void write_block_sync(""",
               """static uint32_t mm_write_seq = 0;

template <uint32_t M_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void write_block_sync(""")

CPP_EDITS = [("""    noc_async_write_barrier();
    noc_async_atomic_barrier();""",
              """    MM_WRITE_BARRIER();
    noc_async_atomic_barrier();""")]


def apply(text, edits, path):
    for old, new in edits:
        if text.count(old) != 1:
            raise SystemExit("patch site not unique in %s: %r" % (path, old[:60]))
        text = text.replace(old, new)
    return text


def main():
    sys.path.insert(0, ".")
    from tt_bio.mm_generic import _kernel_dir
    src = _kernel_dir()
    OUT.mkdir(parents=True, exist_ok=True)

    anchor = '#include "api/dataflow/dataflow_api.h"\n'
    hpp = (src / "matmul_dataflow_common.hpp").read_text()
    if hpp.count(anchor) != 1:
        raise SystemExit("include anchor not unique in matmul_dataflow_common.hpp")
    hpp = hpp.replace(anchor, anchor + HELPER)
    hpp = apply(hpp, [HPP_COUNTER] + HPP_EDITS, "matmul_dataflow_common.hpp")
    (OUT / "matmul_dataflow_common.hpp").write_text(hpp)

    for name in ("dm_in0_sender.cpp", "dm_in1_sender_out.cpp"):
        t = (src / name).read_text()
        t = apply(t, CPP_EDITS, name)
        (OUT / name).write_text(t)

    print("wrote", OUT, "from", src)


if __name__ == "__main__":
    main()
