#!/usr/bin/env python3
"""Generate `tt_bio/kernels/triatt/`'s two operand readers from the wheel's own kernels.

The head-major guards K1 needs are all in this directory's hand-written
`matmul_dataflow_common.hpp`; the two `.cpp` files were byte-identical copies of the wheel's until
the multicast arm, which is shared with `mm_split` and `trimul_tail` and lives in
`tt_bio/kernels/mm_mcast.py`. Run from the repo root; overwrites the two `.cpp` files and leaves
the header alone.
"""

import sys
from pathlib import Path

OUT = Path("tt_bio/kernels/triatt")


def main():
    sys.path.insert(0, ".")
    from tt_bio.mm_generic import _kernel_dir
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tt_bio/kernels is not a package
    import mm_mcast
    src = _kernel_dir()
    for name in ("dm_in0_sender.cpp", "dm_in1_sender_out.cpp"):
        (OUT / name).write_text(mm_mcast.apply((src / name).read_text(), name))
    print("wrote", OUT, "from", src)


if __name__ == "__main__":
    main()
