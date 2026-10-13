"""No tt-bio data-movement kernel copies within its own core over the NoC while it also writes.

A same-core L1 -> L1 copy issued as a NoC read (`get_noc_addr(<local address>)`) on the RISC whose NoC
also carries non-posted DRAM writes deadlocked reblock_permute_gated inside a Protenix-v2 fold on one
Blackhole chip (qb2 chip 2). Such copies go through `kernels/common/l1_copy.hpp` on the RISC instead.

Host only: the check reads the kernel sources. The exempt files are ttnn's own matmul and SDPA data
movers, kept as upstream wrote them; their only local reads are zero fills from MEM_ZEROS.
"""

import re
from pathlib import Path

import tt_bio

KERNELS = Path(tt_bio.__file__).resolve().parent / "kernels"
LOCAL_NOC_ADDR = re.compile(r"(?<![.\w])get_noc_addr\(\s*[^,()]+(?:\([^()]*\))?\s*\)")
VENDORED = {
    "mm_split/matmul_dataflow_common.hpp",
    "swiglu_fused/matmul_dataflow_common.hpp",
    "triatt/matmul_dataflow_common.hpp",
    "trimul_tail/matmul_dataflow_common.hpp",
    "triatt_sdpa/dataflow/dataflow_common.hpp",
}


def test_no_local_noc_copy_beside_writes():
    bad = []
    for f in sorted(KERNELS.rglob("*.[ch]pp")):
        rel = f.relative_to(KERNELS).as_posix()
        if rel in VENDORED:
            continue
        src = re.sub(r"//.*", "", f.read_text())
        if "noc_async_write" in src and LOCAL_NOC_ADDR.search(src):
            bad.append(rel)
    assert not bad, f"same-core NoC copy in a kernel that also writes, use l1_copy: {bad}"
