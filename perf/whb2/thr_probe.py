"""Print the thresholds actually in effect on this part, after the device is open."""
import sys
from pathlib import Path
tree = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(tree))
import ttnn                                                              # noqa: E402
import tt_bio.tenstorrent as T                                           # noqa: E402
assert Path(T.__file__).resolve().is_relative_to(tree), T.__file__
d = T.get_device()
mv = ttnn.get_memory_view(d, ttnn.BufferType.DRAM)
dram = int(mv.total_bytes_per_bank) * int(mv.num_banks)
print(f"RESULT arch={T.arch_name()} grid={tuple(T.COMPUTE_GRID_MAIN)} "
      f"small_grid={T._IS_SMALL_GRID} dram_gib={dram / 2**30:.2f} "
      f"SEQ_LEN_MORE_CHUNKING={T.SEQ_LEN_MORE_CHUNKING}")
T.cleanup()
