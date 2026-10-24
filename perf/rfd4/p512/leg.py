"""One A/B leg of rfd4-perf-p512: `perf/rfd4/fair/run.py` as it is, plus two records it does not keep.

    TT_VISIBLE_DEVICES=<chip> python perf/rfd4/p512/leg.py [--dram] <run.py args>

* Every batch's design tensors go to `designs.pt` beside its CIFs, so two arms can be compared tensor for
  tensor (`compare.py`), not only through the CIF's printed precision.
* With `--dram`, the allocator's DRAM in use is read before every `ttnn.deallocate` and after every upload,
  and the largest reading (bytes a bank x banks) is written to `<out>/dram.json`. A tensor freed by Python
  refcount alone is not sampled, so this is a lower bound of the true peak; both arms get the same instrument.
  The readings make the leg slower: a `--dram` leg is never a timed one.
"""
import json
import runpy
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
dram = "--dram" in sys.argv
if dram:
    sys.argv.remove("--dram")

from tt_bio import rfd4  # noqa: E402

_write = rfd4.write_designs


def write_designs(o, features, atom_array, paths):
    torch.save(o, Path(paths[0]).parent / "designs.pt")
    return _write(o, features, atom_array, paths)


rfd4.write_designs = write_designs

if dram:
    import ttnn
    from tt_bio import tenstorrent

    peak = {"bytes": 0, "samples": 0}
    _dealloc, _upload = ttnn.deallocate, tenstorrent.upload

    def _read():
        dev = tenstorrent._device
        if dev is None:
            return
        mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        peak["bytes"] = max(peak["bytes"], int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks))
        peak["samples"] += 1

    def deallocate(t, *a, **k):
        _read()
        return _dealloc(t, *a, **k)

    def upload(*a, **k):
        out = _upload(*a, **k)
        _read()
        return out

    ttnn.deallocate = deallocate
    tenstorrent.upload = upload
    import tt_bio.rfd4.trunk as trunk_mod
    trunk_mod.upload = upload

try:
    sys.argv[0] = str(HERE.parent / "fair" / "run.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    if dram:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
        (out / "dram.json").write_text(json.dumps({**peak, "gib": round(peak["bytes"] / 2**30, 3)}) + "\n")
