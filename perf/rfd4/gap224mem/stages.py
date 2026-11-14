"""Card-free: memory of one writer process stage by stage (import, preload, warm, first write). Prints JSON lines."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
def mem(tag):
    d = {}
    for line in open("/proc/self/smaps_rollup"):
        k, *v = line.split()
        if k in ("Rss:", "Pss:", "Anonymous:"): d[k[:-1].lower() + "_mib"] = int(v[0]) // 1024
    d["stage"] = tag; d["mods"] = len(sys.modules); print(json.dumps(d), flush=True)
mem("python")
import torch; mem("torch")
import tt_bio._vendor.rfd4_prep.processing; mem("preload processing")
from tt_bio.rfd4 import io as IO
IO._warm_worker(); mem("warm")
if len(sys.argv) > 2:
    import tempfile
    from tt_bio import rfd4
    out = Path(tempfile.mkdtemp(dir=Path(__file__).parent))
    ex = rfd4.prepare_examples(sys.argv[2], out / "inputs")[0]
    f, aa = rfd4.featurize(ex.cif, seed=0); mem("featurize (parent only)")
    sys.path.insert(0, str(Path(sys.argv[1]) / "tests"))
    from test_rfd4_write_designs import _stand_in
    o = _stand_in(f, 1)
    rfd4.write_design(o, f, aa, out / "d0.cif", row=0); mem("write")
