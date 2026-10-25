"""Card-free: a writer killed mid-life does not leave write_designs broken for the rest of the process."""
import os, signal, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, str(Path(sys.argv[1]) / "tests"))
from tt_bio import rfd4
from tt_bio.rfd4 import io as IO
from test_rfd4_write_designs import _stand_in
out = Path(__file__).parent / "runs" / f"broken-{os.getpid()}"
ex = rfd4.prepare_examples(sys.argv[2], out / "inputs")[0]
f, aa = rfd4.featurize(ex.cif, seed=0)
o = _stand_in(f, 3)
IO.write_designs(o, f, aa, [out / "a" / f"d{k}.cif" for k in range(3)])
pool = IO.writer_pool(3)
os.kill(next(iter(pool._processes)), signal.SIGKILL)
try:
    IO.write_designs(o, f, aa, [out / "b" / f"d{k}.cif" for k in range(3)])
except Exception as e:
    print("batch during the kill:", type(e).__name__)
print("after:", [p.name for p in IO.write_designs(o, f, aa, [out / "c" / f"d{k}.cif" for k in range(3)])], IO.writer_pool(3) is not pool)
