"""Card-free: memory of the writer preload, then after dropping biotite's decoded CCD."""
import gc, sys, time
sys.path.insert(0, sys.argv[1])
def m(t):
    d = {l.split()[0]: int(l.split()[1]) >> 10 for l in open("/proc/self/smaps_rollup") if l.startswith(("Rss", "Pss:"))}
    print(t, d, flush=True)
t = time.time(); import tt_bio.rfd4._writer_preload; m(f"preload {time.time()-t:.1f}s")
from biotite.structure.info.ccd import get_ccd
get_ccd.cache_clear(); gc.collect(); m("ccd dropped")
import ctypes; ctypes.CDLL("libc.so.6").malloc_trim(0); m("malloc_trim")
