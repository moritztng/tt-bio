"""Which import in the writer preload leaves threads running (a forkserver must fork single-threaded)."""
import sys, threading, os
sys.path.insert(0, sys.argv[1])
def n(tag): print(tag, "os threads", len(os.listdir("/proc/self/task")), [t.name for t in threading.enumerate()], flush=True)
n("start")
import torch; n("torch")
torch.set_num_threads(1); n("set_num_threads")
from tt_bio._vendor.atomworks.io.template import get_empty_ccd_template; n("template")
from tt_bio._vendor.rfd4_prep._cif import transforms_unindexing; n("transforms")
from tt_bio._vendor.rfd4_prep.processing import write_cif; n("processing")
from tt_bio.rfd4.io import write_design; n("io")
get_empty_ccd_template("ALA"); n("ALA")
