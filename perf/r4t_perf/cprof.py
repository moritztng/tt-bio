"""cProfile a trunk_step run; dump stats to $CPROF_OUT. Profiling only."""
import sys, os, cProfile, runpy
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
sys.argv = sys.argv[1:]
pr = cProfile.Profile(); pr.enable()
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    pr.disable(); pr.dump_stats(os.environ["CPROF_OUT"])
