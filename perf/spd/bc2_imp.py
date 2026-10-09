"""Import the modules named in SPD_IMPORTS (comma list) first, set SPD_GC ("disable" or a gen-0
threshold), then run perf/bgx_size/rung.py."""
import gc, importlib, os, runpy, sys
for m in filter(None, os.environ.get("SPD_IMPORTS", "").split(",")):
    importlib.import_module(m)
g = os.environ.get("SPD_GC")
if g == "disable":
    gc.disable()
elif g:
    gc.set_threshold(int(g), 2, 2)
sys.argv = ["perf/bgx_size/rung.py"] + sys.argv[1:]
runpy.run_path("perf/bgx_size/rung.py", run_name="__main__")
