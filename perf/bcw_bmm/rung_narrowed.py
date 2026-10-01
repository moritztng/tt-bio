"""perf/bgx_size/rung.py, plus what `bmm_program_config` narrowed, written beside rung.json.

`autograd.BMM_NARROWED` is empty on every axis whose plans the L1 pricing left alone, so this
is the per-axis evidence that a sweep changed nothing where it should not have.
"""
import atexit, json, pathlib, runpy, sys
import tt_bio.autograd as ag

out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1])
atexit.register(lambda: (out.mkdir(parents=True, exist_ok=True), (out / "narrowed.json").write_text(
    json.dumps({str(k): v for k, v in getattr(ag, "BMM_NARROWED", {"-": "tree predates the pricing"}).items()}, indent=1))))
sys.argv = ["rung.py"] + sys.argv[1:]
runpy.run_path("perf/bgx_size/rung.py", run_name="__main__")
