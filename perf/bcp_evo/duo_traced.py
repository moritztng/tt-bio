#!/usr/bin/env python3
"""`duo_round.py` with `TracedEvo` in place of the shipped splice when BCP_EVO_TRACE=1.

The round harness builds its splice through `bindcraft2.predictor`, which names
`EvoformerOnDevice` at call time, so rebinding the module attribute before the harness runs is
the whole hook; tt_bio is not edited. With BCP_EVO_TRACE unset or 0 this is duo_round.py
exactly. Each wire's `stats()` is written beside the arm's other artifacts at exit.
"""
import atexit, json, os, pathlib, runpy, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

if os.environ.get("BCP_EVO_TRACE") == "1":
    from tt_bio import bindcraft2 as B
    import traced_evo

    B.EvoformerOnDevice = traced_evo.TracedEvo
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "."

    def dump():
        stats = {f"evo{i}:{slot}": w.stats() for i, e in enumerate(traced_evo.LIVE)
                 for slot, w in e._wires.items()}
        pathlib.Path(out, "trace_wire.json").write_text(json.dumps(stats, indent=1, default=str))

    atexit.register(dump)

sys.argv[0] = str(ROOT / "perf/bcx_p10_duotraj/duo_round.py")
runpy.run_path(sys.argv[0], run_name="__main__")
