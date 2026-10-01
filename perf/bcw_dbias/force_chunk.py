"""campaign_run.py with the query loop forced on, so acceptance can be graded where it is known.

No campaign has accepted a design above 288 tokens on either arm, and at 288 the serving plan is
the whole-query form, which this branch leaves bit-identical. This makes `serving_plan` take the
largest dividing chunk below Nt (Qt=3 at Nt=9) whenever a chunk fits, so the PD-L1 288 campaign
runs every fused backward through the loop. Measurement only; nothing imports it.
"""
import atexit, runpy, sys
import tt_bio.triatt_bw as tb

_orig = tb.serving_plan
calls = {"forced": 0}


def _forced(B, H, N, d, grid, wormhole=None, **kw):
    p = _orig(B, H, N, d, grid, wormhole, **kw)
    if p is None or p["Qt"] < p["Nt"]:
        return p
    Nt = p["Nt"]
    for qt in range(Nt - 1, 0, -1):
        if Nt % qt == 0:
            q = tb.plan(B, H, N, d, grid, q_chunk_tiles=qt, **kw)
            if tb.fits_l1(q, wormhole):
                if calls["forced"] == 0:
                    print(f"[force_chunk] N={N} Nt={Nt} forced Qt={qt}", flush=True)
                calls["forced"] += 1
                return q
    return p


tb.serving_plan = _forced
atexit.register(lambda: print(f"[force_chunk] forced {calls['forced']} plans", flush=True))
sys.argv = ["campaign_run.py"] + sys.argv[1:]
runpy.run_path("perf/bcx_p10_campaign/campaign_run.py", run_name="__main__")
