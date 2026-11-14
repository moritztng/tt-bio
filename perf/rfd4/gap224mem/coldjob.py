"""Run inside a writer by cold.py: one write_design, optionally profiled."""
import time


def job(row, mask, aa, p, prof):
    from tt_bio.rfd4 import io as IO
    t0 = time.perf_counter()
    if prof:
        import cProfile, io, pstats
        pr = cProfile.Profile()
        pr.enable()
    IO.write_design(row, mask, aa, p)
    dt = time.perf_counter() - t0
    if prof:
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(18)
        pstats.Stats(pr, stream=s).print_callers("get_from_ccd|_residue_index")
        print(s.getvalue()[-3000:], flush=True)
    import os
    return dt, len(os.listdir("/proc/self/task"))
