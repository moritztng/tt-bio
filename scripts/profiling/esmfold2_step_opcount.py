"""Graph-capture ONE warm diffusion step of a real tt-bio model (ESMFold2-Fast).

Answers "how many ttnn ops does this model dispatch per diffusion step?" — the
number the device profiler cannot reach, because it hard-fails at 1000 programs.
"""
import collections, os, sys, time

import ttnn
import tt_bio.esmfold2 as E

SEQ = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQ"
       "HDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIH"
       "FVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFEL"
       "SSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL")
TARGET_STEP = int(os.environ.get("CAPTURE_STEP", "3"))


def _rss_mb():
    """CURRENT rss. Not resource.ru_maxrss -- that is a high-water mark, so on a
    process that already peaked during weight load it reports a delta of 0."""
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
_orig = E.sample_structure


def patched(denoise_fn, *a, **kw):
    n = {"i": 0}

    def wrapped(x, t):
        n["i"] += 1
        if n["i"] != TARGET_STEP:      # steps 1..N-1 warm this exact shape first
            return denoise_fn(x, t)
        r0 = _rss_mb()
        t0 = time.perf_counter()
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        out = denoise_fn(x, t)
        g = ttnn.graph.end_graph_capture()
        wall = time.perf_counter() - t0
        r1 = _rss_mb()
        ops = [nd for nd in g
               if nd.get("node_type") == "function_start"
               and str((nd.get("params") or {}).get("name", "")).startswith("ttnn.")]
        c = collections.Counter((nd.get("params") or {}).get("name") for nd in ops)
        print("\n=== REAL MODEL: ESMFold2-Fast, ONE diffusion step (L=%d) ===" % len(SEQ))
        print("TTNN_OPS_PER_DIFFUSION_STEP %d" % len(ops))
        print("GRAPH_NODES %d  NODES_PER_OP %.1f" % (len(g), len(g) / max(1, len(ops))))
        print("CAPTURE_RSS_DELTA_MB %.0f  BYTES_PER_OP %.0f"
              % (r1 - r0, (r1 - r0) * 1e6 / max(1, len(ops))))
        print("CAPTURED_STEP_WALL_S %.2f (instrumented — NOT a timing number)" % wall)
        print("DISTINCT_OP_TYPES %d" % len(c))
        print("TOP_OPS " + ", ".join("%s=%d" % kv for kv in c.most_common(15)))
        sys.stdout.flush()
        return out

    return _orig(wrapped, *a, **kw)


def main():
    E.sample_structure = patched
    from tt_bio import esmfold2_runtime as R
    t0 = time.perf_counter()
    model = R.load_ttnn_esmfold2(esmfold2_repo="biohub/ESMFold2-Fast", fast=True)
    print("LOAD_S %.1f" % (time.perf_counter() - t0), flush=True)
    t0 = time.perf_counter()
    res = R.fold_complex(model, [("A", SEQ)], num_loops=1, num_sampling_steps=8)
    print("FOLD_S %.1f  plddt=%.3f" % (time.perf_counter() - t0, float(res.plddt.mean())))


if __name__ == "__main__":
    main()
