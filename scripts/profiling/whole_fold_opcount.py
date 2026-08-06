"""Total dispatched-op count for a WHOLE real fold -> sets --op-support-count and the disk budget."""
import collections, os, sys, time
import ttnn

SEQ = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQ"
       "HDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIH"
       "FVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFEL"
       "SSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL")

def _rss_mb():
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6

def main():
    steps = int(os.environ.get("STEPS", "8"))
    from tt_bio import esmfold2_runtime as R
    m = R.load_ttnn_esmfold2(esmfold2_repo="biohub/ESMFold2-Fast", fast=True)
    print("LOADED", flush=True)
    r0, t0 = _rss_mb(), time.perf_counter()
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    res = R.fold_complex(m, [("A", SEQ)], num_loops=1, num_sampling_steps=steps)
    g = ttnn.graph.end_graph_capture()
    wall, r1 = time.perf_counter() - t0, _rss_mb()
    ops = [n for n in g if n.get("node_type") == "function_start"
           and str((n.get("params") or {}).get("name", "")).startswith("ttnn.")]
    c = collections.Counter((n.get("params") or {}).get("name") for n in ops)
    print("WHOLE_FOLD_TTNN_OPS %d" % len(ops))
    print("WHOLE_FOLD_GRAPH_NODES %d" % len(g))
    print("CAPTURE_RSS_DELTA_MB %.0f" % (r1 - r0))
    print("INSTRUMENTED_FOLD_S %.1f (NOT a timing number)" % wall)
    print("PROJECTED_DEVICE_CSV_GB %.2f  (at 350 KB/op)" % (len(ops) * 350e3 / 1e9))
    print("TOP_OPS " + ", ".join("%s=%d" % kv for kv in c.most_common(15)))

if __name__ == "__main__":
    main()
