"""Warm op-count + timing probe for the embed models (esmc-600m, saprot-650m).

Mirrors the perf gate protocol (8x ubiquitin 76 aa, batch 8, warmup-then-time)
and adds a ttnn.graph capture of ONE warm call to count dispatched ttnn ops —
the structural input to "is this model dispatch-bound?" (skill ttnn-perf-profiling:
graph capture is the wheel-only instrument for op counts; its timings are lies,
so timing comes from the bare warm calls, which end in a host read = honest sync).

Usage:
  TT_VISIBLE_DEVICES=0 PYTHONPATH=<repo> python3 scripts/profiling/embed_opcount.py esmc-600m
  TT_VISIBLE_DEVICES=0 PYTHONPATH=<repo> python3 scripts/profiling/embed_opcount.py saprot-650m
"""
import collections, os, sys, time

UBIQUITIN = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTL"
             "LHLVLRLRGG")  # 76 aa — same fixture as scripts/perf_regression.py

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "esmc-600m"
    import torch
    torch.set_grad_enabled(False)
    from tt_bio.tenstorrent import get_device
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    get_device()

    if model.startswith("saprot"):
        from tt_bio import saprot
        seqs = {f"u{i}": UBIQUITIN for i in range(8)}
        m = saprot.load_saprot(model)
        call = lambda: saprot.embed_sequences(m, seqs, pool="mean", batch_size=8)
    else:
        from tt_bio import esmc
        seqs = {f"u{i}": UBIQUITIN for i in range(8)}
        m = esmc.load_esmc(model)
        call = lambda: esmc.embed_sequences(m, seqs, batch_size=8)

    call(); call()  # warmup: absorb first-kernel compile + program-cache fill
    times = []
    for _ in range(3):
        t0 = time.perf_counter(); call(); times.append(time.perf_counter() - t0)
    warm_ms = sorted(times)[1] * 1000.0

    import ttnn
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    call()
    g = ttnn.graph.end_graph_capture()
    ops = [n for n in g if n.get("node_type") == "function_start"
           and str((n.get("params") or {}).get("name", "")).startswith("ttnn.")]
    c = collections.Counter((n.get("params") or {}).get("name") for n in ops)
    tag = model.upper().replace("-", "")
    print(f"{tag}_WARM_BATCH8_MS {warm_ms:.2f}  (per-seq {warm_ms/8:.2f} ms, {8*1000/warm_ms:.1f} seq/s)")
    print(f"{tag}_TTNN_OPS_PER_BATCH8 {len(ops)}  ({len(ops)/8:.0f} ops/seq)")
    print(f"{tag}_US_PER_OP_ENQUEUE_AVG {warm_ms*1000/len(ops):.1f}")
    print("TOP " + ", ".join("%s=%d" % kv for kv in c.most_common(12)))

if __name__ == "__main__":
    main()
