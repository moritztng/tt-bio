import collections, os, time, ttnn
SEQ = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQ"
       "HDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIH")
def main():
    from tt_bio import esmc
    m = esmc.load_esmc("esmc-300m")
    esmc.embed_sequences(m, {"w": SEQ})           # warm
    t0=time.perf_counter(); esmc.embed_sequences(m, {"w": SEQ}); bare=time.perf_counter()-t0
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    esmc.embed_sequences(m, {"x": SEQ})
    g = ttnn.graph.end_graph_capture()
    ops=[n for n in g if n.get("node_type")=="function_start"
         and str((n.get("params") or {}).get("name","")).startswith("ttnn.")]
    c=collections.Counter((n.get("params") or {}).get("name") for n in ops)
    print("ESMC300M_L%d_WARM_BARE_S %.2f"%(len(SEQ),bare))
    print("ESMC300M_TTNN_OPS %d"%len(ops))
    print("ESMC300M_PROJECTED_CSV_GB %.2f"%(len(ops)*350e3/1e9))
    print("TOP "+", ".join("%s=%d"%kv for kv in c.most_common(10)))
if __name__=="__main__": main()
