"""One warm ESMC-300M forward, for the device profiler to attribute."""
SEQ = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQ"
       "HDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIH")
def main():
    from tt_bio import esmc
    m = esmc.load_esmc("esmc-300m")
    esmc.embed_sequences(m, {"warm": SEQ})     # warm: JIT + program cache
    esmc.embed_sequences(m, {"prof": SEQ})     # the one we attribute
    print("PROFILE_TARGET_DONE")
if __name__ == "__main__": main()
