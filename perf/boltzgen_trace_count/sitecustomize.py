"""Confirm the trace path is actually exercised when BOLTZGEN_DIFFUSION_TRACE=1.
Counts TTDiffusionModule.forward_traced calls and prints the count at exit."""
import os


def _install():
    try:
        import tt_bio.tenstorrent as TT
    except Exception:
        return
    counts = {"traced": 0, "untraced": 0}
    DiffusionModule = TT.DiffusionModule
    orig_traced = DiffusionModule.forward_traced
    orig_forward = DiffusionModule.forward

    def traced(self, *a, **k):
        counts["traced"] += 1
        return orig_traced(self, *a, **k)

    def forward(self, *a, **k):
        counts["untraced"] += 1
        return orig_forward(self, *a, **k)

    DiffusionModule.forward_traced = traced
    DiffusionModule.forward = forward

    import atexit

    @atexit.register
    def _dump():
        msg = f"[TRACE_COUNT] forward_traced={counts['traced']} forward(untraced)={counts['untraced']}"
        print(msg, flush=True)
        with open("/tmp/trace_count.txt", "w") as f:
            f.write(msg + "\n")


_install()
