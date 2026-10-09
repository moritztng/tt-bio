"""Run perf/bgx_size/rung.py with one pair_transpose variable changed (SPD_PTLEAK=mode).

nol1: no L1 destination or L1 stage for pair transposes. bwperm: the taped entry's backward
always uses ttnn.permute. nocache: the fused kernel rebuilds its descriptors every call.
fwdperm: the taped entry's forward runs ttnn.permute instead of the kernel (backward unchanged).
"""
import os, runpy, sys
mode = os.environ["SPD_PTLEAK"]
import ttnn
import tt_bio.tenstorrent as T, tt_bio.taped_ttnn as TT, tt_bio.pair_transpose as PT, tt_bio.autograd as ag
if mode == "nol1":
    T._TRANSPOSE_L1_HEADROOM = 1e6
    T._l1_memory_config_if_it_fits = lambda *a, **k: ttnn.DRAM_MEMORY_CONFIG
elif mode == "nocache":
    class _D(dict):
        def get(self, k, d=None):
            return None
    PT._CACHE = _D()
elif mode in ("bwperm", "fwdperm"):
    def entry(shipped, args, kwargs):
        x = TT._wrap(args[0])
        ra, rk = TT._raw(args, kwargs)
        perm = (1, 0, 2) if len(x.value.shape) == 3 else (0, 2, 1, 3)
        out_v = ttnn.permute(ra[0], perm, memory_config=(ra[1] if len(ra) > 1 else rk.get("memory_config")) or ttnn.DRAM_MEMORY_CONFIG) \
            if mode == "fwdperm" else shipped(*ra, **rk)
        def make():
            def bw(g):
                x.add_grad(ttnn.permute(g, perm) if mode == "bwperm" or not PT.shape_ok(g)
                           else PT.pair_transpose.__wrapped__(g))
            return bw
        return ag._tape(out_v, [x], make, reads=())
    TT._KERNELS["pair_transpose"] = entry
else:
    raise SystemExit(mode)
sys.argv = ["perf/bgx_size/rung.py"] + sys.argv[1:]
runpy.run_path("perf/bgx_size/rung.py", run_name="__main__")
