"""Record every ttnn.softmax call a fold makes: shape, dtype, whether a compute kernel
config reached the kernel, and the call chain. site_softmax is a wrapper, so one frame is
not enough to name the construction site; three are."""
import atexit
import json
import os
import traceback


def _install():
    out_dir = os.environ.get("TT_BIO_SMX_SPY")
    if not out_dir:
        return
    try:
        import ttnn
    except Exception:
        return
    real = ttnn.softmax
    seen = {}

    def spy(x, *a, **kw):
        try:
            st = traceback.extract_stack(limit=6)[:-1]
            chain = " <- ".join("%s:%d" % (os.path.basename(f.filename), f.lineno)
                                for f in reversed(st[-3:]))
            cfg = kw.get("compute_kernel_config")
            desc = "none" if cfg is None else "|".join(
                "%s=%s" % (f, getattr(cfg, f, "?")) for f in
                ("math_fidelity", "math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc"))
            key = "%s|%s|%s|%s" % (chain, tuple(x.shape), x.dtype, desc)
            r = seen.setdefault(key, {"chain": chain, "shape": list(x.shape),
                                      "dtype": str(x.dtype),
                                      "compute_kernel_config": desc, "calls": 0})
            r["calls"] += 1
        except Exception:
            pass
        return real(x, *a, **kw)

    ttnn.softmax = spy

    @atexit.register
    def _dump():
        if not seen:
            return
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "%d.json" % os.getpid()), "w") as f:
            json.dump(sorted(seen.values(), key=lambda r: -r["calls"]), f, indent=1)


_install()
