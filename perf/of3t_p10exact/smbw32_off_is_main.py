"""Does TT_BIO_SOFTMAX_BW_FP32=0 reach `ttnn.sum` with exactly the arguments main uses?

The flag's own commit (`b252620cb` on `wk/of3t-p10exact`) made `softmax_bw_inner` pass its
caller's `compute_kernel_config` to the `sum(g y)` reduction UNCONDITIONALLY. That is not a
default-off change: `softmax` and `triangle_attention` both call it with `config or
precise_config()`, so `config` is never None on a shipped path, and the fidelity of every taped
softmax backward in the package would move on the shipped default.

This records the real kwargs by swapping `autograd.ttnn` for a recorder -- no device, no card.
Four arms, and the two OFF ones are asserted against what `origin/main` emits rather than
against a description of it.
"""
import importlib
import os
import sys


class T:
    """A tensor stand-in. `dtype` is what the fp32 branch tests."""

    def __init__(self, dtype, name="t"):
        self.dtype = dtype
        self.name = name

    def volume(self):
        return 4096


class Rec:
    """Records every `ttnn` call `autograd` makes, and returns tensors that keep going."""

    float32 = "FLOAT32"
    bfloat16 = "BFLOAT16"

    class MathFidelity:
        HiFi4 = "HiFi4"

    class WormholeComputeKernelConfig:
        def __init__(self, **kw):
            self.kw = kw

    BlackholeComputeKernelConfig = WormholeComputeKernelConfig

    def __init__(self):
        self.calls = []

    def _mk(self, op):
        def f(*a, **kw):
            self.calls.append((op, {k: ("<cfg>" if k == "compute_kernel_config" and v is not None
                                        else v) for k, v in kw.items()}))
            if op == "typecast":
                return T(a[1], "cast")
            return T(getattr(a[0], "dtype", self.bfloat16), op)
        return f

    def __getattr__(self, op):
        return self._mk(op)


def arm(flag, cfg_present):
    """The `ttnn.sum` calls one `softmax_bw` makes, as (op, compute_kernel_config)."""
    os.environ["TT_BIO_SOFTMAX_BW_FP32"] = "1" if flag else "0"
    for m in [m for m in sys.modules if m.startswith("tt_bio")]:
        del sys.modules[m]
    ag = importlib.import_module("tt_bio.autograd")
    rec = Rec()
    ag.ttnn = rec
    cfg = "<cfg>" if cfg_present else None
    ag.softmax_bw(T(rec.bfloat16, "y"), T(rec.bfloat16, "g"), dim=-1, config=cfg)
    return [(op, kw.get("compute_kernel_config", "<absent>"))
            for op, kw in rec.calls if op == "sum"]


MAIN_OFF = [("sum", "<absent>"), ("sum", "<cfg>")]


def main():
    """Print all four arms and assert the two OFF ones against what origin/main emits.

    Read 2026-09-26 by swapping tt_bio/autograd.py for each tree in turn and running this same
    recorder against it:

        tree                     flag=0 cfg     flag=0 no-cfg   flag=1 cfg
        origin/main 868a2eaab    absent, cfg    absent, cfg     absent, cfg
        b252620cb as handed      CFG,    cfg    None,   cfg     cfg,    cfg
        this tree                absent, cfg    absent, cfg     cfg,    cfg

    Row 2 column 1 is the defect. The second `sum` is the renorm divisor, which has carried a
    config on main since D56 and is not touched here.
    """
    bad = []
    print("flag  config     sum() calls as (op, compute_kernel_config)")
    for flag in (False, True):
        for cfg_present in (True, False):
            got = arm(flag, cfg_present)
            print("%-5s %-10s %s" % (flag, cfg_present, got))
            if not flag and got != MAIN_OFF:
                bad.append("flag=0 config=%s emits %s, main emits %s"
                           % (cfg_present, got, MAIN_OFF))
    for b in bad:
        print("FAIL " + b)
    print("PASS TT_BIO_SOFTMAX_BW_FP32=0 reaches ttnn.sum exactly as origin/main does"
          if not bad else "FAIL the flag is not default-off")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
