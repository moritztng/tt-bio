"""Run perf/rfd4/fair/run.py with every failing ttnn.linear logged (operand shapes, dtypes, kwargs).

    TT_VISIBLE_DEVICES=0 python perf/rfd4/l1decline/repro.py <run.py args>
"""
import runpy
import sys
from pathlib import Path

import ttnn

_linear = ttnn.linear


def linear(a, b, *args, **kw):
    try:
        return _linear(a, b, *args, **kw)
    except Exception as e:
        print("LINEAR-REFUSED", tuple(a.padded_shape), a.dtype, tuple(b.padded_shape), b.dtype,
              {k: v for k, v in kw.items() if k != "compute_kernel_config"}, str(e).splitlines()[:3],
              file=sys.stderr, flush=True)
        from tt_bio import trimul_tail
        print("T2-REJECTS", trimul_tail.REJECTS, "T2_STATS", trimul_tail.T2_STATS, file=sys.stderr, flush=True)
        raise


ttnn.linear = linear

from tt_bio import tenstorrent as _tt  # noqa: E402
from tt_bio import trimul_tail as _tail  # noqa: E402

_seen = set()


def _trace(name, fn):
    def wrapped(*a, **kw):
        out = fn(*a, **kw)
        sig = (name, tuple(tuple(t.padded_shape) for t in a[:1] if hasattr(t, "padded_shape")),
               tuple(str(t.dtype) for t in a[:1] if hasattr(t, "dtype")), out is None)
        if sig not in _seen:
            _seen.add(sig)
            print("PATH", *sig, file=sys.stderr, flush=True)
        return out
    return wrapped


_tail.fused_tail = _trace("fused_tail", _tail.fused_tail)
_tail.fused_tail_ln = _trace("fused_tail_ln", _tail.fused_tail_ln)
_tt._trimul_tail = _tail
_tt._trimul_out_proj = _trace("trimul_out_proj", _tt._trimul_out_proj)
_tt.TriangleMultiplication._tail_rows = _trace("tail_rows", _tt.TriangleMultiplication._tail_rows)
run = Path(__file__).resolve().parents[1] / "fair" / "run.py"
sys.argv = [str(run)] + sys.argv[1:]
runpy.run_path(str(run), run_name="__main__")
