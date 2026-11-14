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
        sig = (name, tuple(tuple(t.padded_shape) for t in a[:2] if hasattr(t, "padded_shape")),
               tuple(str(t.dtype) for t in a[:2] if hasattr(t, "dtype")),
               out if isinstance(out, (bool, int, str, type(None))) else type(out).__name__,
               tuple(x for x in a if isinstance(x, (bool, int))))
        if sig not in _seen:
            _seen.add(sig)
            print("PATH", *sig, file=sys.stderr, flush=True)
        return out
    return wrapped


_tail.fused_tail = _trace("fused_tail", _tail.fused_tail)
_tail.fused_tail_ln = _trace("fused_tail_ln", _tail.fused_tail_ln)
from tt_bio import back_ln as _bl  # noqa: E402
_bl.eligible = _trace("back_ln.eligible", _bl.eligible)
_tail.t2m_eligible = _trace("t2m_eligible", _tail.t2m_eligible)
_tt._trimul_route_n = _trace("route_n", _tt._trimul_route_n)
_tt._triangle_mul_memory_config = _trace("tri_memcfg", _tt._triangle_mul_memory_config)
_tt.TriangleMultiplication._t2_serves = _trace("t2_serves", _tt.TriangleMultiplication._t2_serves)
_tt.TriangleMultiplication._t2g_exact = _trace("t2g_exact", _tt.TriangleMultiplication._t2g_exact)
_tt.TriangleMultiplication._t1_in = _trace("t1_in", _tt.TriangleMultiplication._t1_in)
_tt._trimul_out_proj = _trace("trimul_out_proj", _tt._trimul_out_proj)
_tt.TriangleMultiplication._tail_rows = _trace("tail_rows", _tt.TriangleMultiplication._tail_rows)
run = Path(__file__).resolve().parents[1] / "fair" / "run.py"
sys.argv = [str(run)] + sys.argv[1:]
runpy.run_path(str(run), run_name="__main__")
