"""Run a script with ONE OuterProductMean whole-z allocation refused, the way the allocator refuses it.

    python perf/spd_msa/opm_force_refusal.py STAGE perf/spd/bench.py --out ... --inputs 9W89 ...

STAGE picks which whole-call allocation after the contraction is refused, at the first OPM call whose
token count is at least 256: `rm` the row-major copy of z (I*C, D*J), `tile` its tiled (I, C*D, J) view,
`permute` the (I, J, C*D) permute. These are the three z-sized buffers a released whole call asks for;
on Wormhole staging11b 0d2737a5b died on the second and third (9W89, 9W8A). The fold must complete.

The patch goes in after tt_bio.tenstorrent is imported by the script itself, so the arm's environment
is in place before anything of tt_bio loads. The refusal is logged to stderr with the shape it hit.
"""
import importlib.abc
import runpy
import sys

STAGE, SCRIPT = sys.argv[1], sys.argv[2]
assert STAGE in ("rm", "tile", "permute"), STAGE
sys.argv = sys.argv[2:]
MSG = "TT_FATAL: Out of Memory: Not enough space to allocate (opm_force_refusal %s %s)"


class _Refuse:
    """The engine's `ttnn`, refusing the chosen whole-z allocation once."""

    def __init__(self, ttnn):
        self._ttnn, self.done = ttnn, False

    def __getattr__(self, name):
        return getattr(self._ttnn, name)

    def _hit(self, stage, shape):
        if self.done or stage != STAGE:
            return
        self.done = True
        print(f"opm_force_refusal: refused {stage} {shape}", file=sys.stderr, flush=True)
        raise RuntimeError(MSG % (stage, shape))

    def to_layout(self, t, layout, *a, **kw):
        s = tuple(t.shape)
        if layout == self._ttnn.ROW_MAJOR_LAYOUT and len(s) == 2 and s[0] == s[1] and s[0] >= 256 * 32:
            self._hit("rm", s)
        if layout == self._ttnn.TILE_LAYOUT and len(s) == 3 and s[0] == s[2] >= 256 and s[1] == 1024:
            self._hit("tile", s)
        return self._ttnn.to_layout(t, layout, *a, **kw)

    def permute(self, t, dims, *a, **kw):
        s = tuple(t.shape)
        if tuple(dims) == (0, 2, 1) and len(s) == 3 and s[0] == s[2] >= 256 and s[1] == 1024:
            self._hit("permute", s)
        return self._ttnn.permute(t, dims, *a, **kw)


class _Patch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "tt_bio.tenstorrent":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        run = spec.loader.exec_module

        def exec_module(mod):
            run(mod)
            mod.ttnn = _Refuse(mod.ttnn)
        spec.loader.exec_module = exec_module
        return spec


import importlib.util  # noqa: E402

sys.meta_path.insert(0, _Patch())
runpy.run_path(SCRIPT, run_name="__main__")
