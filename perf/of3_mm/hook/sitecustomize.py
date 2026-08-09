"""ttnn.matmul census that survives tt-bio's spawned fold worker.

tt-bio runs the fold in a spawned child, so patching ttnn.matmul in the parent records nothing.
Python imports sitecustomize in every interpreter it starts, including that child, so putting the
patch here catches the process that actually issues the matmuls.

Enable by putting this directory first on PYTHONPATH and setting OF3_MM_CENSUS to an output
directory. Each process writes census_<pid>.json at exit; the fold child is the big one.
"""
import os, sys

_OUT = os.environ.get("OF3_MM_CENSUS")

if _OUT:
    import atexit, builtins, collections, json

    _counter = collections.Counter()
    _state = {"patched": False}

    def _shape(t):
        for attr in ("padded_shape", "shape_with_tile_padding"):
            v = getattr(t, attr, None)
            if v is not None:
                try:
                    return tuple(int(x) for x in v)
                except TypeError:
                    pass
        return tuple(int(x) for x in t.shape)

    def _patch(ttnn):
        if _state["patched"] or not hasattr(ttnn, "matmul"):
            return
        _state["patched"] = True
        real = ttnn.matmul

        def patched(a, b, *args, **kw):
            try:
                fr = sys._getframe(1)
                site = "%s:%d" % (fr.f_code.co_filename.split("/")[-1], fr.f_lineno)
                hint = ("core_grid" if kw.get("core_grid") is not None else
                        "program_config" if kw.get("program_config") is not None else "none")
                _counter[(site, _shape(a), _shape(b), str(a.dtype), str(b.dtype),
                          str(a.memory_config().buffer_type),
                          str(b.memory_config().buffer_type), hint)] += 1
            except Exception:
                _counter[("introspect-failed", (), (), "", "", "", "", "")] += 1
            return real(a, b, *args, **kw)

        ttnn.matmul = patched

    _real_import = builtins.__import__

    def _hooked_import(name, *a, **kw):
        mod = _real_import(name, *a, **kw)
        if not _state["patched"] and "ttnn" in sys.modules:
            _patch(sys.modules["ttnn"])
        return mod

    builtins.__import__ = _hooked_import

    @atexit.register
    def _dump():
        if not _counter:
            return
        rows = []
        for (site, ash, bsh, ad, bd, ab, bb, hint), n in sorted(_counter.items(), key=lambda kv: -kv[1]):
            row = dict(site=site, a=list(ash), b=list(bsh), a_dtype=ad, b_dtype=bd,
                       a_buf=ab, b_buf=bb, hint=hint, calls=n)
            if len(ash) >= 2 and len(bsh) >= 2:
                B = 1
                for d in ash[:-2]:
                    B *= d
                row.update(B=B, Mt=ash[-2] // 32, Kt=ash[-1] // 32, Nt=bsh[-1] // 32)
            rows.append(row)
        os.makedirs(_OUT, exist_ok=True)
        with open(os.path.join(_OUT, "census_%d.json" % os.getpid()), "w") as f:
            json.dump(rows, f, indent=1)
