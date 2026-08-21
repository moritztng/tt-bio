"""Startup hook that installs the softmax census in EVERY process of a fold, children included.

`tt_bio predict` runs its device work in `mp.get_context("spawn")` children (main.py:1105), so a
monkeypatch applied in the CLI process measures nothing: the spawned interpreter re-imports
everything from scratch. A `sitecustomize` on `PYTHONPATH` is imported by `site` at startup in
the parent and in every spawned child, which is the only hook that covers all of them.

Activated only when TT_BIO_SM_CENSUS_DIR is set, so having this directory on PYTHONPATH is inert
otherwise. Each process writes `census_pid<N>.json` into that directory; the runner merges them.

ttnn is not imported here -- it is expensive and this hook runs in processes that never touch a
device. Instead `builtins.__import__` is wrapped until ttnn shows up, then the patch is applied
and the wrapper removed.
"""
import builtins
import os
import sys

_DIR = os.environ.get("TT_BIO_SM_CENSUS_DIR")

if _DIR:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from census_core import Census

    try:  # so a missing process is distinguishable from a process that made no calls
        with open(os.path.join(_DIR, "procs.log"), "a") as _fh:
            _fh.write("%d\t%s\n" % (os.getpid(), " ".join(sys.argv)[:300]))
    except Exception:
        pass

    _root = os.environ.get("TT_BIO_SM_CENSUS_ROOT", os.getcwd())
    _census = Census(_root, int(os.environ.get("TT_BIO_SM_CENSUS_MAX", "24")))
    _real_import = builtins.__import__
    _state = {"done": False}

    def _try_install():
        """Idempotent, and re-run on every import on purpose.

        ttnn puts itself in sys.modules before its body finishes, so the first import event
        that reaches here can find a half-built module: the wrapper goes on, and then ttnn's
        own initialisation rebinds `ttnn.softmax` back to the raw op and the census goes
        silent (measured: ESMFold2 recorded 0 calls at a site that runs unconditionally,
        while OpenFold3's spawned child -- which imports ttnn fully before the hook fires --
        recorded all six of its sites). So never latch: re-assert the wrapper on every import
        and let the marker make it a no-op once it has stuck.
        """
        if "ttnn" not in sys.modules:
            return
        try:
            if _census.install() and not _state["done"]:
                _state["done"] = True
                import atexit
                import signal
                _census.out_dir = _DIR
                atexit.register(_census.dump, _DIR)
                # Chain onto, never replace, tt_bio's own shutdown handling.
                for _sig in (signal.SIGTERM, signal.SIGINT):
                    _prev = signal.getsignal(_sig)

                    def _handler(sig, frm, _prev=_prev):
                        try:
                            _census.dump(_DIR)
                        except Exception:
                            pass
                        if callable(_prev):
                            return _prev(sig, frm)
                        if _prev == signal.SIG_DFL:
                            signal.signal(sig, signal.SIG_DFL)
                            os.kill(os.getpid(), sig)

                    try:
                        signal.signal(_sig, _handler)
                    except (ValueError, OSError):
                        pass  # not the main thread
        except Exception:
            pass

    def _hooked_import(name, *a, **kw):
        m = _real_import(name, *a, **kw)
        _try_install()
        return m

    builtins.__import__ = _hooked_import
    _try_install()
