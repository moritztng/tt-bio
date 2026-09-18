"""Install the AdaLN memo counter in EVERY process, including spawned fold workers.

`tt_bio/main.py:1135` runs the fold in a `multiprocessing.get_context("spawn")` worker. A spawn
child re-execs python and re-imports every module from source, so a monkeypatch applied in the
parent is gone by the time the model runs -- the first version of this census patched the parent
and read 0 calls in all four arms, which is a DARK instrument, not an absent lever.

`site` imports `sitecustomize` in every interpreter, and spawn children inherit `PYTHONPATH`, so
putting the patch here is what reaches the process that actually folds. The patch itself is
deferred to a meta-path finder because `sitecustomize` runs long before `tt_bio` is importable.

Reads `ADALN_CENSUS_DIR` and writes `<dir>/<pid>.json` at exit. Off unless that variable is set.
"""
import atexit
import json
import os
import sys

TARGET = "tt_bio.tenstorrent"
_DIR = os.environ.get("ADALN_CENSUS_DIR")


def _bytes(t) -> int:
    n = 1
    for d in tuple(t.shape):
        n *= int(d)
    name = str(t.dtype).lower()
    if "float32" in name or "uint32" in name or "int32" in name:
        return n * 4
    if "bfloat8" in name:
        return n
    return n * 2


def _install(T):
    rec = {"pid": os.getpid(), "calls": 0, "atom_calls": 0, "token_calls": 0,
           "hits": 0, "stores": 0, "pair_bytes": None, "pair_shapes": None,
           "pair_dtype": None, "eager": bool(getattr(T, "_B2_ADALN_MEMO_EAGER", False)),
           "memo_on": bool(getattr(T, "_B2_ADALN_S_MEMO", False))}
    orig = T.AdaLN.s_terms

    def wrapped(self, s, large_seq_len=False):
        memo_before = self._s_memo
        out = orig(self, s, large_seq_len)
        rec["calls"] += 1
        if self.atom_level:
            rec["atom_calls"] += 1
        else:
            rec["token_calls"] += 1
        # OBSERVED, not transcribed: a hit returns the object that was already there.
        if memo_before is not None and out is memo_before:
            rec["hits"] += 1
        elif (self._s_memo is not None and self._s_memo is not memo_before
              and self._s_memo_src is s):
            rec["stores"] += 1
            if rec["pair_bytes"] is None:
                sc, bi = self._s_memo
                rec["pair_bytes"] = _bytes(sc) + _bytes(bi)
                rec["pair_shapes"] = [list(map(int, tuple(sc.shape))),
                                      list(map(int, tuple(bi.shape)))]
                rec["pair_dtype"] = str(sc.dtype)
        return out

    T.AdaLN.s_terms = wrapped

    @atexit.register
    def _dump():
        try:
            os.makedirs(_DIR, exist_ok=True)
            with open(os.path.join(_DIR, f"{os.getpid()}.json"), "w") as fh:
                json.dump(rec, fh)
        except Exception:
            pass


class _Patcher:
    """Delegates to the real finder, then wraps `exec_module` so the patch lands after import."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            found = getattr(finder, "find_spec", None)
            if found is None:
                continue
            spec = found(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if loader is None or getattr(loader, "_adaln_census", False):
            return spec
        orig_exec = loader.exec_module

        def exec_module(module, _o=orig_exec):
            _o(module)
            _install(module)

        loader.exec_module = exec_module
        loader._adaln_census = True
        return spec


if _DIR:
    sys.meta_path.insert(0, _Patcher())
