#!/usr/bin/env python3
"""Detect, at runtime, whether OpenFold3's permutation alignment took its silent fallback.

`safe_multi_chain_permutation_alignment` catches ANY exception out of the real alignment,
logs one `logger.error` line and continues on `naive_alignment` with a differently-aligned
ground truth. Nothing in the batch, the loss or the returned features records that this
happened, so a run that fell back is indistinguishable from one that did not unless the log
line is read. D117 is exactly that: two rows drove the fallback for eighty passes and the only
trace was that line.

Two detectors, deliberately independent, because either alone can lie:

  LOG      a `logging.Handler` attached to `openfold3.core.utils.permutation_alignment`'s own
           logger. It reads precisely what upstream emits and patches nothing, so it is the
           one a production run can use. It is blind if a caller has disabled or reconfigured
           that logger, which is the failure mode the second detector covers.
  WRAP     pass-through wrappers around `multi_chain_permutation_alignment` and
           `naive_alignment` that record entry, normal return and the re-raised exception with
           its traceback. They change no behaviour: each calls the original and returns or
           re-raises what it got. This is where the traceback comes from; the log line only
           carries `str(e)`.

The two must agree. If they do not, the guard says so rather than picking one.
"""
from __future__ import annotations

import importlib
import logging
import traceback


class AlignmentGuard:
    """Records whether the real alignment completed, per call."""

    def __init__(self, module_name: str = "openfold3.core.utils.permutation_alignment"):
        self.module_name = module_name
        self.mod = importlib.import_module(module_name)
        self.calls: list[dict] = []
        self.log_records: list[dict] = []
        self._orig: dict = {}
        self._handler = None

    # --- detector LOG -------------------------------------------------------------
    class _Recorder(logging.Handler):
        def __init__(self, sink):
            super().__init__(level=logging.WARNING)
            self.sink = sink

        def emit(self, record):
            self.sink.append({"level": record.levelname,
                              "logger": record.name,
                              "message": record.getMessage()})

    # --- detector WRAP ------------------------------------------------------------
    def _wrap(self, name: str, kind: str):
        orig = getattr(self.mod, name)
        self._orig[name] = orig

        def wrapped(*a, **kw):
            rec = {"path": kind, "completed": False, "error": None, "traceback": None}
            self.calls.append(rec)
            try:
                out = orig(*a, **kw)
            except BaseException as e:               # noqa: BLE001 - re-raised below
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["traceback"] = traceback.format_exc()
                raise
            rec["completed"] = True
            return out

        wrapped.__name__ = name
        setattr(self.mod, name, wrapped)

    def __enter__(self):
        self._handler = self._Recorder(self.log_records)
        logger = logging.getLogger(self.module_name)
        logger.addHandler(self._handler)
        self._logger = logger
        self._wrap("multi_chain_permutation_alignment", "full")
        self._wrap("naive_alignment", "naive")
        return self

    def __exit__(self, *exc):
        for name, orig in self._orig.items():
            setattr(self.mod, name, orig)
        self._orig.clear()
        self._logger.removeHandler(self._handler)
        return False

    # --- verdict ------------------------------------------------------------------
    @property
    def full_calls(self):
        return [c for c in self.calls if c["path"] == "full"]

    @property
    def naive_calls(self):
        return [c for c in self.calls if c["path"] == "naive"]

    @property
    def fell_back_wrap(self) -> bool:
        return any(not c["completed"] for c in self.full_calls) or bool(self.naive_calls)

    @property
    def fell_back_log(self) -> bool:
        return any("falling back to naive alignment" in r["message"]
                   for r in self.log_records)

    def verdict(self) -> dict:
        agree = self.fell_back_wrap == self.fell_back_log
        return {
            "n_alignment_calls": len(self.full_calls),
            "n_completed": sum(1 for c in self.full_calls if c["completed"]),
            "naive_fallback_invoked": bool(self.naive_calls),
            "fell_back_by_wrapper": self.fell_back_wrap,
            "fell_back_by_log": self.fell_back_log,
            "detectors_agree": agree,
            "completed": (not self.fell_back_wrap) and agree and bool(self.full_calls),
            "errors": [{"error": c["error"], "traceback": c["traceback"]}
                       for c in self.calls if c["error"]],
            "log_records": self.log_records,
        }
