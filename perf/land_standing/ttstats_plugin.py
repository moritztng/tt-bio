"""Dump the triangle-attention counters at pytest SESSION FINISH, not at interpreter exit.

The first attempt used `atexit`. This test aborts in device teardown
(`pthread_mutex_unlock failed for mutex CHIP_IN_USE_3_PCIe` inside `_close_device_locked`), and
SIGABRT skips atexit handlers, so both arms produced no counters at all — "one test passed in each
arm" with no proof the n=288 path was even reached, which is the exact ambiguity this run exists
to remove.

`pytest_sessionfinish` runs while the interpreter is still healthy, before the close that aborts.
"""
import json
import os
import sys


def pytest_runtest_logreport(report):
    # Fires right after the test BODY, before any teardown. sessionfinish was too
    # late: this test aborts in device close during teardown, so the process dies
    # before the session ever finishes and both arms produced no counters twice.
    if report.when != "call":
        return
    exitstatus = report.outcome
    out = os.environ.get("TT_STOCK_STATS_DUMP")
    if not out:
        return
    T = sys.modules.get("tt_bio.tenstorrent")
    TS = sys.modules.get("tt_bio.triatt_sdpa")
    rec = {
        "pid": os.getpid(),
        "outcome": str(exitstatus),
        "nodeid": getattr(report, "nodeid", None),
        "dividing_k_env": os.environ.get("TT_BIO_TRIATT_DIVIDING_K"),
        "hifi_stats": dict(getattr(T, "TRIATT_FUSED_HIFI_STATS", {}) or {}),
        "hifi_picks": {str(k): v for k, v in
                       (getattr(T, "TRIATT_FUSED_HIFI_PICKS", {}) or {}).items()},
        "chunk_picks": {str(k): v for k, v in
                        (getattr(T, "SDPA_CHUNK_PICKS", {}) or {}).items()},
        "route_counts": dict(getattr(T, "SDPA_ROUTE_COUNTS", {}) or {}),
        "k_chunk_stats": list(getattr(T, "SDPA_K_CHUNK_STATS", []) or []),
        "kernel_stats": list(getattr(TS, "STATS", []) or []),
    }
    try:
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:                                              # noqa: BLE001
        pass
