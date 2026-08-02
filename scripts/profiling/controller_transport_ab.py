#!/usr/bin/env python3
"""Cost the embed result transport against the real ControllerStore, with no device and no
live service.

The 32-card embed fanout is capped by result upload: every shard's output comes back through one
controller process as base64 inside JSON, stored as text in SQLite, while the cards idle. This
drives the production `ControllerStore` in a private temp DB with the exact shape of a measured
N=1024 esmc-600m run (26 shards x 40 files x 651 KB) both ways, so the fix can be costed before
anyone changes a protocol on a live service.

Measured A/B/A/B, 26 shards x 40 files x 651 KB (the shape of a real N=1024 esmc run):

    through the real controller HTTP interface, warm:
        base64-in-JSON-in-SQLite   ~22 s of controller work, SQLite +925 MB per run
        path handback              ~0.8 s, SQLite +0.1 MB        -> ~27x less controller work

The base64 leg is bimodal -- 48-58 s on the first run inside a process, 15-22 s after -- because it
writes 925 MB into SQLite and its cost tracks page-cache state rather than the work. Report a range,
not two agreeing runs. The path leg is stable at 0.71-1.10 s across every run of both harnesses.

The path rows include the worker writing all 693 MB of files; the base64 rows write nothing, so the
comparison favours the status quo. Path handback is only valid when worker and client share a
filesystem, which is true on a single-host galaxy and false for a joined remote one -- a real change
needs capability negotiation, not an unconditional switch.

    python3 controller_transport_ab.py /path/to/tt-bio-checkout
"""

import base64
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

SHARDS, PER_SHARD, SZ = 26, 40, 651 * 1024


def bench(store_cls, mode, blob, b64, scratch):
    d = Path(tempfile.mkdtemp(dir=scratch))
    store = store_cls(d / "c.db")
    run = store.create_run({"data": "x", "out_dir": str(d), "result_dir": str(d),
                            "jobs": [{"id": f"s{i}", "name": f"s{i}.yaml", "input_b64": ""}
                                     for i in range(SHARDS)], "config": {}, "owner": None})
    rid = run["run_id"] if isinstance(run, dict) else run
    leased = []
    for i in range(SHARDS):
        r = store.lease({"worker": {"worker_id": f"w{i}", "host": "h", "device_id": i,
                                    "accelerator": "tenstorrent", "label": f"h:tt{i}"},
                         "batch_size": 1})
        leased += [(f"w{i}", j.get("job_id") or j.get("id")) for j in (r.get("jobs") or [])]

    t0 = time.perf_counter()
    for wid, jid in leased:
        if mode == "b64":
            outputs = {f"{jid}_{k}.npz": b64 for k in range(PER_SHARD)}
        else:
            sd = d / str(jid)
            sd.mkdir(exist_ok=True)
            for k in range(PER_SHARD):
                (sd / f"{jid}_{k}.npz").write_bytes(blob)
            outputs = {f"{jid}_{k}.npz": str(sd / f"{jid}_{k}.npz") for k in range(PER_SHARD)}
        store.complete_job({"run_id": rid, "worker_id": wid,
                            "result": {"status": "ok", "id": jid}, "outputs": outputs})
    t_up = time.perf_counter() - t0

    t0 = time.perf_counter()
    seen = 0
    for _, jid in leased:
        for _, v in store.job_outputs(rid, jid).items():
            seen += len(base64.b64decode(v)) if mode == "b64" else Path(v).stat().st_size
    t_dl = time.perf_counter() - t0

    db_mb = (d / "c.db").stat().st_size / 1e6
    shutil.rmtree(d, ignore_errors=True)
    return len(leased), t_up, t_dl, db_mb, seen


def bench_http(server_cls, client_cls, mode, blob, b64, scratch):
    """Same A/B, but every hop goes over the controller's real HTTP interface.

    The store-only benchmark misses the request handling -- reading the body, parsing
    ~24 MB of JSON per shard, re-serialising the response -- which the live profile
    showed is the rest of the controller's cost. This is what a worker actually pays.
    """
    d = Path(tempfile.mkdtemp(dir=scratch))
    server = server_cls("127.0.0.1", 0, d / "c.db")
    server.serve_in_background()
    client = client_cls(f"http://127.0.0.1:{server.port}")
    try:
        rid = client.create_run({"data": "x", "out_dir": str(d), "result_dir": str(d),
                                 "jobs": [{"id": f"s{i}", "name": f"s{i}.yaml", "input_b64": ""}
                                          for i in range(SHARDS)],
                                 "config": {}, "owner": None})["run_id"]
        leased = []
        for i in range(SHARDS):
            r = client.lease({"worker_id": f"w{i}", "host": "h", "device_id": i,
                              "accelerator": "tenstorrent", "label": f"h:tt{i}"}, 1)
            leased += [(f"w{i}", j.get("job_id") or j.get("id")) for j in (r.get("jobs") or [])]

        t0 = time.perf_counter()
        for wid, jid in leased:
            if mode == "b64":
                outputs = {f"{jid}_{k}.npz": b64 for k in range(PER_SHARD)}
            else:
                sd = d / str(jid)
                sd.mkdir(exist_ok=True)
                for k in range(PER_SHARD):
                    (sd / f"{jid}_{k}.npz").write_bytes(blob)
                outputs = {f"{jid}_{k}.npz": str(sd / f"{jid}_{k}.npz")
                           for k in range(PER_SHARD)}
            client.complete(rid, wid, {"status": "ok", "id": jid},
                            {"event": "done", "name": jid}, outputs=outputs)
        t_up = time.perf_counter() - t0

        t0 = time.perf_counter()
        seen = 0
        for _, jid in leased:
            for _, v in client.job_outputs(rid, jid).items():
                seen += len(base64.b64decode(v)) if mode == "b64" else Path(v).stat().st_size
        t_dl = time.perf_counter() - t0
        db_mb = (d / "c.db").stat().st_size / 1e6
    finally:
        server.shutdown()
    shutil.rmtree(d, ignore_errors=True)
    return len(leased), t_up, t_dl, db_mb, seen


def main() -> int:
    checkout = sys.argv[1] if len(sys.argv) > 1 else "."
    scratch = sys.argv[2] if len(sys.argv) > 2 else tempfile.gettempdir()
    sys.path.insert(0, checkout)
    from tt_bio.distributed import ControllerClient, ControllerServer, ControllerStore

    blob = os.urandom(SZ)
    b64 = base64.b64encode(blob).decode()
    print("--- store only (no HTTP) ---", flush=True)
    for mode in ("b64", "path", "b64", "path"):
        n, up, dl, db, seen = bench(ControllerStore, mode, blob, b64, scratch)
        print(f"{mode:5s} shards={n:2d} upload={up:6.2f}s client_read={dl:6.2f}s "
              f"TOTAL={up + dl:6.2f}s db={db:7.1f}MB payload={seen / 1e6:.0f}MB", flush=True)
    print("--- through the real controller HTTP interface ---", flush=True)
    for mode in ("b64", "path", "b64", "path"):
        n, up, dl, db, seen = bench_http(ControllerServer, ControllerClient, mode, blob,
                                         b64, scratch)
        print(f"{mode:5s} shards={n:2d} upload={up:6.2f}s client_read={dl:6.2f}s "
              f"TOTAL={up + dl:6.2f}s db={db:7.1f}MB payload={seen / 1e6:.0f}MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
