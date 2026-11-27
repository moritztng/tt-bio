"""Cross-host gradient sync, hierarchical, host only: what a second Galaxy adds to a DP step.

Two hosts with L ranks each (global rank = host * L + local). Per step every rank:
  1. reduce-scatters its gradient with the host's other ranks (`ProcessAxis.reduce_scatter`,
     /dev/shm, the launcher's own collective), leaving it this host's partial sum of 1/L;
  2. swaps that slice with the same local rank on the other host over one TCP stream (L streams
     in parallel per host pair) and adds the two partials in host order (host 0 + host 1);
  3. all-gathers the summed slices on its host (`ProcessAxis.all_gather`).
Deterministic and identical on every rank of both hosts, but NOT the flat rank-order sum, so
the reference is the hierarchical one: (sum over host 0 ranks) + (sum over host 1 ranks).
No chips are opened. Plain TCP on the LAN, one listening port per local rank on host 0,
bound to --bind, and a shared token checked on connect.

    host0: python3 xhost_hier.py --host 0 --local 16 --peer 10.0.0.2 --bind 10.0.0.1 --token T --out h0.json
    host1: python3 xhost_hier.py --host 1 --local 16 --peer 10.0.0.1 --token T --out h1.json

Gates: every rank of both hosts ends each step on the same digest (checked over the stream);
rank 0 of host 0 rebuilds the hierarchical reference on steps 2 and the last and must match
bit for bit (timings are medians of steps 4.., clear of the step-2 rebuild every rank waits
out). --broken (on either host, step 2): local rank 0 keeps its own partial instead of adding
the other host's, which must FAIL (on host 0 the reference too; exit 1 on PASS when broken).
"""
import argparse, hashlib, json, multiprocessing as mp, os, shutil, socket, struct, sys, threading, time
from pathlib import Path

import numpy as np

N_RFD4 = 178_448_304


def grad(rank, step, n):
    """Deterministic per-(global rank, step) fill, no RNG cost on 178 M elements."""
    v = np.arange(n, dtype=np.float32)
    v *= np.float32(1e-9 * (rank + 1))
    v += np.float32(step * 0.25 + rank)
    return v


def _recv_exact(sock, buf):
    mv, got = memoryview(buf).cast("B"), 0
    while got < len(mv):
        k = sock.recv_into(mv[got:])
        if not k:
            raise ConnectionError("peer closed mid-message")
        got += k


def _connect(host, local, peer, bind, port, token):
    if host == 0:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((bind, port))
        srv.listen(1)
        srv.settimeout(600)
        s, _ = srv.accept()
        srv.close()
        got = s.recv(len(token))
        if got != token:
            s.close()
            raise PermissionError("peer sent the wrong token")
    else:
        t_end = time.time() + 600
        while True:
            try:
                s = socket.create_connection((peer, port), timeout=10)
                break
            except OSError:
                if time.time() > t_end:
                    raise
                time.sleep(0.5)
        s.sendall(token)
    s.settimeout(600)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        s.setsockopt(socket.SOL_SOCKET, opt, 64 << 20)
    return s


def _swap(sock, mine, theirs):
    """Send `mine`, receive the same number of bytes into `theirs`, concurrently."""
    err = []

    def tx():
        try:
            sock.sendall(memoryview(mine).cast("B"))
        except Exception as e:  # noqa: BLE001 - surfaced below
            err.append(e)
    t = threading.Thread(target=tx)
    t.start()
    _recv_exact(sock, theirs)
    t.join()
    if err:
        raise err[0]


def rank_main(host, local_rank, L, n, steps, run, peer, bind, port, token, broken, q):
    from tt_bio.train.launcher import ProcessAxis
    ax = ProcessAxis(name="dp", device_ids=tuple(range(L)), dp_rank=local_rank, run=str(run))
    sock = _connect(host, local_rank, peer, bind, port + local_rank, token)
    g = host * L + local_rank
    rows = []
    for s in range(1, steps + 1):
        v = grad(g, s, n)
        t0 = time.perf_counter()
        part = ax.reduce_scatter(v)
        t1 = time.perf_counter()
        other = np.empty_like(part)
        _swap(sock, part, other)
        t2 = time.perf_counter()
        if broken and local_rank == 0 and s == 2:
            summed = part
        else:
            summed = (part + other) if host == 0 else (other + part)  # host order, both sides
        tot = ax.all_gather(summed)
        t3 = time.perf_counter()
        for k in ("_pub", "_wait"):
            ax.state.pop(k, None)
        dig = hashlib.blake2b(tot.tobytes(), digest_size=8).digest()
        peer_dig = bytearray(8)
        _swap(sock, np.frombuffer(dig, np.uint8), np.frombuffer(peer_dig, np.uint8))
        row = {"step": s, "rs_s": t1 - t0, "xhost_s": t2 - t1, "ag_s": t3 - t2, "wall_s": t3 - t0,
               "xhost_bytes": int(part.nbytes), "digest": dig.hex(), "peer_agrees": dig == bytes(peer_dig)}
        if host == 0 and local_rank == 0 and s in (2, steps):
            h0 = np.zeros(n, np.float32)
            h1 = np.zeros(n, np.float32)
            for r in range(L):
                h0 += grad(r, s, n)
                h1 += grad(L + r, s, n)
            row["ref_ok"] = bool(np.array_equal(h0 + h1, tot))
            del h0, h1
        rows.append(row)
        del tot, v
    sock.close()
    q.put({"rank": local_rank, "rows": rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=int, choices=(0, 1), required=True)
    ap.add_argument("--local", type=int, required=True, help="ranks on this host")
    ap.add_argument("--peer", required=True, help="the other host's LAN address")
    ap.add_argument("--bind", default="", help="host 0: address to listen on")
    ap.add_argument("--port", type=int, default=29500)
    ap.add_argument("--token", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--n", type=int, default=N_RFD4)
    ap.add_argument("--shm", default="/dev/shm")
    ap.add_argument("--broken", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.shm) / f"r4t-scale-xh-{os.getpid()}"
    run.mkdir(parents=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    load0 = os.getloadavg()
    t0 = time.perf_counter()
    tok = a.token.encode()
    ps = [ctx.Process(target=rank_main, args=(a.host, r, a.local, a.n, a.steps, run, a.peer, a.bind,
                                              a.port, tok, a.broken, q)) for r in range(a.local)]
    try:
        for p in ps:
            p.start()
        res = sorted((q.get(timeout=3600) for _ in ps), key=lambda r: r["rank"])
        for p in ps:
            p.join()
    finally:
        for p in ps:
            if p.is_alive():
                p.kill()
        shutil.rmtree(run, ignore_errors=True)
    agree = all(len({r["rows"][i]["digest"] for r in res}) == 1 and
                all(r["rows"][i]["peer_agrees"] for r in res) for i in range(a.steps))
    ref_ok = all(row.get("ref_ok", True) for row in res[0]["rows"])

    def med(key):  # steps 4..: past fresh-slot faults, connect skew and the step-2 rebuild
        xs = [row[key] for r in res for row in r["rows"][3:]]
        return float(np.median(xs)) if xs else None
    walls = [max(r["rows"][i]["wall_s"] for r in res) for i in range(3, a.steps)]
    xb = res[0]["rows"][0]["xhost_bytes"]
    mx = med("xhost_s")
    out = {"host": socket.gethostname(), "role": a.host, "local": a.local, "world": 2 * a.local,
           "n": a.n, "steps": a.steps, "broken": a.broken, "ranks_agree_both_hosts": agree,
           "ref_sum_exact": ref_ok if a.host == 0 else None,
           "PASS": agree and ref_ok,
           "median_rs_s": med("rs_s"), "median_xhost_s": mx, "median_ag_s": med("ag_s"),
           "median_wall_s": med("wall_s"), "median_slowest_rank_wall_s": float(np.median(walls)) if walls else None,
           "xhost_bytes_per_rank_each_way": xb,
           "xhost_GBps_per_host_each_way": (xb * a.local / 1e9 / mx) if mx else None,
           "run_wall_s": time.perf_counter() - t0, "loadavg_before": load0, "per_rank": res}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_rank"}))
    sys.exit(0 if out["PASS"] != a.broken else 1)


if __name__ == "__main__":
    main()
