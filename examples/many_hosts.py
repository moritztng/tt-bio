#!/usr/bin/env python3
"""Fold a directory of inputs across several machines, with no platform.

Each machine runs its own `tt-bio controller` on loopback with a worker per chip
(docs/multi-host.md). Everything that crosses machines is here: ask each host how
many chips it can compute on, give it a share of the inputs in proportion, fold the
share through that host's controller over ssh, copy the results back. A host that
fails hands its share to the hosts that did not.

    python examples/many_hosts.py ./proteins ./out --model esmfold2 \\
        --host ubuntu@galaxy1:8765 --host ubuntu@galaxy2:8765 ...
"""
import argparse, json, shlex, subprocess, threading, uuid
from pathlib import Path


def ssh(host, cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], text=True, capture_output=True)


def usable_chips(host, port):
    r = ssh(host, f"curl -sf http://127.0.0.1:{port}/cluster")
    return json.loads(r.stdout)["online_workers"] if r.returncode == 0 else 0


def fold(host, port, files, out, a):
    """Fold ``files`` on ``host``; False if the host itself failed, not a target."""
    tmp = f"/tmp/many-hosts-{uuid.uuid4().hex[:8]}"
    if ssh(host, f"mkdir -p {tmp}/in").returncode or subprocess.run(
            ["scp", "-q", *map(str, files), f"{host}:{tmp}/in/"]).returncode:
        return False
    r = ssh(host, f"{a.tt_bio} predict {tmp}/in --out_dir {tmp}/out --model {a.model} "
                  f"--controller http://127.0.0.1:{port} --owner {shlex.quote(a.owner)} {a.args}")
    (out / host).mkdir(parents=True, exist_ok=True)
    ok = r.returncode in (0, 1, 2) and not subprocess.run(   # 1, 2: some targets failed
        ["scp", "-qr", f"{host}:{tmp}/out/.", str(out / host)]).returncode
    ssh(host, f"rm -rf {tmp}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--host", action="append", required=True, help="user@machine:port, repeatable")
    p.add_argument("--model", default="esmfold2")
    p.add_argument("--owner", default="me", help="fairness key the controllers share chips by")
    p.add_argument("--tt-bio", dest="tt_bio", default="tt-bio", help="how to run tt-bio on the hosts")
    p.add_argument("--args", default="", help="extra predict flags")
    a = p.parse_args()
    hosts = {h: int(port) for h, _, port in (x.rpartition(":") for x in a.host)}
    todo = sorted(f for f in a.inputs.iterdir() if f.is_file())
    while todo:
        chips = {h: n for h in hosts if (n := usable_chips(h, hosts[h]))}
        if not chips:
            raise SystemExit(f"no host can compute; {len(todo)} inputs left")
        share = {h: [] for h in chips}              # largest first, to the least loaded
        for f in sorted(todo, key=lambda f: -f.stat().st_size):
            share[min(chips, key=lambda h: sum(g.stat().st_size for g in share[h]) / chips[h])].append(f)
        done = {}
        runs = [threading.Thread(target=lambda h=h: done.update({h: fold(h, hosts[h], share[h], a.out, a)}))
                for h in chips if share[h]]
        for t in runs:
            t.start()
        for t in runs:
            t.join()
        todo = [f for h, ok in done.items() if not ok for f in share[h]]
        hosts = {h: hosts[h] for h in hosts if done.get(h, True)}
        print({h: f"{len(share[h])} inputs on {chips[h]} chips, {'ok' if ok else 'FAILED'}" for h, ok in done.items()})


if __name__ == "__main__":
    main()
