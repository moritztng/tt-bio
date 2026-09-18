"""Node-parameterised evidence helpers: holder census, during-fold clock sampling, coverage.

Generalised from `perf/c10_bare_baseline/control.py` (commit bc66f7d6d), which hardcodes node 0
and a 1350 MHz target in `coverage`. This row needs a node argument (qb2's live card moved after
the 2026-09-17 reboot) and a per-fold clock target (two arms, 1350 and 800 MHz), so the two
functions that carried those constants take them as parameters here. Everything else is the same
code path; `imports.json` records the parent digest so the drift is visible.
"""
from __future__ import annotations
import hashlib, json, os, select, subprocess, sys, threading, time
from pathlib import Path

# The tt-kmd srcversions this box is known to boot. `candidate` is the qualified module
# `qb2-endpoint-containment.service` inserts with `qb_endpoint_quarantine=1 dma_address_bits=0
# power_policy=0 fw_log_level=3`; `stock` is the DKMS 2.11.0 build with default parameters.
# c12-host-decomp's whole table was taken on `candidate`. On the 2026-09-18 boot the unit is
# DISABLED and never started, so the box runs `stock`, i.e. no endpoint quarantine and the
# stock power policy rather than 0.
KNOWN_SRCVERSIONS = {"A10759A24565BC5BBE903C5": "candidate (qualified, power_policy=0)",
                     "28CFF5A6678E4F2D87F6383": "stock DKMS 2.11.0, default parameters"}
# Latched from the first snapshot of a capture. The invariant a timed capture needs is that the
# driver does not change UNDER it, which is what this enforces; pinning one accepted value
# instead would refuse every capture on whichever module the box happens to have booted, which
# is how this check spent a pass refusing a healthy chip on 2026-09-18.
SESSION_SRCVERSION: list = []
GAP_LIMIT_NS = 10_000_000

HOLDERS = """
import json, os
from pathlib import Path
rows=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    nodes=set()
    try:
        for fd in (p/'fd').iterdir():
            try: target=os.readlink(fd)
            except OSError: continue
            if target.startswith('/dev/tenstorrent/'): nodes.add(target)
        if not nodes: continue
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        env=dict(e.split('=',1) for e in (p/'environ').read_text().split(chr(0)) if '=' in e)
        rows.append({'pid':int(p.name),'nodes':sorted(nodes),
                     'state':stat[0], 'utime_ticks':int(stat[11]), 'stime_ticks':int(stat[12]),
                     'starttime_ticks':int(stat[19]),
                     'argv':(p/'cmdline').read_text().replace(chr(0),' '),
                     'env':{k:env.get(k) for k in ('TT_VISIBLE_DEVICES','TT_BIO_LEASE_CARDS','TT_BIO_LEASE_HOLDER')}})
    except (OSError,PermissionError): continue
print(json.dumps(rows))
"""


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size}


def holders():
    return json.loads(subprocess.check_output(["sudo", "-n", "python3", "-c", HOLDERS], text=True))


def own_nodes(pid=None):
    nodes = set()
    for fd in Path(f"/proc/{pid or os.getpid()}/fd").iterdir():
        try:
            target = os.readlink(fd)
            if target.startswith("/dev/tenstorrent/"):
                nodes.add(target)
        except OSError:
            pass
    return sorted(nodes)


def load_accounting(node, dt=2.0):
    """Split the 1-minute loadavg into load this row can account for and load it cannot.

    An absolute loadavg bar cannot tell a co-tenant that would corrupt a host-time measurement
    from a wedged chip holder spinning on another chip whose pid, owner and node are already
    known. Both read as load, and on this box two known orphans pin loadavg at 2.00 for as long
    as they live, so an absolute bar of 2.0 can never pass and stops being a statement about
    contention. Price every foreign device holder off its own /proc ticks and gate on the
    REMAINDER, recording the accounted part beside it so a reader sees what was on the box.
    """
    ticks = os.sysconf("SC_CLK_TCK")
    before = {h["pid"]: h for h in holders()}
    t0 = time.monotonic()
    time.sleep(dt)
    after = {h["pid"]: h for h in holders()}
    span = time.monotonic() - t0
    me = os.getpid()
    rows = []
    for pid, h in after.items():
        if pid == me or pid not in before:
            continue
        used = ((h["utime_ticks"] + h["stime_ticks"])
                - (before[pid]["utime_ticks"] + before[pid]["stime_ticks"])) / ticks
        rows.append({"pid": pid, "nodes": h["nodes"], "state": h["state"],
                     "cpu_cores": round(used / span, 3),
                     "holder": (h.get("env") or {}).get("TT_BIO_LEASE_HOLDER"),
                     "argv": h["argv"][:160]})
    accounted = sum(max(0.0, r["cpu_cores"]) for r in rows)
    load1 = os.getloadavg()[0]
    return {"loadavg": os.getloadavg(), "dt_s": round(span, 3), "node": node,
            "foreign_device_holders": sorted(rows, key=lambda r: -r["cpu_cores"]),
            "accounted_cores": round(accounted, 3),
            "unaccounted_load": round(load1 - accounted, 3)}


def containment_unit():
    """`systemctl is-active qb2-endpoint-containment.service`, RECORDED, never gated on.

    Two independent reasons this string cannot be a precondition, both of them properties of the
    unit rather than of the hardware:

    1. `is-active` exits NONZERO for `inactive`, which is an answer and not a failure.
       `check_output` raised `CalledProcessError` on it and killed the capture before its first
       fold on 2026-09-18, when the unit was `disabled` and had not started on that boot.
    2. The unit is `Type=oneshot RemainAfterExit=yes` with NO `ExecStop`, so a stop transitions
       systemd's own bookkeeping to inactive while undoing nothing in config space. The journal
       shows exactly that four times over three boots. So `active` does not prove containment is
       applied and `inactive` does not prove it is not.

    The hardware fact the capture actually needs is the assigned node's upstream port keeping
    Memory Space Enable, and that is read per node in `validate_snapshot` below.
    """
    r = subprocess.run(["systemctl", "is-active", "qb2-endpoint-containment.service"],
                       text=True, capture_output=True)
    return {"is_active": (r.stdout or "").strip() or f"rc={r.returncode}", "rc": r.returncode}


def port_command(node):
    """The assigned node's upstream-port COMMAND register, read through `dispatch_probe`'s own
    reader rather than a second copy of it, so the guard and the probe cannot drift apart.

    Bit 1, not bit 2: QB quarantine goes 0x0407 -> 0x0405, which drops Memory Space Enable and
    keeps Bus Master Enable, so a bus-master test passes a quarantined port.
    """
    try:
        from dispatch_probe import upstream_port_state
        return upstream_port_state(node)
    except Exception as e:
        return {"error": repr(e), "memory_space": None}


def snapshot(node=None):
    return {"monotonic_ns": time.monotonic_ns(), "utc_ns": time.time_ns(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "module_srcversion": Path("/sys/module/tenstorrent/srcversion").read_text().strip(),
            "containment_unit": containment_unit(),
            "port": port_command(node) if node is not None else None,
            "loadavg": os.getloadavg(), "holders": holders(), "own_nodes": own_nodes()}


def validate_snapshot(s, node, opened=False):
    sv = s["module_srcversion"]
    if not SESSION_SRCVERSION:
        SESSION_SRCVERSION.append(sv)
    if sv != SESSION_SRCVERSION[0]:
        raise RuntimeError(f"tt-kmd changed mid-capture: {SESSION_SRCVERSION[0]} -> {sv}")
    if sv not in KNOWN_SRCVERSIONS:
        raise RuntimeError(f"unknown tt-kmd srcversion {sv}: qualify it before timing on it")
    port = s.get("port") or port_command(node)
    if not port.get("memory_space"):
        raise RuntimeError(f"assigned node {node} upstream port quarantined or unreadable: {port}"
                           " -- that state needs a REBOOT, not a reset")
    dev = f"/dev/tenstorrent/{node}"
    for h in s["holders"]:
        if dev in h["nodes"] and h["pid"] != os.getpid():
            raise RuntimeError(f"assigned node busy: {h}")
    if opened and s["own_nodes"] != [dev]:
        raise RuntimeError(f"unexpected device opens: {s['own_nodes']}")


def coverage(samples, interval, target_MHz):
    """During-fold clock coverage scored against THIS fold's own requested target.

    The c10 parent hardcoded 1350. An 800 MHz arm must pass at 800 and fail at 1350, and a
    1350 arm the reverse, or the two-clock control cannot be trusted.
    """
    start, end = interval["start_monotonic_ns"], interval["end_monotonic_ns"]
    during = [s for s in samples
              if s.get("read_start_ns", -1) >= start and s.get("read_end_ns", end + 1) <= end]
    valid = [s for s in during if "MHz" in s]
    centers = [(s["read_start_ns"] + s["read_end_ns"]) // 2 for s in valid]
    points = [start] + centers + [end]
    r = {"target_MHz": target_MHz, "samples": len(valid),
         "min_MHz": min((s["MHz"] for s in valid), default=None),
         "max_MHz": max((s["MHz"] for s in valid), default=None),
         "min_W": min((s["W"] for s in valid if "W" in s), default=None),
         "max_W": max((s["W"] for s in valid if "W" in s), default=None),
         "max_gap_ns": max(b - a for a, b in zip(points, points[1:])),
         "sample_span_fraction": (centers[-1] - centers[0]) / (end - start) if centers else 0,
         "errors": [s for s in during if "error" in s]}
    r["pass"] = (len(valid) >= 3 and r["min_MHz"] == r["max_MHz"] == target_MHz
                 and r["max_gap_ns"] <= GAP_LIMIT_NS and not r["errors"])
    return r


def clock_worker(path, owner, node):
    """~1 kHz aiclk + board power on `node`, plus a 10 Hz holder census in a second thread."""
    root = Path(f"/sys/class/tenstorrent/tenstorrent!{node}")
    clk = root / "tt_aiclk"
    try:
        hw = next(root.glob("device/hwmon/hwmon*"))
        pwr = hw / "power1_input"
    except StopIteration:
        pwr = None
    stop = threading.Event()

    def observe():
        with Path(path).with_name("holders.jsonl").open("w") as out:
            while not stop.is_set():
                row = {"monotonic_ns": time.monotonic_ns(), "utc_ns": time.time_ns()}
                try:
                    row.update(holders=holders(), owner_nodes=own_nodes(owner))
                except BaseException as e:
                    row["error"] = repr(e)
                out.write(json.dumps(row) + "\n")
                out.flush()
                stop.wait(0.1)

    monitor = threading.Thread(target=observe)
    monitor.start()
    try:
        with Path(path).open("w") as out:
            while not select.select([sys.stdin], [], [], 0.001)[0]:
                row = {"read_start_ns": time.monotonic_ns(), "utc_ns": time.time_ns(),
                       "node": str(clk)}
                try:
                    row["MHz"] = int(clk.read_text())
                    if pwr is not None:
                        row["W"] = int(pwr.read_text()) / 1e6
                except (OSError, ValueError) as e:
                    row["error"] = repr(e)
                row["read_end_ns"] = time.monotonic_ns()
                out.write(json.dumps(row) + "\n")
                out.flush()
    finally:
        stop.set()
        monitor.join(timeout=10)


if __name__ == "__main__":
    clock_worker(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
