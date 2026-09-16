#!/usr/bin/env python3
"""Does holding the Blackhole ARC clock at burst cost reliability on qb2?

`b2z2-aiclk-burst-pin` measured the speed (1.34x through the shipped knob at 512 aa) and left the
reliability question open on purpose: qb2 resets on its own from the open PCIe/NoC fault closed as
hardware-only by `qbroot-verdict-pcie-link-failure`, at an inherited 23.93 resets/day, which swamps
anything a ten-minute run could see. Two things have changed since, both recorded in
`state/qb2-vendor-handoff.md`: the four chips now run firmware 19.15.0.0 instead of 19.11.0.0, and
the host boots with failed-endpoint containment that survives an endpoint loss without a reset.
So the ambient rate that any burst-clock rate has to be compared against is no longer the rate that
was measured, and the comparison has to be made inside this host state rather than against a
remembered number.

That is what this soak does. One card is held at burst continuously, idle included, which is
strictly harsher than default-on would ever be (default-on holds the clock only while a run has the
device open). The other card on the same board runs the identical fold cadence on the governor. Both
cards see the same work; one differs in clock. The reliability readout is not this process's own
logging: `~/qbcard/cardtel.tsv` and `~/qbcard-bisect/stalls.log` have been sampling all four cards
at 2 Hz across every boot since 2026-09-15, which is the instrument the qbroot campaign was closed
on, and it costs no extra ARC traffic. This process only has to put the clock where it says it is
and say when each window started and ended.

Roles:
  hold  -- forces AICLK on one node and keeps it there, sampling every card's clock/temp/power
  fold  -- one device open, one model load, then a 512 aa fold every `--interval` seconds

Both roles run to an absolute `--until` epoch, so the wrapper can restart them after a host reset
and the campaign still ends when it was supposed to.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, shutil, socket, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "_cell_now", REPO / "perf" / "b2z2_cell_recheck" / "cell_now.py")
CN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CN)
LEV, FIX = CN.LEV, CN.FIX

CARDS = (0, 1, 2, 3)


def sysfs(node: int, attr: str):
    try:
        return int(Path(f"/sys/class/tenstorrent/tenstorrent!{node}/{attr}").read_text())
    except (OSError, ValueError):
        return None


def hwmon(node: int, attr: str):
    try:
        p = next(Path(f"/sys/class/tenstorrent/tenstorrent!{node}/device/hwmon").glob(f"hwmon*/{attr}"))
        return int(p.read_text())
    except (OSError, ValueError, StopIteration):
        return None


def boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()[:8]
    except OSError:
        return "?"


def emit(log: Path, rec: dict) -> None:
    rec["t"] = round(time.time(), 3)
    rec["boot"] = boot_id()
    with log.open("a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def own_node() -> int:
    """Which /dev/tenstorrent/N ttnn opened. TT_VISIBLE_DEVICES is a UMD logical id, not the node."""
    nodes = set()
    for fd in Path("/proc/self/fd").iterdir():
        try:
            t = os.readlink(fd)
        except OSError:
            continue
        if t.startswith("/dev/tenstorrent/"):
            nodes.add(int(t.rsplit("/", 1)[1]))
    assert len(nodes) == 1, f"expected exactly one open chip, got {sorted(nodes)}"
    return nodes.pop()


def role_hold(args) -> int:
    from tt_bio import aiclk
    log = args.log
    emit(log, {"ev": "hold_start", "node": args.node, "mhz": args.mhz,
               "until": args.until, "host": socket.gethostname(),
               "fw": Path(f"/sys/class/tenstorrent/tenstorrent!{args.node}/tt_fw_bundle_ver").read_text().strip(),
               "serial": Path(f"/sys/class/tenstorrent/tenstorrent!{args.node}/tt_serial").read_text().strip(),
               "uptime_s": round(float(Path("/proc/uptime").read_text().split()[0]), 1)})
    hold = aiclk._Hold([args.node], args.mhz)
    below = 0
    samples = 0
    try:
        while time.time() < args.until:
            time.sleep(1.0)
            samples += 1
            clk = sysfs(args.node, "tt_aiclk")
            if clk is not None and clk < args.mhz - 50:
                below += 1
            if samples % args.sample_every == 0:
                emit(log, {"ev": "hold_sample", "reasserts": hold.reasserts,
                           "below_target_s": below, "samples": samples,
                           "aiclk": {c: sysfs(c, "tt_aiclk") for c in CARDS},
                           "heartbeat": {c: sysfs(c, "tt_heartbeat") for c in CARDS},
                           "temp_c": {c: (hwmon(c, "temp1_input") or 0) / 1e3 for c in CARDS},
                           "power_w": {c: (hwmon(c, "power1_input") or 0) / 1e6 for c in CARDS},
                           "loadavg1": round(os.getloadavg()[0], 2)})
    finally:
        emit(log, {"ev": "hold_end", "reasserts": hold.reasserts, "below_target_s": below,
                   "samples": samples})
        hold.release()
    return 0


def role_fold(args) -> int:
    log = args.log
    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this tree")
    _E.set_progress(lambda *a, **k: None)
    LEV.SAMPLING_STEPS, LEV.RECYCLING_STEPS = args.steps, args.recycles

    dev = get_device()
    node = own_node()
    emit(log, {"ev": "fold_start", "arm": args.arm, "node": node,
               "visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
               "aiclk_knob": os.environ.get("TT_BIO_AICLK"),
               "serial": Path(f"/sys/class/tenstorrent/tenstorrent!{node}/tt_serial").read_text().strip(),
               "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
               "until": args.until, "interval": args.interval})

    work = Path(tempfile.mkdtemp(prefix=f"b2z2-soak-{args.arm}-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    LEV._seed_msa(FIX / f"{args.fixture}.yaml", (FIX / f"{args.fixture}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    LEV._ensure_local_artifacts = _ensure_local_artifacts
    _ensure_local_artifacts(cfg)
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-aiclk-default-decision", cfg)

    i = 0
    while time.time() < args.until:
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        clk0 = sysfs(node, "tt_aiclk")
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(FIX / f"{args.fixture}.yaml", cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t0
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        emit(log, {"ev": "fold", "arm": args.arm, "i": i, "node": node,
                   "fold_s": round(wall, 3),
                   "aiclk_at_start": clk0, "aiclk_after": sysfs(node, "tt_aiclk"),
                   "temp_c": (hwmon(node, "temp1_input") or 0) / 1e3,
                   "power_w": (hwmon(node, "power1_input") or 0) / 1e6,
                   "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
                   "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest()[:16],
                   "loadavg1": round(os.getloadavg()[0], 2)})
        i += 1
        # Idle between folds on purpose: the gap is where a default-on lever would hand the clock
        # back and where this soak deliberately does not.
        nap = args.interval - wall
        while nap > 0 and time.time() < args.until:
            time.sleep(min(5.0, nap))
            nap -= 5.0
    emit(log, {"ev": "fold_stop", "arm": args.arm, "folds": i})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=("hold", "fold"), required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--until", type=float, required=True, help="absolute epoch to stop at")
    ap.add_argument("--node", type=int, default=1, help="hold role: device node to force")
    ap.add_argument("--mhz", type=int, default=1350)
    ap.add_argument("--sample-every", type=int, default=30, help="hold role: seconds per log line")
    ap.add_argument("--arm", default="held", help="fold role: label for this card's arm")
    ap.add_argument("--interval", type=float, default=240.0, help="fold role: seconds per fold slot")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--fixture", default="cdk2x2_512")
    args = ap.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    if time.time() >= args.until:
        print("past --until, nothing to do")
        return 0
    return role_hold(args) if args.role == "hold" else role_fold(args)


if __name__ == "__main__":
    raise SystemExit(main())
