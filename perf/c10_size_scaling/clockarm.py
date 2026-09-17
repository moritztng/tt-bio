"""Per-arm clock telemetry and the inverse-clock fit.

`control.py` is reused byte-identical from c10-bare-baseline, so its 1350-only `coverage` cannot
score an 800 MHz arm. Everything clock-arm specific lives here instead of being patched into the
pinned helper. The sampler is a strict superset of the baseline's: same 1 kHz aiclk read with
monotonic brackets and the same 10 Hz all-chip holder census, plus board power and temperature,
which are what prove the arm physically happened rather than only the aiclk register moving.
"""
from __future__ import annotations
import json, math, select, sys, threading, time
from pathlib import Path

from control import holders, own_nodes

NODE = 0
SYS = Path(f"/sys/class/tenstorrent/tenstorrent!{NODE}")
MAX_GAP_NS = 10_000_000
MIN_SAMPLES = 3


def hwmon():
    return next(SYS.glob("device/hwmon/hwmon*"))


def coverage(samples, interval, target):
    """Clock coverage of one fold interval, scored against that fold's OWN requested clock."""
    start, end = interval["start_monotonic_ns"], interval["end_monotonic_ns"]
    during = [s for s in samples
              if s.get("read_start_ns", -1) >= start and s.get("read_end_ns", end + 1) <= end]
    valid = [s for s in during if "MHz" in s]
    centers = [(s["read_start_ns"] + s["read_end_ns"]) // 2 for s in valid]
    points = [start] + centers + [end]
    watts = [s["W"] for s in valid if s.get("W") is not None]
    result = {
        "target_MHz": target,
        "samples": len(valid),
        "min_MHz": min((s["MHz"] for s in valid), default=None),
        "max_MHz": max((s["MHz"] for s in valid), default=None),
        "max_gap_ns": max(b - a for a, b in zip(points, points[1:])),
        "first_offset_ns": centers[0] - start if centers else None,
        "last_offset_ns": end - centers[-1] if centers else None,
        "sample_span_fraction": (centers[-1] - centers[0]) / (end - start) if centers else 0,
        "min_W": min(watts, default=None), "max_W": max(watts, default=None),
        "mean_W": sum(watts) / len(watts) if watts else None,
        "max_C": max((s["C"] for s in valid if s.get("C") is not None), default=None),
        "errors": [s for s in during if "error" in s],
    }
    result["pass"] = (len(valid) >= MIN_SAMPLES
                      and result["min_MHz"] == result["max_MHz"] == target
                      and result["max_gap_ns"] <= MAX_GAP_NS and not result["errors"])
    return result


def sampler_main(path, owner):
    """1 kHz aiclk/power/temp with monotonic read brackets, plus a 10 Hz all-chip holder census."""
    stop = threading.Event()
    hw = hwmon()

    def census():
        with Path(path).with_name("holders.jsonl").open("w") as out:
            while not stop.is_set():
                row = {"monotonic_ns": time.monotonic_ns(), "utc_ns": time.time_ns()}
                try:
                    row.update(holders=holders(), owner_nodes=own_nodes(owner))
                except BaseException as e:
                    row["error"] = repr(e)
                out.write(json.dumps(row) + "\n"); out.flush()
                stop.wait(0.1)

    monitor = threading.Thread(target=census)
    monitor.start()
    try:
        with Path(path).open("w") as out:
            while not select.select([sys.stdin], [], [], 0.001)[0]:
                row = {"read_start_ns": time.monotonic_ns(), "utc_ns": time.time_ns(),
                       "node": str(SYS / "tt_aiclk")}
                try:
                    row["MHz"] = int((SYS / "tt_aiclk").read_text())
                    row["W"] = int((hw / "power1_input").read_text()) / 1e6
                    row["C"] = int((hw / "temp1_input").read_text()) / 1e3
                except (OSError, ValueError) as e:
                    row["error"] = repr(e)
                row["read_end_ns"] = time.monotonic_ns()
                out.write(json.dumps(row) + "\n"); out.flush()
    finally:
        stop.set(); monitor.join(timeout=10)


def settle(target, timeout_s=5.0, poll_s=0.02):
    """Wait for the requested clock to appear in telemetry. Returns the settle record."""
    t0 = time.monotonic_ns()
    reads = []
    while (time.monotonic_ns() - t0) / 1e9 < timeout_s:
        mhz = int((SYS / "tt_aiclk").read_text())
        reads.append(mhz)
        if mhz == target:
            ok_since = time.monotonic_ns()
            # require it to stay there for 200 ms before the fold starts
            while (time.monotonic_ns() - ok_since) / 1e9 < 0.2:
                mhz = int((SYS / "tt_aiclk").read_text())
                reads.append(mhz)
                if mhz != target:
                    break
                time.sleep(poll_s)
            else:
                return {"target_MHz": target, "settled": True, "reads": len(reads),
                        "settle_ns": time.monotonic_ns() - t0, "distinct": sorted(set(reads))}
        time.sleep(poll_s)
    return {"target_MHz": target, "settled": False, "reads": len(reads),
            "settle_ns": time.monotonic_ns() - t0, "distinct": sorted(set(reads))}


def fit_inverse_clock(points):
    """Least squares T = F + C/f. `points` are (f_MHz, T_s).

    Returns F in seconds, C in Mcycles (MHz*s), their standard errors from the residual variance,
    and the residuals. With two distinct clocks this is the exact closed form and the residual is
    zero by construction, so the standard errors come out None -- say so rather than printing a
    zero uncertainty.
    """
    n = len(points)
    xs = [1.0 / f for f, _ in points]
    ys = [t for _, t in points]
    clocks = sorted({f for f, _ in points})
    if len(clocks) < 2:
        raise ValueError("a fixed term needs at least two distinct clocks")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    C = sxy / sxx
    F = my - C * mx
    res = [y - (F + C * x) for x, y in zip(xs, ys)]
    out = {"n": n, "clocks_MHz": clocks, "F_s": F, "C_Mcycles": C,
           "residual_s": res, "max_abs_residual_s": max(abs(r) for r in res),
           "rms_residual_s": math.sqrt(sum(r * r for r in res) / n)}
    dof = n - 2
    if dof > 0 and len(clocks) > 2:
        s2 = sum(r * r for r in res) / dof
        out["se_C_Mcycles"] = math.sqrt(s2 / sxx)
        out["se_F_s"] = math.sqrt(s2 * (1.0 / n + mx * mx / sxx))
    else:
        out["se_C_Mcycles"] = out["se_F_s"] = None
        out["se_note"] = ("two distinct clocks only: the fit is the exact closed form, residual is "
                          "zero by construction and carries no information about uncertainty")
    return out


def propagate_two_clock(f1, t1, s1, f2, t2, s2):
    """Closed-form F from two clock arms, with F's uncertainty propagated from the arm errors.

    F = (t1/f2 - t2/f1) / (1/f2 - 1/f1); dF/dt1 = (1/f2)/(1/f2-1/f1), dF/dt2 = -(1/f1)/(...).
    """
    u, v = 1.0 / f1, 1.0 / f2
    den = v - u
    F = (t1 * v - t2 * u) / den
    C = (t1 - t2) / (u - v)
    g1, g2 = v / den, -u / den
    return {"clocks_MHz": [f1, f2], "F_s": F, "C_Mcycles": C,
            "se_F_s": math.sqrt((g1 * s1) ** 2 + (g2 * s2) ** 2),
            "dF_dt_gain": [g1, g2],
            "se_C_Mcycles": math.sqrt((s1 / (u - v)) ** 2 + (s2 / (u - v)) ** 2)}


if __name__ == "__main__":
    sampler_main(sys.argv[1], int(sys.argv[2]))
