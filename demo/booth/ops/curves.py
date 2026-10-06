"""Draw the watchdog's resource samples over a run, and say for each whether it is flat.

    python3 demo/booth/ops/curves.py ~/booth-logs/watchdog.jsonl --since 2026-10-06T00:00 --out curves.png

A leak is a slope, so each series gets one: the least-squares trend over the run's second half
(after warm-up), per hour and per booth day of 10 h, next to the series' median. Needs matplotlib
(tt-bio's environment has it); prints the table without it.
"""
import argparse
import json
import statistics
from datetime import datetime, timezone

SERIES = [  # (key, label, unit); a list value is summed (the chip workers)
    ("engine_rss_mb", "engine RSS", "MB"), ("worker_rss_mb", "chip workers RSS, sum", "MB"),
    ("rss_browser_mb", "browser RSS", "MB"), ("engine_mb", "engine unit memory, with page cache", "MB"),
    ("engine_anon_mb", "engine unit memory, anon", "MB"), ("engine_file_mb", "engine unit page cache", "MB"),
    ("vram_mb", "GPU memory", "MB"),
    ("engine_fds", "engine fds", ""), ("worker_fds", "chip workers fds, sum", ""),
    ("browser_fds", "browser fds", ""), ("mem_avail_gb", "host memory free", "GB"),
    ("disk_free_gb", "disk free", "GB"), ("logs_mb", "demo logs", "MB"), ("fps", "page frame rate", "fps"),
]


def load(path, since, until):
    """The watchdog's checks, and everything else it logged (restarts, failures), in the window."""
    rows, other = [], []
    with open(path, "rb") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if since <= r.get("t", 0) <= until:
                (rows if r.get("ev") == "tick" else other).append(r)
    return rows, other


def screen(rows, other):
    """What the screen did on every check: how often each state, the frame rate's spread, every
    stretch it was not moving, and every recovery the watchdog made."""
    states = {}
    for r in rows:
        states[r.get("screen")] = states.get(r.get("screen"), 0) + 1
    fps = sorted(r["fps"] for r in rows if r.get("fps") is not None)
    pct = lambda q: fps[min(len(fps) - 1, int(q * len(fps)))] if fps else None
    stretches, cur = [], None
    for r in rows:
        if r.get("screen") not in ("moving", None):
            cur = cur or [r["t"], r["t"], r.get("screen")]
            cur[1] = r["t"]
        elif cur:
            stretches.append(cur)
            cur = None
    if cur:
        stretches.append(cur)
    acts = {}
    for r in other:
        if r.get("ev") == "restart":
            k = f"{r.get('unit')}: {r.get('why', '').split(' for ')[0]}"
            acts[k] = acts.get(k, 0) + 1
    return {"checks": len(rows), "states": states,
            "fps_p1_p5_p50": [pct(0.01), pct(0.05), pct(0.5)], "fps_checks": len(fps),
            "not_moving": [{"from": datetime.fromtimestamp(a, timezone.utc).strftime("%m-%d %H:%M:%SZ"),
                            "s": round(b - a + 10), "as": w} for a, b, w in stretches],
            "restarts": acts}


def slope(ts, vs):
    """Least-squares slope per hour."""
    n = len(ts)
    if n < 3:
        return None
    mt, mv = sum(ts) / n, sum(vs) / n
    den = sum((t - mt) ** 2 for t in ts)
    return sum((t - mt) * (v - mv) for t, v in zip(ts, vs)) / den * 3600 if den else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--since", default="", help="ISO time, UTC")
    ap.add_argument("--until", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="", help="write the table here too")
    a = ap.parse_args()
    iso = lambda s, d: datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() if s else d
    rows, other = load(a.log, iso(a.since, 0), iso(a.until, 1e12))
    if not rows:
        raise SystemExit("no samples in that window")
    t0 = rows[0]["t"]
    table, series = [], {}
    for key, label, unit in SERIES:
        pts = [((r["t"] - t0) / 3600, sum(r[key]) if isinstance(r[key], list) else r[key])
               for r in rows if r.get(key) is not None]
        if not pts:
            continue
        series[key] = pts
        half = [p for p in pts if p[0] >= pts[-1][0] / 2]
        s = slope([p[0] * 3600 for p in half], [p[1] for p in half])
        med = statistics.median(p[1] for p in pts)
        table.append({"series": label, "unit": unit, "n": len(pts), "first": pts[0][1], "median": round(med, 1),
                      "last": pts[-1][1], "min": min(p[1] for p in pts), "max": max(p[1] for p in pts),
                      "per_h_2nd_half": None if s is None else round(s, 3),
                      "per_10h_pct_of_median": None if s is None or not med else round(1000 * s / med, 2)})
    hours = (rows[-1]["t"] - t0) / 3600
    print(f"{len(rows)} samples over {hours:.1f} h from {datetime.fromtimestamp(t0, timezone.utc):%Y-%m-%d %H:%MZ}")
    print(f"{'series':26} {'first':>9} {'median':>9} {'last':>9} {'max':>9} {'/h (2nd half)':>14} {'%/10h':>7}")
    for r in table:
        print(f"{r['series']:26} {r['first']:>9} {r['median']:>9} {r['last']:>9} {r['max']:>9} "
              f"{r['per_h_2nd_half'] if r['per_h_2nd_half'] is not None else '-':>14} "
              f"{r['per_10h_pct_of_median'] if r['per_10h_pct_of_median'] is not None else '-':>7}")
    scr = screen(rows, other)
    print(f"screen on {scr['checks']} checks: {scr['states']}; page fps p1/p5/p50 {scr['fps_p1_p5_p50']}")
    for x in scr["not_moving"]:
        print(f"  not moving from {x['from']} for ~{x['s']} s ({x['as']})")
    for k, n in sorted(scr["restarts"].items()):
        print(f"  restart {k}: {n}x")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"hours": round(hours, 2), "t0": t0, "table": table, "screen": scr}, f, indent=1)
    if not a.out:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found: no plot")
        return
    keys = [k for k, _, _ in SERIES if k in series]
    fig, axes = plt.subplots((len(keys) + 1) // 2, 2, figsize=(13, 2.1 * ((len(keys) + 1) // 2)), sharex=True)
    for ax, key in zip(axes.flat, keys):
        label, unit = next((l, u) for k, l, u in SERIES if k == key)
        xs, ys = zip(*series[key])
        ax.plot(xs, ys, lw=0.8, color="#2b6cb0")
        ax.set_title(f"{label}" + (f" ({unit})" if unit else ""), fontsize=9, loc="left")
        ax.tick_params(labelsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("hours", fontsize=8)
    fig.suptitle(f"TT-Bio booth demo, {hours:.1f} h from {datetime.fromtimestamp(t0, timezone.utc):%Y-%m-%d %H:%MZ}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
