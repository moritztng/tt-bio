"""Say what the screen showed at every sample of a chaos run, from the saved shots.

    python3 demo/sc26/ops/screens.py ~/sc26-logs/soak-1005T2100

chaos.py calls a sample "moving" when it differs from the one before, so the sway poster that
shows for a second or two after a browser crash counts as moving. This labels each shot instead:
page (the live app), poster (sway's background still, the browser is not up), flat (one colour,
a blank frame), same (identical to the previous shot of a page: frozen), or what chaos.py saw
when there was no shot (display lost, compositor stuck, nothing). blank and same are failures.

Firefox's own error page ("Unable to connect") looks like a page to these pixel rules, so a shot
whose last watchdog page check (at most 15 s before it) left the browser on an error page is
labelled error, also a failure. The 10-06 soak's first report missed exactly that.
"""
import argparse
import bisect
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageOps, ImageStat

POSTER = Path(__file__).parent / "session" / "poster.png"


def label(path, poster, prev):
    img = Image.open(path).convert("RGB")
    if img.resize((img.width // 7, img.height // 7), Image.NEAREST).getcolors(3) is not None:
        return "flat", img
    if poster is not None and img.size == poster.size and \
            sum(ImageStat.Stat(ImageChops.difference(img, poster)).mean) / 3 < 8:
        return "poster", img
    if prev is not None and prev.size == img.size and ImageChops.difference(prev, img).getbbox() is None:
        return "same", img
    return "page", img


def error_checks(log):
    """Every watchdog page check (its tick time) and whether it left Firefox on an error page,
    in time order."""
    errs, ticks = [], []
    if not log.exists():
        return []
    for line in open(log, errors="replace"):
        if '"tick"' not in line and "error page" not in line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("ev") == "tick":
            ticks.append(r["t"])
        elif "error page" in r.get("err", ""):   # a navigation that landed on one; it stays up
            errs.append(r["t"])
    return [(t, any(t - 2 <= e <= t for e in errs)) for t in ticks]


def on_error(checks, t):
    """What the last page check at or before t found."""
    i = bisect.bisect_right(checks, (t, True)) - 1
    return i >= 0 and t - checks[i][0] < 15 and checks[i][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--watchdog", default="~/sc26-logs/watchdog.jsonl")
    a = ap.parse_args()
    run = Path(a.run).expanduser()
    checks = error_checks(Path(a.watchdog).expanduser())
    poster, counts, bad = None, {}, []
    for line in open(run / "events.jsonl"):
        e = json.loads(line)
        prev, row = None, []
        for s in e.get("screen", []):
            shot = run / "shots" / f"{e['i']:03d}-{e['event']}-{s['t']:03d}s.ppm"
            if not shot.exists():
                lab = s.get("display", "none")
            else:
                if poster is None:
                    size = Image.open(shot).size
                    poster = ImageOps.fit(Image.open(POSTER).convert("RGB"), size)
                lab, img = label(shot, poster, prev)
                prev = img if lab == "page" else None
                if lab == "page" and on_error(checks, e["t_wall"] + s["t"]):
                    lab = "error"
            counts[lab] = counts.get(lab, 0) + 1
            row.append(f"{s['t']}s:{lab}")
            if lab in ("flat", "same", "error"):
                bad.append(f"{e['i']} {e['event']} +{s['t']}s {lab}")
        live = next((r.split(":")[0] for r in row if r.endswith(":page")), "never")
        print(f"{e['i']:3d} {e['event']:15s} live page from {live:>5s}  {' '.join(row)}")
    print("samples:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print("blank, frozen or error:", "; ".join(bad) if bad else "none")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
