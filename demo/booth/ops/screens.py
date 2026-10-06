"""Say what the screen showed at every sample of a chaos run, from the saved shots.

    python3 demo/booth/ops/screens.py ~/booth-logs/soak-1005T2100

chaos.py calls a sample "moving" when it differs from the one before, so the sway poster that
shows for a second or two after a browser crash counts as moving. This labels each shot instead:
page (the live app), poster (sway's background still, the browser is not up), flat (one colour,
a blank frame), same (identical to the previous shot of a page: frozen), or what chaos.py saw
when there was no shot (display lost, compositor stuck, nothing). blank and same are failures.
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args()
    run = Path(a.run).expanduser()
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
            counts[lab] = counts.get(lab, 0) + 1
            row.append(f"{s['t']}s:{lab}")
            if lab in ("flat", "same"):
                bad.append(f"{e['i']} {e['event']} +{s['t']}s {lab}")
        live = next((r.split(":")[0] for r in row if r.endswith(":page")), "never")
        print(f"{e['i']:3d} {e['event']:15s} live page from {live:>5s}  {' '.join(row)}")
    print("samples:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print("blank or frozen:", "; ".join(bad) if bad else "none")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
