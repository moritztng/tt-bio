"""Compose the session the video plays: four chip lanes of real folds recorded on this box, as one
loop of exactly --period seconds.

    python3 demo/booth/video/compose.py plan --tap ~/booth-video/tap --period 130 --out ~/booth-video/s1
    python3 demo/booth/video/compose.py build --out ~/booth-video/s1 --period 129.85
    python3 demo/booth/video/compose.py seam --out ~/booth-video/s1

Every fold a lane shows is one the booth engine ran on a chip of this QuietBox and capture.py heard:
its messages as the engine sent them, at the times they arrived, and its coordinates from GET /fold.
Lanes 1, 2 and 4 are chips 0, 1 and 3 and show folds those chips ran. Chip 2 is out of service, so
lane 3 shows folds chips 0, 1 and 3 ran at other times; only the lane they appear in is staged.

`plan` picks each lane's folds: one engine rule (at most three chips on a long fold), no protein on
two lanes at once, every protein once a loop, each recorded fold used once. `build` writes the
session for a period, spreading the slack over the short "Ready" gaps between folds, so a lane's
folds repeat exactly every period. `seam` reads a timing run's slot log (render.py) and names the
two slot changes, one period apart, that show the same fold: the loop's cut.
"""
import argparse
import base64
import gzip
import json
import random
import struct
import sys
from collections import defaultdict
from pathlib import Path

LONG = 400          # engine --long-res: a longer protein is a long fold
GAP = (0.05, 0.9)   # seconds a lane says "Ready" between two folds
EPOCH = 1791244800.0  # the session's wall clock at its start (only differences are ever shown)


def folds_from_tap(tap):
    """Every complete live attract fold in the tap: its messages from the chip taking it to the chip
    being ready again, timed from the moment it was taken, and its coordinates."""
    lines = [json.loads(x) for x in open(tap / "tap.jsonl")]
    bodies = {}
    for f in (tap / "folds").glob("*.bin"):
        at, fid = f.stem.split("-", 1)
        bodies[(fid, at)] = f
    conn, open_, out = 0, {}, []
    bad = set()
    for x in lines:
        at, m = x["at"], x["m"]
        t = m["type"]
        if t == "_connected":
            conn += 1
            open_ = {}
            continue
        if t == "chip":
            c = m["chip"]
            if m["state"] == "busy" and m.get("doing"):
                open_[c] = {"conn": conn, "id": m["doing"]["id"], "chip": c, "t0": at, "msgs": [(0.0, m)], "done": None}
            elif m["state"] == "ready" and c in open_ and open_[c]["done"]:
                r = open_.pop(c)
                r["msgs"].append((at - r["t0"], m))
                r["dur"] = at - r["t0"]
                out.append(r)
            elif m["state"] != "busy":
                open_.pop(c, None)   # warming, resetting, recovering...: not a clean fold
            continue
        c = m.get("chip")
        r = open_.get(c)
        if r is None or m.get("id") != r["id"]:
            continue
        if t == "fold_error":
            open_.pop(c)
            continue
        r["msgs"].append((at - r["t0"], m))
        if t == "fold_done":
            r["done"] = m
            r["body"] = bodies.get((m["id"], f"{at:.3f}"))
    good = []
    for r in out:
        d = r["done"]
        start = next((m for _, m in r["msgs"] if m["type"] == "fold_start"), None)
        if (start is None or d.get("kind") != "attract" or d.get("source") != "live" or not r.get("body")
                or (d.get("aiclk_mhz") or {}).get("median", 0) < 1300):
            continue
        r.update(name=d["name"], n_res=d["n_res"], seconds=d["seconds"], aiclk=d["aiclk_mhz"],
                 key=f"{r['conn']}:{r['id']}", body=str(r["body"]))
        good.append(r)
    return good


def overlaps(a0, a1, b0, b1, period):
    """Do [a0,a1) and [b0,b1) meet on a circle of length period?"""
    for k in (-1, 0, 1):
        if a0 < b1 + k * period and b0 + k * period < a1:
            return True
    return False


def plan(args):
    folds = folds_from_tap(Path(args.tap).expanduser())
    names = sorted({f["name"] for f in folds})
    P = args.period
    print(f"{len(folds)} folds, {len(names)} proteins, by chip:",
          {c: sum(1 for f in folds if f["chip"] == c) for c in (0, 1, 3)})
    rng = random.Random(args.seed)
    by_chip = defaultdict(list)
    for f in folds:
        by_chip[f["chip"]].append(f)

    def lane_cycle(pool, used):
        """A natural run of folds from the pool, as the engine ran them, summing to just under P."""
        pool = sorted((f for f in pool if f["key"] not in used), key=lambda f: f["t0"])
        for _ in range(200):
            i = rng.randrange(len(pool))
            seq, tot = [], 0.0
            for f in pool[i:]:
                if f["key"] in used or any(g["name"] == f["name"] and g["n_res"] == f["n_res"] for g in seq[-1:]):
                    continue
                seq.append(f)
                tot += f["dur"]
                n = len(seq)
                if P - n * GAP[1] <= tot <= P - n * GAP[0]:
                    return seq
                if tot > P:
                    break
        return None

    best = None
    for trial in range(args.trials):
        used, lanes = set(), {}
        for lane in (0, 1, 3, 2):
            pool = by_chip[lane] if lane != 2 else folds
            seq = lane_cycle(pool, used)
            if seq is None:
                break
            used |= {f["key"] for f in seq}
            lanes[lane] = seq
        if len(lanes) < 4:
            continue
        offs = {l: rng.uniform(0, P) for l in lanes}
        iv = []   # (lane, name, long, start, end) on the circle, gaps spread evenly
        for l, seq in lanes.items():
            gap = (P - sum(f["dur"] for f in seq)) / len(seq)
            t = offs[l]
            for f in seq:
                iv.append((l, f["name"], f["n_res"] > LONG, t % P, t % P + f["dur"]))
                t += f["dur"] + gap
        clash = sum(1 for i, a in enumerate(iv) for b in iv[i + 1:]
                    if a[0] != b[0] and a[1] == b[1] and overlaps(a[3], a[4], b[3], b[4], P))
        # the engine never puts all four chips on long folds
        longs = [a for a in iv if a[2]]
        worst = max(sum(1 for b in longs if overlaps(b[3], b[4], t, t + 1e-3, P)) for t in
                    [a[3] + 0.01 for a in iv] + [k * P / 400 for k in range(400)])
        shown = {a[1] for a in iv}
        # as the engine runs them: three chips on long folds whenever the rotation allows
        score = clash * 100 + max(0, worst - 3) * 100 + (len(names) - len(shown)) * 10 + (3 - min(worst, 3)) * 5
        if best is None or score < best[0]:
            best = (score, lanes, offs, clash, worst, len(shown))
            print(f"trial {trial}: score {score} clashes {clash} max long {worst} proteins {len(shown)}/{len(names)}")
        if score == 0:
            break
    score, lanes, offs, clash, worst, shown = best
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    keep = {"lanes": {l: [f["key"] for f in seq] for l, seq in lanes.items()}, "offsets": offs,
            "tap": str(Path(args.tap).expanduser()), "score": score}
    (out / "plan.json").write_text(json.dumps(keep, indent=1))
    for l in sorted(lanes):
        print(f"lane {l + 1} (chip {l}):", ", ".join(f"{f['name'][:18]}/{f['chip']} {f['dur']:.1f}s" for f in lanes[l]),
              f"sum {sum(f['dur'] for f in lanes[l]):.2f}")


def build(args):
    out = Path(args.out).expanduser()
    pl = json.loads((out / "plan.json").read_text())
    folds = {f["key"]: f for f in folds_from_tap(Path(pl["tap"]))}
    P = args.period
    lanes = {int(l): [folds[k] for k in ks] for l, ks in pl["lanes"].items()}
    end = args.periods * P
    msgs, fold_files, rewrite, jobs = [], {}, {}, []   # jobs: (lane, start, end, fold, id)
    for l, seq in lanes.items():
        gap = (P - sum(f["dur"] for f in seq)) / len(seq)
        assert GAP[0] <= gap <= GAP[1] + 0.2, f"lane {l}: gap {gap:.3f} s for period {P}"
        cum = 0.0
        for j, f in enumerate(seq):
            for r in range(-1, args.periods + 1):
                s = pl["offsets"][str(l)] - P + r * P + cum
                if s + f["dur"] < 0 or s > end:
                    continue
                fid = f"L{l}F{j}P{r + 1}"
                jobs.append((l, s, s + f["dur"], f, fid))
                fold_files[fid] = f["body"]
                rewrite[fid] = {"id": fid, "chip": l, "t_wall": EPOCH + s + f["dur"]}
                for dt, m in f["msgs"]:
                    t = s + dt
                    m = json.loads(json.dumps(m))
                    if "id" in m:
                        m["id"] = fid
                    if "chip" in m:
                        m["chip"] = l
                    if m.get("job"):
                        m["job"] = fid
                    if m.get("doing"):
                        m["doing"].update(id=fid, t_wall=EPOCH + s)
                    if "t_wall" in m:
                        m["t_wall"] = EPOCH + t
                    msgs.append((t, m))
            cum += f["dur"] + gap
    jobs.sort(key=lambda j: j[1])

    def lane_state(l, t):
        """The chip's status entry at time t: what it is on, and its last finished fold."""
        cur = next((j for j in jobs if j[0] == l and j[1] <= t < j[2]), None)
        prev = max((j for j in jobs if j[0] == l and j[2] <= t), key=lambda j: j[2], default=None)
        folds_done = sum(1 for j in jobs if j[0] == l and j[2] <= t)
        last = prev and {"name": prev[3]["name"], "n_res": prev[3]["n_res"], "seconds": prev[3]["seconds"],
                         "aiclk_mhz": prev[3]["aiclk"], "t_wall": EPOCH + prev[2]}
        aiclk = (cur or prev or jobs[0])[3]["aiclk"]["median"]
        if cur:
            busy = next(m for _, m in cur[3]["msgs"] if m["type"] == "chip")
            doing = dict(busy["doing"], id=cur[4], t_wall=EPOCH + cur[1])
            return {"chip": l, "state": "busy", "job": cur[4], "doing": doing, "aiclk_mhz": aiclk,
                    "folds": 40 + folds_done, "restarts": 0, "last_fold": last}
        return {"chip": l, "state": "ready", "job": None, "doing": None, "aiclk_mhz": aiclk,
                "folds": 40 + folds_done, "restarts": 0, "last_fold": last}

    n = round(P / 2)
    for k in range(int(end / (P / n)) + 1):
        t = k * P / n + 0.5 * P / n
        msgs.append((t, {"type": "status", "models": ["openfold3"], "chips": [lane_state(l, t) for l in range(4)],
                         "queue": 0, "replays": 11, "t_wall": EPOCH + t}))
    hello = {"type": "hello", "protocol": 2, "models": ["openfold3"], "chips": [lane_state(l, 0.025) for l in range(4)],
             "queue": 0, "replays": 11, "t_wall": EPOCH + 0.025, "folds": []}
    msgs = [(0.025, hello)] + sorted((x for x in msgs if 0.03 <= x[0] <= end), key=lambda x: x[0])
    session = {"t0": EPOCH * 1000, "period": P,
               "messages": [[round(1000 * t, 3), json.dumps(m, separators=(",", ":"))] for t, m in msgs],
               "folds": fold_files, "rewrite": rewrite}
    (out / "session.json").write_text(json.dumps(session, separators=(",", ":")))
    print(f"period {P} s, {len(msgs)} messages over {end:.0f} s, {len(fold_files)} fold instances")


def load_fold(path, rewrite):
    """A pulled fold (GET /fold/<id>?every=1) as server.Folds holds one, relabelled for its lane."""
    buf = Path(path).read_bytes()
    ml = struct.unpack("<I", buf[:4])[0]
    meta = json.loads(gzip.decompress(buf[4:4 + ml]))
    at = 4 + ml + (-(4 + ml) % 4)
    n = meta["n_atoms"]
    xyz, q16 = buf[at:at + 12 * n], buf[at + 12 * n:]
    frames = {"step": meta["step"], "t": meta["t"], "origin": meta["origin"], "scale": meta["scale"]}
    meta = {k: v for k, v in meta.items() if k not in ("every", "step", "t", "origin", "scale")} | rewrite
    assert len(q16) == 6 * n * len(frames["scale"]) and len(frames["step"]) == len(frames["scale"]) + 1
    return {"meta": meta, "frames": frames, "q16": q16, "xyz": xyz, "n_atoms": n}


def seam(args):
    """Slot changes one period apart that put the same recorded fold on the stage."""
    out = Path(args.out).expanduser()
    P = json.loads((out / "session.json").read_text())["period"]
    N = round(P * args.fps)
    slots = [json.loads(x) for x in open(out / "slots.jsonl") if '"id"' in x]
    split = lambda i: (i.rsplit("P", 1)[0], int(i.rsplit("P", 1)[1])) if i and i.startswith("L") else (i, None)
    best = []
    for a in slots:
        if a["k"] < args.warm * N:
            continue
        fa, ra = split(a["id"])
        for b in slots:
            fb, rb = split(b["id"])
            if b["k"] > a["k"] and fb == fa and ra is not None and rb == ra + 1:
                best.append((abs(b["k"] - a["k"] - N), a["k"], b["k"], a["name"]))
    best.sort()
    print(f"period {P} s = {N} frames")
    for d, ka, kb, name in best[:8]:
        print(f"  cut at frame {ka} -> {kb}: {kb - ka} frames ({(kb - ka) / args.fps:.4f} s), off by {d}, {name}")
    if best:
        print("BEST", *best[0][:3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "build", "seam"])
    ap.add_argument("--tap", default="~/booth-video/tap")
    ap.add_argument("--out", required=True)
    ap.add_argument("--period", type=float, default=130.0)
    ap.add_argument("--periods", type=int, default=5, help="how many loops the session runs")
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--warm", type=float, default=2.5, help="loops before the cut may start")
    args = ap.parse_args()
    {"plan": plan, "build": build, "seam": seam}[args.cmd](args)


if __name__ == "__main__":
    main()
