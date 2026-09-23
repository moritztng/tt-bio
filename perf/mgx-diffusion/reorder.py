"""Reorder a plan so the points that state the law and the ceiling run first.

    python perf/mgx-diffusion/reorder.py <plan> ...

The box is shared by a dozen rows and a lane gets a chip for a few folds at a time, so order is
what decides which cells exist when a pass ends. Both ends of the sample axis (memory flat past W is
read off S=1 against the largest S), then the time law's first two points, then the rest. Stable
within a rank, comments stay at the top, and ladder.py skips what is already recorded.
"""
import sys
from pathlib import Path


def parse(line):
    # ladder.py's parse without its engine imports, which need the tt_bio env
    model, tokens, samples, *kv = line.split("#")[0].split()
    return {"model": model, "tokens": int(tokens), "samples": int(samples),
            **{k: int(v) for k, v in (x.split("=") for x in kv)}}


def rank(p, top):
    s, probe = p["samples"], p.get("probe")
    if probe and s in (1, top[(p["model"], p["tokens"])]):
        return 0
    if not probe and "steps" not in p and s in (1, 5):
        return 1
    if probe and s != 10:
        return 2
    if s != 10:
        return 3
    return 4


def main():
    for path in map(Path, sys.argv[1:]):
        lines = path.read_text().splitlines()
        head = [l for l in lines if l.startswith("#") or not l.strip()]
        body = [(l, parse(l)) for l in lines if l not in head]
        top = {}
        for _, p in body:
            k = (p["model"], p["tokens"])
            top[k] = max(top.get(k, 0), p["samples"])
        body.sort(key=lambda lp: rank(lp[1], top))
        path.write_text("\n".join(head + [l for l, _ in body]) + "\n")


if __name__ == "__main__":
    main()
