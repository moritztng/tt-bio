#!/usr/bin/env python3
"""List the module constants in tt_bio/tenstorrent.py whose comment names a part or an envelope.

The scan the ROOF audit is built from, kept as code so the table can be re-derived rather than
trusted. Six lines of lookback and the brief's own keyword set; it returns 48 at 3df8e8cd4.
"""
import argparse
import pathlib
import re

PAT = re.compile(r"^([A-Z_0-9]+) *=")
ENV = re.compile(r"wormhole|galaxy|8x9|72 core|p150|p300|clash|verified envelope|measured on",
                 re.I)


def scan(path: pathlib.Path, lookback: int = 6):
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        m = PAT.match(line)
        if m and ENV.search("\n".join(lines[max(0, i - lookback):i + 1])):
            yield i + 1, m.group(1), line.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="tt_bio/tenstorrent.py")
    ap.add_argument("--lookback", type=int, default=6)
    a = ap.parse_args()
    hits = list(scan(pathlib.Path(a.file), a.lookback))
    for ln, name, text in hits:
        print(f"{ln:6d}  {name:38s}  {text[:100]}")
    print(f"\n{len(hits)} constants")


if __name__ == "__main__":
    main()
