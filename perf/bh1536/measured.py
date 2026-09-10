#!/usr/bin/env python3
"""Exit 0 if (model, size) already has a MEASURED rung in results.jsonl, 1 otherwise.

`chain.sh` calls this to decide whether to walk a rung again. Measured means the run reached
the card and produced an outcome that says something about capacity: PASS, OOM, STALLED,
TIMEOUT or FAIL -- a stall is a result, the model did not complete at that size on this
silicon, which is exactly what the ladder asks. CONTENDED and WEDGED do not count -- a co-tenant
held the card, or the card would not come up at all, so nothing ran; treating either as a result
retires the rung on the one class of outcome that carries no information about capacity.

Only untagged rows (`tag: ""`) count, so a deliberate `--tag debug` re-run never retires the
rung it was investigating.
"""
import json
import sys
from pathlib import Path

MEASURED = {"PASS", "OOM", "STALLED", "TIMEOUT", "FAIL"}


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {Path(sys.argv[0]).name} <model> <size>", file=sys.stderr)
        return 2
    model, size = sys.argv[1], int(sys.argv[2])
    jl = Path(__file__).resolve().parent / "results.jsonl"
    if not jl.is_file():
        return 1
    for line in jl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if (row.get("model") == model and row.get("size") == size
                and not row.get("tag") and row.get("verdict") in MEASURED):
            return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
