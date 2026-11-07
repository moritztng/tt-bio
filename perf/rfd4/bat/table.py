"""The climb table from climb.sh pass legs: traced trunk call (3 passes, 2 recycles) per design per pass, achieved
TFLOP/s on ARCH.md section 2 FLOP model, eager/traced (host dispatch share), op count, AICLK during.
    python perf/rfd4/bat/table.py <run dir>"""
import json
import sys
from pathlib import Path


def tflop_pass(n):
    # state/rfd4/ARCH.md section 2 census, d 768, 14 blocks, 7 pair updates, c_z 384, trimul c 256: 8.09 at 512 (ARCH 8.15)
    return (302e6 * n + 26.6e6 * n * n + 7168 * n ** 3) / 1e12


rows = []
for f in sorted(Path(sys.argv[1]).glob("pass-*.json")):
    r = json.loads(f.read_text())
    b, c = r["batch"], r["call_traced_ms"]["median"]
    per = c / 3 / b
    rows.append((r["n"], r["real"], b, round(c, 1), round(per, 2), round(tflop_pass(r["n"]) / per * 1e3, 1),
                 round(r["call_eager_ms"]["median"] / c, 3), r["ops_per_call_depth3"],
                 r["aiclk"]["0"]["median"], r["aiclk"]["0"]["min"]))
rows.sort()
base = {}
print("| n (real) | b | traced call ms | ms/design/pass | gain vs b1 | TFLOP/s | eager/traced | ops/call | AICLK med (min) |")
print("|---|--:|--:|--:|--:|--:|--:|--:|---|")
for n, real, b, c, per, tf, et, ops, ck, ckmin in rows:
    base.setdefault((n, real), per)
    print(f"| {n} ({real}) | {b} | {c} | {per} | {base[(n, real)] / per:.2f}x | {tf} | {et} | {ops} | {ck} ({ckmin}) |")
