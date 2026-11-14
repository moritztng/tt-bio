#!/usr/bin/env python3
"""Two gap_tracy.py censuses of the same tree side by side (Wormhole against p150a), every op on its roofline.

    gap_tracy_cmp.py WH.json BH.json [--top=40] [--md=OUT.md]

Per region and per op signature: device ms/step on each chip, the WH/BH ratio, and the WH roof for that op,
max(bytes / DRAM bandwidth, matmul flops / math peak). An op whose WH/BH ratio is above 2x is slower on WH than the
hardware ratio explains (WH-only slow); one near ~1.8x is just the bandwidth ratio. Host idle per step is the plain
wall minus device kernel time. Peaks are measured, not datasheet: WH .107 roof_wh.json (8x9 grid, 1000 MHz) add/clone
248 GB/s, bf16 HiFi4 matmul 56 TF; p150a 448 GB/s, 119 TF HiFi4. The roof is an under-estimate of an op's floor where
it re-reads an operand from DRAM (the signature counts each tensor once), so a ratio near 1 is not proof of roof.
"""
import json
import sys

PEAK = {"wh": (248e9, 56e12), "bh": (448e9, 119e12)}
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
wh, bh = (json.load(open(p)) for p in args[:2])
TOP = int(opts.get("top", 40))


def roof(arch, b, f):
    bw, pk = PEAK[arch]
    return 1e3 * max(b / bw, f / pk)


def rows(c):
    return {(s["region"], s["sig"]): s for s in c["sigs"]}


W, B = rows(wh), rows(bh)
out = []
p = out.append
for name, c in (("WH", wh), ("p150a", bh)):
    p("%s: plain %.2f s/design (%.1f ms/step), device %.1f ms/step, host idle %.1f ms/step (%.0f %% busy), %d ops, "
      "%d unread" % (name, c["plain_s"], c["plain_wall_ms_per_step"], c["device_ms_per_step"],
                     c["plain_wall_ms_per_step"] - c["device_ms_per_step"],
                     100 * c["device_ms_per_step"] / c["plain_wall_ms_per_step"], c["ops"], c["ops_missing"]))
p("")
p("| region | WH ms/step | p150a ms/step | WH/BH | WH roof | WH %roof |")
p("|---|---|---|---|---|---|")
reg_roof = {}
for (r, g), s in W.items():
    reg_roof[r] = reg_roof.get(r, 0) + s["calls_per_step"] * roof("wh", s["bytes_per_call"], s["flops_per_call"])
for r, ms in sorted(wh["regions_ms_per_step"].items(), key=lambda kv: -kv[1]):
    b = bh["regions_ms_per_step"].get(r, 0)
    rr = reg_roof.get(r, 0)
    p("| %s | %.2f | %.2f | %s | %.2f | %.0f %% |" % (r, ms, b, "%.2f" % (ms / b) if b else "-", rr,
                                                  100 * rr / ms if ms else 0))
p("")
p("| region | op signature | calls | WH ms | BH ms | WH/BH | WH roof ms | WH %roof |")
p("|---|---|---|---|---|---|---|---|")
for (r, g), s in sorted(W.items(), key=lambda kv: -kv[1]["ms_per_step"])[:TOP]:
    b = B.get((r, g))
    rf = s["calls_per_step"] * roof("wh", s["bytes_per_call"], s["flops_per_call"])
    p("| %s | `%s` | %.0f | %.2f | %s | %s | %.2f | %.0f %% |" % (
        r, g[:110], s["calls_per_step"], s["ms_per_step"], "%.2f" % b["ms_per_step"] if b else "-",
        "%.2f" % (s["ms_per_step"] / b["ms_per_step"]) if b and b["ms_per_step"] else "-", rf,
        100 * rf / s["ms_per_step"] if s["ms_per_step"] else 0))
# by op kind: where WH loses more than the hardware ratio
kind = {}
for key, s in W.items():
    k = (key[0], key[1].split(" |")[0])
    e = kind.setdefault(k, [0, 0, 0])
    e[0] += s["ms_per_step"]
    e[2] += s["calls_per_step"] * roof("wh", s["bytes_per_call"], s["flops_per_call"])
for key, s in B.items():     # BH totals by kind on their own, so an arch-specific shape still lands in its kind
    kind.setdefault((key[0], key[1].split(" |")[0]), [0, 0, 0])[1] += s["ms_per_step"]
p("")
p("| region | op | WH ms | BH ms | WH/BH | WH roof | WH-only slow |")
p("|---|---|---|---|---|---|---|")
for (r, o), (w, b, rf) in sorted(kind.items(), key=lambda kv: -kv[1][0])[:TOP]:
    p("| %s | %s | %.2f | %.2f | %s | %.2f | %s |" % (r, o, w, b, "%.2f" % (w / b) if b else "-", rf,
                                                   "YES" if b and w / b > 2 else ""))
text = "\n".join(out)
print(text)
if "md" in opts:
    open(opts["md"], "w").write(text + "\n")
