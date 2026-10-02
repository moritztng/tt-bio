"""Read the two-axis float64 VJP grade and say whether the gradient degrades at 800.

The question this answers is cause (c) in the state doc's VERDICT: 60 % of arm B's screen-gate
deficit is optimisation gaining less than arm A's, which is the shape a DEGRADED GRADIENT makes,
and nothing had graded this gradient above 288 tokens before 2026-10-02.

Read it against the control, never on its own. Every block prints the device VJP's rel L2 against
float64 AND `*_torch_bf16`, which is torch's own bf16 VJP against the same float64 reference. That
control is the whole instrument: bf16 representational error grows with n by itself, so a device
number that rises from 352 to 800 means nothing until you know whether the bf16 control rose with
it. What indicts us is the RATIO device/bf16 growing, not the device number growing.
"""
import json
import re
import sys
from pathlib import Path

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/home/ttuser/.coworker/wt/bcw-accept/perf/bcw_accept/out/grade_axes.log")


def blocks_by_axis(text):
    axis, out = None, {}
    for line in text.splitlines():
        m = re.match(r"===== n=(\d+) begin", line)
        if m:
            axis = int(m.group(1))
            out.setdefault(axis, [])
            continue
        if not line.startswith('{"block"'):
            continue
        try:
            out[axis].append(json.loads(line))
        except (json.JSONDecodeError, KeyError):
            pass
    return out


def main():
    if not LOG.exists():
        print(f"no log at {LOG}")
        return 1
    axes = blocks_by_axis(LOG.read_text(errors="replace"))
    rows = []
    for axis in sorted(axes):
        for b in axes[axis]:
            for field in ("dz", "dm"):
                dev = b.get(field, {}).get("rel_l2")
                ctl = b.get(f"{field}_torch_bf16", {}).get("rel_l2")
                if dev is None or ctl is None:
                    continue
                rows.append((axis, b.get("block"), field, dev, ctl, dev / ctl if ctl else float("nan")))
    if not rows:
        print("no graded blocks yet")
        return 1
    print(f"{'axis':>5} {'block':>6} {'t':>3} {'device rel L2':>14} {'torch bf16':>12} {'device/bf16':>12}")
    for axis, blk, field, dev, ctl, ratio in rows:
        print(f"{axis:>5} {blk:>6} {field:>3} {dev:>14.5f} {ctl:>12.5f} {ratio:>12.3f}")
    print()
    for axis in sorted(axes):
        sel = [r for r in rows if r[0] == axis]
        if not sel:
            continue
        md = sum(r[3] for r in sel) / len(sel)
        mc = sum(r[4] for r in sel) / len(sel)
        mr = sum(r[5] for r in sel) / len(sel)
        print(f"n={axis}: mean device {md:.5f}  mean bf16 control {mc:.5f}  mean ratio {mr:.3f}"
              f"  ({len(sel)} readings)")
    done = [a for a in sorted(axes) if axes[a]]
    if len(done) == 2:
        lo, hi = done
        rl = sum(r[5] for r in rows if r[0] == lo) / len([r for r in rows if r[0] == lo])
        rh = sum(r[5] for r in rows if r[0] == hi) / len([r for r in rows if r[0] == hi])
        print()
        print(f"device/bf16 ratio {rl:.3f} at n={lo} -> {rh:.3f} at n={hi}, change {rh - rl:+.3f}")
        print("A ratio that holds means the device VJP tracks torch's own bf16 at both axes, so the")
        print("gradient is NOT worse at the large axis and cause (c) is closed. A ratio that climbs")
        print("means the degradation is ours and is a defect, not a property of the model.")
    else:
        print(f"only n={done} graded so far; the comparison needs both axes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
