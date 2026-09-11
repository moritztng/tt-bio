#!/usr/bin/env python3
"""Turn host_phase_512_*.json into the tables this task owes: per-phase host CPU against wall,
the injected-host-CPU slope per phase, and the pairformer block's device floor."""
from __future__ import annotations
import json
import sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
W = []


def p(s=""):
    W.append(s)
    print(s)


p("## env")
p("`" + json.dumps(d["env"]) + "`")
p()
p("## plain folds (no instrument, benchlock)")
p("| arm | wall s | main-thread CPU s | CPU % | loadavg |")
p("|---|---|---|---|---|")
for k, v in d.get("plain", {}).items():
    p(f"| {k} | {v['wall_s']:.3f} | {v['cpu_s']:.3f} | {v['cpu_pct']:.1f} | {' '.join(v['loadavg'])} |")

t = d["table"]
cen = d.get("census", {})
byp = cen.get("by_phase", {})
p()
p(f"## per-phase wall and main-thread CPU (bracketed fold, wall {t['fold_wall_s']:.3f} s, "
  f"CPU {t['fold_cpu_s']:.3f} s = {t['fold_cpu_pct']:.1f} %)")
p("| phase | calls | wall s | host CPU s | CPU % of wall | blocked s | ttnn calls | us/call CPU |")
p("|---|---|---|---|---|---|---|---|")
top = {k: v for k, v in t["tree"].items() if "/" not in k}
for k, v in sorted(top.items(), key=lambda kv: -kv[1]["incl_wall_s"]):
    n = byp.get(k, {}).get("calls", 0)
    cpu_ttnn = byp.get(k, {}).get("cpu_s", 0.0)
    p(f"| {k} | {v['calls']} | {v['incl_wall_s']:.3f} | {v['incl_cpu_s']:.3f} | "
      f"{100*v['incl_cpu_s']/v['incl_wall_s']:.1f} | {v['incl_wall_s']-v['incl_cpu_s']:.3f} | "
      f"{n} | {1e6*cpu_ttnn/max(n,1):.1f} |")
p(f"| **residual** | | {t['residual_wall_s']:.3f} | {t['residual_cpu_s']:.3f} | "
  f"{100*t['residual_cpu_s']/t['residual_wall_s']:.1f} | "
  f"{t['residual_wall_s']-t['residual_cpu_s']:.3f} | "
  f"{byp.get('(glue)',{}).get('calls',0)} | |")

p()
p("## where the trunk's host CPU is, by sub-unit (census fold)")
p("| path | ttnn calls | CPU inside ttnn s | us/call | bracket incl CPU s | bracket incl wall s |")
p("|---|---|---|---|---|---|")
ctree = cen.get("tree", {})
rows = sorted(byp.items(), key=lambda kv: -kv[1]["cpu_s"])[:28]
for k, v in rows:
    bt = ctree.get(k, {})
    p(f"| {k} | {v['calls']} | {v['cpu_s']:.3f} | {v['us_per_call_cpu']:.1f} | "
      f"{bt.get('incl_cpu_s','')} | {bt.get('incl_wall_s','')} |")

p()
p("## the ops that carry the host CPU")
p("| op | calls | CPU s | us/call |")
p("|---|---|---|---|")
for r in cen.get("by_op", [])[:20]:
    p(f"| {r['op']} | {r['n']} | {r['cpu_s']:.3f} | {r['us_per_call']:.2f} |")

p()
p("## slope: inject measured host CPU per ttnn call, read d(wall)/d(CPU)")
sl = d.get("slope", [])
base = sl[0] if sl else None
p("| spin us/call | fold wall s | fold CPU s | added CPU s | added wall s | slope |")
p("|---|---|---|---|---|---|")
for r in sl:
    dc = r["fold_cpu_s"] - base["fold_cpu_s"]
    dw = r["fold_wall_s"] - base["fold_wall_s"]
    p(f"| {r['spin_us']:.1f} | {r['fold_wall_s']:.3f} | {r['fold_cpu_s']:.3f} | {dc:+.3f} | "
      f"{dw:+.3f} | {dw/dc if abs(dc) > 0.05 else float('nan'):.3f} |")
p()
p("| phase | spin | wall s | CPU s | added CPU s | added wall s | slope |")
p("|---|---|---|---|---|---|---|")
for ph in sorted(base["phases"]):
    b = base["phases"][ph]
    for r in sl[1:]:
        v = r["phases"].get(ph)
        if not v:
            continue
        dc = v["cpu_s"] - b["cpu_s"]
        dw = v["wall_s"] - b["wall_s"]
        p(f"| {ph} | {r['spin_us']:.0f} | {v['wall_s']:.3f} | {v['cpu_s']:.3f} | {dc:+.3f} | "
          f"{dw:+.3f} | {dw/dc if abs(dc) > 0.05 else float('nan'):.3f} |")

p()
p("## controls: is main-thread CPU a host/device discriminator on this stack?")
p("| loop | reps | host us/call | device us/call | CPU/wall | drain s |")
p("|---|---|---|---|---|---|")
for k, v in d.get("controls", {}).items():
    if not isinstance(v, dict) or "reps" not in v:
        continue
    p(f"| {k} | {v['reps']} | {v['us_per_call_cpu']:.2f} | {v['us_per_call_device']:.2f} | "
      f"{v['cpu_over_total_wall']:.3f} | {v['drain_s']:.3f} |")

p()
p("## pairformer block device floor (ttnn trace capture of shipped PairformerLayer.__call__)")
p("`" + json.dumps(d.get("block", {})) + "`")

if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text("\n".join(W) + "\n")
