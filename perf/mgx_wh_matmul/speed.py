#!/usr/bin/env python3
"""Speed-bar verdict for the dest-carry guard at 1536 tokens, per model.

The guard's own 1536 cost is measured here as a ratio (fold_time.py, guard off then on, same
commit, chip and fixture, AICLK DURING). It is applied to mgx-speed's C11 1536 rung and that rung is
judged by scripts/speed_bar.py against C11's own 512-1024 fit (results/c11_verdicts.jsonl, copied
from wk/mgx-speed 32e1c189d). The fit rungs are left as C11 measured them: the guard slows them
too, which would steepen the fit and loosen the bar, so leaving them is the stricter reading.

Nesso-1 is not in C11; its 1536 rung is mgx-speed's C10 route (a): 130.6 s at ratio 1.150 against
an allowed 1.954 (fit on the clock-matched 896/928/960/992 rungs), so its ceiling is 1.954 / 1.150.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from speed_bar import judge  # noqa: E402

R = HERE / "results"
c11 = {d["model"]: d for d in map(json.loads, open(R / "c11_verdicts.jsonl"))}


def wall(model, g, s=0):
    p = R / f"time_{model}_g{g}_s{s}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text().splitlines()[-1])
    return r if r["rc"] == 0 else None


for model in ("boltz2", "openfold3", "protenix-v2", "opendde", "nesso1"):
    off, on = wall(model, 0), wall(model, 1)
    if not (off and on):
        print(f"{model}: not measured (off {bool(off)}, on {bool(on)})")
        continue
    ratio = on["wall_s"] / off["wall_s"]
    clk = {k: (r["aiclk"] or {}).get("0", {}).get("median") for k, r in (("off", off), ("on", on))}
    load = max(r["load"]["max"] / r["load"]["nproc"] for r in (off, on))
    head = (f"{model}: 3abq_1536 off {off['wall_s']} s, on {on['wall_s']} s, x{ratio:.3f} "
            f"(AICLK median {clk['off']}/{clk['on']}, load max {load:.2f}x, card {on['card']})")
    if model == "nesso1":
        print(head, f"-> C10 1536 130.6 s x{ratio:.3f} = {130.6 * ratio:.1f} s, ratio {1.150 * ratio:.3f} "
                    f"vs allowed 1.954: {'PASS' if 1.150 * ratio <= 1.954 else 'FAIL'}")
        continue
    d = c11[model]
    rt = {int(k): v for k, v in d["runtimes"].items()}
    t = rt[1536] * ratio
    v = judge({k: v for k, v in rt.items() if k <= 1024}, 1536, t, sigma=d["sigma"])
    base = d["rungs"]["1536"]
    print(head, f"-> C11 1536 {rt[1536]} s ({base['verdict']}) becomes {t:.1f} s: {v['verdict']} "
                f"(ratio {v['ratio']} vs allowed {v['allowed']}, ceiling {v['ceiling_s']} s, k_fit {v['k_fit']})")
