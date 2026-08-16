"""BoltzGen TT dispatch census: programs per diffusion step, K, per rung.

`tt_bio/boltzgen/model/modules/diffusion.py:369` emits one progress call per diffusion step, so
the interval between two consecutive emits is exactly one step. Patching that emitter lets graph
capture bracket a single warm step in a live design run -- no differential needed and no guessing
where a step starts.

Capture opens on step START_AT and closes on START_AT+1, well past kernel compile. Two
consecutive steps are captured and both counts recorded; they must agree, and a disagreement is
reported rather than averaged. Allocations and views are not programs and are excluded.

D = K * t_d / step_wall against the MEASURED t_d = 19.179 us and the MEASURED step wall from
perf/dsfix/results/bg_tt.jsonl.
"""
import json, os, pathlib, sys
from collections import Counter

sys.path.insert(0, os.getcwd())

import ttnn
import tt_bio.boltzgen.model.modules.diffusion as DIFF

T_D = 19.179e-6
START_AT = 12
OUT = pathlib.Path(os.environ.get("BGC_OUT", "perf/dsfix/results/bg_census.jsonl"))
OUT.parent.mkdir(parents=True, exist_ok=True)
RUNG = sys.argv[1]

SKIP = ("create_device_tensor", "Tensor::deallocate", "Tensor::reshape")


def count(captured):
    py, dev = Counter(), Counter()
    for n in captured:
        if n.get("node_type") != "function_start":
            continue
        nm = (n.get("params") or {}).get("name", "?")
        if any(s in nm for s in SKIP):
            continue
        if nm.startswith("ttnn.prim.") or "::prim::" in nm:
            dev[nm] += 1
        elif nm.startswith("ttnn."):
            py[nm] += 1
    return py, dev


class Done(Exception):
    """Abort the design run once the census is complete; the rest of the steps are not needed."""


state = {"open": False, "caps": [], "tops": []}
_orig = DIFF._emit_progress


def hooked(kind, i, total, *a, **kw):
    if kind == "diffusion":
        if state["open"]:
            py, dev = count(ttnn.graph.end_graph_capture())
            state["caps"].append((sum(py.values()), sum(dev.values())))
            state["tops"].append(py.most_common(10))
            state["open"] = False
            if len(state["caps"]) >= 2:
                raise Done()
        if i in (START_AT, START_AT + 1):
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            state["open"] = True
    return _orig(kind, i, total, *a, **kw)


DIFF._emit_progress = hooked

sys.argv = ["tt_bio", "design", "perf/dsfix/fixtures/bg_%s.yaml" % RUNG,
            "--model", "boltzgen", "--steps", "design", "--num_designs", "1",
            "--out_dir", "/tmp/bgc_%s" % RUNG,
            "--config", "design", "sampling_steps=40",
            "--debug", "--log"]

from tt_bio.main import cli as app

try:
    app(standalone_mode=False)
except Done:
    pass
except SystemExit:
    pass

if len(state["caps"]) < 2:
    print("[bgc] %s FAILED: only %d captures" % (RUNG, len(state["caps"])), flush=True)
    sys.exit(1)

k1, k2 = state["caps"][0][0], state["caps"][1][0]
K = max(k1, k2)
wall = None
p = pathlib.Path(os.environ.get("BGC_WALLS", "perf/dsfix/results/bg_tt.jsonl"))
if p.exists():
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r["rung"] == RUNG and r["batch"] == 1 and not r["trace"]:
            wall = r["step_ms_median"] / 1000.0
rec = {"rung": RUNG, "K_step1": k1, "K_step2": k2, "K": K, "agree": k1 == k2,
       "t_d_s": T_D, "step_wall_s": wall,
       "D": round(K * T_D / wall, 4) if wall else None,
       "us_per_program": round(wall / K * 1e6, 2) if wall and K else None,
       "top_ops": [{"op": o, "n": c} for o, c in state["tops"][0]],
       "grid": None, "host": os.environ.get("BGC_HOST", "qb1"),
       "card": os.environ.get("TT_VISIBLE_DEVICES", "0"),
       "ttnn": os.environ.get("BGC_TTNN", "0.67.4")}
try:
    import tt_bio.tenstorrent as _T
    rec["grid"] = list(_T.COMPUTE_GRID_MAIN)
except Exception:                                         # noqa: BLE001
    pass
with OUT.open("a") as fh:
    fh.write(json.dumps(rec) + "\n")
print("[bgc] %s K=%d (steps %d/%d agree=%s) D=%s us/prog=%s"
      % (RUNG, K, k1, k2, k1 == k2, rec["D"], rec["us_per_program"]), flush=True)
