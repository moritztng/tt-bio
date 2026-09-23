"""Fold one model at the given rungs with the size guard OFF, through the ladder's own fold.

The ladder records what a user can submit, so a model whose size_limits row caps it below a
rung records a refusal there and never touches the device. That is the right record and says
nothing about whether the cap is still true. This asks the second question with the same
fixture, fold config, census and clock sampling the arm uses (release_gate._run_census_fold),
so a probe cell and a ladder cell differ only in TT_BIO_SIZE_LIMIT.

    TT_VISIBLE_DEVICES=<c> ... python perf/sizegate/mgx/probe.py <model> <rung>[,<rung>...]
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["TT_BIO_SIZE_LIMIT"] = "0"
import release_gate as rg  # noqa: E402

model, rungs = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
out = ROOT / "perf" / "sizegate" / "mgx" / "probe"
out.mkdir(parents=True, exist_ok=True)
work = Path(os.environ.get("RELEASE_GATE_SIZE_WORKDIR", ROOT / "perf" / "sizegate" / f"probe-{model}"))
for rung in rungs:
    t0 = time.time()
    r = rg._run_census_fold(model, rung, work, "probe")
    cell = {"model": model, "rung": rung, "size_limit": "off", "card": os.environ.get("TT_VISIBLE_DEVICES"),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "wall_s": round(time.time() - t0, 1)}
    for k in ("error", "refused", "runtime_s", "aiclk", "grid", "structure"):
        if r.get(k) is not None:
            cell[k] = r[k]
    (out / f"{model}_{rung}.json").write_text(json.dumps(cell, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in cell.items() if k != "structure"}, default=str), flush=True)
