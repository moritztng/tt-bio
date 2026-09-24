"""Does an aligned (unpadded, so unmasked-at-the-caller) ESMC/SaProt forward still trace, and does
the replay return what the eager call returned?

    python3 perf/mgx_sdpa/trace_probe.py --model esmc-300m --lengths 126,1534 --out rows.jsonl

Each length is embedded four times in one process: eager (compile), capture, replay, replay.
The model captures a trace on the second sighting of a shape and falls back to eager for the rest
of the process if capture throws, so a mask allocated inside the capture could cost the trace
without failing anything. The row records whether the trace was taken and whether every call is
bit-equal to the first.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.mgx_sdpa.make_inputs import CDK2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="126,1534")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    torch.set_grad_enabled(False)
    if a.model.startswith("saprot"):
        from tt_bio import saprot as mod
        model = mod.load_saprot(a.model)
    else:
        from tt_bio import esmc as mod
        model = mod.load_esmc(a.model)
    for L in (int(x) for x in a.lengths.split(",")):
        seqs = {"s": (CDK2 * (L // len(CDK2) + 1))[:L]}
        vs = [torch.as_tensor(mod.embed_sequences(model, seqs, batch_size=1)[0].per_residue)
              for _ in range(4)]
        equal = [bool(torch.equal(vs[0], v)) for v in vs[1:]]
        row = {"model": a.model, "L": L, "root": str(ROOT),
               "rev": (ROOT / "REV").read_text().strip() if (ROOT / "REV").exists() else "",
               "traces": len(getattr(model, "_trace_cache", {}) or {}),
               "trace_broken": bool(getattr(model, "_trace_broken", False)),
               "equal_to_first": equal, "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES")}
        print(json.dumps(row), flush=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
