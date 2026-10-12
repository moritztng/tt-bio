#!/usr/bin/env python3
"""gap_pws_probe: what pair_swiglu_ws is handed inside a real fold, and what each call costs there.

Wraps `Transition._weight_stationary` (synchronize on both sides) and records, per distinct input signature, x's
shape / padded shape / dtype / buffer type, the weights' shapes and dtypes, and the synced ms of every call. The
op bench (gap_pws_rms.py) reads 3.95 ms for hidden 256 at 685 tokens on WH; gap_itemise read 13.3 ms a call in
--fast. This says which input differs.

    gap_pws_probe.py ARM OUT.json [STEPS=6] [--fast=1] [--fixture=perf/dsfix/fixtures/rfd3_R4.json]
"""
import collections, json, os, pathlib, shutil, statistics, sys, time

ENV_KNOBS = {"pzc": "RFD3_PROCESS_Z_COLLAPSE", "asdpa": "RFD3_ATOM_SDPA", "tb8": "RFD3_TRANSITION_B8",
             "dsdpa": "RFD3_DIT_SDPA", "sf": "RFD3_SIGMOID_FOLD", "rs": "RFD3_RESIDENT_STEP", "pws": "RFD3_PAIR_WS",
             "pwr": "RFD3_PAIR_WS_RMS", "wf": "RFD3_DIT_MASK_WFOLD", "kd": "RFD3_ATTN_KDTREE"}
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
ARM, OUT = args[0], pathlib.Path(args[1])
STEPS = int(args[2]) if len(args) > 2 else 6
for _k, _env in ENV_KNOBS.items():
    os.environ[_env] = "1" if _k in set(ARM.split("+")) else "0"   # read at import

sys.path.insert(0, os.getcwd())
from tt_bio.main import ensure_p300_mesh_descriptor                      # noqa: E402

ensure_p300_mesh_descriptor()
import ttnn                                                              # noqa: E402
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3 import model as M                                       # noqa: E402

WEIGHTS = pathlib.Path(os.environ.get("BOLTZ_CACHE", pathlib.Path.home() / ".boltz")) / "rfd3" / "weights"
FIXTURE = opts.get("fixture", "perf/dsfix/fixtures/rfd3_R4.json")
CALLS = collections.defaultdict(list)


def _sig(t):
    return {"shape": [int(d) for d in t.shape], "padded": [int(d) for d in t.padded_shape], "dtype": str(t.dtype),
            "buffer": str(t.memory_config().buffer_type), "layout": str(t.memory_config().memory_layout)}


def main():
    dev = M.get_device()
    inner = M.Transition._weight_stationary

    def probe(self, x):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = inner(self, x)
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) * 1e3
        w12 = getattr(self, "_ws12", None)
        key = json.dumps({"x": _sig(x), "w12": None if w12 is None else _sig(w12), "w3": _sig(self.fc3_w),
                          "served": out is not None}, sort_keys=True)
        CALLS[key].append(ms)
        return out
    M.Transition._weight_stationary = probe

    specs = json.loads(pathlib.Path(FIXTURE).read_text())
    out_dir = "/tmp/gap_pws_probe_%s" % ARM
    shutil.rmtree(out_dir, ignore_errors=True)
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=str(WEIGHTS), from_pdb=True, num_timesteps=STEPS, seed=42,
                           num_designs=1, batch_size=1, fast=opts.get("fast") == "1", verbose=False)
    rec = {"arm": ARM, "fast": opts.get("fast") == "1", "steps": STEPS, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "host": os.uname().nodename, "arch": str(dev.arch()), "signatures": []}
    for key, ms in CALLS.items():
        rec["signatures"].append({**json.loads(key), "calls": len(ms), "ms_first": ms[0],
                                  "ms_median_after_first": statistics.median(ms[1:]) if len(ms) > 1 else None})
        print(json.dumps(rec["signatures"][-1]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
