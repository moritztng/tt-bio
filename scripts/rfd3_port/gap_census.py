#!/usr/bin/env python3
"""RFD3 per-step census on the page workload: where one design step's wall time goes.

    gap_census.py OUT.json [STEPS] [--fixture=perf/dsfix/fixtures/rfd3_R4.json]

Two folds of the page fixture (seed 42, 1 design, batch 1), after one discarded warmup:

  1. plain: the shipped path, timed only at the sampler boundary (the page number);
  2. census: every region below wrapped with `synchronize_device` at entry and exit, so a region's
     wall includes its own device work and its own host work, and the entry sync charges queued work
     to the caller. The census fold is slower than the plain one by the syncs it adds; the gap
     between the census total and the plain total is reported, not hidden.
  3. drain: only the step is synced, at entry and exit. `return_ms` is the time until the step's
     Python returns (host torch, op dispatch, and any sync the step itself does); `drain_ms` is the
     device work still queued after it returns. A step with a small drain is waiting on its host,
     which is what trace capture would remove; a large drain means the device is the bound.

Regions are class methods, patched on the class, so the census sees every instance. The residual
(step wall minus the sum of the top-level regions) is the host glue in RFD3DiffusionModule.__call__
and _process_ that no region owns.
"""
import collections
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time

sys.path.insert(0, os.getcwd())
import ttnn                                                               # noqa: E402
from tt_bio import tenstorrent as T                                       # noqa: E402
from tt_bio.rfd3 import design as rfd3_design                             # noqa: E402
from tt_bio.rfd3 import model as M                                        # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                               # noqa: E402

WEIGHTS = pathlib.Path(os.environ.get("BOLTZ_CACHE", pathlib.Path.home() / ".boltz")) / "rfd3" / "weights"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
OUT = pathlib.Path(args[0])
STEPS = int(args[1]) if len(args) > 1 else 200
FIXTURE = opts.get("fixture", "perf/dsfix/fixtures/rfd3_R4.json")

# (label, class, method, top-level?) -- top-level regions partition the step; nested ones split them.
REGIONS = [
    ("step", M.RFD3DiffusionModule, "__call__", False),
    ("encoder", M.LocalAtomTransformer, "__call__", True),
    ("token_encoder", M.DiffusionTokenEncoder, "run_device", True),
    ("dit", M.LocalTokenTransformer, "__call__", True),
    ("decoder", M.CompactStreamingDecoder, "run_full_device", True),
    ("seq_head", M.LinearSequenceHead, "__call__", True),
    ("downcast_c", M.RFD3DiffusionModule, "_downcast_c", True),
    ("downcast_q", M.RFD3DiffusionModule, "_downcast_q", True),
    ("atom_block", M.RFD3AtomBlock, "__call__", False),
    ("gca", M.GatedCrossAttention, "__call__", False),
]
ATTN_IDX = ("attn_indices", M, "_create_attention_indices", True)

acc = collections.defaultdict(float)
cnt = collections.Counter()
ON = [False]
DRAIN = [False, [], []]


def sync():
    ttnn.synchronize_device(T._device)


def wrap(label, owner, name):
    fn = getattr(owner, name)

    def timed(*a, **k):
        if DRAIN[0] and label == "step":
            sync()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            t1 = time.perf_counter()
            sync()
            DRAIN[1].append(t1 - t0)
            DRAIN[2].append(time.perf_counter() - t1)
            return out
        if not ON[0]:
            return fn(*a, **k)
        sync()
        t0 = time.perf_counter()
        out = fn(*a, **k)
        sync()
        acc[label] += time.perf_counter() - t0
        cnt[label] += 1
        return out
    setattr(owner, name, timed)


for label, owner, name, _top in REGIONS + [ATTN_IDX]:
    wrap(label, owner, name)

WALL = []
_sample = RFD3Sampler.sample


def _timed(self, *a, **k):
    t0 = time.perf_counter()
    out = _sample(self, *a, **k)
    WALL.append(time.perf_counter() - t0)
    return out


RFD3Sampler.sample = _timed


def fold(tag):
    specs = json.loads(pathlib.Path(FIXTURE).read_text())
    out_dir = "/tmp/gap_census_%s" % tag
    shutil.rmtree(out_dir, ignore_errors=True)
    WALL.clear()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=str(WEIGHTS), from_pdb=True,
                           num_timesteps=STEPS, seed=42, num_designs=1, batch_size=1, verbose=False)
    cifs = sorted(pathlib.Path(out_dir).glob("*.cif"))
    dig = "|".join(hashlib.sha256(c.read_bytes()).hexdigest()[:16] for c in cifs) or "NO CIF"
    return WALL[0], dig


warm_s, warm_d = fold("warm")
print("[census] warmup %.3f s %s" % (warm_s, warm_d), flush=True)
plain_s, plain_d = fold("plain")
print("[census] plain  %.3f s %s  (%.2f ms/step)" % (plain_s, plain_d, 1e3 * plain_s / STEPS), flush=True)
ON[0] = True
cen_s, cen_d = fold("census")
ON[0] = False
print("[census] census %.3f s %s" % (cen_s, cen_d), flush=True)
DRAIN[0] = True
drain_s, drain_d = fold("drain")
DRAIN[0] = False
ret_ms, drn_ms = (1e3 * sum(x) / max(len(x), 1) for x in DRAIN[1:])
print("[census] drain  %.3f s %s  step returns after %.2f ms, then drains %.2f ms" % (drain_s, drain_d, ret_ms, drn_ms),
      flush=True)

top = [r[0] for r in REGIONS + [ATTN_IDX] if r[3]]
step_total = acc["step"]
rows = {k: {"ms_per_step": 1e3 * acc[k] / STEPS, "calls_per_step": cnt[k] / STEPS,
            "share_of_step": acc[k] / step_total if step_total else None} for k in acc}
resid = step_total - sum(acc[k] for k in top)
rows["glue (step - top regions)"] = {"ms_per_step": 1e3 * resid / STEPS,
                                     "share_of_step": resid / step_total if step_total else None}
for k, v in sorted(rows.items(), key=lambda kv: -kv[1]["ms_per_step"]):
    print("  %-28s %8.2f ms/step  %5.1f %%  %s" % (k, v["ms_per_step"], 100 * (v["share_of_step"] or 0),
                                                  "calls/step %.1f" % v["calls_per_step"]
                                                  if "calls_per_step" in v else ""))
print("  sampler outside the module: %.2f ms/step" % (1e3 * (cen_s - step_total) / STEPS))
rec = {"fixture": FIXTURE, "steps": STEPS, "warm_s": warm_s, "plain_s": plain_s, "census_s": cen_s,
       "digests": {"warm": warm_d, "plain": plain_d, "census": cen_d},
       "sync_inflation_s": cen_s - plain_s, "regions": rows,
       "drain": {"fold_s": drain_s, "digest": drain_d, "steps": len(DRAIN[1]), "return_ms": ret_ms, "drain_ms": drn_ms},
       "card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rec, indent=1))
print("wrote", OUT)
